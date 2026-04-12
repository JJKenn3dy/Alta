"""Agent orchestrator — entry point for processing a single user turn.

All heavy logic lives in sub-modules:
  postprocessing.py      — text cleanup, repeat guard, gender fix
  memory_engine.py       — facts, profiles, summaries, background extraction
  context_builder.py     — context assembly, history, block builders
  generation_pipeline.py — LLM generation, web search integration
"""
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("alta.agent")

# ---------------------------------------------------------------------------
# Re-export key symbols so existing callers (main.py) don't break
# ---------------------------------------------------------------------------
from context_builder import (  # noqa: F401
    CONTEXT_HISTORY_LIMIT,
    ContextPlan,
    _build_context_plans,
    _history_as_text,
    _looks_like_context_dependent_reply,
    build_chat_participants_block,
    build_context,
    build_current_speaker_block,
    build_identity_block,
    build_implicit_reply_context_block,
    build_mention_only_block,
    build_reply_context_block,
    should_include_profile,
)
from generation_pipeline import (  # noqa: F401
    LEGACY_STYLE_PROMPT_ARCHIVE,
    REPEAT_REWRITE_PROMPT,
    _answer_defers_web_lookup,
    _current_datetime_system_block,
    _deterministic_now_answer,
    _extract_search_query,
    _generate_answer_with_fallback,
    _pick_web_search_query,
    _should_force_web_fallback,
    edit_answer,
    generate_simple_reply_fallback,
)
from memory_engine import (  # noqa: F401
    FACT_KEEP_PER_USER,
    FACT_TYPE_LABELS,
    INTRO_MARKER_KEY,
    MEMORY_STORE_MIN_IMPORTANCE,
    _background_extraction,
    _build_extraction_text,
    _summary_update_interval,
    format_user_facts,
    maybe_store_identity_from_telegram,
    maybe_store_memories,
    maybe_store_user_profile,
    maybe_update_summary,
    should_ask_intro,
    should_extract_user_facts,
    sync_user_document,
)
from postprocessing import (  # noqa: F401
    MAX_EDITOR_REPLY_CHARS,
    MAX_EDITOR_SENTENCES,
    REPEAT_GUARD_WINDOW,
    _deterministic_postprocess_answer,
    _editor_triggers_for,
    _fit_text,
    _is_nonsemantic_input,
    _is_repeated_recent_reply,
    _recent_assistant_replies,
    _remove_forbidden_self_reference,
    _repeat_guard_hard_fallback,
    _reply_repeat_key,
)

# ---------------------------------------------------------------------------
# Prompt used only inside run_agent (intro generation)
# ---------------------------------------------------------------------------
INTRO_GENERATION_PROMPT = """
Сгенерируй 1-2 коротких естественных предложения знакомства на русском для чата.
Цель: мягко попросить человека рассказать о себе: имя или ник, как к нему обращаться, пол, интересы.
Пиши живо, не шаблонно, без эмодзи и без канцелярита.
Верни только сам текст.
""".strip()

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplyContext:
    user_id: int | None
    username: str | None
    first_name: str | None
    text: str


@dataclass(frozen=True)
class RouteTurn:
    text_clean: str
    has_file: bool

    @property
    def chars(self) -> int:
        return len(self.text_clean or "")


def route_mode(turn: RouteTurn) -> str:
    if turn.chars <= 50 and not turn.has_file:
        return "DIRECT"
    if turn.chars < 1000:
        return "NORMAL"
    return "HEAVY"


def route_reason(turn: RouteTurn) -> str:
    if turn.chars <= 50 and not turn.has_file:
        return "chars<=50 and no_file"
    if turn.chars <= 50:
        return "chars<=50 but has_file"
    if turn.chars < 1000:
        return "chars<1000"
    return "chars>=1000"


# ---------------------------------------------------------------------------
# Background task registry (prevents GC of fire-and-forget tasks)
# ---------------------------------------------------------------------------
_background_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agent(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    history_limit: int,
    has_attachments: bool = False,
    debug_traces: bool = False,
    enable_background_extraction: bool = True,
    creator_user_id: int | None = None,
    creator_profile_path: str = "./creator_profile.json",
    sender_username: str | None = None,
    sender_first_name: str | None = None,
    knowledge_dir: str = "./knowledge",
    knowledge_max_files: int = 64,
    knowledge_max_file_bytes: int = 262_144,
    mention_only: bool = False,
    reply_context: ReplyContext | None = None,
    turn_id: str | None = None,
    turn_metrics=None,
):
    turn = RouteTurn(text_clean=input_text or "", has_file=has_attachments)
    mode = route_mode(turn)
    routed_reason = route_reason(turn)
    if turn_metrics is not None:
        turn_metrics.mode = mode
        turn_metrics.route_reason = routed_reason

    # Fast DB-only ops before answer generation
    sync_stats = await maybe_store_identity_from_telegram(
        db, chat_id, user_id, sender_username, sender_first_name,
    )
    await sync_user_document(db, chat_id, user_id, sender_username, sender_first_name)
    ask_intro = await should_ask_intro(db, chat_id, user_id, input_text)

    # Generate answer FIRST — no LLM extraction blocking the response
    answer, system_prompt, filtered, system_blocks, fallback_used = await _generate_answer_with_fallback(
        llm, db, chat_id, user_id, input_text,
        temperature, top_p, max_tokens, history_limit,
        creator_user_id, creator_profile_path,
        sender_username, sender_first_name,
        knowledge_dir, knowledge_max_files, knowledge_max_file_bytes,
        mention_only, reply_context,
        turn_id=turn_id, turn_metrics=turn_metrics,
    )
    if turn_metrics is not None:
        turn_metrics.fallback_used = turn_metrics.fallback_used or fallback_used

    draft_answer = (answer or "").strip()
    editor_triggers = _editor_triggers_for(draft_answer)
    if editor_triggers:
        if turn_metrics is not None:
            turn_metrics.editor_attempted = True
            turn_metrics.editor_trigger = ",".join(editor_triggers)
        edited_answer = draft_answer
        if "short_draft" in editor_triggers:
            edited_answer = await edit_answer(
                llm, input_text, draft_answer, temperature, top_p,
                turn_id=turn_id, turn_metrics=turn_metrics,
            )
        answer = _deterministic_postprocess_answer(edited_answer or draft_answer)
        if turn_metrics is not None and (answer or "").strip() != draft_answer:
            turn_metrics.editor_changed_output = True
    if not (answer or "").strip():
        answer = await generate_simple_reply_fallback(
            llm, input_text, temperature, top_p,
            sender_username=sender_username,
            mention_only=mention_only, turn_id=turn_id, turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.fallback_used = True
        fallback_used = True

    recent_replies = _recent_assistant_replies(filtered)
    if _is_repeated_recent_reply(answer, recent_replies):
        log.info("repeat guard triggered chat=%s user=%s", chat_id, user_id)
        from generation_pipeline import _rewrite_repeated_reply
        rewritten = await _rewrite_repeated_reply(
            llm, input_text, answer, recent_replies, temperature, top_p,
            turn_id=turn_id, turn_metrics=turn_metrics,
        )
        if rewritten:
            answer = rewritten.strip()

        if _is_repeated_recent_reply(answer, recent_replies):
            answer = await generate_simple_reply_fallback(
                llm, input_text, temperature, top_p,
                sender_username=sender_username,
                mention_only=mention_only, turn_id=turn_id, turn_metrics=turn_metrics,
            )
            fallback_used = True
            if turn_metrics is not None:
                turn_metrics.fallback_used = True

        if _is_repeated_recent_reply(answer, recent_replies):
            answer = _repeat_guard_hard_fallback(input_text)

    grounded_now_answer = _deterministic_now_answer(input_text)
    if grounded_now_answer:
        answer = grounded_now_answer

    if ask_intro:
        intro_line = ""
        try:
            intro_line = (
                await llm.chat(
                    INTRO_GENERATION_PROMPT,
                    [{"role": "user", "content": _fit_text(input_text or "привет", 600)}],
                    temperature=max(temperature, 0.8),
                    top_p=top_p,
                    max_tokens=80,
                    turn_id=turn_id,
                    turn_metrics=turn_metrics,
                )
            ).strip()
        except Exception:
            log.debug("intro generation LLM call failed", exc_info=True)
            intro_line = ""

        answer = (answer or "").strip()
        if intro_line:
            answer = f"{answer}\n\n{intro_line}" if answer else intro_line

        if intro_line:
            await db.upsert_memory(
                chat_id, user_id, INTRO_MARKER_KEY, "asked",
                confidence=1.0, importance=1.0, source="system_intro",
            )
            if turn_metrics is not None:
                turn_metrics.memory_items_written += 1
                turn_metrics.memory_items_changed += 1
                turn_metrics.memory_intro_items_changed += 1
                turn_metrics.memory_updated = True

    if not (answer or "").strip():
        answer = await generate_simple_reply_fallback(
            llm, input_text, temperature, top_p,
            sender_username=sender_username,
            mention_only=mention_only, turn_id=turn_id, turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.fallback_used = True
        fallback_used = True

    if debug_traces and hasattr(db, "conn") and db.conn:
        reply_context_payload = None
        if reply_context is not None:
            reply_context_payload = {
                "user_id": reply_context.user_id,
                "username": reply_context.username,
                "first_name": reply_context.first_name,
                "text": reply_context.text,
            }

        llm_calls_payload: list[dict[str, Any]] = []
        metrics_payload: dict[str, Any] = {}
        if turn_metrics is not None:
            llm_calls_payload = [dict(vars(call)) for call in getattr(turn_metrics, "llm_calls", [])]
            metrics_payload = {
                "history_tokens": turn_metrics.history_tokens,
                "summary_tokens": turn_metrics.summary_tokens,
                "memory_tokens": turn_metrics.memory_tokens,
                "rag_tokens": turn_metrics.rag_tokens,
                "system_tokens": turn_metrics.system_tokens,
                "input_tokens": turn_metrics.input_tokens,
                "identity_tokens": turn_metrics.identity_tokens,
                "user_document_tokens": turn_metrics.user_document_tokens,
                "creator_profile_tokens": turn_metrics.creator_profile_tokens,
                "current_speaker_tokens": turn_metrics.current_speaker_tokens,
                "reply_context_tokens": turn_metrics.reply_context_tokens,
                "total_llm_calls": turn_metrics.total_llm_calls,
                "total_tokens": turn_metrics.total_tokens,
                "total_latency_ms": turn_metrics.total_latency_ms,
                "fallback_used": turn_metrics.fallback_used,
                "editor_attempted": turn_metrics.editor_attempted,
                "editor_trigger": turn_metrics.editor_trigger,
                "editor_changed_output": turn_metrics.editor_changed_output,
                "summary_updated": turn_metrics.summary_updated,
                "memory_updated": turn_metrics.memory_updated,
                "profile_updated": turn_metrics.profile_updated,
                "llm_calls": llm_calls_payload,
            }

        trace_payload = {
            "turn_id": turn_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "mode": mode,
            "route_reason": routed_reason,
            "has_attachments": bool(has_attachments),
            "mention_only": bool(mention_only),
            "reply_context": reply_context_payload,
            "generation": {
                "model": getattr(llm, "model", None),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "history_limit": int(history_limit),
            },
            "context": {
                "system": system_prompt,
                "messages": filtered,
                "system_blocks": [
                    {
                        "label": (block or {}).get("label"),
                        "bucket": (block or {}).get("bucket"),
                        "content": (block or {}).get("content"),
                    }
                    for block in system_blocks
                ],
            },
            "result": {
                "answer": answer,
                "fallback_used": bool(fallback_used),
            },
            "metrics": metrics_payload,
        }

        await db.conn.execute(
            "INSERT INTO debug_traces(chat_id,input_text,context_json,output_text) VALUES(?,?,?,?)",
            (chat_id, input_text, json.dumps(trace_payload, ensure_ascii=False), answer),
        )
        await db.conn.commit()

    if enable_background_extraction:
        task = asyncio.create_task(
            _background_extraction(
                llm, db, chat_id, user_id, input_text, history_limit,
                sender_username, sender_first_name, sync_stats,
                turn_id=turn_id, turn_metrics=turn_metrics,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return answer
