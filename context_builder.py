"""Context builder: assembles system prompt blocks and message history.

Builds the structured context (system_blocks + filtered history) that
generation_pipeline feeds to the LLM on each turn.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

log = logging.getLogger("alta.state")

from file_context import build_file_context_block, retrieve_context_snippets, sync_context_dir
from memory_engine import (
    FACT_CONTEXT_MIN_IMPORTANCE,
    FACT_TYPE_LABELS,
    RELATIONSHIP_NOTES_CHAR_LIMIT,
    PROFILE_FIELD_CHAR_LIMIT,
    USER_DOCUMENT_CHAR_LIMIT,
    format_user_facts,
    should_include_user_document,
    _clean_identity_profile_text,
    _clean_profile_text,
    _trusted_sender_identity,
    _profile_text_or_list,
)
from postprocessing import _fit_text, _clip_text
from telemetry import approx_messages_tokens, approx_text_tokens

# ---------------------------------------------------------------------------
# Context budget constants
# ---------------------------------------------------------------------------
CONTEXT_HISTORY_LIMIT = 15
SUMMARY_CONTEXT_CHAR_LIMIT = 700
HISTORY_MESSAGE_CHAR_LIMIT = 1200
DEFAULT_CONTEXT_CHAR_BUDGET = 12_000
COMPACT_CONTEXT_CHAR_BUDGET = 7_000
MINIMAL_CONTEXT_CHAR_BUDGET = 3_500
EMERGENCY_CONTEXT_CHAR_BUDGET = 1_500
DEFAULT_FILE_CONTEXT_CHUNKS = 3
COMPACT_FILE_CONTEXT_CHUNKS = 2
MINIMAL_FILE_CONTEXT_CHUNKS = 1
DEFAULT_FILE_CONTEXT_CHARS = 1_800
COMPACT_FILE_CONTEXT_CHARS = 1_000
MINIMAL_FILE_CONTEXT_CHARS = 500
RAG_MIN_SNIPPET_SCORE = 3.0
MAX_CHAT_PARTICIPANTS = 8
FLIRT_POLICY_EXCEPTION_USERNAME = "jjkenn9dy"

# Patterns
USER_PREFIX_ID_PATTERN = re.compile(r"\[[^\]]*\bid=(\d+)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Alta internal state — signal detection patterns
# ---------------------------------------------------------------------------
_SIGNAL_AGGRESSION = (
    re.compile(
        r"\b(иди\s+нахуй|иди\s+нафиг|пошла?\s+нахуй|заткнись|ты\s+тупая?"
        r"|ублюдок|мразь|тварь|бесишь|надоела)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(shut\s*up|fuck\s+you|go\s+fuck|you\s+suck|fuck\s+off)\b",
        re.IGNORECASE,
    ),
)
# Ambiguous words — часто используются по-дружески в русском чате
_SIGNAL_AGGRESSION_WEAK = (
    re.compile(r"\b(дура\b|дебил\b|идиот\b)\b", re.IGNORECASE),
)
_SIGNAL_GRATITUDE = (
    re.compile(r"\b(спасибо|спс|благодарю|thanks?|thank\s+you)\b", re.IGNORECASE),
)
_SIGNAL_CREATIVE = (
    re.compile(
        r"\b(расскажи|объясни|что\s+такое|почему|как\s+работает|философ|теория|идея|концепция)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(придумай|представь|если\s+бы|а\s+что\s+если|гипотетически|допустим)\b",
        re.IGNORECASE,
    ),
)

USER_DOCUMENT_TRIGGER_PATTERNS = (
    re.compile(r"\b(кто я|что ты обо мне|что знаешь обо мне|что помнишь обо мне)\b", re.IGNORECASE),
    re.compile(r"\b(как меня зовут|как ко мне обращаться|обращайся ко мне)\b", re.IGNORECASE),
    re.compile(r"\b(мой|моя|мне|обо мне|про меня|меня)\b", re.IGNORECASE),
    re.compile(r"\b(интерес|люблю|предпочитаю|нравит)\b", re.IGNORECASE),
)

CREATOR_PROFILE_PATTERNS = (
    re.compile(r"\bкто тебя (создал|сделал)\b"),
    re.compile(r"\bкто твой (создатель|владелец|хозяин|owner|creator)\b"),
    re.compile(r"\bтвой (создатель|владелец|хозяин|owner|creator)\b"),
    re.compile(r"\bзахар\b.*\b(создал|сделал|владелец|хозяин)\b"),
    re.compile(r"\b(создал|сделал|владелец|хозяин)\b.*\bзахар\b"),
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextPlan:
    name: str
    history_messages: int
    history_chars: int
    summary_chars: int
    memory_lines: int
    file_chunks: int
    file_chars: int
    max_tokens: int
    include_creator_profile: bool
    include_participants: bool = True


# ---------------------------------------------------------------------------
# Context plan builder
# ---------------------------------------------------------------------------

def _build_context_plans(max_tokens: int, history_limit: int) -> list[ContextPlan]:
    safe_max_tokens = max(48, int(max_tokens))
    return [
        ContextPlan(
            name="full",
            history_messages=min(history_limit, CONTEXT_HISTORY_LIMIT),
            history_chars=DEFAULT_CONTEXT_CHAR_BUDGET,
            summary_chars=SUMMARY_CONTEXT_CHAR_LIMIT,
            memory_lines=16,
            file_chunks=DEFAULT_FILE_CONTEXT_CHUNKS,
            file_chars=DEFAULT_FILE_CONTEXT_CHARS,
            max_tokens=safe_max_tokens,
            include_creator_profile=True,
            include_participants=False,
        ),
        ContextPlan(
            name="compact",
            history_messages=min(history_limit, 20),
            history_chars=COMPACT_CONTEXT_CHAR_BUDGET,
            summary_chars=min(450, SUMMARY_CONTEXT_CHAR_LIMIT),
            memory_lines=4,
            file_chunks=COMPACT_FILE_CONTEXT_CHUNKS,
            file_chars=COMPACT_FILE_CONTEXT_CHARS,
            max_tokens=min(safe_max_tokens, 280),
            include_creator_profile=True,
            include_participants=False,
        ),
        ContextPlan(
            name="minimal",
            history_messages=min(history_limit, 10),
            history_chars=MINIMAL_CONTEXT_CHAR_BUDGET,
            summary_chars=min(300, SUMMARY_CONTEXT_CHAR_LIMIT),
            memory_lines=2,
            file_chunks=MINIMAL_FILE_CONTEXT_CHUNKS,
            file_chars=MINIMAL_FILE_CONTEXT_CHARS,
            max_tokens=min(safe_max_tokens, 180),
            include_creator_profile=True,
            include_participants=False,
        ),
        ContextPlan(
            name="emergency",
            history_messages=min(history_limit, 2),
            history_chars=EMERGENCY_CONTEXT_CHAR_BUDGET,
            summary_chars=120,
            memory_lines=0,
            file_chunks=0,
            file_chars=0,
            max_tokens=min(safe_max_tokens, 120),
            include_creator_profile=False,
            include_participants=False,
        ),
    ]


# ---------------------------------------------------------------------------
# History utilities
# ---------------------------------------------------------------------------

TIME_GAP_THRESHOLD_MINUTES = 30


def _history_line(role: str, content: str) -> str:
    return f"{role}: {content}"


def _parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def _insert_time_gaps(messages: list[dict]) -> list[dict]:
    if len(messages) < 2:
        return messages
    result: list[dict] = [messages[0]]
    for i in range(1, len(messages)):
        prev_ts = _parse_ts(messages[i - 1].get("ts", ""))
        curr_ts = _parse_ts(messages[i].get("ts", ""))
        if prev_ts and curr_ts:
            gap_minutes = (curr_ts - prev_ts).total_seconds() / 60
            if gap_minutes >= TIME_GAP_THRESHOLD_MINUTES:
                if gap_minutes >= 1440:
                    gap_text = f"[... прошло {int(gap_minutes // 1440)} дн. ...]"
                elif gap_minutes >= 60:
                    gap_text = f"[... прошло {int(gap_minutes // 60)} ч. ...]"
                else:
                    gap_text = f"[... прошло {int(gap_minutes)} мин. ...]"
                result.append({"role": "system", "content": gap_text})
        result.append(messages[i])
    return result


def _shrink_message(message: dict, limit: int = HISTORY_MESSAGE_CHAR_LIMIT) -> dict:
    result: dict = {
        "role": message["role"],
        "content": _fit_text(message.get("content", ""), limit),
    }
    if "target_user_id" in message:
        result["target_user_id"] = message["target_user_id"]
    return result


def _take_history_with_budget(history: list[dict], max_messages: int, max_chars: int) -> list[dict]:
    if not history or max_messages <= 0 or max_chars <= 0:
        return []

    selected: list[dict] = []
    skipped_count = 0
    chars_used = 0
    recent = list(reversed(history[-max_messages:]))

    for index, message in enumerate(recent):
        remaining = max_chars - chars_used
        if remaining <= 80:
            skipped_count += len(recent) - index
            break

        per_message_limit = min(HISTORY_MESSAGE_CHAR_LIMIT, max(120, remaining - 24))
        shrunk = _shrink_message(message, per_message_limit)
        cost = len(shrunk["content"]) + 24
        if chars_used + cost > max_chars:
            skipped_count += 1
            continue

        selected.append(shrunk)
        chars_used += cost

    result = list(reversed(selected))

    if skipped_count > 0 and result:
        gap_note = {
            "role": "system",
            "content": f"[... пропущено {skipped_count} сообщений из-за лимита контекста ...]",
        }
        result.insert(0, gap_note)

    return result


def _merge_histories(primary: list[dict], secondary: list[dict], limit: int, char_budget: int) -> list[dict]:
    merged: list[dict] = []
    seen_ids: set[int] = set()

    for item in primary + secondary:
        message_id = int(item.get("id") or 0)
        if message_id and message_id in seen_ids:
            continue
        if message_id:
            seen_ids.add(message_id)
        merged.append(item)

    merged.sort(key=lambda item: int(item.get("id") or 0))
    trimmed = _take_history_with_budget(merged, limit, char_budget)
    return _insert_time_gaps(trimmed)


def _history_as_text(history: list[dict], max_chars: int) -> str:
    selected = _take_history_with_budget(history, len(history), max_chars)
    return "\n".join(_history_line(item["role"], item["content"]) for item in selected)


# ---------------------------------------------------------------------------
# Reply context blocks
# ---------------------------------------------------------------------------

def _looks_like_context_dependent_reply(text: str) -> bool:
    value = " ".join(str(text or "").strip().lower().split())
    if not value:
        return False
    if len(value) > 48:
        return False
    words = value.split()
    if len(words) > 6:
        return False
    return True


def _format_reply_author(reply_context) -> str:
    if reply_context.username:
        return f"@{reply_context.username}"
    if reply_context.first_name:
        return reply_context.first_name
    if reply_context.user_id is not None:
        return f"id={reply_context.user_id}"
    return "unknown"


def build_reply_context_block(reply_context) -> str:
    if not reply_context:
        return ""
    reply_text = _fit_text(reply_context.text, 400)
    if not reply_text:
        return ""
    return (
        "Текущее сообщение является ответом на конкретную реплику в чате.\n"
        f"Автор исходной реплики: {_format_reply_author(reply_context)}\n"
        f"Текст исходной реплики: {reply_text}\n"
        "Учитывай это при интерпретации коротких и зависимых от контекста ответов."
    ).strip()


def build_implicit_reply_context_block(history: list[dict], input_text: str) -> str:
    if not _looks_like_context_dependent_reply(input_text):
        return ""
    if len(history) < 2:
        return ""

    last_message = history[-1]
    prev_message = history[-2]
    if last_message.get("role") != "user":
        return ""
    if prev_message.get("role") != "assistant":
        return ""

    prev_bot_text = _fit_text(prev_message.get("content", ""), 300)
    if not prev_bot_text:
        return ""

    prev_user_text = ""
    for i in range(len(history) - 3, -1, -1):
        if history[i].get("role") == "user":
            prev_user_text = _fit_text(history[i].get("content", ""), 200)
            break

    parts = ["Текущее сообщение короткое и зависит от контекста предыдущих реплик."]
    if prev_user_text:
        parts.append(f"Предыдущее сообщение пользователя: {prev_user_text}")
    parts.append(f"Предыдущая реплика бота: {prev_bot_text}")
    parts.append(
        "Считай текущее сообщение прежде всего ответом, уточнением или продолжением этого диалога, "
        "а не новой отдельной темой."
    )
    return "\n".join(parts)


def build_mention_only_block(mention_only: bool) -> str:
    if not mention_only:
        return ""
    return (
        "Пользователь обратился к тебе по имени, но не написал содержательный текст.\n"
        "Ответь коротко и естественно, как на оклик или привлечение внимания.\n"
        "Не воспринимай это как слово 'ping' и не отвечай 'понг'."
    ).strip()


# ---------------------------------------------------------------------------
# Alta internal state logic (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _hours_since_state(ts_str: str | None) -> float:
    """Hours elapsed since the given SQLite datetime string.

    SQLite datetime('now') returns UTC, so we compare against utcnow()
    to avoid spurious timezone offsets on non-UTC systems.
    """
    if not ts_str:
        return 0.0
    last_dt = _parse_ts(ts_str)
    if not last_dt:
        return 0.0
    return max(0.0, (datetime.utcnow() - last_dt).total_seconds() / 3600.0)


def apply_time_decay_chat(state: dict, hours: float) -> dict:
    """Decay chat-level state toward neutral baseline over elapsed hours."""
    irr = _clamp(float(state.get("irritation", 0.0)) * (0.65 ** max(hours, 0.0)))
    sb = float(state.get("social_battery", 0.8))
    en = float(state.get("energy", 0.7))
    play = float(state.get("playfulness", 0.5))
    if hours > 0:
        # Social battery and energy recover toward baseline
        sb = _clamp(sb + min(0.3, 0.12 * hours) * (1.0 if sb < 0.8 else -0.3))
        en = _clamp(en + min(0.2, 0.08 * hours) * (1.0 if en < 0.7 else -0.5))
        # Playfulness drifts toward 0.5 (same speed as warmth)
        play = _clamp(play + (0.5 - play) * min(0.3, 0.04 * hours))
    return {
        "energy": en,
        "irritation": irr,
        "playfulness": play,
        "social_battery": sb,
    }


def apply_time_decay_relation(rel: dict, hours: float) -> dict:
    """Decay user-relation state toward neutral baseline over elapsed hours."""
    ann = _clamp(float(rel.get("annoyance", 0.0)) * (0.60 ** max(hours, 0.0)))
    cau = _clamp(float(rel.get("caution", 0.0)) * (0.80 ** max(hours, 0.0)))
    warmth = float(rel.get("warmth", 0.5))
    trust = float(rel.get("trust", 0.5))
    if hours > 0:
        # Warmth drifts slowly toward 0.5
        warmth = _clamp(warmth + (0.5 - warmth) * min(0.3, 0.04 * hours))
        # Trust drifts slowly toward 0.5 (half the speed of warmth)
        trust = _clamp(trust + (0.5 - trust) * min(0.15, 0.02 * hours))
    return {
        "trust": trust,
        "warmth": warmth,
        "caution": cau,
        "annoyance": ann,
        "interaction_count": int(rel.get("interaction_count", 0)),
    }


def compute_state_updates(
    input_text: str,
    chat_state: dict,
    user_relation: dict,
) -> tuple[dict, dict, str | None, float]:
    """Deterministic state update from incoming message.

    Returns (new_chat_state, new_user_relation, reaction_kind | None, reaction_strength).
    Returned dicts never include 'updated_at'.
    """
    text = str(input_text or "")
    aggression = 1.0 if any(p.search(text) for p in _SIGNAL_AGGRESSION) else 0.0
    aggression_weak = 0.4 if (not aggression and any(p.search(text) for p in _SIGNAL_AGGRESSION_WEAK)) else 0.0
    gratitude = 1.0 if any(p.search(text) for p in _SIGNAL_GRATITUDE) else 0.0
    creative = 1.0 if any(p.search(text) for p in _SIGNAL_CREATIVE) else 0.0
    has_question = "?" in text

    # Scale aggression effect down for trusted users — "дура" from a friend hurts less
    trust = float(user_relation.get("trust", 0.5))
    trust_discount = 1.0 - trust * 0.5  # 1.0 at trust=0, 0.5 at trust=1.0
    agg = (aggression + aggression_weak) * trust_discount

    # Chat state
    new_chat = {
        "energy": _clamp(
            float(chat_state.get("energy", 0.7)) - 0.003 - agg * 0.03 + creative * 0.01
        ),
        "irritation": _clamp(
            float(chat_state.get("irritation", 0.0)) + agg * 0.12 - 0.015
        ),
        "playfulness": min(0.9, _clamp(
            float(chat_state.get("playfulness", 0.5))
            + creative * 0.06
            - agg * 0.05
        )),
        "social_battery": _clamp(
            float(chat_state.get("social_battery", 0.8))
            - 0.005
            - agg * 0.06
        ),
    }

    # User relation
    prev_annoyance = float(user_relation.get("annoyance", 0.0))
    new_rel = {
        "trust": min(0.85, _clamp(
            trust
            + (0.02 if not aggression else -0.03 * trust_discount)
            + gratitude * 0.015
        )),
        "warmth": _clamp(
            float(user_relation.get("warmth", 0.5))
            + gratitude * 0.05
            - agg * 0.05
        ),
        "caution": min(0.75, _clamp(float(user_relation.get("caution", 0.0)) + agg * 0.05)),
        "annoyance": _clamp(prev_annoyance + agg * 0.15 - 0.01),
        "interaction_count": int(user_relation.get("interaction_count", 0)) + 1,
    }

    # Determine reaction
    reaction_kind: str | None = None
    reaction_strength = 0.0
    new_annoyance = new_rel["annoyance"]
    if aggression and new_annoyance > 0.35:
        reaction_kind = "annoyed"
        reaction_strength = min(1.0, new_annoyance + 0.1)
    elif creative or (has_question and not aggression):
        reaction_kind = "curious"
        reaction_strength = 0.5 + creative * 0.1
    elif gratitude:
        reaction_kind = "amused"
        reaction_strength = 0.4

    return new_chat, new_rel, reaction_kind, reaction_strength


def build_alta_state_block(
    chat_state: dict,
    user_relation: dict,
    active_reaction: dict | None,
) -> str:
    """Format Alta's internal state as a compact system block.

    Returns empty string when state is near-neutral (no token waste).
    """
    energy = float(chat_state.get("energy", 0.7))
    irritation = float(chat_state.get("irritation", 0.0))
    playfulness = float(chat_state.get("playfulness", 0.5))
    social_battery = float(chat_state.get("social_battery", 0.8))
    trust = float(user_relation.get("trust", 0.5))
    warmth = float(user_relation.get("warmth", 0.5))
    annoyance = float(user_relation.get("annoyance", 0.0))

    noteworthy = (
        energy < 0.4
        or irritation > 0.2
        or playfulness > 0.72
        or social_battery < 0.35
        or trust < 0.3
        or trust > 0.75
        or warmth < 0.3
        or warmth > 0.75
        or annoyance > 0.25
        or active_reaction is not None
    )
    if not noteworthy:
        return ""

    lines = ["[Твоё состояние — влияй на тон, не упоминай вслух]"]

    chat_parts: list[str] = []
    if energy < 0.35:
        chat_parts.append("устала")
    elif energy < 0.5:
        chat_parts.append("чуть устала")
    if irritation > 0.5:
        chat_parts.append("раздражена")
    elif irritation > 0.2:
        chat_parts.append("немного раздражена")
    if social_battery < 0.25:
        chat_parts.append("соц. батарея пустая")
    elif social_battery < 0.4:
        chat_parts.append("соц. батарея низкая")
    if playfulness > 0.72 and irritation < 0.3:
        chat_parts.append("игривое настроение")
    if chat_parts:
        lines.append("Фон: " + ", ".join(chat_parts))

    user_parts: list[str] = []
    if annoyance > 0.55:
        user_parts.append("сильно раздражает")
    elif annoyance > 0.25:
        user_parts.append("немного раздражает")
    if trust < 0.25:
        user_parts.append("почти нет доверия")
    elif trust < 0.35:
        user_parts.append("мало доверия")
    elif trust > 0.75:
        user_parts.append("высокое доверие")
    if warmth < 0.25:
        user_parts.append("холодно")
    elif warmth > 0.75:
        user_parts.append("тепло")
    if user_parts:
        lines.append("К собеседнику: " + ", ".join(user_parts))

    if active_reaction:
        kind = str(active_reaction.get("reaction_kind", ""))
        strength = float(active_reaction.get("reaction_strength", 0.5))
        _REACTION_LABELS = {
            "annoyed": "раздражена",
            "curious": "любопытна",
            "amused": "смешит",
            "concerned": "тревожно",
            "embarrassed": "неловко",
        }
        label = _REACTION_LABELS.get(kind, kind)
        prefix = "сильно " if strength > 0.7 else ""
        lines.append(f"Реакция: {prefix}{label}")

    tone_parts: list[str] = []
    if irritation > 0.5 or annoyance > 0.55:
        tone_parts.append("суше и короче")
    if social_battery < 0.3:
        tone_parts.append("без охоты развивать тему")
    if warmth > 0.75 and annoyance < 0.2:
        tone_parts.append("чуть теплее обычного")
    if playfulness > 0.72 and irritation < 0.3:
        tone_parts.append("можно немного поиграть")
    if tone_parts:
        lines.append("→ " + ", ".join(tone_parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System block builders
# ---------------------------------------------------------------------------

def _normalize_username(username: str | None) -> str:
    return str(username or "").strip().lstrip("@").lower()


def build_flirt_policy_block(sender_username: str | None) -> str:
    if _normalize_username(sender_username) == FLIRT_POLICY_EXCEPTION_USERNAME:
        return (
            "Политика флирта для текущего собеседника:\n"
            "- ИСКЛЮЧЕНИЕ: @JJKenn9dy. Для этого пользователя ответный флирт допустим по ситуации.\n"
            "- Даже в этом случае не уходи в эротические подробности и не разгоняй интимный отыгрыш."
        )
    return (
        "Политика флирта для текущего собеседника:\n"
        "- Флирт, романтические заигрывания и сексуализированные намёки в ответе запрещены.\n"
        "- На флирт/интим-провокации отвечай нейтрально, с иронией или коротким отказом без вовлечения."
    )


def build_identity_block(
    user_profile: dict,
    sender_username: str | None,
    sender_first_name: str | None,
) -> str:
    preferred_name, nickname = _trusted_sender_identity(user_profile, sender_username, sender_first_name)
    gender = _clip_text(user_profile.get("gender"), PROFILE_FIELD_CHAR_LIMIT)
    interests = _clip_text(user_profile.get("interests"), PROFILE_FIELD_CHAR_LIMIT)
    communication_style = _clip_text(user_profile.get("communication_style"), PROFILE_FIELD_CHAR_LIMIT)
    response_preference = _clip_text(user_profile.get("response_preference"), PROFILE_FIELD_CHAR_LIMIT)
    relationship_notes = _clip_text(user_profile.get("relationship_notes"), RELATIONSHIP_NOTES_CHAR_LIMIT)

    lines = []
    if preferred_name and preferred_name.strip() != (sender_first_name or "").strip():
        lines.append(f"- Предпочтительное обращение: {preferred_name}")
    if gender:
        lines.append(f"- Пол: {gender}")
    if interests:
        lines.append(f"- Интересы: {interests}")
    if communication_style:
        lines.append(f"- Как человек общается: {communication_style}")
    if response_preference:
        lines.append(f"- Как ему лучше отвечать: {response_preference}")
    if relationship_notes:
        lines.append(f"- Заметка о динамике общения: {relationship_notes}")

    if not lines:
        return ""

    return (
        "Профиль текущего собеседника:\n"
        + "\n".join(lines)
        + "\nУчитывай это в обращениях, интонации и длине ответа. Не проговаривай этот блок пользователю."
    )


def build_current_speaker_block(
    user_id: int,
    sender_username: str | None,
    sender_first_name: str | None,
    creator_user_id: int | None = None,
) -> str:
    lines = [
        "Текущий собеседник для этого ответа:",
        f"- user_id: {user_id}",
    ]
    if sender_username:
        lines.append(f"- username: @{sender_username.lstrip('@')}")
    if sender_first_name:
        lines.append(f"- имя в Telegram: {sender_first_name.strip()}")

    is_creator = creator_user_id is not None and user_id == creator_user_id
    if is_creator:
        lines.append(
            "- Это твой создатель Захар (свой, близкий). Характер НЕ теряй: не "
            "подлизывайся, не сюсюкай, держи дерзость. НЕ лепи обращение/кличку в "
            "каждую реплику — общайся по-простому. «Теплее» не значит пресно."
        )
    else:
        lines.append(
            "- Это НЕ твой создатель. Обращения «Шеф», «Захар» и любые ники/клички "
            "из истории к нему НЕ относятся — не вешай их на этого человека."
        )
    lines.extend(
        [
            "- Отвечай именно этому человеку.",
            "- Никогда не переноси имя, ник, пол, интересы и факты другого участника на текущего собеседника.",
            "- Если спрашивает «кто я / помнишь меня / как меня зовут» — отвечай ТОЛЬКО по "
            "данным этого блока и его профилю. Нет данных — честно скажи что не знаешь "
            "(«хз, ты кто вообще»), не выдумывай и не приписывай чужой ник.",
            "- Имя используй только если без него непонятно к кому обращаешься. Не начинай с него каждую реплику.",
        ]
    )
    return "\n".join(lines)


def build_chat_participants_block(
    participants: list[dict],
    current_user_id: int,
) -> str:
    others = [p for p in participants if p.get("user_id") != current_user_id]
    if not others:
        return ""

    lines = ["Другие недавно активные участники этого чата:"]
    for p in others:
        name_parts = []
        display = (p.get("display_name") or "").strip()
        username = (p.get("username") or "").strip()
        if display:
            name_parts.append(display)
        if username:
            name_parts.append(f"@{username}")
        identifier = " / ".join(name_parts) if name_parts else f"id={p['user_id']}"

        detail_parts = []
        gender = (p.get("gender") or "").strip()
        if gender:
            detail_parts.append(f"пол: {gender}")
        interests = (p.get("interests") or "").strip()
        if interests:
            detail_parts.append(f"интересы: {_clip_text(interests, 80)}")

        detail = "; ".join(detail_parts)
        if detail:
            lines.append(f"- {identifier} ({detail})")
        else:
            lines.append(f"- {identifier}")

    lines.append("Не путай факты этих людей с фактами текущего собеседника.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Participants and miscellaneous helpers
# ---------------------------------------------------------------------------

def _extract_prefixed_user_id(content: str) -> int | None:
    match = USER_PREFIX_ID_PATTERN.search(str(content or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _should_include_participants_context(
    input_text: str,
    prompt_history: list[dict],
    current_user_id: int,
) -> bool:
    normalized = " ".join((input_text or "").lower().split())
    if not normalized:
        return False
    trivial_turns = {
        "привет", "ку", "ок", "оке", "ага", "угу", "ясно", "пон", "понял",
        "поняла", "спс", "спасибо", "hi", "hello",
    }
    if normalized in trivial_turns:
        return False

    other_user_ids: set[int] = set()
    for item in prompt_history:
        if item.get("role") != "user":
            continue
        uid = _extract_prefixed_user_id(item.get("content", ""))
        if uid is None or uid == current_user_id:
            continue
        other_user_ids.add(uid)

    if not other_user_ids:
        return False

    if _looks_like_context_dependent_reply(input_text):
        return True

    ambiguous_markers = (
        "он", "она", "его", "ее", "её", "ему", "ей", "они", "их",
        "кто", "кому", "чей", "чья", "чьи",
    )
    if any(marker in normalized for marker in ambiguous_markers):
        return True
    if "@" in normalized:
        return True
    return False


def _max_context_snippet_score(snippets: list[dict]) -> float:
    best = 0.0
    for snippet in snippets:
        try:
            score = float((snippet or {}).get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if score > best:
            best = score
    return best


def _is_creator_profile_query(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    return any(pattern.search(normalized) for pattern in CREATOR_PROFILE_PATTERNS)


def should_include_profile(input_text: str, user_id: int, creator_user_id: int | None) -> bool:
    if creator_user_id is not None and user_id == creator_user_id:
        return True
    return _is_creator_profile_query(input_text)


# ---------------------------------------------------------------------------
# Main context assembly
# ---------------------------------------------------------------------------

async def build_context(
    db,
    chat_id: int,
    user_id: int,
    history_limit: int,
    sender_username: str | None,
    sender_first_name: str | None,
    plan: ContextPlan,
    knowledge_dir: str,
    knowledge_max_files: int,
    knowledge_max_file_bytes: int,
    input_text: str,
    mention_only: bool = False,
    reply_context=None,
    creator_user_id: int | None = None,
    *,
    user_profile: dict | None = None,
    user_facts: list | None = None,
    user_document: dict | None = None,
) -> tuple[list[dict[str, str]], list[dict], dict[str, int], dict]:
    summary = await db.get_summary(chat_id)
    speaker_summary = await db.get_summary(chat_id, user_id)
    if user_facts is None:
        user_facts = await db.get_user_facts(chat_id, user_id, limit=80)  # FACT_KEEP_PER_USER
    if user_profile is None:
        user_profile = await db.get_user_profile_merged(chat_id, user_id)
    if user_document is None:
        user_document = await db.get_user_document(chat_id, user_id)
    focused_history = await db.get_last_messages(
        chat_id,
        min(history_limit, max(plan.history_messages * 2, 12)),
        focus_user_id=user_id,
    )
    shared_history = await db.get_last_messages(
        chat_id,
        min(history_limit, max(plan.history_messages + 2, 10)),
        focus_user_id=None,
    )
    history = _merge_histories(focused_history, shared_history, plan.history_messages, plan.history_chars)
    prompt_history = []
    for item in history:
        role = item["role"]
        content = item["content"]
        if role == "assistant":
            target = item.get("target_user_id")
            if target is not None and int(target) != user_id:
                content = f"[->id={target}] {content}"
        prompt_history.append({"role": role, "content": content})
    focused_prompt_history = [{"role": item["role"], "content": item["content"]} for item in focused_history]

    implicit_reply_block = ""
    if not reply_context:
        implicit_reply_block = build_implicit_reply_context_block(focused_prompt_history, input_text)

    inline_reply_prefix = ""
    if reply_context:
        author = _format_reply_author(reply_context)
        reply_snippet = _fit_text(reply_context.text, 200)
        if reply_snippet:
            inline_reply_prefix = f"[отвечает на сообщение {author}: «{reply_snippet}»]"
    elif implicit_reply_block:
        inline_reply_prefix = f"[контекст: {implicit_reply_block}]"

    if inline_reply_prefix and prompt_history:
        for i in range(len(prompt_history) - 1, -1, -1):
            if prompt_history[i].get("role") == "user":
                original_content = prompt_history[i]["content"]
                prompt_history[i] = {
                    "role": "user",
                    "content": f"{inline_reply_prefix}\n{original_content}",
                }
                break

    system_blocks = []

    # 1. Личность собеседника
    identity_block = build_identity_block(user_profile, sender_username, sender_first_name)
    if identity_block:
        system_blocks.append({"content": identity_block, "bucket": "system", "label": "identity"})

    # 2. Кто пишет сейчас + speaker_guard
    system_blocks.append({
        "content": (
            build_current_speaker_block(user_id, sender_username, sender_first_name, creator_user_id)
            + "\nВ истории реплики помечены [id=123 @username]. Строго различай авторов."
        ),
        "bucket": "system",
        "label": "current_speaker",
    })

    flirt_policy_block = build_flirt_policy_block(sender_username)
    if flirt_policy_block:
        system_blocks.append({
            "content": flirt_policy_block,
            "bucket": "system",
            "label": "flirt_policy",
        })

    mention_only_block = build_mention_only_block(mention_only)
    if mention_only_block:
        system_blocks.append({"content": mention_only_block, "bucket": "system", "label": "mention_only"})

    # 3. Контекст ответа на конкретную реплику (если есть)
    explicit_reply_block = build_reply_context_block(reply_context)
    if explicit_reply_block:
        system_blocks.append({
            "content": explicit_reply_block,
            "bucket": "system",
            "label": "reply_context",
        })

    # 5. Память / факты о собеседнике
    mem_text = format_user_facts(user_facts, max_lines=plan.memory_lines)
    if mem_text:
        system_blocks.append({
            "content": f"Заметки о собеседнике:\n{mem_text}".strip(),
            "bucket": "memory",
            "label": "memory",
        })

    # 5.1 Пользовательское досье (включается только по релевантным запросам)
    user_document_text = _clip_text(user_document.get("document_text"), USER_DOCUMENT_CHAR_LIMIT)
    if user_document_text and should_include_user_document(input_text, user_document_text):
        system_blocks.append({
            "content": (
                "Профиль-досье собеседника (используй это только как внутренний контекст; "
                "не цитируй дословно и не перечисляй без запроса):\n"
                + user_document_text
            ),
            "bucket": "memory",
            "label": "user_document",
        })

    # 6. Резюме диалога с текущим собеседником
    speaker_summary_budget = max(plan.summary_chars * 3 // 5, 200)
    general_summary_budget = max(plan.summary_chars - speaker_summary_budget, 120)
    clipped_speaker_summary = _clip_text(speaker_summary, speaker_summary_budget)
    if clipped_speaker_summary:
        system_blocks.append({
            "bucket": "summary",
            "label": "speaker_summary",
            "content": (
                "Контекст прошлых разговоров с текущим собеседником (только для справки). "
                "Не перечисляй эти темы сам — отвечай только на то, о чём спрашивают сейчас.\n"
                f"{clipped_speaker_summary}"
            ).strip(),
        })

    # 7. Общее резюме чата
    clipped_summary = _clip_text(summary, general_summary_budget)
    if clipped_summary:
        system_blocks.append({
            "bucket": "summary",
            "label": "summary",
            "content": (
                "Краткое резюме общего чата. В нем могут быть разные участники. "
                "Не переноси чужие факты на текущего собеседника.\n"
                f"{clipped_summary}"
            ).strip(),
        })

    include_participants = plan.include_participants or _should_include_participants_context(
        input_text, prompt_history, user_id,
    )
    if include_participants:
        chat_participants = await db.get_chat_participants(
            chat_id, exclude_user_id=user_id, limit=MAX_CHAT_PARTICIPANTS,
        )
        participants_block = build_chat_participants_block(chat_participants, user_id)
        if participants_block:
            system_blocks.append({
                "content": participants_block, "bucket": "system", "label": "participants",
            })

    if plan.file_chunks > 0 and plan.file_chars > 0:
        await sync_context_dir(
            db, knowledge_dir,
            max_files=knowledge_max_files, max_file_bytes=knowledge_max_file_bytes,
        )
        snippets = await retrieve_context_snippets(
            db, input_text, max_chunks=plan.file_chunks, max_chars=plan.file_chars,
        )
        if _max_context_snippet_score(snippets) >= RAG_MIN_SNIPPET_SCORE:
            file_block = build_file_context_block(snippets)
            if file_block:
                system_blocks.append({"content": file_block, "bucket": "rag", "label": "rag"})

    # Alta internal state: load → decay → compute (writes deferred to background)
    chat_state_raw = await db.get_alta_chat_state(chat_id)
    user_rel_raw = await db.get_alta_user_relation(chat_id, user_id)
    hours_chat = _hours_since_state(chat_state_raw.get("updated_at"))
    hours_rel = _hours_since_state(user_rel_raw.get("updated_at"))
    chat_state_decayed = apply_time_decay_chat(chat_state_raw, hours_chat)
    user_rel_decayed = apply_time_decay_relation(user_rel_raw, hours_rel)
    new_chat_state, new_user_rel, reaction_kind, reaction_strength = compute_state_updates(
        input_text, chat_state_decayed, user_rel_decayed,
    )
    # Creator always gets max trust/warmth — not affected by decay or signals
    if creator_user_id is not None and user_id == creator_user_id:
        new_user_rel = {
            **new_user_rel,
            "trust": 1.0,
            "warmth": 1.0,
            "caution": 0.0,
            "annoyance": 0.0,
        }
    # Read active reaction (pre-tick, at most 1 turn stale — acceptable for TTL=2)
    active_reaction = await db.get_active_alta_reaction(chat_id, user_id)
    state_block = build_alta_state_block(new_chat_state, new_user_rel, active_reaction)
    if state_block:
        system_blocks.append({"content": state_block, "bucket": "state", "label": "alta_state"})

    # Fresh-session guard: if there's been a significant gap since last interaction,
    # explicitly tell Alta not to proactively surface old topics/conflicts.
    _SESSION_GAP_THRESHOLD_HOURS = 4.0
    if hours_rel >= _SESSION_GAP_THRESHOLD_HOURS:
        if hours_rel >= 48:
            gap_label = f"{int(round(hours_rel / 24))} дн."
        elif hours_rel >= 24:
            gap_label = "больше суток"
        elif hours_rel >= 1:
            gap_label = f"{int(hours_rel)} ч."
        else:
            gap_label = f"{int(hours_rel * 60)} мин."
        system_blocks.append({
            "content": (
                f"[Прошло {gap_label} с последнего сообщения — человек пишет снова. "
                "Встречай свежо. Не поднимай сама прошлые темы, недосказанности или конфликты — "
                "особенно если они были давно. Реагируй только на то, что написано сейчас.]"
            ),
            "bucket": "state",
            "label": "session_gap",
        })

    log.info(
        "mood  chat=%s user=%s | "
        "energy=%.2f irr=%.2f play=%.2f sbat=%.2f | "
        "trust=%.2f warmth=%.2f ann=%.2f | "
        "reaction=%s",
        chat_id, user_id,
        new_chat_state.get("energy", 0.7),
        new_chat_state.get("irritation", 0.0),
        new_chat_state.get("playfulness", 0.5),
        new_chat_state.get("social_battery", 0.8),
        new_user_rel.get("trust", 0.5),
        new_user_rel.get("warmth", 0.5),
        new_user_rel.get("annoyance", 0.0),
        active_reaction.get("reaction_kind") if active_reaction else "—",
    )

    context_metrics = {
        "history_tokens": approx_messages_tokens(prompt_history),
        "summary_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("bucket") == "summary"
        ),
        "memory_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("bucket") == "memory"
        ),
        "rag_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("bucket") == "rag"
        ),
        "identity_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "identity"
        ),
        "user_document_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "user_document"
        ),
        "current_speaker_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "current_speaker"
        ),
        "reply_context_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "reply_context"
        ),
        "participants_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "participants"
        ),
    }

    mood_pending = {
        "new_chat_state": new_chat_state,
        "new_user_rel": new_user_rel,
        "reaction_kind": reaction_kind,
        "reaction_strength": reaction_strength,
    }
    return [item for item in system_blocks if item.get("content")], prompt_history, context_metrics, mood_pending


# ---------------------------------------------------------------------------
# Mood state persistence (called in background after reply is sent)
# ---------------------------------------------------------------------------

async def _save_mood_state(db, chat_id: int, user_id: int, mood_pending: dict) -> None:
    """Persist computed mood state to DB — runs in background, off the hot path."""
    try:
        new_chat_state = mood_pending["new_chat_state"]
        new_user_rel = mood_pending["new_user_rel"]
        reaction_kind = mood_pending.get("reaction_kind")
        reaction_strength = mood_pending.get("reaction_strength", 0.5)
        await db.save_alta_chat_state(
            chat_id,
            energy=new_chat_state["energy"],
            irritation=new_chat_state["irritation"],
            playfulness=new_chat_state["playfulness"],
            social_battery=new_chat_state["social_battery"],
        )
        await db.save_alta_user_relation(
            chat_id, user_id,
            trust=new_user_rel["trust"],
            warmth=new_user_rel["warmth"],
            caution=new_user_rel["caution"],
            annoyance=new_user_rel["annoyance"],
            interaction_count=new_user_rel["interaction_count"],
        )
        if reaction_kind:
            await db.add_alta_reaction(
                chat_id, reaction_kind, reaction_strength,
                ttl_messages=2, target_user_id=user_id,
            )
        await db.tick_alta_reactions(chat_id)
    except Exception:
        log.warning("_save_mood_state failed chat=%s user=%s", chat_id, user_id, exc_info=True)
