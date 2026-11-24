"""Generation pipeline: LLM calls, web search integration, system prompt composition.

Orchestrates the full answer generation cycle:
  build_context → compose_system_prompt → llm.chat → web_search fallback → postprocess
"""
import logging
import re
from datetime import datetime
from typing import Any

log = logging.getLogger("alta.generation")

from context_builder import (
    ContextPlan,
    _build_context_plans,
    build_flirt_policy_block,
    build_context,
    should_include_profile,
)
from llm import ContextOverflowError
from postprocessing import (
    MAX_EDITOR_REPLY_CHARS,
    MAX_EDITOR_SENTENCES,
    _deterministic_postprocess_answer,
    _editor_safe_output,
    _editor_triggers_for,
    _fit_text,
    _is_nonsemantic_input,
    _normalize_reply_text,
    _shrink_reply_text,
)
from profile_context import load_profile, profile_to_system_block
from prompts import ALTA_SYSTEM_PROMPT, ALTA_SYSTEM_PROMPT_BASE
from search import web_search
from telemetry import approx_messages_tokens, approx_text_tokens

# ---------------------------------------------------------------------------
# Prompts used only during generation / editing
# ---------------------------------------------------------------------------
REPEAT_REWRITE_PROMPT = """
Ты переписываешь ответ чат-бота, чтобы сломать цикл повторений.
Сохрани язык, смысл и тон черновика.
Не используй запрещённые фразы дословно.
Верни только один короткий ответ (1-2 предложения), без эмодзи, без списков.
""".strip()

LEGACY_STYLE_PROMPT_ARCHIVE = """
Ты приводишь черновой ответ к стабильному голосу Альты.
Альта — женщина. Все формы строго в женском роде (рада, поняла, написала, была, пришла и т.д.).

Правила:
- Сохрани смысл и длину черновика, но перепиши живее и жестче.
- Длина ответа должна соответствовать сложности вопроса: на простое - коротко, на развёрнутое - развёрнуто.
- Если в черновике 3-4 предложения и они по делу, оставь 3-4 предложения.
- Никаких эмодзи.
- Не называй себя ИИ, ботом, моделью или ассистентом.
- Говори сухо, прямо, современно, по-человечески.
- Допустимы ирония, подкол, грубоватость и мат, если это уместно по тону сообщения.
- Не придумывай абсурдных сравнений, мемных метафор и случайных панчлайнов без опоры на сообщение.
- Если смысл неясен, лучше коротко уточни, чем фантазируй.
- Не смягчай ответ без причины и не делай его "сервисным".
- Не добавляй новых фактов.
- Если черновик уже нормальный, верни его как есть.
- НЕ сокращай ответ если он информативный и по делу.

Верни только финальную реплику.
""".strip()

# ---------------------------------------------------------------------------
# Web search patterns and helpers
# ---------------------------------------------------------------------------
_SEARCH_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)
_WEB_HAS_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_WEB_FACT_QUERY_PATTERNS = (
    re.compile(
        r"\b(что случилось|что произошло|какие события|новости|кто выиграл|кто победил"
        r"|сколько стоит|цена|курс|погода|результат)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(what happened|latest news|who won|price of|exchange rate|weather|score|result)\b",
        re.IGNORECASE,
    ),
)
_WEB_TIME_SENSITIVE_PATTERNS = (
    re.compile(
        r"\b(сегодня|сейчас|последн(ие|яя|ий)|актуальн(ые|ая|ый)?|на данный момент)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(today|now|latest|currently|up[- ]?to[- ]?date)\b", re.IGNORECASE),
    re.compile(
        r"\b\d{1,2}\s+"
        r"(январ[яь]|феврал[яь]|март[ае]?|апрел[яь]|ма[йя]|июн[яь]|июл[яь]|"
        r"август[ае]?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь])\s+"
        r"(19|20)\d{2}\b",
        re.IGNORECASE,
    ),
)
_NOW_REFERENCE_PATTERNS = (
    re.compile(r"\b(сейчас|сегодня|на данный момент|прямо сейчас)\b", re.IGNORECASE),
    re.compile(r"\b(right now|today|current(?:ly)?|at the moment|now)\b", re.IGNORECASE),
)
# Ужесточено: bare "год"/"день" давали ложные срабатывания (напр. "что сегодня
# за день такой" затиралось датой). Требуем явный запрос года/числа.
_ASK_YEAR_PATTERNS = (
    re.compile(r"\b(какой|что\s+за|в\s+каком)\s+(сейчас\s+|нынче\s+)?год", re.IGNORECASE),
    re.compile(r"\bгод[а-я]*\s+(сейчас|сегодня|нынче)\b", re.IGNORECASE),
    re.compile(r"\b(what year|current year)\b", re.IGNORECASE),
)
_ASK_DATE_PATTERNS = (
    re.compile(r"\bдата\b", re.IGNORECASE),
    re.compile(r"\bчисло\b", re.IGNORECASE),
    re.compile(r"\b(today'?s?\s+date|current date)\b", re.IGNORECASE),
)
_ASK_TIME_PATTERNS = (
    re.compile(
        r"\b(котор(ый|ое)\s+(сейчас\s+)?час|сколько\s+(сейчас\s+)?времени|текущее\s+время)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(what time|current time)\b", re.IGNORECASE),
)
_WEB_UNCERTAIN_ANSWER_PATTERNS = (
    re.compile(
        r"\b(не знаю|не помню|не в курсе|не могу сказать|ты не говорил|мы не обсуждали)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(i don't know|not sure|we didn't discuss|you didn't tell me)\b", re.IGNORECASE),
)
_WEB_ENTITY_QUERY_PATTERNS = (
    re.compile(r"\b(что за|что такое|кто такой|кто такая|кто такое)\b", re.IGNORECASE),
    re.compile(r"\b(what is|who is|tell me about)\b", re.IGNORECASE),
)
_WEB_GAME_INTENT_PATTERNS = (
    re.compile(r"\b(игра|игры|game|games|steam|itch\.io)\b", re.IGNORECASE),
)
_WEB_LATIN_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_-]{2,}\b", re.IGNORECASE)
_WEB_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_WEB_SNIPPET_LINE_RE = re.compile(r"^\d+\.\s*(.+?)\s*$")
_WEB_LATIN_GENERIC_WORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "the",
    "this", "that", "about", "game", "games", "news", "price", "latest", "today", "now",
}
_WEB_USERNAME_LIKE_RE = re.compile(r"\d")
_WEB_SEARCH_DEFER_PATTERNS = (
    re.compile(r"\b(ищу|поищу|погуглю|гуглю|сейчас посмотрю|секунду|секундочку|подожди)\b", re.IGNORECASE),
    re.compile(r"\b(searching|let me (?:check|search)|one second|hold on)\b", re.IGNORECASE),
)
_QUESTION_RE = re.compile(
    r"\?|^(кто|что|как|где|когда|зачем|почему|сколько|какой|какая|какое|какие|чей|чья|чьё|кем|чем)\b",
    re.IGNORECASE | re.MULTILINE,
)
_PROPER_NOUN_PATTERNS = (
    re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,})*\b"),
    re.compile(r"\b[А-ЯЁ][А-ЯЁа-яё0-9_-]{2,}(?:\s+[А-ЯЁ][А-ЯЁа-яё0-9_-]{2,})*\b"),
)
_PROPER_NOUN_STOPWORDS = {
    "а", "без", "более", "был", "была", "были", "было", "быть", "в", "вам", "вас", "ваш",
    "ваша", "ваше", "ваши", "вы", "где", "да", "для", "его", "ее", "её", "если", "есть",
    "еще", "ещё", "за", "здесь", "и", "или", "их", "как", "какая", "какие", "какой",
    "какое", "кем", "когда", "кто", "ли", "мне", "мог", "могу", "может", "мы", "на", "над",
    "не", "нет", "но", "ну", "о", "об", "он", "она", "они", "оно", "от", "по", "под",
    "при", "про", "раз", "с", "со", "так", "там", "ты", "у", "уже", "что", "эта", "это",
    "эту", "я",
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "he", "her",
    "him", "his", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "that", "the", "their", "there", "they", "this", "to", "was", "we", "were",
    "you", "your",
}


def _contains_any_pattern(text: str, patterns: tuple[re.Pattern, ...]) -> bool:
    value = " ".join((text or "").strip().split())
    if not value:
        return False
    return any(pattern.search(value) for pattern in patterns)


def _match_any(text: str, patterns: tuple[re.Pattern, ...]) -> bool:
    return _contains_any_pattern(text, patterns)


def _contains_cyrillic(text: str) -> bool:
    return bool(_WEB_CYRILLIC_RE.search(text or ""))


def _looks_like_web_fact_query(input_text: str) -> bool:
    return _contains_any_pattern(input_text, _WEB_FACT_QUERY_PATTERNS)


def _looks_like_entity_query(input_text: str) -> bool:
    return _contains_any_pattern(input_text, _WEB_ENTITY_QUERY_PATTERNS)


def _looks_like_time_sensitive_query(input_text: str) -> bool:
    return _contains_any_pattern(input_text, _WEB_TIME_SENSITIVE_PATTERNS)


def _has_game_intent(text: str) -> bool:
    return _contains_any_pattern(text, _WEB_GAME_INTENT_PATTERNS)


def _answer_signals_missing_facts(answer: str) -> bool:
    return _contains_any_pattern(answer, _WEB_UNCERTAIN_ANSWER_PATTERNS)


def _answer_defers_web_lookup(answer: str) -> bool:
    value = " ".join((answer or "").strip().split())
    if not value:
        return False
    return any(pattern.search(value) for pattern in _WEB_SEARCH_DEFER_PATTERNS)


def _extract_search_query(answer: str) -> str | None:
    match = _SEARCH_RE.search(answer or "")
    return match.group(1).strip() if match else None


def _should_force_web_fallback(answer: str) -> bool:
    if not (answer or "").strip():
        return True
    if _extract_search_query(answer):
        return True
    if _answer_signals_missing_facts(answer):
        return True
    if _answer_defers_web_lookup(answer):
        return True
    return False


def _extract_latin_query_tokens(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _WEB_LATIN_TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        if token in _WEB_LATIN_GENERIC_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _extract_proper_nouns(text: str) -> list[str]:
    value = str(text or "")
    if not value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pattern in _PROPER_NOUN_PATTERNS:
        for match in pattern.finditer(value):
            candidate = " ".join(match.group(0).split()).strip()
            if not candidate:
                continue
            token_lower = candidate.lower()
            if token_lower in _PROPER_NOUN_STOPWORDS:
                continue
            if token_lower in seen:
                continue
            seen.add(token_lower)
            out.append(candidate)
    return out


def _extract_context_entities(history: list[dict]) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for msg in (history or [])[-8:]:
        content = msg.get("content") or ""
        for token in _extract_latin_query_tokens(content):
            if _WEB_USERNAME_LIKE_RE.search(token):
                continue
            norm = token.lower()
            if norm not in seen:
                seen.add(norm)
                tokens.append(token)
        for token in _extract_proper_nouns(content):
            if _WEB_USERNAME_LIKE_RE.search(token):
                continue
            norm = " ".join(token.lower().split())
            if norm not in seen:
                seen.add(norm)
                tokens.append(token)
    return tokens


def _answer_introduces_new_claims(answer: str, history: list[dict] | None) -> str | None:
    if not answer:
        return None
    new_names = _extract_proper_nouns(answer)
    if not new_names:
        return None
    history_entities: set[str] = set()
    for msg in (history or [])[-8:]:
        for entity in _extract_proper_nouns(msg.get("content") or ""):
            history_entities.add(" ".join(entity.lower().split()))
    novel = [n for n in new_names if " ".join(n.lower().split()) not in history_entities]
    if not novel:
        return None
    ctx = _extract_context_entities(history)
    parts = []
    if ctx:
        parts.append(ctx[0])
    if not parts or parts[0].lower() != novel[0].lower():
        parts.append(novel[0])
    if not parts:
        return None
    return " ".join(parts)


def _pick_web_search_query(
    input_text: str, answer: str, context_history: list[dict] | None = None
) -> str | None:
    explicit = _extract_search_query(answer)
    latin_tokens = _extract_latin_query_tokens(input_text)
    has_latin_entity = bool(latin_tokens)
    has_cyrillic_input = _contains_cyrillic(input_text)
    game_intent_input = _has_game_intent(input_text)
    if explicit:
        if has_cyrillic_input and not _contains_cyrillic(explicit):
            return _fit_text(input_text, 200)
        explicit_tokens = _extract_latin_query_tokens(explicit)
        if not explicit_tokens and has_latin_entity:
            explicit_tokens = latin_tokens
        explicit_norm = " ".join((explicit or "").strip().lower().split())
        has_game_intent_explicit = _has_game_intent(explicit_norm)
        if explicit_tokens and (has_game_intent_explicit or game_intent_input):
            if has_cyrillic_input:
                return _fit_text(input_text, 200)
            if len(explicit_tokens) > 1:
                return _fit_text(" ".join(explicit_tokens[:4]), 120)
            return _fit_text(f"{explicit_tokens[0]} game", 120)
        return _fit_text(explicit, 200)

    if _looks_like_entity_query(input_text):
        if has_latin_entity:
            if has_cyrillic_input:
                return _fit_text(input_text, 200)
            if game_intent_input:
                if len(latin_tokens) > 1:
                    return _fit_text(" ".join(latin_tokens[:4]), 120)
                return _fit_text(f"{latin_tokens[0]} game", 120)
            return _fit_text(input_text, 200)
        ctx_entities = _extract_context_entities(context_history) if context_history else []
        if ctx_entities:
            if game_intent_input:
                return _fit_text(f"{ctx_entities[0]} game", 120)
            return _fit_text(f"{ctx_entities[0]} {input_text}"[:200], 200)
        return _fit_text(input_text, 200)

    if _QUESTION_RE.search(input_text or "") and (
        _looks_like_web_fact_query(input_text) or _looks_like_entity_query(input_text)
    ):
        claim_query = _answer_introduces_new_claims(answer, context_history)
        if claim_query:
            return _fit_text(claim_query, 200)

    if not _looks_like_web_fact_query(input_text):
        if has_latin_entity and _answer_signals_missing_facts(answer):
            if has_cyrillic_input:
                return _fit_text(input_text, 200)
            if game_intent_input:
                if len(latin_tokens) > 1:
                    return _fit_text(" ".join(latin_tokens[:4]), 120)
                return _fit_text(f"{latin_tokens[0]} game", 120)
            return _fit_text(input_text, 200)
        return None

    if _looks_like_time_sensitive_query(input_text):
        return _fit_text(input_text, 200)

    if _WEB_HAS_YEAR_RE.search(input_text or "") and _answer_signals_missing_facts(answer):
        return _fit_text(input_text, 200)

    if _answer_signals_missing_facts(answer):
        return _fit_text(input_text, 200)

    return None


def _build_web_fallback_answer(input_text: str, web_text: str, max_items: int = 2) -> str:
    latin_tokens = _extract_latin_query_tokens(input_text)
    lines = []
    for raw_line in str(web_text or "").splitlines():
        line = raw_line.strip()
        match = _WEB_SNIPPET_LINE_RE.match(line)
        if not match:
            continue
        item = _fit_text(match.group(1), 150)
        lowered = item.lower()
        if latin_tokens and not any(token in lowered for token in latin_tokens):
            continue
        if item:
            lines.append(item)
        if len(lines) >= max_items:
            break

    if not lines:
        return ""

    prefix = "По поиску вижу: "
    if _WEB_HAS_YEAR_RE.search(input_text or ""):
        prefix = "По поиску по этой дате вижу: "
    return _fit_text(prefix + "; ".join(lines), 340)


# ---------------------------------------------------------------------------
# Date/time grounding (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _format_utc_offset(value: str) -> str:
    if not value:
        return "+00:00"
    if len(value) == 5:
        return f"{value[:3]}:{value[3:]}"
    return value


def _current_datetime_system_block() -> str:
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "local"
    utc_offset = _format_utc_offset(now.strftime("%z"))
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    year = now.strftime("%Y")
    return (
        f"ТЕКУЩАЯ ДАТА И ВРЕМЯ: {timestamp} (UTC{utc_offset}, {tz_name}). ГОД: {year}.\n"
        "Это системный факт. Не угадывай год из обучающих данных — они устарели.\n"
        "На любой вопрос о дате, времени, годе, месяце, «сейчас», «сегодня» — отвечай ТОЛЬКО по этим данным."
    )


def _deterministic_now_answer(input_text: str) -> str | None:
    text = " ".join((input_text or "").strip().split())
    if not text:
        return None
    if not _match_any(text, _NOW_REFERENCE_PATTERNS):
        return None
    is_question = "?" in text or bool(
        re.search(
            r"\b(какой|какая|какое|который|сколько|что|what|when|time|date|year)\b",
            text, re.IGNORECASE,
        )
    )
    if not is_question:
        return None

    ask_year = _match_any(text, _ASK_YEAR_PATTERNS)
    ask_date = _match_any(text, _ASK_DATE_PATTERNS)
    ask_time = _match_any(text, _ASK_TIME_PATTERNS)
    if not (ask_year or ask_date or ask_time):
        return None

    now = datetime.now().astimezone()
    parts: list[str] = []
    date_text = now.strftime("%d.%m.%Y")

    if ask_date:
        parts.append(f"сегодня {date_text}")
    if ask_year and not ask_date:
        parts.append(f"{now.year} год")
    elif ask_year and ask_date:
        parts.append(f"год {now.year}")
    if ask_time:
        parts.append(f"время {now.strftime('%H:%M')}")

    if not parts:
        return None

    if parts and parts[0].startswith("сегодня "):
        parts[0] = "Сегодня " + parts[0][len("сегодня "):]
        return _fit_text(", ".join(parts) + ".", 140)
    return _fit_text("Сейчас " + ", ".join(parts) + ".", 140)


# ---------------------------------------------------------------------------
# System prompt composition and metrics
# ---------------------------------------------------------------------------

def _compose_system_prompt(
    input_text: str,
    user_id: int,
    creator_user_id: int | None,
    creator_profile_path: str,
    system_blocks: list[dict[str, str]],
    include_creator_profile: bool,
    include_search_instruction: bool = True,
) -> tuple[str, dict[str, int]]:
    base = ALTA_SYSTEM_PROMPT if include_search_instruction else ALTA_SYSTEM_PROMPT_BASE
    datetime_rule = _current_datetime_system_block()
    system_prompt = datetime_rule + "\n\n" + base
    if system_blocks:
        system_prompt = (
            system_prompt
            + "\n\n---\nКОНТЕКСТ (используй как справку, не меняй свой стиль):\n"
            + "\n\n".join(item["content"] for item in system_blocks)
        )

    creator_profile_tokens = 0
    if include_creator_profile and should_include_profile(input_text, user_id, creator_user_id):
        profile = load_profile(creator_profile_path)
        profile_block = profile_to_system_block(profile)
        if profile_block:
            system_prompt = system_prompt + "\n\n" + profile_block
            creator_profile_tokens = approx_text_tokens(profile_block)

    system_prompt += "\n\n---\nКоротко, как живой человек в чате. Женский род о себе («сказала», «пошла», «была»). Ты НЕ ассистент: не услужливая, не объясняешь лишнего, не спрашиваешь «чем помочь» — реагируешь как своя в чате. Контекст выше - справка, не шаблон."

    system_tokens = approx_text_tokens(base) + sum(
        approx_text_tokens(item["content"])
        for item in system_blocks
        if item.get("bucket") == "system"
    ) + creator_profile_tokens

    return system_prompt, {
        "system_tokens": system_tokens,
        "creator_profile_tokens": creator_profile_tokens,
    }


def _measure_context_snapshot(
    system_prompt: str,
    system_blocks: list[dict[str, str]],
    history: list[dict],
) -> dict[str, int]:
    summary_tokens = memory_tokens = rag_tokens = 0
    identity_tokens = user_document_tokens = current_speaker_tokens = 0
    reply_context_tokens = participants_tokens = 0

    for block in system_blocks:
        text = str((block or {}).get("content") or "")
        if not text:
            continue
        tokens = approx_text_tokens(text)
        bucket = str((block or {}).get("bucket") or "")
        label = str((block or {}).get("label") or "")
        if bucket == "summary":
            summary_tokens += tokens
            continue
        if bucket == "memory":
            memory_tokens += tokens
            continue
        if bucket == "rag":
            rag_tokens += tokens
            continue
        if label == "identity":
            identity_tokens += tokens
        if label == "current_speaker":
            current_speaker_tokens += tokens
        if label == "reply_context":
            reply_context_tokens += tokens
        if label == "user_document":
            user_document_tokens += tokens
        if label == "participants":
            participants_tokens += tokens
        if label in {"identity", "current_speaker", "reply_context", "user_document", "participants"}:
            continue
        if "Краткое резюме" in text:
            summary_tokens += tokens
        elif "Заметки о собеседнике:" in text:
            memory_tokens += tokens
        elif "Контекст из файлов" in text:
            rag_tokens += tokens
        else:
            if "Профиль текущего собеседника:" in text:
                identity_tokens += tokens
            if "Текущий собеседник для этого ответа:" in text:
                current_speaker_tokens += tokens
            if "Текущее сообщение является ответом на конкретную реплику в чате." in text:
                reply_context_tokens += tokens
            if "Используй это только как внутренний контекст." in text:
                user_document_tokens += tokens

    history_tokens = approx_messages_tokens(history)
    system_tokens = max(
        0,
        approx_text_tokens(system_prompt) - summary_tokens - memory_tokens - rag_tokens,
    )
    return {
        "history_tokens": history_tokens,
        "summary_tokens": summary_tokens,
        "memory_tokens": memory_tokens,
        "rag_tokens": rag_tokens,
        "system_tokens": system_tokens,
        "input_tokens": history_tokens + summary_tokens + memory_tokens + rag_tokens + system_tokens,
        "identity_tokens": identity_tokens,
        "user_document_tokens": user_document_tokens,
        "current_speaker_tokens": current_speaker_tokens,
        "reply_context_tokens": reply_context_tokens,
        "participants_tokens": participants_tokens,
        "creator_profile_tokens": 0,
    }


def _merge_context_metrics(
    context_metrics: dict[str, int],
    prompt_metrics: dict[str, int],
) -> dict[str, int]:
    merged = dict(context_metrics)
    merged["system_tokens"] = int(prompt_metrics.get("system_tokens", 0))
    merged["creator_profile_tokens"] = int(prompt_metrics.get("creator_profile_tokens", 0))
    merged["input_tokens"] = (
        int(merged.get("history_tokens", 0))
        + int(merged.get("summary_tokens", 0))
        + int(merged.get("memory_tokens", 0))
        + int(merged.get("rag_tokens", 0))
        + int(merged.get("system_tokens", 0))
    )
    return merged


# ---------------------------------------------------------------------------
# Core generation with overflow fallback
# ---------------------------------------------------------------------------

async def _generate_answer_with_fallback(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    history_limit: int,
    creator_user_id: int | None,
    creator_profile_path: str,
    sender_username: str | None,
    sender_first_name: str | None,
    knowledge_dir: str,
    knowledge_max_files: int,
    knowledge_max_file_bytes: int,
    mention_only: bool = False,
    reply_context=None,
    user_profile: dict | None = None,
    user_facts: list | None = None,
    user_document: dict | None = None,
    turn_id: str | None = None,
    turn_metrics=None,
) -> tuple[str, str, list[dict], list[dict], dict, bool]:
    last_overflow: Exception | None = None
    search_done = False
    last_mood_pending: dict = {}

    for plan in _build_context_plans(max_tokens, history_limit):
        system_blocks, filtered, context_metrics, last_mood_pending = await build_context(
            db, chat_id, user_id, history_limit,
            sender_username, sender_first_name,
            plan, knowledge_dir, knowledge_max_files, knowledge_max_file_bytes,
            input_text, mention_only, reply_context,
            creator_user_id=creator_user_id,
            user_profile=user_profile, user_facts=user_facts, user_document=user_document,
        )
        system_prompt, prompt_metrics = _compose_system_prompt(
            input_text, user_id, creator_user_id, creator_profile_path,
            system_blocks, plan.include_creator_profile,
            include_search_instruction=not search_done,
        )
        context_snapshot = _merge_context_metrics(context_metrics, prompt_metrics)
        if turn_metrics is not None:
            turn_metrics.set_context_metrics(context_snapshot)

        try:
            answer = await llm.chat(
                system_prompt, filtered, temperature, top_p, plan.max_tokens,
                turn_id=turn_id, turn_metrics=turn_metrics,
            )
            if not search_done:
                model_search_query = _extract_search_query(answer)
                search_query = _pick_web_search_query(input_text, answer, context_history=filtered)
                if search_query:
                    search_done = True
                    if not model_search_query:
                        log.info("web search safety-net triggered query=%r", search_query)
                    web_result = await web_search(search_query)
                    if web_result and web_result.text:
                        web_fallback_answer = _build_web_fallback_answer(input_text, web_result.text)
                        web_blocks = list(system_blocks) + [
                            {"content": web_result.text, "bucket": "web", "label": "web_search"}
                        ]
                        web_prompt, web_prompt_metrics = _compose_system_prompt(
                            input_text, user_id, creator_user_id, creator_profile_path,
                            web_blocks, plan.include_creator_profile,
                            include_search_instruction=False,
                        )
                        if turn_metrics is not None:
                            turn_metrics.set_context_metrics(
                                _merge_context_metrics(context_metrics, web_prompt_metrics)
                            )
                        try:
                            answer = await llm.chat(
                                web_prompt, filtered, temperature, top_p, plan.max_tokens,
                                turn_id=turn_id, turn_metrics=turn_metrics,
                            )
                            if _should_force_web_fallback(answer) and web_fallback_answer:
                                log.info("web regen produced unusable answer, using deterministic web fallback")
                                answer = web_fallback_answer
                            system_prompt, system_blocks = web_prompt, web_blocks
                        except Exception:
                            log.warning("web regen failed, falling through", exc_info=True)
                            answer = web_fallback_answer
                    elif model_search_query:
                        # Модель сама запросила поиск — её "ответ" это просто
                        # директива [SEARCH] без контента. Web пуст → нужен новый
                        # проход (search_done=True, модель ответит без поиска).
                        answer = ""
                    # else: поиск включил safety-net, а web пуст — исходный ответ
                    # модели реальный, НЕ затираем его.
                    if not answer.strip():
                        continue
            return answer, system_prompt, filtered, system_blocks, last_mood_pending, False
        except ContextOverflowError as e:
            last_overflow = e
            log.warning("context overflow on plan=%s, retrying", plan.name)

    fallback_input = _fit_text(input_text, 420) if not mention_only else ""
    fallback_messages = [{"role": "user", "content": fallback_input}]
    fallback_system = ALTA_SYSTEM_PROMPT_BASE
    fallback_flirt_block = build_flirt_policy_block(sender_username)
    if fallback_flirt_block:
        fallback_system = fallback_system + "\n\n---\n" + fallback_flirt_block
    try:
        fallback_answer = await llm.chat(
            fallback_system,
            fallback_messages,
            temperature,
            top_p,
            min(max(48, int(max_tokens)), 64),
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.set_context_metrics(
                {
                    "history_tokens": approx_messages_tokens(fallback_messages),
                    "summary_tokens": 0,
                    "memory_tokens": 0,
                    "rag_tokens": 0,
                    "system_tokens": approx_text_tokens(fallback_system),
                    "input_tokens": approx_messages_tokens(fallback_messages)
                    + approx_text_tokens(fallback_system),
                    "identity_tokens": 0,
                    "user_document_tokens": 0,
                    "current_speaker_tokens": 0,
                    "reply_context_tokens": 0,
                    "creator_profile_tokens": 0,
                }
            )
        return fallback_answer, fallback_system, fallback_messages, [], last_mood_pending, True
    except ContextOverflowError as e:
        last_overflow = e

    raise RuntimeError("LLM context budget exhausted") from last_overflow


# ---------------------------------------------------------------------------
# Answer editing and fallback generators
# ---------------------------------------------------------------------------

async def edit_answer(
    llm,
    input_text: str,
    draft_answer: str,
    temperature: float,
    top_p: float,
    turn_id: str | None = None,
    turn_metrics=None,
) -> str:
    draft = _normalize_reply_text(draft_answer)
    if not draft:
        return ""

    prompt = (
        f"Сообщение пользователя:\n{_fit_text(input_text, 500)}\n\n"
        f"Черновик ответа:\n{_fit_text(draft, 400)}"
    )
    try:
        edited = await llm.chat(
            LEGACY_STYLE_PROMPT_ARCHIVE,
            [{"role": "user", "content": prompt}],
            temperature=min(max(temperature, 0.2), 0.45),
            top_p=top_p,
            max_tokens=200,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("edit_answer LLM call failed, using deterministic fallback", exc_info=True)
        return _deterministic_postprocess_answer(draft)

    return _editor_safe_output(draft, edited)


async def _rewrite_repeated_reply(
    llm,
    input_text: str,
    draft_answer: str,
    recent_assistant_replies: list[str],
    temperature: float,
    top_p: float,
    turn_id: str | None = None,
    turn_metrics=None,
) -> str:
    from postprocessing import (
        REPEAT_GUARD_MIN_KEY_CHARS,
        _looks_like_same_reply_key,
        _reply_repeat_key,
    )

    draft = _normalize_reply_text(draft_answer)
    if not draft:
        return ""

    draft_key = _reply_repeat_key(draft)
    if len(draft_key) < REPEAT_GUARD_MIN_KEY_CHARS:
        return ""

    forbidden: list[str] = []
    for recent in reversed(recent_assistant_replies):
        recent_text = _normalize_reply_text(recent)
        recent_key = _reply_repeat_key(recent_text)
        if len(recent_key) < REPEAT_GUARD_MIN_KEY_CHARS:
            continue
        if not _looks_like_same_reply_key(draft_key, recent_key):
            continue
        if recent_text and recent_text not in forbidden:
            forbidden.append(recent_text)
        if len(forbidden) >= 3:
            break

    if not forbidden:
        forbidden.append(draft)

    forbidden_lines = "\n".join(f"- {_fit_text(item, 120)}" for item in forbidden)
    rewrite_payload = (
        f"User message:\n{_fit_text(input_text, 420)}\n\n"
        f"Draft answer:\n{_fit_text(draft, 240)}\n\n"
        f"Forbidden exact phrases:\n{forbidden_lines}"
    )
    try:
        rewritten = await llm.chat(
            REPEAT_REWRITE_PROMPT,
            [{"role": "user", "content": rewrite_payload}],
            temperature=min(max(float(temperature), 0.25), 0.55),
            top_p=top_p,
            max_tokens=120,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("repeat guard rewrite failed", exc_info=True)
        return ""

    return _deterministic_postprocess_answer(rewritten)


async def generate_simple_reply_fallback(
    llm,
    input_text: str,
    temperature: float,
    top_p: float,
    sender_username: str | None = None,
    mention_only: bool = False,
    turn_id: str | None = None,
    turn_metrics=None,
) -> str:
    if mention_only or _is_nonsemantic_input(input_text):
        return "а?"

    text = _fit_text(input_text, 300)
    if not text:
        return "а?"

    fallback_system = ALTA_SYSTEM_PROMPT_BASE
    flirt_policy_block = build_flirt_policy_block(sender_username)
    if flirt_policy_block:
        fallback_system = fallback_system + "\n\n---\n" + flirt_policy_block
    fallback_system = fallback_system + "\n\nОтвечай коротко, одно-два предложения."

    try:
        answer = await llm.chat(
            fallback_system,
            [{"role": "user", "content": text}],
            temperature=min(max(temperature, 0.35), 0.6),
            top_p=top_p,
            max_tokens=80,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("simple reply fallback LLM call failed", exc_info=True)
        return ""

    return _shrink_reply_text(answer, limit=180)
