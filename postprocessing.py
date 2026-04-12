"""Deterministic text postprocessing, repeat guard, editor helpers.

No async, no DB, no LLM calls — pure text transformation.
"""
import re
from typing import Any

# ---------------------------------------------------------------------------
# Text size constants
# ---------------------------------------------------------------------------
REPEAT_GUARD_WINDOW = 6
REPEAT_GUARD_MAX_CHARS = 400
REPEAT_GUARD_MIN_KEY_CHARS = 3
REPEAT_GUARD_SHORT_EXACT_MIN_KEY_CHARS = 2
REPEAT_GUARD_SUBSTRING_MAX_CHARS = 28

MIN_EDITOR_DRAFT_CHARS = 15
MAX_EDITOR_REPLY_CHARS = 700
MAX_EDITOR_SENTENCES = 8

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAD6"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    re.UNICODE,
)
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
REPLY_REPEAT_PREFIX_PATTERN = re.compile(r"^\s*\[(?:to[^]]+|id=\d+[^]]*)\]\s*", re.IGNORECASE)
REPLY_REPEAT_TOKEN_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
FORBIDDEN_SELF_REFERENCE_PATTERNS = (
    re.compile(r"\bя\s+(?:ии|ai|бот|модель|ассистент)\b", re.IGNORECASE),
    re.compile(r"\b(?:как|будучи)\s+(?:ии|ai|бот(?:ом)?|модель(?:ю)?|ассистент(?:ом)?)\b", re.IGNORECASE),
    re.compile(r"\bя\s+(?:языковая|нейро)\s*(?:модель|сеть)\b", re.IGNORECASE),
)

OVER_PERSONA_STRONG_PATTERNS = (
    re.compile(r"[!?]{3,}"),
    re.compile(r"\*[^*]{1,80}\*"),
    re.compile(r"([а-яё])\1{3,}", re.IGNORECASE),
)
OVER_PERSONA_SOFT_PATTERNS = (
    re.compile(r"\b(?:с удовольствием|буду рада помочь|всегда к услугам)\b", re.IGNORECASE),
    re.compile(r"\b(?:чем.{0,5}помочь|обращайся|не стесняйся)\b", re.IGNORECASE),
)

_GENDER_FIX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bя\s+рад\b", re.IGNORECASE), "я рада"),
    (re.compile(r"\bя\s+готов\b", re.IGNORECASE), "я готова"),
    (re.compile(r"\bя\s+понял\b", re.IGNORECASE), "я поняла"),
    (re.compile(r"\bя\s+сделал\b", re.IGNORECASE), "я сделала"),
    (re.compile(r"\bя\s+написал\b", re.IGNORECASE), "я написала"),
    (re.compile(r"\bя\s+пришёл\b", re.IGNORECASE), "я пришла"),
    (re.compile(r"\bя\s+пошёл\b", re.IGNORECASE), "я пошла"),
    (re.compile(r"\bя\s+был\b", re.IGNORECASE), "я была"),
    (re.compile(r"\bя\s+сказал\b", re.IGNORECASE), "я сказала"),
    (re.compile(r"\bя\s+увидел\b", re.IGNORECASE), "я увидела"),
    (re.compile(r"\bя\s+решил\b", re.IGNORECASE), "я решила"),
    (re.compile(r"\bя\s+хотел\b", re.IGNORECASE), "я хотела"),
    (re.compile(r"\bя\s+знал\b", re.IGNORECASE), "я знала"),
    (re.compile(r"\bя\s+думал\b", re.IGNORECASE), "я думала"),
    (re.compile(r"\bя\s+забыл\b", re.IGNORECASE), "я забыла"),
    (re.compile(r"\bя\s+узнал\b", re.IGNORECASE), "я узнала"),
    (re.compile(r"\bя\s+нашёл\b", re.IGNORECASE), "я нашла"),
    (re.compile(r"\bя\s+успел\b", re.IGNORECASE), "я успела"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Рад\b"), "Рада"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Готов\b"), "Готова"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Понял\b"), "Поняла"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Сделал\b"), "Сделала"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Написал\b"), "Написала"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Решил\b"), "Решила"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Думал\b"), "Думала"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Забыл\b"), "Забыла"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Узнал\b"), "Узнала"),
]

_SEARCH_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Basic text utilities
# ---------------------------------------------------------------------------

def _fit_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _normalize_reply_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _shrink_reply_text(text: str, limit: int = 400, max_sentences: int = 5) -> str:
    value = " ".join((text or "").strip().split())
    if not value:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", value)
    compact = " ".join(part.strip() for part in parts[:max_sentences] if part.strip())
    if compact:
        value = compact
    return _fit_text(value, limit)


def _split_reply_sentences(text: str) -> list[str]:
    value = _normalize_reply_text(text)
    if not value:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]


def _sentence_count(text: str) -> int:
    value = _normalize_reply_text(text)
    if not value:
        return 0
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]
    return max(1, len(parts))


def _is_nonsemantic_input(text: Any) -> bool:
    normalized = str(text or "").strip().lower()
    return normalized in {"", "ping"}


# ---------------------------------------------------------------------------
# Boolean checks on reply text
# ---------------------------------------------------------------------------

def _contains_emoji(text: str) -> bool:
    return bool(EMOJI_PATTERN.search(text or ""))


def _has_forbidden_self_reference(text: str) -> bool:
    value = _normalize_reply_text(text)
    if not value:
        return False
    return any(pattern.search(value) for pattern in FORBIDDEN_SELF_REFERENCE_PATTERNS)


def _has_format_violation(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    value = _normalize_reply_text(raw)
    if _contains_emoji(value):
        return True
    if _has_forbidden_self_reference(value):
        return True
    if len(value) > MAX_EDITOR_REPLY_CHARS:
        return True
    if _sentence_count(value) > MAX_EDITOR_SENTENCES:
        return True
    if LIST_PREFIX_PATTERN.search(raw) is not None:
        return True
    return False


def _has_over_persona(text: str) -> bool:
    value = _normalize_reply_text(text)
    if not value or len(value) < MIN_EDITOR_DRAFT_CHARS:
        return False
    if any(pattern.search(value) for pattern in OVER_PERSONA_STRONG_PATTERNS):
        return True
    soft_hits = sum(1 for pattern in OVER_PERSONA_SOFT_PATTERNS if pattern.search(value))
    return soft_hits >= 2


# ---------------------------------------------------------------------------
# Editor helpers
# ---------------------------------------------------------------------------

def _editor_triggers_for(draft_answer: str) -> list[str]:
    raw = str(draft_answer or "").strip()
    if not raw:
        return ["empty_draft"]
    value = _normalize_reply_text(raw)
    triggers: list[str] = []
    if len(value) < MIN_EDITOR_DRAFT_CHARS:
        triggers.append("short_draft")
    if _has_format_violation(raw):
        triggers.append("format_violation")
    if _has_over_persona(raw):
        triggers.append("over_persona")
    return triggers


def _editor_safe_output(draft_answer: str, candidate_answer: str) -> str:
    draft = _normalize_reply_text(draft_answer)
    if not draft:
        return ""
    candidate = _normalize_reply_text(candidate_answer)
    if not candidate:
        return _fix_gender(draft)
    max_candidate_len = int(len(draft) * 1.3)
    if len(draft) <= 24:
        max_candidate_len = max(max_candidate_len, len(draft) + 80)
    if len(candidate) > max_candidate_len:
        return _fix_gender(draft)
    if _contains_emoji(candidate) or _has_forbidden_self_reference(candidate):
        return _fix_gender(draft)
    if _sentence_count(candidate) > max(MAX_EDITOR_SENTENCES, _sentence_count(draft) + 1):
        return _fix_gender(draft)
    return _fix_gender(candidate)


# ---------------------------------------------------------------------------
# Cleanup passes
# ---------------------------------------------------------------------------

def _fix_gender(text: str) -> str:
    for pattern, replacement in _GENDER_FIX_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _strip_list_prefixes(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = LIST_PREFIX_PATTERN.sub("", raw_line).strip()
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def _remove_forbidden_self_reference(text: str) -> str:
    value = text
    value = re.sub(r"\bкак\s+(?:ии|ai|бот(?:ом)?|модель(?:ю)?|ассистент(?:ом)?)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bбудучи\s+(?:ии|ai|бот(?:ом)?|модель(?:ю)?|ассистент(?:ом)?)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bя\s+(?:ии|ai|бот|модель|ассистент)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bя\s+(?:языковая|нейро)\s*(?:модель|сеть)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value if value else text


def _deterministic_postprocess_answer(draft_answer: str) -> str:
    raw = str(draft_answer or "").strip()
    if not raw:
        return ""
    value = _strip_list_prefixes(raw) if "\n" in raw or LIST_PREFIX_PATTERN.search(raw) else raw
    value = _SEARCH_RE.sub("", value).strip()
    value = re.sub(r"^\[\d{1,2}:\d{2}\]\s*", "", value)
    value = EMOJI_PATTERN.sub("", value)
    value = re.sub(r"\s*[—–]\s*", " - ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*([,.:;!?])\s*", r"\1 ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"([!?])\1{2,}", r"\1\1", value)
    value = re.sub(r"\.{4,}", "...", value)

    if _has_forbidden_self_reference(value):
        value = _remove_forbidden_self_reference(value)

    if _has_over_persona(value):
        value = re.sub(r"\*[^*]{1,80}\*", "", value)
        value = re.sub(r"([А-Яа-яЁё])\1{3,}", r"\1\1", value, flags=re.IGNORECASE)
        value = re.sub(r"[!?]{3,}", "!!", value)
        value = _normalize_reply_text(value)

    value = _fix_gender(value)
    value = _shrink_reply_text(value, limit=MAX_EDITOR_REPLY_CHARS, max_sentences=MAX_EDITOR_SENTENCES)
    # Strip trailing punctuation that signals AI-generated text (.  ,  ;  :)
    # Preserve: ellipsis (...), !, ?
    value = re.sub(r"(?<!\.)\.(?!\.)(\s*)$", r"\1", value).rstrip()
    value = re.sub(r"[,;:]\s*$", "", value).rstrip()
    return value


# ---------------------------------------------------------------------------
# Repeat guard
# ---------------------------------------------------------------------------

def _reply_repeat_key(text: str) -> str:
    value = _normalize_reply_text(text).lower()
    if not value:
        return ""
    value = REPLY_REPEAT_PREFIX_PATTERN.sub("", value)
    value = REPLY_REPEAT_TOKEN_PATTERN.sub(" ", value)
    return " ".join(value.split())


def _looks_like_same_reply_key(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter = left if len(left) <= len(right) else right
    longer = right if len(left) <= len(right) else left
    if len(shorter) < REPEAT_GUARD_MIN_KEY_CHARS:
        return False
    if len(shorter) > REPEAT_GUARD_SUBSTRING_MAX_CHARS:
        return False
    return shorter in longer


def _recent_assistant_replies(messages: list[dict], limit: int = REPEAT_GUARD_WINDOW) -> list[str]:
    replies: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content", "")).strip()
        if text:
            replies.append(text)
    if limit <= 0:
        return replies
    return replies[-limit:]


def _is_repeated_recent_reply(answer: str, recent_assistant_replies: list[str]) -> bool:
    candidate = _normalize_reply_text(answer)
    if not candidate:
        return False
    if len(candidate) > REPEAT_GUARD_MAX_CHARS:
        return False

    candidate_key = _reply_repeat_key(candidate)
    recent_keys: list[str] = []
    for item in recent_assistant_replies:
        key = _reply_repeat_key(item)
        if len(key) >= REPEAT_GUARD_SHORT_EXACT_MIN_KEY_CHARS:
            recent_keys.append(key)
    if not recent_keys:
        return False

    if len(candidate_key) < REPEAT_GUARD_MIN_KEY_CHARS:
        if len(candidate_key) < REPEAT_GUARD_SHORT_EXACT_MIN_KEY_CHARS:
            return False
        if candidate_key == recent_keys[-1]:
            return True
        repeats = sum(1 for key in recent_keys if key == candidate_key)
        return repeats >= 2

    if _looks_like_same_reply_key(candidate_key, recent_keys[-1]):
        return True

    repeats = sum(1 for key in recent_keys if _looks_like_same_reply_key(candidate_key, key))
    return repeats >= 2


def _repeat_guard_hard_fallback(input_text: str) -> str:
    normalized = _normalize_reply_text(input_text)
    if not normalized:
        return "Скажи конкретнее, что нужно."
    if "?" in normalized:
        return "Уточни вопрос чуть конкретнее."
    return "Давай по делу: что именно нужно?"
