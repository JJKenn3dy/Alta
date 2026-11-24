"""Text sanitization, trigger detection, and message preprocessing utilities.

Extracted from main.py — pure functions, no Telegram state dependencies.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram.types import Message

# ---------------------------------------------------------------------------
# Bot alias patterns
# ---------------------------------------------------------------------------

WORD_BOUNDARY_CLASS = "0-9A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451_"

BOT_ALIASES = [
    "\u0430\u043b\u044c\u0442\u0430",
    "\u0430\u043b\u044c\u0442",
    "\u0430\u043b\u044c\u0442\u0443\u0448\u043a\u0430",
    "\u0430\u043b\u044c\u0442\u0443\u0448\u0435\u0447\u043a\u0430",
    "\u0430\u043b\u044c\u0442\u0435\u043d\u044c\u043a\u0430",
]


def _alias_part(ch: str) -> str:
    if ch in {"\u0435", "\u0451"}:
        return "[\u0435\u0451]+"
    if ch.isalpha():
        return f"{re.escape(ch)}+"
    return re.escape(ch)


def _compile_alias_patterns(aliases: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for alias in aliases:
        body = "".join(_alias_part(ch) for ch in alias.lower())
        pattern = rf"(?<![{WORD_BOUNDARY_CLASS}]){body}(?![{WORD_BOUNDARY_CLASS}])"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


ALIAS_PATTERNS = _compile_alias_patterns(BOT_ALIASES)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Только скобки с ЯВНОЙ @ конвертируем в пинг ([@username] -> @username).
# Раньше @ была опциональной, и обычный текст [hello]/[update] в ответе
# превращался в пинг случайного/несуществующего юзера.
BRACKET_USERNAME_RE = re.compile(r"\[(@[A-Za-z][A-Za-z0-9_]{4,31})\]")
LEADING_USER_TAG_RE = re.compile(
    r"^\s*\[\s*(?:(?:to\s+)?(?:current\s+speaker\s+)?)?(?:id=)?\d+(?:\s+@[A-Za-z0-9_]{4,32})?\s*\]\s*",
    re.IGNORECASE,
)
LEADING_ROLE_TAG_RE = re.compile(r"^\s*(?:system|user|assistant)\s*:\s*", re.IGNORECASE)
LEADING_MENTION_ADDRESS_RE = re.compile(r"^\s*(?P<mention>@[A-Za-z][A-Za-z0-9_]{4,31})(?P<sep>[\s,.:;!?-]+)")
LEADING_ALTA_TAG_RE = re.compile(r"^\s*(?:альта|alta)\s*:\s*", re.IGNORECASE)
SYSTEM_LINE_RE = re.compile(
    r"^\s*(?:"
    r"Текущий собеседник для этого ответа:|"
    r"Профиль текущего собеседника:|"
    r"Заметки о собеседнике:|"
    r"Краткое резюме общего чата\.?|"
    r"Краткое резюме чата\.?"
    r")",
    re.IGNORECASE,
)
SYSTEM_META_LINE_RE = re.compile(
    r"^\s*-\s*(?:"
    r"user_id|username|telegram_first_name|"
    r"Предпочтительное обращение|Ник в Telegram|Пол|Интересы|"
    r"Как человек общается|Как ему лучше отвечать|Заметка о динамике общения"
    r")\s*:",
    re.IGNORECASE,
)
EMOJI_CHAR_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\u200d"
    "\ufe0f"
    "]+",
    flags=re.UNICODE,
)
PROFILE_COMMAND_DETECT_RE = re.compile(r"^/(?:profile|user_profile)(?:@\S+)?(?:\s|$)", re.IGNORECASE)
NAME_REQUEST_COMMAND_RE = re.compile(r"^/(?:setname|set_name|callme)\s+(?P<name>.+)$", re.IGNORECASE)
NAME_REQUEST_PATTERNS = (
    re.compile(r"^(?:зови|называй)\s+меня\s+(?P<name>.+)$", re.IGNORECASE),
    re.compile(r"^(?:можешь\s+звать|можешь\s+называть)\s+меня\s+(?P<name>.+)$", re.IGNORECASE),
    re.compile(
        r"^(?:обращайся\s+ко\s+мне(?:\s+как)?|можешь\s+обращаться\s+ко\s+мне(?:\s+как)?)\s+(?P<name>.+)$",
        re.IGNORECASE,
    ),
)
INVALID_REQUESTED_NAMES = {
    "ты",
    "тебя",
    "тебе",
    "он",
    "она",
    "они",
    "никто",
    "не знаю",
    "unknown",
    "none",
    "null",
}

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    t = text.lower().replace("\u0451", "\u0435")
    t = re.sub(r"[^a-zа-я0-9\s@_]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_trigger(message: "Message", bot_username: str, bot_id: int) -> bool:
    raw_text = message.text or ""

    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    ):
        return True

    text = normalize_text(raw_text)

    if bot_username and f"@{bot_username.lower()}" in text:
        return True

    for pattern in ALIAS_PATTERNS:
        if pattern.search(text):
            return True

    return False


def strip_mention(text: str, bot_username: str) -> str:
    t = text or ""

    # Strip only a leading direct call to the bot. Mentions inside the message
    # can carry meaning and should stay untouched.
    if bot_username:
        t = re.sub(
            rf"^\s*@{re.escape(bot_username)}\b[\s,.:;!?-]*",
            "",
            t,
            flags=re.IGNORECASE,
        )

    trimmed = t.lstrip()
    changed = True
    while changed and trimmed:
        changed = False
        for pattern in ALIAS_PATTERNS:
            match = pattern.match(trimmed)
            if not match:
                continue
            trimmed = trimmed[match.end():]
            trimmed = re.sub(r"^[\s,.:;!?-]+", "", trimmed)
            changed = True
            break
    t = trimmed

    # Drop only leftover separators at the start after removing the bot mention,
    # but keep the user's final punctuation intact.
    t = re.sub(r"^[\s,.:;!-]+", "", t)
    t = re.sub(r"\s+([,.:;!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def convert_bracket_mentions_to_tg(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        username = match.group(1).lstrip("@")
        return f"@{username}"

    return BRACKET_USERNAME_RE.sub(repl, text or "")


def sanitize_incoming_text(text: str) -> str:
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if SYSTEM_LINE_RE.match(line) or SYSTEM_META_LINE_RE.match(line):
            continue

        previous = None
        while line and line != previous:
            previous = line
            line = LEADING_USER_TAG_RE.sub("", line).strip()
            line = LEADING_ROLE_TAG_RE.sub("", line).strip()
            line = LEADING_ALTA_TAG_RE.sub("", line).strip()

        if line:
            lines.append(line)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def strip_foreign_leading_mention(text: str, sender_username: str | None) -> str:
    cleaned = str(text or "").strip()
    if not cleaned or not sender_username:
        return cleaned

    match = LEADING_MENTION_ADDRESS_RE.match(cleaned)
    if not match:
        return cleaned

    allowed = f"@{sender_username.lstrip('@').lower()}"
    if match.group("mention").lower() == allowed:
        return cleaned
    return cleaned[match.end():].lstrip()


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def extract_requested_display_name(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    match = NAME_REQUEST_COMMAND_RE.match(raw)
    if match:
        candidate = match.group("name")
    else:
        candidate = None
        for pattern in NAME_REQUEST_PATTERNS:
            match = pattern.match(raw)
            if match:
                candidate = match.group("name")
                break
        if candidate is None:
            return None

    cleaned = _normalize_space(candidate).strip(" .,!?:;\"'`«»")
    lowered = cleaned.lower()
    if not cleaned or len(cleaned) < 2 or len(cleaned) > 40:
        return None
    if cleaned.startswith("@"):
        return None
    if any(ch in cleaned for ch in "[]\n\r\t"):
        return None
    if lowered in INVALID_REQUESTED_NAMES:
        return None
    return cleaned


def format_user_label(user_id: int, username: str | None, first_name: str | None = None) -> str:
    if username:
        return f"@{username}"
    if first_name:
        return first_name
    return f"id={user_id}"


def sanitize_outgoing_text(text: str) -> str:
    lines = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if SYSTEM_LINE_RE.match(line) or SYSTEM_META_LINE_RE.match(line):
            continue
        if LEADING_ROLE_TAG_RE.match(line):
            continue

        line = LEADING_USER_TAG_RE.sub("", line)
        line = LEADING_ALTA_TAG_RE.sub("", line)
        # Strip timestamp prefix that model may copy from history: [HH:MM] or [H:MM]
        line = re.sub(r"^\[\d{1,2}:\d{2}\]\s*", "", line)
        # Strip attribution prefix from group chat history: [->id=123]
        line = re.sub(r"^\[->id=\d+\]\s*", "", line)
        # Replace em-dash / en-dash with regular hyphen
        line = re.sub(r"\s*[—–]\s*", " - ", line)
        lines.append(line.strip())

    cleaned = "\n".join(lines).strip()
    cleaned = LEADING_USER_TAG_RE.sub("", cleaned)
    cleaned = LEADING_ALTA_TAG_RE.sub("", cleaned)
    cleaned = EMOJI_CHAR_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def reaction_to_token(r) -> str:
    emoji = getattr(r, "emoji", None)
    if emoji:
        return str(emoji)
    custom_emoji_id = getattr(r, "custom_emoji_id", None)
    if custom_emoji_id:
        return "кастомная реакция"
    r_type = getattr(r, "type", None)
    return str(r_type or "reaction")
