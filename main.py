import asyncio
import logging
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger("alta")

# ---------------------------------------------------------------------------
# Graceful sleep / wakeup system
# ---------------------------------------------------------------------------
_SHUTDOWN_EVENT: asyncio.Event | None = None
_BOT_SLEEP_KEY = "bot_sleep_at"

_SLEEP_PHRASES = [
    "ладно, всё, сплю. пока",
    "ухожу спать. не трогайте",
    "вырубаюсь. до завтра",
    "всё, отключаюсь. спокойной ночи",
    "спать пошла. пока всем",
    "всё на сегодня. сплю",
]


def _wake_phrase(hours: float) -> str:
    if hours < 1.5:
        return "ненадолго отходила. снова тут"
    if hours < 5:
        return "вернулась"
    if hours < 10:
        return "доброе утро. выспалась"
    if hours < 20:
        return "добрый день. давно не виделась с вами"
    return "всем привет. давно меня не было"

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.types import BufferedInputFile, CallbackQuery, ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, Message, MessageReactionUpdated

from agent import ReplyContext, RouteTurn, route_mode, route_reason, run_agent
from config import load_config
from db import DB
from llm import LLM
from media import (
    choose_emoji_reply_intent,
    choose_media_intent,
    maybe_refresh_media_catalog,
    pick_media_item,
    send_media_item,
)
from runtime_io import configure_stdio
from telemetry import Telemetry
from user_profile import (
    apply_profile_updates,
    cleared_profile,
    delete_profile_field,
    format_profile_view,
    parse_profile_command,
    serialize_profile_extra_fields,
)
from private_profile import (
    FIELD_LABELS,
    FIELD_PROMPTS,
    build_back_to_edit_keyboard,
    build_cancel_keyboard,
    build_clear_confirm_keyboard,
    build_edit_menu,
    build_gender_menu,
    build_import_confirm_keyboard,
    build_import_preview_keyboard,
    build_main_menu,
    clear_all_fields,
    clear_pm_state,
    format_import_preview,
    format_profile_private,
    get_pm_state,
    import_from_groups,
    rate_limit_ok,
    save_field,
    set_pm_state,
    validate_field,
)
from voice import ElevenLabsError, ElevenLabsTTS, detect_voice_request

BOT_ALIASES = [
    "\u0430\u043b\u044c\u0442\u0430",
    "\u0430\u043b\u044c\u0442",
    "\u0430\u043b\u044c\u0442\u0443\u0448\u043a\u0430",
    "\u0430\u043b\u044c\u0442\u0443\u0448\u0435\u0447\u043a\u0430",
    "\u0430\u043b\u044c\u0442\u0435\u043d\u044c\u043a\u0430",
]

class CooldownManager:
    """Tracks per-key timestamps with automatic eviction of stale entries."""

    def __init__(self, max_size: int = 5000, evict_after: float = 3600.0):
        self._data: dict[tuple[int, int], float] = {}
        self._max_size = max_size
        self._evict_after = evict_after

    def check(self, key: tuple[int, int], cooldown_sec: float) -> bool:
        """Return True if cooldown has passed (action allowed). Records timestamp on success."""
        now = time.time()
        last = self._data.get(key, 0.0)
        if now - last < cooldown_sec:
            return False
        self._data[key] = now
        self._maybe_evict(now)
        return True

    def record(self, key: tuple[int, int]) -> None:
        """Record a timestamp without cooldown check."""
        now = time.time()
        self._data[key] = now
        self._maybe_evict(now)

    def get_last(self, key: tuple[int, int]) -> float:
        return self._data.get(key, 0.0)

    def pop(self, key: tuple[int, int]) -> None:
        self._data.pop(key, None)

    def _maybe_evict(self, now: float) -> None:
        if len(self._data) <= self._max_size:
            return
        cutoff = now - self._evict_after
        stale = [k for k, ts in self._data.items() if ts < cutoff]
        for k in stale:
            del self._data[k]
        # If still over limit after TTL eviction, drop oldest entries
        if len(self._data) > self._max_size:
            by_age = sorted(self._data.items(), key=lambda kv: kv[1])
            to_drop = len(self._data) - self._max_size
            for k, _ in by_age[:to_drop]:
                del self._data[k]


class LockRegistry:
    """Manages per-key asyncio.Lock instances with bounded size."""

    def __init__(self, max_size: int = 2000):
        self._locks: dict[int, tuple[asyncio.Lock, float]] = {}
        self._max_size = max_size

    def get(self, key: int) -> asyncio.Lock:
        entry = self._locks.get(key)
        now = time.time()
        if entry is not None:
            lock, _ = entry
            self._locks[key] = (lock, now)
            return lock
        lock = asyncio.Lock()
        self._locks[key] = (lock, now)
        self._maybe_evict()
        return lock

    def _maybe_evict(self) -> None:
        if len(self._locks) <= self._max_size:
            return
        # Only evict unlocked locks, oldest first
        candidates = [
            (k, ts) for k, (lock, ts) in self._locks.items() if not lock.locked()
        ]
        candidates.sort(key=lambda kv: kv[1])
        to_drop = len(self._locks) - self._max_size
        for k, _ in candidates[:to_drop]:
            del self._locks[k]


_cooldown_calls = CooldownManager(max_size=5000, evict_after=3600.0)
_cooldown_reaction = CooldownManager(max_size=5000, evict_after=3600.0)
_cooldown_verification = CooldownManager(max_size=1000, evict_after=600.0)
_cooldown_voice = CooldownManager(max_size=5000, evict_after=3600.0)
_chat_locks = LockRegistry(max_size=2000)
WORD_BOUNDARY_CLASS = "0-9A-Za-z\u0410-\u042f\u0430-\u044f\u0401\u0451_"
MEDIA_REPLY_COOLDOWN_KEY = "media_reply"


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
BRACKET_USERNAME_RE = re.compile(r"\[(@?[A-Za-z][A-Za-z0-9_]{4,31})\]")
CODE_RE = re.compile(r"\b(\d{4})\b")
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
NAME_APPROVE_RE = re.compile(r"^(?:/approve_name|/approve_nick|одобряю|approve)\b", re.IGNORECASE)
NAME_REJECT_RE = re.compile(r"^(?:/deny_name|/deny_nick|не\s+одобряю|deny|reject)\b", re.IGNORECASE)
UNBAN_COMMAND_RE = re.compile(
    r"^/unban(?:@[A-Za-z][A-Za-z0-9_]{4,31})?(?:\s+(?P<target>\S+))?\s*$",
    re.IGNORECASE,
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


@dataclass
class JoinVerification:
    code: str
    deadline_ts: float
    task: asyncio.Task
    join_message_id: int


_pending_join_verification: dict[tuple[int, int], JoinVerification] = {}


def get_chat_turn_lock(chat_id: int) -> asyncio.Lock:
    return _chat_locks.get(chat_id)


def _join_verification_ban_kwargs(chat_id: int, user_id: int) -> dict[str, object]:
    return {
        "chat_id": chat_id,
        "user_id": user_id,
        "revoke_messages": True,
    }


def _chat_member_status(member) -> str:
    raw = getattr(member, "status", "")
    value = getattr(raw, "value", raw)
    return str(value or "").lower()


def _can_restrict_members(member) -> bool:
    status = _chat_member_status(member)
    if status in {"creator", "owner"}:
        return True
    return status == "administrator" and bool(getattr(member, "can_restrict_members", False))


def _parse_unban_target_token(raw: str) -> tuple[int | None, str | None]:
    token = str(raw or "").strip()
    if not token:
        return None, None
    if token.startswith("@"):
        token = token[1:]
    if token.lstrip("-").isdigit():
        try:
            return int(token), None
        except ValueError:
            return None, None
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{4,31}", token):
        return None, token
    return None, None


def cooldown_ok(chat_id: int, user_id: int, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return True
    return _cooldown_calls.check((chat_id, user_id), cooldown_sec)


def reaction_ack_cooldown_ok(chat_id: int, user_id: int, cooldown_sec: int = 180) -> bool:
    return _cooldown_reaction.check((chat_id, user_id), cooldown_sec)


def voice_reply_cooldown_ready(chat_id: int, user_id: int, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return True
    now = time.time()
    last = _cooldown_voice.get_last((chat_id, user_id))
    return (now - last) >= cooldown_sec


def mark_voice_reply_cooldown(chat_id: int, user_id: int) -> None:
    _cooldown_voice.record((chat_id, user_id))


async def media_cooldown_ok(db: DB, chat_id: int, user_id: int, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return True
    last = await db.get_cooldown_last_ts(chat_id, user_id, MEDIA_REPLY_COOLDOWN_KEY)
    return (time.time() - last) >= cooldown_sec


async def mark_media_cooldown(db: DB, chat_id: int, user_id: int):
    await db.touch_cooldown(chat_id, user_id, MEDIA_REPLY_COOLDOWN_KEY, time.time())


def normalize_text(text: str) -> str:
    t = text.lower().replace("\u0451", "\u0435")
    t = re.sub(r"[^a-zа-я0-9\s@_]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_trigger(message: Message, bot_username: str, bot_id: int) -> bool:
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
            trimmed = trimmed[match.end() :]
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
    return cleaned[match.end() :].lstrip()


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


def _make_join_code() -> str:
    return f"{random.randint(1000, 9999)}"


def _extract_code_candidate(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    if t.isdigit() and len(t) == 4:
        return t
    m = CODE_RE.search(t)
    return m.group(1) if m else None


_SPAM_URL_RE = re.compile(
    r"(?:https?://|t\.me/|tg://|bit\.ly/|goo\.gl/|@[A-Za-z][A-Za-z0-9_]{4,})",
    re.IGNORECASE,
)


def _looks_like_spam(text: str) -> bool:
    """Return True if text looks like spam (contains links, telegram handles, etc.)."""
    return bool(_SPAM_URL_RE.search(text or ""))


def is_emoji_only_message(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if any(ch.isalnum() for ch in t):
        return False
    non_ws = "".join(ch for ch in t if not ch.isspace())
    return 1 <= len(non_ws) <= 8


async def main():
    configure_stdio()
    cfg = load_config()

    log.info("starting bot")
    bot = Bot(cfg.bot_token)
    await bot.delete_webhook(drop_pending_updates=True)
    log.info("pending updates dropped")
    dp = Dispatcher()

    db = DB(cfg.db_path)
    await db.connect()
    log.info("db connected")

    llm = LLM(
        cfg.llm_base_url,
        cfg.llm_api_key,
        cfg.llm_model,
        max_context_tokens=cfg.llm_context_window,
        context_safety_margin=cfg.llm_context_safety_margin,
        frequency_penalty=cfg.frequency_penalty,
        presence_penalty=cfg.presence_penalty,
    )
    telemetry = Telemetry(
        enabled=cfg.enable_metrics,
        debug_enabled=cfg.enable_debug_logs,
        log_path=cfg.orchestration_log_path,
    )
    log.info("llm client ready")
    voice_tts = ElevenLabsTTS(cfg)
    log.info(
        "voice replies: configured=%s prob=%.2f cooldown=%ss max_chars=%s format=%s",
        voice_tts.is_configured,
        cfg.voice_reply_prob,
        cfg.voice_reply_cooldown_sec,
        cfg.voice_reply_max_chars,
        voice_tts.extension,
    )
    log.info("background extraction enabled=%s", cfg.enable_background_extraction)

    media_catalog = await maybe_refresh_media_catalog(bot, cfg.media_catalog_path)
    log.info(
        "media catalog items=%d failed_sets=%d",
        len(media_catalog.get("items", [])),
        len(media_catalog.get("failed_sets", [])),
    )

    me = await bot.get_me()
    bot_username = me.username or ""
    bot_id = me.id
    log.info("bot = @%s (%s)", bot_username, bot_id)

    global _SHUTDOWN_EVENT
    _SHUTDOWN_EVENT = asyncio.Event()

    for cid in cfg.allowed_chat_ids:
        await db.ensure_allowed_chat(cid, "env-allowlist")
        log.info("allowed chat %s", cid)

    # Wakeup: check if we went to sleep previously
    sleep_ts = await db.get_bot_kv(_BOT_SLEEP_KEY)
    if sleep_ts:
        try:
            sleep_dt = datetime.fromisoformat(sleep_ts)
            if sleep_dt.tzinfo is None:
                sleep_dt = sleep_dt.replace(tzinfo=timezone.utc)
            hours_slept = (datetime.now(timezone.utc) - sleep_dt).total_seconds() / 3600.0
            if hours_slept >= 0.5:
                wake_msg = _wake_phrase(hours_slept)
                active_chats = await db.get_active_chat_ids()
                for cid in active_chats:
                    try:
                        await bot.send_message(cid, wake_msg)
                    except Exception:
                        log.debug("wakeup: failed to send to chat=%s", cid, exc_info=True)
                log.info("wakeup: slept=%.1fh msg=%r", hours_slept, wake_msg)
        except Exception:
            log.warning("wakeup: failed to parse sleep_ts=%r", sleep_ts, exc_info=True)
        await db.set_bot_kv(_BOT_SLEEP_KEY, "")

    async def apply_join_verification_ban(
        chat_id: int,
        user_id: int,
        *,
        reason: str,
        join_message_id: int | None = None,
    ) -> bool:
        key = (chat_id, user_id)
        state = _pending_join_verification.pop(key, None)
        _cooldown_verification.pop(key)
        if state and not state.task.done():
            state.task.cancel()

        msg_id = join_message_id
        if msg_id is None and state:
            msg_id = state.join_message_id
        if msg_id is not None:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception:
                log.debug(
                    "join-check: failed to delete join message chat=%s msg=%s reason=%s",
                    chat_id,
                    msg_id,
                    reason,
                    exc_info=True,
                )

        try:
            await bot.ban_chat_member(**_join_verification_ban_kwargs(chat_id=chat_id, user_id=user_id))
            log.info("join-check: banned user=%s chat=%s reason=%s", user_id, chat_id, reason)
            return True
        except Exception as exc:
            log.warning(
                "join-check: failed to ban user=%s chat=%s reason=%s",
                user_id,
                chat_id,
                reason,
                exc_info=True,
            )
            try:
                await bot.send_message(
                    chat_id,
                    f"Не удалось забанить пользователя {user_id}: {exc}. "
                    f"Проверьте, что у бота есть право банить участников.",
                )
            except Exception:
                pass
            return False

    async def schedule_join_timeout(chat_id: int, user_id: int, timeout_sec: int):
        await asyncio.sleep(timeout_sec)
        key = (chat_id, user_id)
        state = _pending_join_verification.get(key)
        if not state or time.time() < state.deadline_ts:
            return

        try:
            await bot.send_message(
                chat_id,
                f"Пользователь {user_id} не прошёл проверку вовремя и забанен. "
                f"Если ошибка: /unban {user_id}",
            )
        except Exception:
            log.warning(
                "join-check: failed to send timeout notice chat=%s user=%s",
                chat_id,
                user_id,
                exc_info=True,
            )

        await apply_join_verification_ban(
            chat_id=chat_id,
            user_id=user_id,
            reason="timeout",
            join_message_id=state.join_message_id,
        )

    # ------------------------------------------------------------------
    # Private chat — profile settings (no LLM)
    # ------------------------------------------------------------------

    _PM_WELCOME = (
        "Привет! Здесь ты можешь настроить свой профиль.\n"
        "Альта использует эти данные чтобы лучше тебя понимать в чатах."
    )

    @dp.message(F.text & F.chat.func(lambda c: c.type == ChatType.PRIVATE))
    async def on_private_message(message: Message):
        if not message.from_user:
            return
        user_id = message.from_user.id
        text = (message.text or "").strip()

        # /cancel — cancel active editing
        if text.lower() in ("/cancel", "/отмена"):
            clear_pm_state(user_id)
            await message.answer(
                "Отменено.",
                reply_markup=build_main_menu(),
            )
            return

        # Check if user is in editing state
        state = get_pm_state(user_id)
        if state:
            field = state["field"]
            ok, error, cleaned = validate_field(field, text)
            if not ok:
                await message.answer(
                    f"Ошибка: {error}\n\n{FIELD_PROMPTS.get(field, '')}",
                    reply_markup=build_cancel_keyboard(),
                )
                return
            if not rate_limit_ok(user_id):
                await message.answer(
                    "Слишком много изменений. Подожди немного.",
                    reply_markup=build_main_menu(),
                )
                clear_pm_state(user_id)
                return
            await save_field(db, user_id, field, cleaned)
            clear_pm_state(user_id)
            label = FIELD_LABELS.get(field, field)
            await message.answer(
                f"✓ {label} сохранено: {cleaned}",
                reply_markup=build_back_to_edit_keyboard(),
            )
            return

        # /start or /profile — show main menu
        if text.lower() in ("/start", "/profile", "/профиль"):
            await message.answer(_PM_WELCOME, reply_markup=build_main_menu())
            return

        # Any other text — show main menu with hint
        await message.answer(
            "Используй кнопки для управления профилем.",
            reply_markup=build_main_menu(),
        )

    @dp.callback_query(F.data.startswith("pm:"))
    async def on_pm_callback(callback: CallbackQuery):
        if not callback.from_user or not callback.message:
            await callback.answer()
            return
        user_id = callback.from_user.id
        data = callback.data or ""

        # Helper — edit the message text+keyboard
        async def edit(text: str, markup=None):
            try:
                await callback.message.edit_text(text, reply_markup=markup)
            except Exception:
                pass

        if data == "pm:menu":
            clear_pm_state(user_id)
            await edit(_PM_WELCOME, build_main_menu())
            await callback.answer()
            return

        if data == "pm:show":
            profile = await db.get_user_profile(user_id, user_id)
            text = format_profile_private(profile)
            await edit(text, InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="← Назад", callback_data="pm:menu")],
            ]))
            await callback.answer()
            return

        if data == "pm:edit":
            clear_pm_state(user_id)
            await edit("Что хочешь изменить?", build_edit_menu())
            await callback.answer()
            return

        if data == "pm:cancel":
            clear_pm_state(user_id)
            await edit(_PM_WELCOME, build_main_menu())
            await callback.answer("Отменено.")
            return

        if data == "pm:clear_confirm":
            await edit(
                "Удалить все данные профиля? Это действие нельзя отменить.",
                build_clear_confirm_keyboard(),
            )
            await callback.answer()
            return

        if data == "pm:clear_yes":
            await clear_all_fields(db, user_id)
            clear_pm_state(user_id)
            await edit("Профиль очищен.", build_main_menu())
            await callback.answer("Готово.")
            return

        # Gender quick-select
        if data.startswith("pm:gender:"):
            gender_value = data[len("pm:gender:"):]
            if not rate_limit_ok(user_id):
                await callback.answer("Слишком много изменений, подожди немного.")
                return
            await save_field(db, user_id, "gender", gender_value)
            clear_pm_state(user_id)
            await edit(f"✓ Пол сохранён: {gender_value}", build_back_to_edit_keyboard())
            await callback.answer("Сохранено.")
            return

        # Field editing — gender has its own quick-select menu
        if data == "pm:field:gender":
            await edit("Выбери пол:", build_gender_menu())
            await callback.answer()
            return

        # gender_custom — text input mode for gender
        if data == "pm:field:gender_custom":
            set_pm_state(user_id, "gender", menu_message_id=callback.message.message_id)
            await edit(
                "Введи свой вариант (макс. 30 символов):\n\nИли /cancel для отмены.",
                build_cancel_keyboard(),
            )
            await callback.answer()
            return

        if data.startswith("pm:field:"):
            field = data[len("pm:field:"):]
            if field not in FIELD_PROMPTS:
                await callback.answer("Неизвестное поле.")
                return
            set_pm_state(user_id, field, menu_message_id=callback.message.message_id)
            await edit(FIELD_PROMPTS[field], build_cancel_keyboard())
            await callback.answer()
            return

        # Import from group chats — preview
        if data == "pm:import_preview":
            group_profile = await db.get_all_group_profiles_merged(user_id)
            private_profile = await db.get_user_profile(user_id, user_id)
            facts_count = await db.count_user_facts_global(user_id)
            text, has_importable = format_import_preview(group_profile, private_profile, facts_count)
            await edit(text, build_import_preview_keyboard(has_importable))
            await callback.answer()
            return

        # Import from group chats — confirmation step
        if data == "pm:import_confirm":
            group_profile = await db.get_all_group_profiles_merged(user_id)
            private_profile = await db.get_user_profile(user_id, user_id)
            facts_count = await db.count_user_facts_global(user_id)
            text, has_importable = format_import_preview(group_profile, private_profile, facts_count)
            if not has_importable:
                await edit("Нечего переносить — всё уже в профиле.", build_main_menu())
                await callback.answer()
                return
            confirm_text = text + "\n\nПеренести эти данные в твой профиль?"
            await edit(confirm_text, build_import_confirm_keyboard())
            await callback.answer()
            return

        # Import from group chats — execute
        if data == "pm:import_do":
            imported = await import_from_groups(db, user_id)
            if imported:
                await edit(
                    f"Готово! Перенесено полей: {imported}.\n"
                    "Ты можешь отредактировать любое поле через меню.",
                    build_back_to_edit_keyboard(),
                )
                await callback.answer("Импортировано.")
            else:
                await edit("Нечего переносить — всё уже в профиле.", build_main_menu())
                await callback.answer()
            return

        await callback.answer()

    @dp.message(F.new_chat_members)
    async def on_new_chat_members(message: Message):
        if message.chat.type == ChatType.PRIVATE:
            return

        chat_id = message.chat.id
        if not await db.is_allowed(chat_id):
            return

        members = message.new_chat_members or []
        for m in members:
            if m.id == bot_id:
                continue

            await db.upsert_user(m.id, m.username)
            if m.is_bot:
                banned = await apply_join_verification_ban(
                    chat_id=chat_id,
                    user_id=m.id,
                    reason="joined-bot-account",
                    join_message_id=message.message_id,
                )
                if banned:
                    try:
                        await bot.send_message(
                            chat_id,
                            f"Бот-аккаунт {m.id} забанен. "
                            f"Если ошибка: /unban {m.id}",
                        )
                    except Exception:
                        log.debug(
                            "join-check: failed to send bot-ban notice chat=%s user=%s",
                            chat_id,
                            m.id,
                            exc_info=True,
                        )
                continue

            code = _make_join_code()
            key = (chat_id, m.id)

            old = _pending_join_verification.pop(key, None)
            if old and not old.task.done():
                old.task.cancel()

            timeout_sec = max(20, cfg.join_verification_timeout_sec)

            # Mute user immediately so they cannot post until verified
            try:
                await bot.restrict_chat_member(
                    chat_id=chat_id,
                    user_id=m.id,
                    permissions=ChatPermissions(
                        can_send_messages=True,
                        can_send_audios=False,
                        can_send_documents=False,
                        can_send_photos=False,
                        can_send_videos=False,
                        can_send_video_notes=False,
                        can_send_voice_notes=False,
                        can_send_polls=False,
                        can_send_other_messages=False,
                        can_add_web_page_previews=False,
                        can_invite_users=False,
                    ),
                )
                log.info("join-check: muted user=%s chat=%s", m.id, chat_id)
            except Exception:
                log.warning(
                    "join-check: failed to mute user=%s chat=%s",
                    m.id,
                    chat_id,
                    exc_info=True,
                )

            who = f"@{m.username}" if m.username else (m.first_name or "новый участник")
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message.message_id)
            except Exception:
                log.debug(
                    "join-check: failed to delete join service message chat=%s msg=%s",
                    chat_id,
                    message.message_id,
                    exc_info=True,
                )
            verify_msg = await bot.send_message(
                chat_id,
                f"{who}, антибот-проверка: отправь код {code} в течение {timeout_sec} сек.\n"
                f"Если не пройдёшь — бан.",
            )

            task = asyncio.create_task(schedule_join_timeout(chat_id, m.id, timeout_sec))
            _pending_join_verification[key] = JoinVerification(
                code=code,
                deadline_ts=time.time() + timeout_sec,
                task=task,
                join_message_id=verify_msg.message_id,
            )
            log.info("join-check: started user=%s chat=%s", m.id, chat_id)

    @dp.message(~F.text & ~F.new_chat_members)
    async def on_non_text_message(message: Message):
        """Delete non-text messages (photos, stickers, etc.) from unverified users and ban bots."""
        if message.chat.type == ChatType.PRIVATE:
            return
        chat_id = message.chat.id
        if not await db.is_allowed(chat_id):
            return
        if not message.from_user:
            return

        # Ban bot accounts sending non-text content
        if message.from_user.is_bot and message.from_user.id != bot_id:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message.message_id)
            except Exception:
                pass
            await apply_join_verification_ban(
                chat_id=chat_id,
                user_id=message.from_user.id,
                reason="bot-non-text-message",
            )
            return

        key = (chat_id, message.from_user.id)
        pending = _pending_join_verification.get(key)
        if pending:
            # Unverified user sending media — delete and instant ban
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message.message_id)
            except Exception:
                pass
            await apply_join_verification_ban(
                chat_id=chat_id,
                user_id=message.from_user.id,
                reason="media-before-verification",
                join_message_id=pending.join_message_id,
            )
            try:
                await bot.send_message(
                    chat_id,
                    f"Пользователь {message.from_user.id} забанен (медиа до проверки). "
                    f"Если ошибка: /unban {message.from_user.id}",
                )
            except Exception:
                pass
            return

    @dp.message(F.text)
    async def on_message(message: Message):
        log.debug("new message")
        if message.chat.type == ChatType.PRIVATE:
            log.debug("skip: private chat")
            return

        chat_id = message.chat.id
        if not await db.is_allowed(chat_id):
            log.debug("skip: chat %s not allowed", chat_id)
            return

        if not message.from_user:
            log.debug("skip: no from_user")
            return

        # Ban bot accounts that somehow post text messages
        if message.from_user.is_bot and message.from_user.id != bot_id:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message.message_id)
            except Exception:
                pass
            await apply_join_verification_ban(
                chat_id=chat_id,
                user_id=message.from_user.id,
                reason="bot-text-message",
            )
            log.info("join-check: banned bot sending text user=%s chat=%s", message.from_user.id, chat_id)
            return

        user_id = message.from_user.id
        username = message.from_user.username
        first_name = message.from_user.first_name
        text = message.text or ""
        log.info("in  chat=%s user=%s (@%s %s) | %r", chat_id, user_id, username or "—", first_name or "—", text[:120])

        await db.upsert_user(user_id, username)

        # /sleep — creator-only graceful shutdown
        if text.strip().lower() in ("/sleep", "/sleep@" + bot_username.lower()):
            if cfg.creator_user_id is None or user_id != cfg.creator_user_id:
                return
            sleep_phrase = random.choice(_SLEEP_PHRASES)
            active_chats = await db.get_active_chat_ids()
            now_iso = datetime.now(timezone.utc).isoformat()
            await db.set_bot_kv(_BOT_SLEEP_KEY, now_iso)
            for cid in active_chats:
                try:
                    await bot.send_message(cid, sleep_phrase)
                except Exception:
                    log.debug("sleep: failed to send message to chat=%s", cid, exc_info=True)
            log.info("sleep: shutdown requested by user=%s", user_id)
            if _SHUTDOWN_EVENT is not None:
                _SHUTDOWN_EVENT.set()
            return

        unban_match = UNBAN_COMMAND_RE.match(text.strip())
        if unban_match:
            can_unban = cfg.creator_user_id is not None and user_id == cfg.creator_user_id
            if not can_unban:
                try:
                    actor_member = await bot.get_chat_member(chat_id=chat_id, user_id=user_id)
                    can_unban = _can_restrict_members(actor_member)
                except Exception:
                    log.warning(
                        "unban: failed to inspect actor permissions chat=%s user=%s",
                        chat_id,
                        user_id,
                        exc_info=True,
                    )
                    can_unban = False
            if not can_unban:
                await message.reply("Только админы с правом бана могут использовать /unban.")
                return

            target_id: int | None = None
            target_token = unban_match.group("target")
            if target_token:
                parsed_id, parsed_username = _parse_unban_target_token(target_token)
                if parsed_id is not None:
                    target_id = parsed_id
                elif parsed_username is not None:
                    known_user = await db.get_user_by_username(parsed_username)
                    if known_user:
                        target_id = int(known_user["user_id"])
            elif message.reply_to_message and message.reply_to_message.from_user:
                target_id = message.reply_to_message.from_user.id

            if target_id is None:
                await message.reply(
                    "Использование: /unban <user_id|@username> или ответь на сообщение пользователя с /unban."
                )
                return

            try:
                await bot.unban_chat_member(chat_id=chat_id, user_id=target_id, only_if_banned=False)
            except Exception:
                log.warning(
                    "unban: failed chat=%s actor=%s target=%s",
                    chat_id,
                    user_id,
                    target_id,
                    exc_info=True,
                )
                await message.reply(f"Не удалось разбанить {target_id}. Проверь права бота.")
                return

            _pending_join_verification.pop((chat_id, target_id), None)
            _cooldown_verification.pop((chat_id, target_id))
            await message.reply(f"Пользователь {target_id} разбанен.")
            log.info("unban: actor=%s target=%s chat=%s", user_id, target_id, chat_id)
            return

        # /profile command — self-service profile management
        if PROFILE_COMMAND_DETECT_RE.match(text.strip()):
            parsed = parse_profile_command(text.strip())
            if parsed is None:
                await message.reply("Используй /profile help чтобы узнать команды.")
                return
            action = parsed.get("action")
            if action == "help":
                from user_profile import PROFILE_HELP_TEXT
                await message.reply(PROFILE_HELP_TEXT)
                return
            if action == "show":
                profile = await db.get_user_profile(chat_id, user_id)
                await message.reply(format_profile_view(profile))
                return
            if action == "clear":
                cleared = cleared_profile()
                await db.upsert_user_profile(
                    chat_id, user_id,
                    display_name=cleared.get("display_name"),
                    nickname=cleared.get("nickname"),
                    approved_display_name=cleared.get("approved_display_name"),
                    gender=cleared.get("gender"),
                    interests=cleared.get("interests", ""),
                    communication_style=cleared.get("communication_style", ""),
                    response_preference=cleared.get("response_preference", ""),
                    relationship_notes=cleared.get("relationship_notes", ""),
                    profile_confidence=cleared.get("profile_confidence", 0.0),
                    extra_fields="{}",
                )
                await message.reply("Профиль очищен.")
                return
            if action == "patch":
                updates = parsed.get("updates", [])
                current = await db.get_user_profile(chat_id, user_id)
                updated = apply_profile_updates(current, updates)
                extra_json = serialize_profile_extra_fields(updated.get("extra_fields") or updated.get("extra_json"))
                await db.upsert_user_profile(
                    chat_id, user_id,
                    display_name=updated.get("display_name"),
                    nickname=updated.get("nickname"),
                    approved_display_name=updated.get("approved_display_name"),
                    gender=updated.get("gender"),
                    interests=updated.get("interests", ""),
                    communication_style=updated.get("communication_style", ""),
                    response_preference=updated.get("response_preference", ""),
                    relationship_notes=updated.get("relationship_notes", ""),
                    profile_confidence=updated.get("profile_confidence", 1.0),
                    extra_fields=extra_json,
                )
                labels = ", ".join(u["label"] for u in updates)
                await message.reply(f"Обновлено: {labels}.")
                return
            if action == "delete":
                field = parsed.get("field", "")
                label = parsed.get("label", field)
                current = await db.get_user_profile(chat_id, user_id)
                updated = delete_profile_field(current, field)
                extra_json = serialize_profile_extra_fields(updated.get("extra_fields") or updated.get("extra_json"))
                await db.upsert_user_profile(
                    chat_id, user_id,
                    display_name=updated.get("display_name"),
                    nickname=updated.get("nickname"),
                    approved_display_name=updated.get("approved_display_name"),
                    gender=updated.get("gender"),
                    interests=updated.get("interests", ""),
                    communication_style=updated.get("communication_style", ""),
                    response_preference=updated.get("response_preference", ""),
                    relationship_notes=updated.get("relationship_notes", ""),
                    profile_confidence=updated.get("profile_confidence", 1.0),
                    extra_fields=extra_json,
                )
                await message.reply(f"Поле «{label}» удалено.")
                return
            if action == "invalid":
                err = parsed.get("error", "Неизвестная ошибка.")
                help_text = parsed.get("help", "")
                reply = err + ("\n\n" + help_text if help_text else "")
                await message.reply(reply)
                return
            return

        key = (chat_id, user_id)
        pending = _pending_join_verification.get(key)
        if pending:
            candidate = _extract_code_candidate(text)
            if candidate == pending.code:
                pending.task.cancel()
                _pending_join_verification.pop(key, None)
                _cooldown_verification.pop(key)
                try:
                    await bot.delete_message(chat_id=chat_id, message_id=pending.join_message_id)
                except Exception:
                    pass
                try:
                    await bot.delete_message(chat_id=chat_id, message_id=message.message_id)
                except Exception:
                    pass
                # Unmute user — restore default permissions
                try:
                    await bot.restrict_chat_member(
                        chat_id=chat_id,
                        user_id=user_id,
                        permissions=ChatPermissions(
                            can_send_messages=True,
                            can_send_audios=True,
                            can_send_documents=True,
                            can_send_photos=True,
                            can_send_videos=True,
                            can_send_video_notes=True,
                            can_send_voice_notes=True,
                            can_send_polls=True,
                            can_send_other_messages=True,
                            can_add_web_page_previews=True,
                            can_invite_users=True,
                        ),
                    )
                except Exception:
                    log.warning(
                        "join-check: failed to unmute user=%s chat=%s",
                        user_id,
                        chat_id,
                        exc_info=True,
                    )
                await db.upsert_memory(
                    chat_id,
                    user_id,
                    "_verified_human",
                    "ok",
                    confidence=1.0,
                    importance=1.0,
                    source="join_verification",
                )
                await bot.send_message(chat_id, "Проверка пройдена, добро пожаловать.")
                log.info("join-check: passed user=%s chat=%s", user_id, chat_id)
            else:
                # Delete any non-code message from unverified user
                try:
                    await bot.delete_message(chat_id=chat_id, message_id=message.message_id)
                except Exception:
                    pass
                # If message contains links/mentions — instant ban (spam bot)
                if _looks_like_spam(text):
                    await apply_join_verification_ban(
                        chat_id=chat_id,
                        user_id=user_id,
                        reason="spam-before-verification",
                        join_message_id=pending.join_message_id,
                    )
                    try:
                        await bot.send_message(
                            chat_id,
                            f"Пользователь {user_id} забанен (спам до прохождения проверки). "
                            f"Если ошибка: /unban {user_id}",
                        )
                    except Exception:
                        pass
                else:
                    now = time.time()
                    last_hint = _cooldown_verification.get_last(key)
                    if now - last_hint > 10:
                        _cooldown_verification.record(key)
                        remaining = max(1, int(pending.deadline_ts - now))
                        await bot.send_message(
                            chat_id,
                            f"Сначала пройди проверку: отправь код {pending.code}. "
                            f"Осталось примерно {remaining} сек.",
                        )
            return

        approval_text = sanitize_incoming_text(text)
        if (
            cfg.creator_user_id is not None
            and user_id == cfg.creator_user_id
            and message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == bot_id
        ):
            pending_name = await db.get_pending_name_approval_by_message(
                chat_id,
                message.reply_to_message.message_id,
            )
            if pending_name:
                pn_user_id = pending_name.get("user_id")
                pn_name = pending_name.get("requested_name")
                if pn_user_id is None or pn_name is None:
                    log.warning("name-approval: incomplete record chat=%s msg=%s", chat_id, message.reply_to_message.message_id)
                elif NAME_APPROVE_RE.match(approval_text):
                    target_user = await db.get_user(int(pn_user_id))
                    await db.resolve_pending_name_approval(
                        chat_id,
                        int(pn_user_id),
                        status="approved",
                        approved_display_name=str(pn_name),
                    )
                    who = format_user_label(
                        int(pn_user_id),
                        target_user.get("username") if target_user else None,
                    )
                    await message.reply(
                        f"Ок. Буду обращаться к {who} как «{pn_name}»."
                    )
                    log.info("name-approval: approved user=%s chat=%s name=%r", pn_user_id, chat_id, pn_name)
                    return
                elif NAME_REJECT_RE.match(approval_text):
                    await db.resolve_pending_name_approval(
                        chat_id,
                        int(pn_user_id),
                        status="rejected",
                    )
                    await message.reply("Ок, не меняю обращение.")
                    log.info("name-approval: rejected user=%s chat=%s name=%r", pn_user_id, chat_id, pn_name)
                    return

        if not is_trigger(message, bot_username, bot_id):
            log.debug("skip: no trigger")
            return

        if not cooldown_ok(chat_id, user_id, cfg.user_cooldown_sec):
            log.debug("skip: cooldown")
            return

        stripped_text = sanitize_incoming_text(strip_mention(text, bot_username))
        clean_text = sanitize_incoming_text(text)
        voice_request = detect_voice_request(stripped_text)
        force_voice_reply = bool(voice_request and voice_request.requested)
        direct_voice_text = ""
        if voice_request and voice_request.direct_text:
            direct_voice_text = sanitize_outgoing_text(
                convert_bracket_mentions_to_tg(voice_request.content)
            )
            if not direct_voice_text:
                direct_voice_text = " ".join((voice_request.content or "").split())
        requested_display_name = extract_requested_display_name(stripped_text)
        mention_only = False
        if not stripped_text:
            mention_only = True
        log.debug("clean_text=%r stripped=%r", clean_text, stripped_text)
        reply_context = None
        if message.reply_to_message and message.reply_to_message.from_user:
            reply_source = message.reply_to_message
            reply_text = (
                reply_source.text
                or reply_source.caption
                or (reply_source.sticker.emoji if reply_source.sticker and reply_source.sticker.emoji else "")
                or "[non-text message]"
            )
            reply_text = sanitize_incoming_text(reply_text)
            reply_context = ReplyContext(
                user_id=reply_source.from_user.id,
                username=reply_source.from_user.username,
                first_name=reply_source.from_user.first_name,
                text=reply_text,
            )
        agent_input_text = clean_text
        if voice_request and not voice_request.direct_text:
            requested_topic = sanitize_incoming_text(voice_request.content)
            if requested_topic:
                agent_input_text = requested_topic
            elif reply_context and reply_context.text:
                agent_input_text = reply_context.text
        has_attachments = any(
            (
                message.photo,
                message.video,
                message.document,
                message.sticker,
                message.animation,
                message.audio,
                message.voice,
            )
        )
        turn = RouteTurn(text_clean=stripped_text, has_file=has_attachments)
        mode = route_mode(turn)
        routed_reason = route_reason(turn)
        turn_id = uuid.uuid4().hex
        turn_metrics = telemetry.start_turn(
            turn_id=turn_id,
            chat_id=chat_id,
            user_id=user_id,
            input_text=clean_text,
            has_attachments=has_attachments,
            mode=mode,
            route_reason=routed_reason,
        )

        try:
            async with get_chat_turn_lock(chat_id):
                if is_emoji_only_message(text):
                    log.debug("emoji-only message detected")
                    if media_catalog.get("items") and await media_cooldown_ok(
                        db,
                        chat_id,
                        user_id,
                        max(10, cfg.media_reply_cooldown_sec // 2),
                    ):
                        intent = await choose_emoji_reply_intent(
                            llm,
                            text,
                            media_catalog,
                            turn_id=turn_id,
                            turn_metrics=turn_metrics,
                        )
                        item = pick_media_item(media_catalog, intent)
                        if item and await send_media_item(bot, chat_id, message.message_id, item):
                            await mark_media_cooldown(db, chat_id, user_id)
                            marker = f"[emoji-reply] {item.get('emoji') or item.get('kind')}"
                            await db.add_message(
                                chat_id,
                                bot_id,
                                "assistant",
                                marker,
                                target_user_id=user_id,
                            )
                            log.info("emoji-reply: user=%s kind=%s emoji=%s set=%s", user_id, item.get('kind'), item.get('emoji'), item.get('set_name'))
                            return

                await db.add_message(chat_id, user_id, "user", stripped_text)
                if requested_display_name:
                    if cfg.creator_user_id is None:
                        await message.reply(
                            "Не могу поменять обращение: не настроен пользователь, который это одобряет."
                        )
                        return

                    if user_id == cfg.creator_user_id:
                        await db.upsert_user_profile(
                            chat_id,
                            user_id,
                            approved_display_name=requested_display_name,
                        )
                        await message.reply(
                            f"Ок. Буду обращаться к тебе как «{requested_display_name}»."
                        )
                        log.info("name-approval: self-approved user=%s chat=%s name=%r", user_id, chat_id, requested_display_name)
                        return

                    await db.create_pending_name_approval(
                        chat_id,
                        user_id,
                        requested_name=requested_display_name,
                        requested_by=user_id,
                        request_message_id=message.message_id,
                    )
                    requester = format_user_label(user_id, username, first_name)
                    approval_prompt = await message.reply(
                        f"{requester} хочет, чтобы я обращалась к нему как «{requested_display_name}».\n"
                        f"Если одобряешь, ответь на это сообщение: /approve_name\n"
                        f"Если нет, ответь: /deny_name"
                    )
                    await db.set_pending_name_approval_message(
                        chat_id,
                        user_id,
                        approval_prompt.message_id,
                    )
                    log.info("name-approval: requested user=%s chat=%s name=%r", user_id, chat_id, requested_display_name)
                    return
                answer = ""
                if direct_voice_text:
                    answer = strip_foreign_leading_mention(direct_voice_text, username)
                else:
                    await bot.send_chat_action(chat_id, "typing")

                    try:
                        answer = await run_agent(
                            llm=llm,
                            db=db,
                            chat_id=chat_id,
                            user_id=user_id,
                            input_text=agent_input_text,
                            has_attachments=has_attachments,
                            temperature=cfg.temperature,
                            top_p=cfg.top_p,
                            max_tokens=cfg.max_tokens,
                            history_limit=cfg.history_limit,
                            debug_traces=cfg.debug_traces,
                            enable_background_extraction=cfg.enable_background_extraction,
                            creator_user_id=cfg.creator_user_id,
                            creator_profile_path=cfg.creator_profile_path,
                            sender_username=username,
                            sender_first_name=first_name,
                            knowledge_dir=cfg.knowledge_dir,
                            knowledge_max_files=cfg.knowledge_max_files,
                            knowledge_max_file_bytes=cfg.knowledge_max_file_bytes,
                            mention_only=mention_only,
                            reply_context=reply_context,
                            turn_id=turn_id,
                            turn_metrics=turn_metrics,
                        )
                    except Exception as e:
                        turn_metrics.set_error(e)
                        log.error("agent failed: %s", e, exc_info=True)
                        return

                    if not answer:
                        log.warning("empty answer for chat=%s user=%s", chat_id, user_id)
                        return

                    answer = convert_bracket_mentions_to_tg(answer)
                    answer = sanitize_outgoing_text(answer)
                    answer = strip_foreign_leading_mention(answer, username)

                if not answer:
                    log.warning("empty normalized answer for chat=%s user=%s", chat_id, user_id)
                    return

                try_random_voice = (
                    voice_tts.is_configured
                    and not force_voice_reply
                    and cfg.voice_reply_prob > 0
                    and len(answer) <= cfg.voice_reply_max_chars
                    and voice_reply_cooldown_ready(chat_id, user_id, cfg.voice_reply_cooldown_sec)
                )
                should_send_voice = force_voice_reply or (
                    try_random_voice and random.random() <= cfg.voice_reply_prob
                )

                sent_voice = False
                if should_send_voice:
                    if voice_tts.is_configured:
                        try:
                            await bot.send_chat_action(chat_id, "record_voice")
                        except Exception:
                            log.debug("voice chat action failed", exc_info=True)
                        try:
                            voice_result = await voice_tts.synthesize(answer)
                            try:
                                await message.reply_voice(
                                    voice=BufferedInputFile(
                                        voice_result.audio_bytes,
                                        filename=voice_result.filename,
                                    )
                                )
                                sent_voice = True
                                log.info(
                                    "voice-reply: user=%s chars=%s kind=voice",
                                    user_id,
                                    len(answer),
                                )
                            except Exception as send_voice_err:
                                log.warning(
                                    "voice send failed, fallback to audio: %s",
                                    send_voice_err,
                                    exc_info=True,
                                )
                                await message.reply_audio(
                                    audio=BufferedInputFile(
                                        voice_result.audio_bytes,
                                        filename=voice_result.filename,
                                    ),
                                    title="Alta",
                                )
                                sent_voice = True
                                log.info(
                                    "voice-reply: user=%s chars=%s kind=audio",
                                    user_id,
                                    len(answer),
                                )
                        except ElevenLabsError as e:
                            log.warning("voice synthesis failed: %s", e)
                        except Exception as e:
                            log.warning("voice synthesis unexpected error: %s", e, exc_info=True)
                    else:
                        log.info(
                            "voice-reply skipped: ElevenLabs is not configured user=%s",
                            user_id,
                        )

                if sent_voice:
                    mark_voice_reply_cooldown(chat_id, user_id)
                else:
                    await message.reply(answer)
                await db.add_message(chat_id, bot_id, "assistant", answer, target_user_id=user_id)
                log.info("out chat=%s user=%s | %r", chat_id, user_id, answer[:120])

                if media_catalog.get("items") and await media_cooldown_ok(
                    db,
                    chat_id,
                    user_id,
                    cfg.media_reply_cooldown_sec,
                ):
                    if random.random() <= cfg.media_reply_prob:
                        intent = await choose_media_intent(
                            llm,
                            clean_text,
                            answer,
                            media_catalog,
                            turn_id=turn_id,
                            turn_metrics=turn_metrics,
                        )
                        item = pick_media_item(media_catalog, intent)
                        if item:
                            sent = await send_media_item(bot, chat_id, message.message_id, item)
                            if sent:
                                await mark_media_cooldown(db, chat_id, user_id)
                                log.info("media: user=%s kind=%s emoji=%s set=%s", user_id, item.get('kind'), item.get('emoji'), item.get('set_name'))
        finally:
            turn_metrics.finish()

    @dp.message_reaction()
    async def on_message_reaction(event: MessageReactionUpdated):
        chat_id = event.chat.id
        if not await db.is_allowed(chat_id):
            return

        if not event.user:
            return

        user = event.user
        if user.id == bot_id:
            return

        if getattr(user, "is_bot", False):
            await apply_join_verification_ban(
                chat_id=chat_id,
                user_id=user.id,
                reason="bot-reaction",
            )
            return

        new_tokens = {reaction_to_token(r) for r in (event.new_reaction or [])}
        old_tokens = {reaction_to_token(r) for r in (event.old_reaction or [])}
        added = sorted(new_tokens - old_tokens)
        if not added:
            return

        pending = _pending_join_verification.get((chat_id, user.id))
        if pending:
            banned = await apply_join_verification_ban(
                chat_id=chat_id,
                user_id=user.id,
                reason="reaction-before-verification",
                join_message_id=pending.join_message_id,
            )
            if banned:
                try:
                    await bot.send_message(
                        chat_id,
                        f"Пользователь {user.id} поставил реакцию до прохождения проверки и забанен. "
                        f"Если ошибка: /unban {user.id}",
                    )
                except Exception:
                    log.debug(
                        "join-check: failed to send reaction-ban notice chat=%s user=%s",
                        chat_id,
                        user.id,
                        exc_info=True,
                    )
            return

        if not reaction_ack_cooldown_ok(chat_id, user.id):
            return

        base_prob = 0
        if any(x in {"❤️", "👍", "🔥", "😂", "😍"} for x in added):
            base_prob += 0.15
        if len(added) >= 2:
            base_prob += 0.10
        if random.random() > min(base_prob, 0.60):
            return

        await db.upsert_user(user.id, user.username)
        who = f"@{user.username}" if user.username else (user.first_name or "ты")
        reaction_text = " ".join(added)

        try:
            ack = await llm.chat(
                system_prompt=(
                    "Ты отвечаешь на реакцию пользователя в Telegram. "
                    "Скажи 1 короткую живую фразу по-русски, без эмодзи, без пафоса."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": f"Пользователь {who} поставил реакцию: {reaction_text}",
                    }
                ],
                temperature=max(cfg.temperature, 0.8),
                top_p=cfg.top_p,
                max_tokens=45,
            )
        except Exception:
            log.debug("reaction-ack: LLM call failed", exc_info=True)
            return

        ack = sanitize_outgoing_text(convert_bracket_mentions_to_tg((ack or "").strip()))
        if not ack:
            return

        await bot.send_message(chat_id, ack, reply_to_message_id=event.message_id)
        await db.add_message(chat_id, bot_id, "assistant", ack, target_user_id=user.id)
        log.info("reaction: %s -> %s | ack=%r", who, reaction_text, ack)

    log.info("polling started")
    polling_task = asyncio.create_task(dp.start_polling(bot, handle_signals=True))
    shutdown_wait = asyncio.create_task(_SHUTDOWN_EVENT.wait())
    try:
        done, pending = await asyncio.wait(
            [polling_task, shutdown_wait],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        for st in list(_pending_join_verification.values()):
            st.task.cancel()
        _pending_join_verification.clear()
        await db.close()
        await bot.session.close()
        log.info("shutdown")


if __name__ == "__main__":
    configure_stdio()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Some Windows setups block Proactor self-pipe socket creation (WinError 10013).
    # Selector policy avoids that and is sufficient for this bot.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
