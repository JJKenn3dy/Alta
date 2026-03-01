import asyncio
import random
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.types import Message, MessageReactionUpdated

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

BOT_ALIASES = [
    "\u0430\u043b\u044c\u0442\u0430",
    "\u0430\u043b\u044c\u0442",
    "\u0430\u043b\u044c\u0442\u0443\u0448\u043a\u0430",
    "\u0430\u043b\u044c\u0442\u0443\u0448\u0435\u0447\u043a\u0430",
    "\u0430\u043b\u044c\u0442\u0435\u043d\u044c\u043a\u0430",
]

_last_call: dict[tuple[int, int], float] = {}
_last_reaction_ack: dict[tuple[int, int], float] = {}
_last_verification_hint: dict[tuple[int, int], float] = {}
_chat_turn_locks: dict[int, asyncio.Lock] = {}
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
    r"^\s*\[\s*(?:id=)?\d+(?:\s+@[A-Za-z0-9_]{4,32})?\s*\]\s*",
    re.IGNORECASE,
)
LEADING_ROLE_TAG_RE = re.compile(r"^\s*(?:system|user|assistant)\s*:\s*", re.IGNORECASE)
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


@dataclass
class JoinVerification:
    code: str
    deadline_ts: float
    task: asyncio.Task


_pending_join_verification: dict[tuple[int, int], JoinVerification] = {}


def get_chat_turn_lock(chat_id: int) -> asyncio.Lock:
    lock = _chat_turn_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _chat_turn_locks[chat_id] = lock
    return lock


def cooldown_ok(chat_id: int, user_id: int, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return True

    now = time.time()
    key = (chat_id, user_id)
    last = _last_call.get(key, 0.0)
    if now - last < cooldown_sec:
        return False
    _last_call[key] = now
    return True


def reaction_ack_cooldown_ok(chat_id: int, user_id: int, cooldown_sec: int = 180) -> bool:
    now = time.time()
    key = (chat_id, user_id)
    last = _last_reaction_ack.get(key, 0.0)
    if now - last < cooldown_sec:
        return False
    _last_reaction_ack[key] = now
    return True


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

    print("[BOOT] starting bot")
    bot = Bot(cfg.bot_token)
    await bot.delete_webhook(drop_pending_updates=True)
    print("[BOOT] pending updates dropped")
    dp = Dispatcher()

    db = DB(cfg.db_path)
    await db.connect()
    print("[BOOT] db connected")

    llm = LLM(
        cfg.llm_base_url,
        cfg.llm_api_key,
        cfg.llm_model,
        max_context_tokens=cfg.llm_context_window,
        context_safety_margin=cfg.llm_context_safety_margin,
    )
    telemetry = Telemetry(
        enabled=cfg.enable_metrics,
        debug_enabled=cfg.enable_debug_logs,
        log_path=cfg.orchestration_log_path,
    )
    print("[BOOT] llm client ready")

    media_catalog = await maybe_refresh_media_catalog(bot, cfg.media_catalog_path)
    print(
        f"[BOOT] media catalog items={len(media_catalog.get('items', []))} "
        f"failed_sets={len(media_catalog.get('failed_sets', []))}"
    )

    me = await bot.get_me()
    bot_username = me.username or ""
    bot_id = me.id
    print(f"[BOOT] bot = @{bot_username} ({bot_id})")

    for cid in cfg.allowed_chat_ids:
        await db.ensure_allowed_chat(cid, "env-allowlist")
        print(f"[BOOT] allowed chat {cid}")

    async def schedule_join_timeout(chat_id: int, user_id: int, timeout_sec: int):
        await asyncio.sleep(timeout_sec)
        key = (chat_id, user_id)
        state = _pending_join_verification.get(key)
        if not state or time.time() < state.deadline_ts:
            return

        _pending_join_verification.pop(key, None)
        _last_verification_hint.pop(key, None)

        try:
            await bot.send_message(
                chat_id,
                f"Проверка не пройдена вовремя. Удаляю пользователя {user_id} из чата.",
            )
        except Exception:
            pass

        try:
            await bot.ban_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                until_date=datetime.now(timezone.utc) + timedelta(minutes=1),
                revoke_messages=False,
            )
            await bot.unban_chat_member(chat_id=chat_id, user_id=user_id, only_if_banned=True)
            print(f"[JOIN-CHECK] kicked user={user_id} chat={chat_id} (timeout)")
        except Exception as e:
            print(f"[JOIN-CHECK] failed to kick user={user_id} chat={chat_id}: {e}")

    @dp.message(F.new_chat_members)
    async def on_new_chat_members(message: Message):
        if message.chat.type == ChatType.PRIVATE:
            return

        chat_id = message.chat.id
        if not await db.is_allowed(chat_id):
            return

        members = message.new_chat_members or []
        for m in members:
            if m.id == bot_id or m.is_bot:
                continue

            await db.upsert_user(m.id, m.username)
            code = _make_join_code()
            key = (chat_id, m.id)

            old = _pending_join_verification.pop(key, None)
            if old:
                old.task.cancel()

            timeout_sec = max(20, cfg.join_verification_timeout_sec)
            task = asyncio.create_task(schedule_join_timeout(chat_id, m.id, timeout_sec))
            _pending_join_verification[key] = JoinVerification(
                code=code, deadline_ts=time.time() + timeout_sec, task=task
            )

            who = f"@{m.username}" if m.username else (m.first_name or "новый участник")
            await message.reply(
                f"{who}, привет. Антибот-проверка: отправь в чат код {code} в течение "
                f"{timeout_sec} сек. Пример: \"{code}\".\n"
                f"Если код не придет вовремя, тебя автоматически удалит из чата."
            )
            print(f"[JOIN-CHECK] started user={m.id} chat={chat_id} code={code}")

    @dp.message(F.text)
    async def on_message(message: Message):
        print("\n[MSG] new message")
        if message.chat.type == ChatType.PRIVATE:
            print("[SKIP] private chat")
            return

        chat_id = message.chat.id
        if not await db.is_allowed(chat_id):
            print(f"[SKIP] chat {chat_id} not allowed")
            return

        if not message.from_user:
            print("[SKIP] no from_user")
            return

        user_id = message.from_user.id
        username = message.from_user.username
        first_name = message.from_user.first_name
        text = message.text or ""
        print(f"[IN] chat={chat_id} user={user_id} text={text!r}")

        await db.upsert_user(user_id, username)

        key = (chat_id, user_id)
        pending = _pending_join_verification.get(key)
        if pending:
            candidate = _extract_code_candidate(text)
            if candidate == pending.code:
                pending.task.cancel()
                _pending_join_verification.pop(key, None)
                _last_verification_hint.pop(key, None)
                await db.upsert_memory(
                    chat_id,
                    user_id,
                    "_verified_human",
                    "ok",
                    confidence=1.0,
                    importance=1.0,
                    source="join_verification",
                )
                await message.reply("Ок, проверка пройдена, добро пожаловать.")
                print(f"[JOIN-CHECK] passed user={user_id} chat={chat_id}")
            else:
                now = time.time()
                last_hint = _last_verification_hint.get(key, 0.0)
                if now - last_hint > 10:
                    _last_verification_hint[key] = now
                    remaining = max(1, int(pending.deadline_ts - now))
                    await message.reply(
                        f"Сначала пройди проверку: отправь код {pending.code}. "
                        f"Осталось примерно {remaining} сек."
                    )
            return

        if not is_trigger(message, bot_username, bot_id):
            print("[SKIP] no trigger")
            return

        if not cooldown_ok(chat_id, user_id, cfg.user_cooldown_sec):
            print("[SKIP] cooldown")
            return

        clean_text = strip_mention(text, bot_username)
        mention_only = False
        if not clean_text:
            mention_only = True
        print(f"[CTX] clean_text={clean_text!r}")
        reply_context = None
        if message.reply_to_message and message.reply_to_message.from_user:
            reply_source = message.reply_to_message
            reply_text = (
                reply_source.text
                or reply_source.caption
                or (reply_source.sticker.emoji if reply_source.sticker else "")
                or "[non-text message]"
            )
            reply_context = ReplyContext(
                user_id=reply_source.from_user.id,
                username=reply_source.from_user.username,
                first_name=reply_source.from_user.first_name,
                text=reply_text,
            )
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
        turn = RouteTurn(text_clean=clean_text, has_file=has_attachments)
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
                    print("[CTX] emoji-only message detected")
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
                            print(
                                f"[EMOJI-REPLY] user={user_id} kind={item.get('kind')} "
                                f"emoji={item.get('emoji')} set={item.get('set_name')}"
                            )
                            return

                await db.add_message(chat_id, user_id, "user", clean_text)
                await bot.send_chat_action(chat_id, "typing")

                try:
                    answer = await run_agent(
                        llm=llm,
                        db=db,
                        chat_id=chat_id,
                        user_id=user_id,
                        input_text=clean_text,
                        has_attachments=has_attachments,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_tokens=cfg.max_tokens,
                        history_limit=cfg.history_limit,
                        debug_traces=cfg.debug_traces,
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
                    print(f"[ERROR] agent failed: {e}")
                    return

                if not answer:
                    print("[WARN] empty answer")
                    return

                answer = sanitize_outgoing_text(convert_bracket_mentions_to_tg(answer))
                await message.reply(answer)
                await db.add_message(chat_id, bot_id, "assistant", answer, target_user_id=user_id)
                print(f"[OUT] {answer!r}")

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
                                print(
                                    f"[MEDIA] user={user_id} kind={item.get('kind')} "
                                    f"emoji={item.get('emoji')} set={item.get('set_name')}"
                                )
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

        new_tokens = {reaction_to_token(r) for r in (event.new_reaction or [])}
        old_tokens = {reaction_to_token(r) for r in (event.old_reaction or [])}
        added = sorted(new_tokens - old_tokens)
        if not added:
            return

        if not reaction_ack_cooldown_ok(chat_id, user.id):
            return

        base_prob = 0.18
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
            return

        ack = sanitize_outgoing_text(convert_bracket_mentions_to_tg((ack or "").strip()))
        if not ack:
            return

        await bot.send_message(chat_id, ack, reply_to_message_id=event.message_id)
        await db.add_message(chat_id, bot_id, "assistant", ack, target_user_id=user.id)
        print(f"[REACTION] {who} -> {reaction_text} | ack={ack!r}")

    print("[BOOT] polling started")
    try:
        await dp.start_polling(bot)
    finally:
        for st in list(_pending_join_verification.values()):
            st.task.cancel()
        _pending_join_verification.clear()
        await db.close()
        await bot.session.close()
        print("[BOOT] shutdown")


if __name__ == "__main__":
    configure_stdio()
    # Some Windows setups block Proactor self-pipe socket creation (WinError 10013).
    # Selector policy avoids that and is sufficient for this bot.
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
