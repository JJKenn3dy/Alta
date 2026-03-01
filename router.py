import time
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class RouteDecision:
    should_respond: bool
    reason: str


def strip_bot_mention(text: str, bot_username: str) -> str:
    if not text:
        return text
    # удаляем @botname из текста (в любом месте)
    pattern = re.compile(rf"@{re.escape(bot_username)}\b", re.IGNORECASE)
    cleaned = pattern.sub("", text)
    return cleaned.strip()


class Router:
    def __init__(self, bot_username: str, bot_user_id: int, cooldown_sec: int):
        self.bot_username = bot_username.lower().lstrip("@")
        self.bot_user_id = bot_user_id
        self.cooldown_sec = cooldown_sec
        # (chat_id, user_id) -> last_ts
        self._last: dict[tuple[int, int], float] = {}

    def _cooldown_ok(self, chat_id: int, user_id: int) -> bool:
        key = (chat_id, user_id)
        now = time.time()
        last = self._last.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self._last[key] = now
        return True

    def _is_mentioning_bot(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return f"@{self.bot_username}" in text.lower()

    def decide(self, chat_id: int, is_private: bool, text: Optional[str], reply_to_user_id: Optional[int], from_user_id: int) -> RouteDecision:
        # личка - всегда игнор
        if is_private:
            return RouteDecision(False, "private_chat_disabled")

        # reply на сообщение бота
        is_reply_to_bot = (reply_to_user_id == self.bot_user_id)

        # mention
        is_mention = self._is_mentioning_bot(text)

        if not (is_reply_to_bot or is_mention):
            return RouteDecision(False, "no_trigger")

        if not self._cooldown_ok(chat_id, from_user_id):
            return RouteDecision(False, "cooldown")

        # пустой пинг
        if is_mention and (text or "").strip().lower() in {f"@{self.bot_username}", f"@{self.bot_username}!"}:
            return RouteDecision(False, "empty_ping")

        return RouteDecision(True, "ok")