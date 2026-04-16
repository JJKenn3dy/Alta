"""Join verification, ban/unban helpers, and spam detection.

Extracted from main.py — moderation utilities with no Telegram Bot dependency.
"""
from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

CODE_RE = re.compile(r"\b(\d{4})\b")
UNBAN_COMMAND_RE = re.compile(
    r"^/unban(?:@[A-Za-z][A-Za-z0-9_]{4,31})?(?:\s+(?P<target>\S+))?\s*$",
    re.IGNORECASE,
)
_SPAM_URL_RE = re.compile(
    r"(?:https?://|t\.me/|tg://|bit\.ly/|goo\.gl/|@[A-Za-z][A-Za-z0-9_]{4,})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Join verification state
# ---------------------------------------------------------------------------


@dataclass
class JoinVerification:
    code: str
    deadline_ts: float
    task: asyncio.Task
    join_message_id: int


_pending_join_verification: dict[tuple[int, int], JoinVerification] = {}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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
