"""Cooldown management and per-chat concurrency locks.

Extracted from main.py — in-memory rate limiting with automatic eviction.
"""
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db import DB

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Module-level instances (shared state for the process)
# ---------------------------------------------------------------------------

_cooldown_calls = CooldownManager(max_size=5000, evict_after=3600.0)
_cooldown_reaction = CooldownManager(max_size=5000, evict_after=3600.0)
_cooldown_verification = CooldownManager(max_size=1000, evict_after=600.0)
_cooldown_voice = CooldownManager(max_size=5000, evict_after=3600.0)
_chat_locks = LockRegistry(max_size=2000)

MEDIA_REPLY_COOLDOWN_KEY = "media_reply"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_chat_turn_lock(chat_id: int) -> asyncio.Lock:
    return _chat_locks.get(chat_id)


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


async def media_cooldown_ok(db: "DB", chat_id: int, user_id: int, cooldown_sec: int) -> bool:
    if cooldown_sec <= 0:
        return True
    last = await db.get_cooldown_last_ts(chat_id, user_id, MEDIA_REPLY_COOLDOWN_KEY)
    return (time.time() - last) >= cooldown_sec


async def mark_media_cooldown(db: "DB", chat_id: int, user_id: int) -> None:
    await db.touch_cooldown(chat_id, user_id, MEDIA_REPLY_COOLDOWN_KEY, time.time())


def verification_pop(chat_id: int, user_id: int) -> None:
    """Remove verification cooldown entry for a user (on ban/unban/pass)."""
    _cooldown_verification.pop((chat_id, user_id))


def verification_record(chat_id: int, user_id: int) -> None:
    """Record that a verification hint was just sent."""
    _cooldown_verification.record((chat_id, user_id))


def verification_get_last(chat_id: int, user_id: int) -> float:
    """Return timestamp of the last verification hint sent (0.0 if none)."""
    return _cooldown_verification.get_last((chat_id, user_id))
