import argparse
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
RECENT_MESSAGES_LIMIT = 40
RECENT_TURNS_LIMIT = 30
TOP_USERS_LIMIT = 12
TOP_FACTS_LIMIT = 20


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def try_parse_iso(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        if "T" in normalized or "+" in normalized:
            return datetime.fromisoformat(normalized)
        return datetime.fromisoformat(normalized.replace(" ", "T"))
    except ValueError:
        return None


def short_text(value: str | None, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def percentile95(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return float(ordered[index])


@dataclass
class LogSnapshot:
    path: str
    exists: bool
    size_bytes: int
    updated_at: str | None
    total_events: int
    total_turns: int
    recent_turns: list[dict[str, Any]]
    turns_by_mode: dict[str, int]
    editor_triggers: dict[str, int]
    daily_turns: list[dict[str, Any]]
    overview: dict[str, Any]
    recent_errors: list[dict[str, Any]]


class LogReader:
    def __init__(self, path: str):
        self.path = Path(path)
        self._cache_key: tuple[bool, int, int] | None = None
        self._cache_value: LogSnapshot | None = None

    def snapshot(self) -> LogSnapshot:
        exists = self.path.exists()
        stat = self.path.stat() if exists else None
        cache_key = (
            exists,
            int(stat.st_mtime_ns) if stat else 0,
            int(stat.st_size) if stat else 0,
        )
        if self._cache_key == cache_key and self._cache_value is not None:
            return self._cache_value

        snapshot = self._build_snapshot(exists, stat)
        self._cache_key = cache_key
        self._cache_value = snapshot
        return snapshot

    def _build_snapshot(self, exists: bool, stat: Any) -> LogSnapshot:
        if not exists:
            return LogSnapshot(
                path=str(self.path),
                exists=False,
                size_bytes=0,
                updated_at=None,
                total_events=0,
                total_turns=0,
                recent_turns=[],
                turns_by_mode={},
                editor_triggers={},
                daily_turns=[],
                overview={},
                recent_errors=[],
            )

        total_events = 0
        turn_end_events: list[dict[str, Any]] = []
        postprocess_events_by_turn: dict[str, dict[str, Any]] = {}
        recent_errors: list[dict[str, Any]] = []
        turns_by_mode: Counter[str] = Counter()
        editor_triggers: Counter[str] = Counter()
        daily_turns_map: defaultdict[str, int] = defaultdict(int)

        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                payload = line.strip()
                if not payload:
                    continue
                total_events += 1
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                event_type = str(event.get("event") or "")
                if event_type == "postprocess_end":
                    turn_id = str(event.get("turn_id") or "").strip()
                    if turn_id:
                        postprocess_events_by_turn[turn_id] = event
                    error_messages = event.get("error_messages") or []
                    if error_messages:
                        recent_errors.append(
                            {
                                "ts": event.get("ts"),
                                "chat_id": event.get("chat_id"),
                                "user_id": event.get("user_id"),
                                "mode": event.get("mode"),
                                "error_type": "postprocess",
                                "error_message": short_text("; ".join(str(item) for item in error_messages), 180),
                                "input_preview": short_text(event.get("input_preview"), 120),
                            }
                        )
                    continue
                if event_type != "turn_end":
                    continue
                turn_end_events.append(event)
                turns_by_mode[str(event.get("mode") or "unknown")] += 1
                trigger_value = str(event.get("editor_trigger") or "").strip()
                if trigger_value:
                    for part in trigger_value.split(","):
                        trigger = part.strip()
                        if trigger:
                            editor_triggers[trigger] += 1
                day = str(event.get("ts") or "")[:10]
                if day:
                    daily_turns_map[day] += 1
                if event.get("error_type"):
                    recent_errors.append(
                        {
                            "ts": event.get("ts"),
                            "chat_id": event.get("chat_id"),
                            "user_id": event.get("user_id"),
                            "mode": event.get("mode"),
                            "error_type": event.get("error_type"),
                            "error_message": short_text(event.get("error_message"), 180),
                            "input_preview": short_text(event.get("input_preview"), 120),
                        }
                    )

        merged_turn_events: list[dict[str, Any]] = []
        for event in turn_end_events:
            merged = dict(event)
            postprocess_event = postprocess_events_by_turn.get(str(event.get("turn_id") or "").strip())
            if postprocess_event:
                merged["summary_updated"] = bool(event.get("summary_updated")) or bool(
                    postprocess_event.get("summary_updated")
                )
                merged["memory_updated"] = bool(event.get("memory_updated")) or bool(
                    postprocess_event.get("memory_updated")
                )
                merged["profile_updated"] = bool(event.get("profile_updated")) or bool(
                    postprocess_event.get("profile_updated")
                )
                merged["memory_extract_items_changed"] = max(
                    to_int(event.get("memory_extract_items_changed")),
                    to_int(postprocess_event.get("memory_extract_items_changed")),
                )
                merged["profile_extract_fields_changed"] = max(
                    to_int(event.get("profile_extract_fields_changed")),
                    to_int(postprocess_event.get("profile_extract_fields_changed")),
                )
                merged["summary_write_count"] = max(
                    to_int(event.get("summary_write_count")),
                    to_int(postprocess_event.get("summary_write_count")),
                )
            merged["postprocess_pending"] = bool(event.get("postprocess_pending")) and not bool(postprocess_event)
            merged_turn_events.append(merged)

        total_tokens = [to_int(item.get("total_tokens")) for item in merged_turn_events]
        total_latency = [to_int(item.get("total_latency_ms")) for item in merged_turn_events]
        total_calls = [to_int(item.get("total_llm_calls")) for item in merged_turn_events]

        overview = {
            "avg_tokens": round(statistics.fmean(total_tokens), 1) if total_tokens else 0.0,
            "p95_tokens": round(percentile95(total_tokens), 1) if total_tokens else 0.0,
            "avg_latency_ms": round(statistics.fmean(total_latency), 1) if total_latency else 0.0,
            "p95_latency_ms": round(percentile95(total_latency), 1) if total_latency else 0.0,
            "avg_llm_calls": round(statistics.fmean(total_calls), 2) if total_calls else 0.0,
            "error_turns": sum(1 for item in merged_turn_events if item.get("error_type")),
            "fallback_turns": sum(1 for item in merged_turn_events if item.get("fallback_used")),
            "summary_updates": sum(1 for item in merged_turn_events if item.get("summary_updated")),
            "memory_updates": sum(1 for item in merged_turn_events if item.get("memory_updated")),
            "profile_updates": sum(1 for item in merged_turn_events if item.get("profile_updated")),
            "editor_attempts": sum(1 for item in merged_turn_events if item.get("editor_attempted")),
        }

        recent_turns: list[dict[str, Any]] = []
        for event in merged_turn_events[-RECENT_TURNS_LIMIT:]:
            recent_turns.append(
                {
                    "ts": event.get("ts"),
                    "chat_id": event.get("chat_id"),
                    "user_id": event.get("user_id"),
                    "mode": event.get("mode"),
                    "input_preview": short_text(event.get("input_preview"), 100),
                    "total_tokens": to_int(event.get("total_tokens")),
                    "total_latency_ms": to_int(event.get("total_latency_ms")),
                    "total_llm_calls": to_int(event.get("total_llm_calls")),
                    "fallback_used": bool(event.get("fallback_used")),
                    "summary_updated": bool(event.get("summary_updated")),
                    "memory_updated": bool(event.get("memory_updated")),
                    "profile_updated": bool(event.get("profile_updated")),
                    "postprocess_pending": bool(event.get("postprocess_pending")),
                    "error_type": event.get("error_type"),
                }
            )

        daily_turns = [
            {"day": day, "turns": count}
            for day, count in sorted(daily_turns_map.items())[-14:]
        ]

        return LogSnapshot(
            path=str(self.path),
            exists=True,
            size_bytes=int(stat.st_size),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            total_events=total_events,
            total_turns=len(turn_end_events),
            recent_turns=list(reversed(recent_turns)),
            turns_by_mode=dict(sorted(turns_by_mode.items())),
            editor_triggers=dict(sorted(editor_triggers.items())),
            daily_turns=daily_turns,
            overview=overview,
            recent_errors=list(reversed(recent_errors[-10:])),
        )


class DashboardStore:
    def __init__(self, db_path: str, log_path: str):
        self.db_path = Path(db_path)
        self.log_reader = LogReader(log_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_overview(self) -> dict[str, Any]:
        log_snapshot = self.log_reader.snapshot()
        db_exists = self.db_path.exists()
        db_size = self.db_path.stat().st_size if db_exists else 0
        counts: dict[str, int] = {}
        latest_message_ts = None
        latest_profile_ts = None
        latest_fact_ts = None
        if db_exists:
            with self._connect() as conn:
                for table in (
                    "messages",
                    "users",
                    "user_profiles",
                    "user_facts",
                    "memories",
                    "chat_summary",
                    "user_chat_summaries",
                    "user_documents",
                    "pending_name_approvals",
                    "context_files",
                    "context_chunks",
                ):
                    counts[table] = to_int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                latest_message_ts = conn.execute("SELECT MAX(ts) FROM messages").fetchone()[0]
                latest_profile_ts = conn.execute("SELECT MAX(updated_at) FROM user_profiles").fetchone()[0]
                latest_fact_ts = conn.execute("SELECT MAX(updated_at) FROM user_facts").fetchone()[0]

        return {
            "generated_at": utc_now_iso(),
            "db": {
                "path": str(self.db_path),
                "exists": db_exists,
                "size_bytes": int(db_size),
                "counts": counts,
                "latest_message_ts": latest_message_ts,
                "latest_profile_ts": latest_profile_ts,
                "latest_fact_ts": latest_fact_ts,
            },
            "log": {
                "path": log_snapshot.path,
                "exists": log_snapshot.exists,
                "size_bytes": log_snapshot.size_bytes,
                "updated_at": log_snapshot.updated_at,
                "total_events": log_snapshot.total_events,
                "total_turns": log_snapshot.total_turns,
                "overview": log_snapshot.overview,
                "turns_by_mode": log_snapshot.turns_by_mode,
                "editor_triggers": log_snapshot.editor_triggers,
                "daily_turns": log_snapshot.daily_turns,
            },
        }

    def list_chats(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.chat_id AS chat_id,
                    COUNT(*) AS messages_count,
                    SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                    SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS user_messages,
                    COUNT(DISTINCT CASE WHEN m.role='user' THEN m.user_id END) AS active_users,
                    MAX(m.ts) AS last_message_ts,
                    COALESCE(LENGTH(cs.summary), 0) AS summary_chars,
                    COALESCE(ac.title, '') AS title
                FROM messages m
                LEFT JOIN chat_summary cs ON cs.chat_id = m.chat_id
                LEFT JOIN allowed_chats ac ON ac.chat_id = m.chat_id
                GROUP BY m.chat_id, cs.summary, ac.title
                ORDER BY last_message_ts DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_chat_details(self, chat_id: int) -> dict[str, Any]:
        if not self.db_path.exists():
            return self._empty_chat(chat_id)
        with self._connect() as conn:
            chat_row = conn.execute(
                """
                SELECT
                    m.chat_id AS chat_id,
                    COUNT(*) AS messages_count,
                    SUM(CASE WHEN m.role='assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                    SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END) AS user_messages,
                    COUNT(DISTINCT CASE WHEN m.role='user' THEN m.user_id END) AS active_users,
                    MAX(m.ts) AS last_message_ts,
                    COALESCE(cs.summary, '') AS summary,
                    COALESCE(ac.title, '') AS title
                FROM messages m
                LEFT JOIN chat_summary cs ON cs.chat_id = m.chat_id
                LEFT JOIN allowed_chats ac ON ac.chat_id = m.chat_id
                WHERE m.chat_id = ?
                GROUP BY m.chat_id, cs.summary, ac.title
                """,
                (chat_id,),
            ).fetchone()
            if chat_row is None:
                return self._empty_chat(chat_id)

            recent_messages = conn.execute(
                """
                SELECT m.id, m.role, m.user_id, m.target_user_id, m.text, m.ts,
                       COALESCE(u.username, '') AS username
                FROM messages m
                LEFT JOIN users u ON u.user_id = m.user_id
                WHERE m.chat_id = ?
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (chat_id, RECENT_MESSAGES_LIMIT),
            ).fetchall()

            top_users = conn.execute(
                """
                SELECT m.user_id, COALESCE(u.username, '') AS username, COUNT(*) AS messages_count,
                       MAX(m.ts) AS last_message_ts,
                       COALESCE(up.display_name, '') AS display_name,
                       COALESCE(up.nickname, '') AS nickname,
                       COALESCE(up.response_preference, '') AS response_preference
                FROM messages m
                LEFT JOIN users u ON u.user_id = m.user_id
                LEFT JOIN user_profiles up ON up.chat_id = m.chat_id AND up.user_id = m.user_id
                WHERE m.chat_id = ? AND m.role = 'user'
                GROUP BY m.user_id, u.username, up.display_name, up.nickname, up.response_preference
                ORDER BY messages_count DESC, last_message_ts DESC
                LIMIT ?
                """,
                (chat_id, TOP_USERS_LIMIT),
            ).fetchall()

            profiles = conn.execute(
                """
                SELECT up.*, COALESCE(u.username, '') AS username
                FROM user_profiles up
                LEFT JOIN users u ON u.user_id = up.user_id
                WHERE up.chat_id = ?
                ORDER BY up.updated_at DESC
                LIMIT 40
                """,
                (chat_id,),
            ).fetchall()

            facts = conn.execute(
                """
                SELECT uf.id, uf.user_id, COALESCE(u.username, '') AS username, uf.fact_type,
                       uf.value, uf.confidence, uf.importance, uf.evidence, uf.source, uf.updated_at
                FROM user_facts uf
                LEFT JOIN users u ON u.user_id = uf.user_id
                WHERE uf.chat_id = ?
                ORDER BY uf.updated_at DESC, uf.importance DESC
                LIMIT ?
                """,
                (chat_id, TOP_FACTS_LIMIT),
            ).fetchall()

            user_summaries = conn.execute(
                """
                SELECT us.user_id, COALESCE(u.username, '') AS username, us.summary, us.updated_at
                FROM user_chat_summaries us
                LEFT JOIN users u ON u.user_id = us.user_id
                WHERE us.chat_id = ?
                ORDER BY us.updated_at DESC
                LIMIT 20
                """,
                (chat_id,),
            ).fetchall()

            pending_name_approvals = conn.execute(
                """
                SELECT pna.*, COALESCE(u.username, '') AS username
                FROM pending_name_approvals pna
                LEFT JOIN users u ON u.user_id = pna.user_id
                WHERE pna.chat_id = ?
                ORDER BY pna.updated_at DESC
                LIMIT 20
                """,
                (chat_id,),
            ).fetchall()

            documents = conn.execute(
                """
                SELECT ud.user_id, COALESCE(u.username, '') AS username, ud.document_text, ud.updated_at
                FROM user_documents ud
                LEFT JOIN users u ON u.user_id = ud.user_id
                WHERE ud.chat_id = ?
                ORDER BY ud.updated_at DESC
                LIMIT 20
                """,
                (chat_id,),
            ).fetchall()

        return {
            "chat_id": chat_id,
            "exists": True,
            "chat": dict(chat_row),
            "recent_messages": [dict(row) for row in recent_messages],
            "top_users": [dict(row) for row in top_users],
            "profiles": [dict(row) for row in profiles],
            "facts": [dict(row) for row in facts],
            "user_summaries": [dict(row) for row in user_summaries],
            "pending_name_approvals": [dict(row) for row in pending_name_approvals],
            "documents": [dict(row) for row in documents],
        }

    def _empty_chat(self, chat_id: int) -> dict[str, Any]:
        return {
            "chat_id": chat_id,
            "exists": False,
            "chat": {},
            "recent_messages": [],
            "top_users": [],
            "profiles": [],
            "facts": [],
            "user_summaries": [],
            "pending_name_approvals": [],
            "documents": [],
        }

    def get_memories(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"recent": [], "top_users": []}
        with self._connect() as conn:
            recent = conn.execute(
                """
                SELECT m.chat_id, m.user_id, COALESCE(u.username, '') AS username,
                       m.key, m.value, m.confidence, m.importance, m.source, m.updated_at
                FROM memories m
                LEFT JOIN users u ON u.user_id = m.user_id
                ORDER BY m.updated_at DESC
                LIMIT 40
                """
            ).fetchall()
            top_users = conn.execute(
                """
                SELECT m.chat_id, m.user_id, COALESCE(u.username, '') AS username,
                       COUNT(*) AS memory_count, ROUND(AVG(m.importance), 3) AS avg_importance,
                       MAX(m.updated_at) AS last_updated_at
                FROM memories m
                LEFT JOIN users u ON u.user_id = m.user_id
                GROUP BY m.chat_id, m.user_id, u.username
                ORDER BY memory_count DESC, last_updated_at DESC
                LIMIT 20
                """
            ).fetchall()
        return {"recent": [dict(row) for row in recent], "top_users": [dict(row) for row in top_users]}

    def get_recent_turns(self) -> dict[str, Any]:
        snapshot = self.log_reader.snapshot()
        return {
            "recent_turns": snapshot.recent_turns,
            "recent_errors": snapshot.recent_errors,
            "turns_by_mode": snapshot.turns_by_mode,
            "editor_triggers": snapshot.editor_triggers,
        }


HTML_PAGE = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alta Dashboard</title>
  <style>
    :root{--bg:#f3ecdf;--panel:rgba(255,249,239,.82);--panel-strong:#fffaf3;--line:rgba(78,53,40,.12);--line-strong:rgba(78,53,40,.22);--ink:#241913;--muted:#766256;--accent:#c45a28;--accent-2:#e2a26f;--accent-soft:rgba(196,90,40,.12);--ok:#2d7a56;--warn:#9e6d12;--bad:#a73b3b;--shadow:0 22px 60px rgba(60,32,18,.08);--radius:26px;--sans:"Bahnschrift","Segoe UI Variable Text","Trebuchet MS",sans-serif;--display:"Georgia","Times New Roman",serif;--mono:"Cascadia Code","Fira Code",monospace}
    *{box-sizing:border-box}
    html{scroll-behavior:smooth}
    body{margin:0;color:var(--ink);font:14px/1.48 var(--sans);background:
      radial-gradient(circle at 8% 12%, rgba(196,90,40,.18), transparent 22%),
      radial-gradient(circle at 92% 18%, rgba(113,78,54,.14), transparent 26%),
      radial-gradient(circle at 50% 100%, rgba(196,90,40,.08), transparent 32%),
      linear-gradient(180deg,#f8f2e9 0%,#f3ecdf 42%,#efe6d8 100%)}
    body:before,body:after{content:"";position:fixed;inset:auto;pointer-events:none;z-index:0}
    body:before{top:24px;right:24px;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle, rgba(255,255,255,.5), transparent 66%);filter:blur(12px)}
    body:after{left:-80px;bottom:-110px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle, rgba(196,90,40,.14), transparent 62%)}
    .wrap{position:relative;z-index:1;width:min(1480px,calc(100vw - 28px));margin:16px auto 30px}
    .hero,.grid{display:grid;gap:16px}
    .hero{grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);margin-bottom:16px}
    .grid{grid-template-columns:repeat(12,1fr)}
    .card{position:relative;overflow:hidden;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px;box-shadow:var(--shadow);backdrop-filter:blur(16px);transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease}
    .card:hover{transform:translateY(-2px);box-shadow:0 28px 70px rgba(60,32,18,.11);border-color:var(--line-strong)}
    .card:before{content:"";position:absolute;inset:0 0 auto 0;height:1px;background:linear-gradient(90deg,transparent, rgba(255,255,255,.9), transparent)}
    .hero-main{min-height:220px;background:
      radial-gradient(circle at right top, rgba(226,162,111,.35), transparent 30%),
      linear-gradient(140deg, rgba(255,250,244,.9), rgba(255,244,233,.68))}
    .hero-main:after{content:"";position:absolute;right:-36px;top:-34px;width:180px;height:180px;border-radius:28px;background:linear-gradient(180deg, rgba(196,90,40,.18), rgba(196,90,40,.02));transform:rotate(18deg)}
    .hero-side{display:grid;align-content:space-between;gap:14px}
    .hero-badge{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border-radius:999px;background:rgba(36,25,19,.07);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
    .hero-title{margin:12px 0 10px;font:600 48px/1 var(--display);letter-spacing:-.05em;max-width:680px}
    .hero-copy{max-width:620px;font-size:16px;color:var(--muted)}
    .hero-copy strong{color:var(--ink)}
    .chips,.toolbar,.head,.split-head{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .chips{margin-top:18px}
    .chip,.tag{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;border-radius:999px;background:var(--accent-soft);font-size:12px;border:1px solid rgba(196,90,40,.12)}
    .tag{padding:5px 9px;margin:6px 6px 0 0;background:rgba(36,25,19,.06);border:none}
    .tag.ok{background:rgba(45,122,86,.12);color:var(--ok)}
    .tag.warn{background:rgba(158,109,18,.14);color:var(--warn)}
    .tag.bad{background:rgba(167,59,59,.14);color:var(--bad)}
    .snapshot-card{display:grid;gap:14px}
    .snapshot-meta{display:grid;gap:10px}
    .snapshot-note{padding:12px 14px;border-radius:18px;background:linear-gradient(180deg, rgba(255,255,255,.7), rgba(255,248,240,.42));border:1px solid var(--line)}
    .eyebrow{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
    .head{justify-content:space-between;margin-bottom:12px}
    .section-title{font-size:18px;letter-spacing:-.025em}
    .section-sub{color:var(--muted);font-size:12px}
    .toolbar{justify-content:space-between;padding:14px 16px;margin-bottom:16px;background:rgba(255,249,239,.58);border:1px solid var(--line);border-radius:22px;backdrop-filter:blur(12px)}
    .toolbar .left,.toolbar .right{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
    .toolbar label,.toolbar span{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
    select,button{appearance:none;border:1px solid var(--line);border-radius:14px;padding:11px 14px;background:var(--panel-strong);color:var(--ink);font:600 13px var(--sans);box-shadow:inset 0 1px 0 rgba(255,255,255,.8)}
    select{min-width:180px}
    button{background:linear-gradient(180deg,#38241b,#1f1511);color:#fff6ef;cursor:pointer;border-color:#1f1511;box-shadow:0 10px 24px rgba(31,21,17,.2)}
    button:hover{transform:translateY(-1px)}
    .s3{grid-column:span 3}.s4{grid-column:span 4}.s5{grid-column:span 5}.s6{grid-column:span 6}.s7{grid-column:span 7}
    .metric{display:flex;flex-direction:column;justify-content:space-between;min-height:132px;background:
      linear-gradient(180deg, rgba(255,255,255,.62), rgba(255,246,236,.44))}
    .metric:after{content:"";position:absolute;right:16px;top:14px;width:54px;height:54px;border-radius:16px;background:linear-gradient(180deg, rgba(196,90,40,.12), rgba(196,90,40,.02))}
    .metric .k{position:relative;z-index:1;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.12em}
    .metric .v{position:relative;z-index:1;font:600 38px/1 var(--display);letter-spacing:-.06em;margin:12px 0 8px}
    .metric .muted{position:relative;z-index:1}
    .muted{color:var(--muted)}
    .stack{display:grid;gap:12px}
    .item{padding:14px 15px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(180deg, rgba(255,255,255,.58), rgba(255,250,243,.36));transition:transform .16s ease,border-color .16s ease}
    .item:hover{transform:translateY(-1px);border-color:var(--line-strong)}
    .item.clickable{cursor:pointer}
    .top{display:flex;justify-content:space-between;gap:10px;align-items:flex-start;color:var(--muted);font-size:12px;margin-bottom:8px}
    .title{font-size:15px;font-weight:700;letter-spacing:-.015em;margin-bottom:6px}
    .body{white-space:pre-wrap;word-break:break-word}
    .bar-group{display:grid;gap:10px}
    .bar{display:grid;grid-template-columns:118px 1fr 60px;gap:10px;align-items:center}
    .track{height:12px;border-radius:999px;background:rgba(74,53,41,.08);overflow:hidden;box-shadow:inset 0 1px 3px rgba(74,53,41,.08)}
    .fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent),var(--accent-2))}
    .days{display:flex;align-items:end;gap:12px;height:190px;padding-top:8px}
    .day{flex:1;display:flex;flex-direction:column;align-items:center;gap:7px;font-size:11px;color:var(--muted)}
    .dayfill{width:100%;min-height:10px;border-radius:14px 14px 6px 6px;background:linear-gradient(180deg,#e08c57,#9f4420);box-shadow:0 10px 18px rgba(159,68,32,.18)}
    .empty{padding:12px 0;color:var(--muted)}
    .mono{font:12px/1.45 var(--mono)}
    .split-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .soft-label{display:inline-flex;align-items:center;gap:8px;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}
    .footer-note{font-size:12px;color:var(--muted)}
    .pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 0 rgba(196,90,40,.4);animation:pulse 1.8s infinite}
    @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(196,90,40,.4)}70%{box-shadow:0 0 0 10px rgba(196,90,40,0)}100%{box-shadow:0 0 0 0 rgba(196,90,40,0)}}
    @media (max-width:1140px){.hero{grid-template-columns:1fr}.s3,.s4,.s5,.s6,.s7{grid-column:span 12}.split-grid{grid-template-columns:1fr}}
    @media (max-width:760px){.wrap{width:min(100vw - 14px,1480px)}.card{padding:14px;border-radius:20px}.hero-title{font-size:34px}.metric .v{font-size:31px}.bar{grid-template-columns:88px 1fr 46px}.days{height:170px;gap:8px}.toolbar{padding:12px}.toolbar .left,.toolbar .right{width:100%}select{min-width:unset;width:100%}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <section class="card hero-main">
        <span class="hero-badge"><span class="pulse"></span>Live Bot Observatory</span>
        <h1 class="hero-title">Alta Dashboard</h1>
        <div class="hero-copy">Визуальная панель для <strong>бота, базы, памяти, summary и LLM-оркестрации</strong>. Без внешних библиотек, но уже с нормальной иерархией и быстрым обзором состояния.</div>
        <div class="chips" id="status"></div>
      </section>
      <section class="card hero-side">
        <div class="head">
          <div>
            <div class="soft-label">Снимок системы</div>
            <h2 class="section-title">Текущий срез</h2>
          </div>
          <button id="reload">Обновить</button>
        </div>
        <div class="snapshot-card">
          <div class="snapshot-note">
            <div class="top"><span>Сгенерировано</span><span id="generatedAt">-</span></div>
            <div class="body mono" id="meta">Загрузка...</div>
          </div>
          <div class="snapshot-meta">
            <div class="footer-note">Ниже можно быстро провалиться в чаты, посмотреть свежие turns, ошибки, профили и то, что реально лежит в БД.</div>
          </div>
        </div>
      </section>
    </div>

    <div class="toolbar">
      <div class="left">
        <label for="chatSelect">Выбранный чат</label>
        <select id="chatSelect"></select>
      </div>
      <div class="right">
        <label for="refreshSelect">Автообновление</label>
        <select id="refreshSelect"><option value="0">выкл</option><option value="5000">5 сек</option><option value="7000" selected>7 сек</option><option value="15000">15 сек</option></select>
      </div>
    </div>

    <div class="grid">
      <section class="card metric s3"><div class="k">Сообщения</div><div class="v" id="messagesCount">0</div><div class="muted" id="messagesMeta">-</div></section>
      <section class="card metric s3"><div class="k">Пользователи</div><div class="v" id="usersCount">0</div><div class="muted" id="usersMeta">-</div></section>
      <section class="card metric s3"><div class="k">Факты / Memory</div><div class="v" id="factMemoryCount">0 / 0</div><div class="muted" id="factsMeta">-</div></section>
      <section class="card metric s3"><div class="k">Turn End</div><div class="v" id="turnCount">0</div><div class="muted" id="turnMeta">-</div></section>

      <section class="card metric s4"><div class="k">Latency</div><div class="v" id="latencyValue">0 ms</div><div class="muted" id="latencyMeta">-</div></section>
      <section class="card metric s4"><div class="k">Tokens</div><div class="v" id="tokenValue">0</div><div class="muted" id="tokenMeta">-</div></section>
      <section class="card metric s4"><div class="k">LLM Calls</div><div class="v" id="callValue">0</div><div class="muted" id="callMeta">-</div></section>

      <section class="card s5">
        <div class="head"><div><div class="soft-label">Маршрутизация</div><h2 class="section-title">Режимы</h2></div></div>
        <div id="modes" class="bar-group"></div>
      </section>
      <section class="card s7">
        <div class="head"><div><div class="soft-label">Нагрузка</div><h2 class="section-title">Turn activity</h2></div></div>
        <div id="days" class="days"></div>
      </section>

      <section class="card s5">
        <div class="head"><div><div class="soft-label">Навигация</div><h2 class="section-title">Чаты</h2></div></div>
        <div id="chatList" class="stack"></div>
      </section>
      <section class="card s7">
        <div class="head"><div><div class="soft-label">Детализация</div><h2 class="section-title">Текущий чат</h2></div></div>
        <div id="chatMeta" class="stack"></div>
      </section>

      <section class="card s6">
        <div class="head"><div><div class="soft-label">Оркестрация</div><h2 class="section-title">Последние turns</h2></div></div>
        <div id="turns" class="stack"></div>
      </section>
      <section class="card s6">
        <div class="head"><div><div class="soft-label">Диагностика</div><h2 class="section-title">Ошибки и editor</h2></div></div>
        <div class="split-grid">
          <div id="errors" class="stack"></div>
          <div id="editor" class="bar-group"></div>
        </div>
      </section>

      <section class="card s6">
        <div class="head"><div><div class="soft-label">Лента</div><h2 class="section-title">Сообщения</h2></div></div>
        <div id="messages" class="stack"></div>
      </section>
      <section class="card s6">
        <div class="head"><div><div class="soft-label">Участники</div><h2 class="section-title">Пользователи</h2></div></div>
        <div id="users" class="stack"></div>
      </section>

      <section class="card s6">
        <div class="head"><div><div class="soft-label">Память о людях</div><h2 class="section-title">Профили и summary</h2></div></div>
        <div id="profiles" class="stack"></div>
      </section>
      <section class="card s6">
        <div class="head"><div><div class="soft-label">Извлеченные сущности</div><h2 class="section-title">Факты и memory</h2></div></div>
        <div id="facts" class="stack"></div>
      </section>
    </div>
  </div>
  <script>
    const $ = id => document.getElementById(id);
    const state = { chats: [], selectedChatId: null, timer: null, memories: null };
    const esc = v => String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
    const fmtTs = v => { if (!v) return '-'; const d = new Date(v); return Number.isNaN(d.getTime()) ? esc(v) : d.toLocaleString(); };
    const fmtBytes = v => { const n = Number(v || 0); if (n < 1024) return `${n} B`; if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`; return `${(n / 1048576).toFixed(2)} MB`; };
    async function getJson(url){ const r = await fetch(url, {cache:'no-store'}); if(!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }
    function fillBars(node, entries){
      if(!entries.length){ node.innerHTML = '<div class="empty">Пусто</div>'; return; }
      const max = Math.max(...entries.map(x => x.value), 1);
      node.innerHTML = entries.map(x => `<div class="bar"><div>${esc(x.label)}</div><div class="track"><div class="fill" style="width:${Math.max(4, Math.round(x.value / max * 100))}%"></div></div><div>${x.value}</div></div>`).join('');
    }
    function fillStack(node, items, empty='Пусто'){ node.innerHTML = items.length ? items.join('') : `<div class="empty">${empty}</div>`; }
    function setOverview(data){
      $('generatedAt').textContent = fmtTs(data.generated_at);
      $('meta').textContent = `DB ${fmtBytes(data.db.size_bytes)} | LOG ${fmtBytes(data.log.size_bytes)} | ${data.db.path} | ${data.log.path}`;
      $('status').innerHTML = `<span class="chip">DB: ${data.db.exists ? 'ok' : 'missing'}</span><span class="chip">LOG: ${data.log.exists ? 'ok' : 'missing'}</span><span class="chip">summary: ${(data.db.counts.chat_summary || 0) + (data.db.counts.user_chat_summaries || 0)}</span><span class="chip">profiles updated: ${data.log.overview.profile_updates || 0}</span><span class="chip">memory updated: ${data.log.overview.memory_updates || 0}</span>`;
      $('messagesCount').textContent = data.db.counts.messages || 0;
      $('messagesMeta').textContent = `последнее: ${fmtTs(data.db.latest_message_ts)}`;
      $('usersCount').textContent = data.db.counts.users || 0;
      $('usersMeta').textContent = `${data.db.counts.user_profiles || 0} профилей`;
      $('factMemoryCount').textContent = `${data.db.counts.user_facts || 0} / ${data.db.counts.memories || 0}`;
      $('factsMeta').textContent = `${data.db.counts.user_documents || 0} user docs`;
      $('turnCount').textContent = data.log.total_turns || 0;
      $('turnMeta').textContent = `log: ${fmtTs(data.log.updated_at)}`;
      $('latencyValue').textContent = `${data.log.overview.avg_latency_ms || 0} ms`;
      $('latencyMeta').textContent = `p95 ${data.log.overview.p95_latency_ms || 0} ms`;
      $('tokenValue').textContent = data.log.overview.avg_tokens || 0;
      $('tokenMeta').textContent = `p95 ${data.log.overview.p95_tokens || 0}`;
      $('callValue').textContent = data.log.overview.avg_llm_calls || 0;
      $('callMeta').textContent = `${data.log.overview.error_turns || 0} ошибок, ${data.log.overview.editor_attempts || 0} editor`;
      fillBars($('modes'), Object.entries(data.log.turns_by_mode || {}).map(([label, value]) => ({label, value})));
      const days = data.log.daily_turns || [];
      if(!days.length){ $('days').innerHTML = '<div class="empty">Нет активности</div>'; }
      else {
        const max = Math.max(...days.map(x => x.turns), 1);
        $('days').innerHTML = days.map(x => `<div class="day" title="${esc(x.day)}: ${x.turns}"><div>${x.turns}</div><div class="dayfill" style="height:${Math.max(8, Math.round(x.turns / max * 130))}px"></div><div>${esc(x.day.slice(5))}</div></div>`).join('');
      }
    }
    function setChats(chats){
      state.chats = chats;
      $('chatSelect').innerHTML = chats.map(chat => `<option value="${chat.chat_id}">${chat.chat_id}${chat.title ? ' | ' + esc(chat.title) : ''}</option>`).join('');
      if(!state.selectedChatId && chats.length) state.selectedChatId = String(chats[0].chat_id);
      if(state.selectedChatId) $('chatSelect').value = String(state.selectedChatId);
      fillStack($('chatList'), chats.map(chat => `<div class="item clickable" data-chat-id="${chat.chat_id}"><div class="top"><span>${esc(chat.chat_id)}</span><span>${fmtTs(chat.last_message_ts)}</span></div><div class="title">${esc(chat.title || 'Без названия')}</div><div class="body">${chat.messages_count} msg | ${chat.active_users} users | ${chat.assistant_messages} bot<div><span class="tag">${chat.summary_chars} chars summary</span></div></div></div>`), 'Чатов нет');
      document.querySelectorAll('[data-chat-id]').forEach(node => node.onclick = () => { state.selectedChatId = node.getAttribute('data-chat-id'); $('chatSelect').value = state.selectedChatId; refreshChat(); });
    }
    function setTurns(data){
      fillStack($('turns'), (data.recent_turns || []).map(turn => `<div class="item"><div class="top"><span>${fmtTs(turn.ts)}</span><span>${esc(turn.mode)} | ${turn.total_latency_ms} ms</span></div><div class="title">${esc(turn.input_preview || '(без preview)')}</div><div class="body">chat ${esc(turn.chat_id)} | user ${esc(turn.user_id)} | ${turn.total_tokens} tok | ${turn.total_llm_calls} calls ${turn.error_type ? `<span class="tag bad">${esc(turn.error_type)}</span>` : ''} ${turn.memory_updated ? '<span class="tag ok">memory</span>' : ''} ${turn.profile_updated ? '<span class="tag ok">profile</span>' : ''} ${turn.summary_updated ? '<span class="tag ok">summary</span>' : ''} ${turn.postprocess_pending ? '<span class="tag warn">postprocess</span>' : ''} ${turn.fallback_used ? '<span class="tag warn">fallback</span>' : ''}</div></div>`), 'Turn-ов нет');
      fillStack($('errors'), (data.recent_errors || []).map(err => `<div class="item"><div class="top"><span>${fmtTs(err.ts)}</span><span>${esc(err.error_type)}</span></div><div class="title">${esc(err.input_preview || 'Без preview')}</div><div class="body">${esc(err.error_message || '')}</div></div>`), 'Ошибок не видно');
      fillBars($('editor'), Object.entries(data.editor_triggers || {}).map(([label, value]) => ({label, value})));
    }
    function setChat(data){
      if(!data.exists){
        ['chatMeta','messages','users','profiles','facts'].forEach(id => $(id).innerHTML = '<div class="empty">Нет данных</div>');
        return;
      }
      const chat = data.chat;
      const summaryBlock = chat.summary ? `<div class="item"><div class="top"><span>chat summary</span><span>${fmtTs(chat.last_message_ts)}</span></div><div class="body">${esc(chat.summary)}</div></div>` : '<div class="empty">Summary пуст</div>';
      fillStack($('chatMeta'), [
        `<div class="item"><div class="top"><span>chat ${esc(chat.chat_id)}</span><span>${fmtTs(chat.last_message_ts)}</span></div><div class="title">${esc(chat.title || 'Без названия')}</div><div class="body">${chat.messages_count} сообщений | ${chat.user_messages} user | ${chat.assistant_messages} assistant | ${chat.active_users} активных</div></div>`,
        summaryBlock,
        ...(data.documents || []).slice(0,3).map(doc => `<div class="item"><div class="top"><span>${esc(doc.username || doc.user_id)}</span><span>${fmtTs(doc.updated_at)}</span></div><div class="body">${esc(doc.document_text || '')}</div></div>`)
      ], 'Нет данных по чату');
      fillStack($('messages'), (data.recent_messages || []).map(m => `<div class="item"><div class="top"><span>${esc(m.role)} | ${esc(m.username || m.user_id)}</span><span>${fmtTs(m.ts)}</span></div><div class="body">${esc(m.text)}</div></div>`), 'Сообщений нет');
      fillStack($('users'), (data.top_users || []).map(u => `<div class="item"><div class="top"><span>${esc(u.username || u.user_id)}</span><span>${u.messages_count} msg</span></div><div class="title">${esc(u.display_name || u.nickname || 'Без имени')}</div><div class="body">last: ${fmtTs(u.last_message_ts)} ${u.response_preference ? `<div><span class="tag">${esc(u.response_preference)}</span></div>` : ''}</div></div>`), 'Пользователей нет');
      const summaryByUser = new Map((data.user_summaries || []).map(x => [String(x.user_id), x]));
      fillStack($('profiles'), (data.profiles || []).map(p => { const s = summaryByUser.get(String(p.user_id)); return `<div class="item"><div class="top"><span>${esc(p.username || p.user_id)}</span><span>${fmtTs(p.updated_at)}</span></div><div class="title">${esc(p.display_name || p.nickname || 'Профиль')}</div><div class="body">${p.gender ? `<span class="tag">${esc(p.gender)}</span>` : ''} ${p.response_preference ? `<span class="tag">${esc(p.response_preference)}</span>` : ''} ${p.communication_style ? `<span class="tag">${esc(p.communication_style)}</span>` : ''} ${p.relationship_notes ? `<span class="tag">${esc(p.relationship_notes)}</span>` : ''} ${s ? `<div style="margin-top:8px">${esc(s.summary)}</div>` : ''}</div></div>`; }), 'Профилей нет');
      const memMap = new Map();
      (state.memories?.recent || []).forEach(m => { if(String(m.chat_id) !== String(data.chat_id)) return; const key = String(m.user_id); const arr = memMap.get(key) || []; if(arr.length < 3) arr.push(m); memMap.set(key, arr); });
      fillStack($('facts'), (data.facts || []).map(f => { const ms = (memMap.get(String(f.user_id)) || []).map(m => `<span class="tag">${esc(m.key)}=${esc(m.value)}</span>`).join(' '); return `<div class="item"><div class="top"><span>${esc(f.username || f.user_id)} | ${esc(f.fact_type)}</span><span>${fmtTs(f.updated_at)}</span></div><div class="title">${esc(f.value)}</div><div class="body"><span class="tag ok">importance ${Number(f.importance || 0).toFixed(2)}</span> <span class="tag ok">confidence ${Number(f.confidence || 0).toFixed(2)}</span>${f.evidence ? `<div style="margin-top:8px">${esc(f.evidence)}</div>` : ''}${ms ? `<div style="margin-top:8px">${ms}</div>` : ''}</div></div>`; }), 'Фактов нет');
    }
    async function refreshOverview(){
      const [overview, chats, turns, memories] = await Promise.all([getJson('/api/overview'), getJson('/api/chats'), getJson('/api/turns'), getJson('/api/memories')]);
      state.memories = memories;
      setOverview(overview);
      setChats(chats);
      setTurns(turns);
    }
    async function refreshChat(){ if(!state.selectedChatId) return; setChat(await getJson(`/api/chat?chat_id=${encodeURIComponent(state.selectedChatId)}`)); }
    async function refreshAll(){ try { await refreshOverview(); await refreshChat(); } catch (e) { $('meta').textContent = `Ошибка обновления: ${e.message}`; } }
    function syncTimer(){ if(state.timer){ clearInterval(state.timer); state.timer = null; } const delay = Number($('refreshSelect').value || 0); if(delay > 0) state.timer = setInterval(refreshAll, delay); }
    $('reload').onclick = refreshAll;
    $('chatSelect').onchange = () => { state.selectedChatId = $('chatSelect').value; refreshChat(); };
    $('refreshSelect').onchange = syncTimer;
    syncTimer();
    refreshAll();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    store: DashboardStore

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/overview":
            self._send_json(self.store.get_overview())
            return
        if parsed.path == "/api/chats":
            self._send_json(self.store.list_chats())
            return
        if parsed.path == "/api/chat":
            params = parse_qs(parsed.query)
            chat_id = to_int((params.get("chat_id") or [0])[0], 0)
            self._send_json(self.store.get_chat_details(chat_id))
            return
        if parsed.path == "/api/memories":
            self._send_json(self.store.get_memories())
            return
        if parsed.path == "/api/turns":
            self._send_json(self.store.get_recent_turns())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args):
        return

    def _send_json(self, payload: Any):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str):
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def build_handler(store: DashboardStore):
    class Handler(DashboardHandler):
        pass

    Handler.store = store
    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local dashboard for Alta bot state")
    parser.add_argument("--db", default="alta_tg_test.db", help="Path to SQLite database")
    parser.add_argument("--log", default="orchestration.log", help="Path to JSONL orchestration log")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = DashboardStore(db_path=args.db, log_path=args.log)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(store))
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print(f"DB: {Path(args.db).resolve()}")
    print(f"LOG: {Path(args.log).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
