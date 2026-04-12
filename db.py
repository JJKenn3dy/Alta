import logging
from contextlib import asynccontextmanager

import aiosqlite
from typing import Optional

log = logging.getLogger("alta.db")

INVALID_IDENTITY_VALUES = (
    "ты",
    "тебя",
    "тебе",
    "твой",
    "твоя",
    "он",
    "она",
    "они",
    "неизвестно",
    "неизвестен",
    "неизвестна",
    "неопределено",
    "не определено",
    "unknown",
    "none",
    "null",
    "никак",
)
INVALID_GENDER_VALUES = INVALID_IDENTITY_VALUES + (
    "не знаю",
    "не указан",
    "не указано",
)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS allowed_chats (
  chat_id INTEGER PRIMARY KEY,
  is_enabled INTEGER NOT NULL DEFAULT 1,
  title TEXT
);

CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY,
  username TEXT,
  first_seen TEXT NOT NULL DEFAULT (datetime('now')),
  last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  target_user_id INTEGER,
  role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
  text TEXT NOT NULL,
  ts TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_target ON messages(chat_id, target_user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_user ON messages(chat_id, user_id, id DESC);

CREATE TABLE IF NOT EXISTS chat_summary (
  chat_id INTEGER PRIMARY KEY,
  summary TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_chat_summaries (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS user_profiles (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  display_name TEXT,
  nickname TEXT,
  approved_display_name TEXT,
  gender TEXT,
  interests TEXT NOT NULL DEFAULT '',
  communication_style TEXT NOT NULL DEFAULT '',
  response_preference TEXT NOT NULL DEFAULT '',
  relationship_notes TEXT NOT NULL DEFAULT '',
  profile_confidence REAL NOT NULL DEFAULT 0.0,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS pending_name_approvals (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  requested_name TEXT NOT NULL,
  request_message_id INTEGER,
  approval_message_id INTEGER UNIQUE,
  requested_by INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (chat_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_pending_name_approvals_chat_status
ON pending_name_approvals(chat_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS user_documents (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  document_json TEXT NOT NULL DEFAULT '{}',
  document_text TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS user_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  fact_type TEXT NOT NULL,
  value TEXT NOT NULL,
  normalized_value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.6,
  importance REAL NOT NULL DEFAULT 0.5,
  evidence TEXT NOT NULL DEFAULT '',
  source TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  last_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.6,
  importance REAL NOT NULL DEFAULT 0.5,
  source TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_chat_user ON memories(chat_id, user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_chat_user_key ON memories(chat_id, user_id, key);
CREATE INDEX IF NOT EXISTS idx_user_facts_chat_user ON user_facts(chat_id, user_id);
CREATE INDEX IF NOT EXISTS idx_user_facts_lookup ON user_facts(chat_id, user_id, fact_type, normalized_value);
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_facts_unique
ON user_facts(chat_id, user_id, fact_type, normalized_value);

CREATE TABLE IF NOT EXISTS debug_traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  input_text TEXT NOT NULL,
  context_json TEXT NOT NULL,
  output_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cooldown_state (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  cooldown_key TEXT NOT NULL,
  last_ts REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (chat_id, user_id, cooldown_key)
);

CREATE TABLE IF NOT EXISTS context_files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL UNIQUE,
  mtime REAL NOT NULL,
  size INTEGER NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS context_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_id INTEGER NOT NULL REFERENCES context_files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  normalized_text TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_context_chunks_file_id ON context_chunks(file_id, chunk_index);
"""

class DB:
    def __init__(self, path: str):
        self.path = path
        self.conn: Optional[aiosqlite.Connection] = None

    @asynccontextmanager
    async def _transaction(self):
        """Explicit transaction: BEGIN … COMMIT, with ROLLBACK on error."""
        assert self.conn
        await self.conn.execute("BEGIN")
        try:
            yield
            await self.conn.commit()
        except BaseException:
            await self.conn.rollback()
            raise

    async def connect(self):
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA foreign_keys = ON")
        await self.conn.executescript(SCHEMA)
        await self._migrate_schema()
        await self._cleanup_identity_noise()
        await self.conn.commit()
        await self.prune_debug_traces(keep=30)

    async def _migrate_schema(self):
        assert self.conn
        cur = await self.conn.execute("PRAGMA table_info(memories)")
        memory_columns = {row["name"] for row in await cur.fetchall()}
        if "importance" not in memory_columns:
            await self.conn.execute(
                "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5"
            )

        cur = await self.conn.execute("PRAGMA table_info(messages)")
        message_columns = {row["name"] for row in await cur.fetchall()}
        if "target_user_id" not in message_columns:
            await self.conn.execute("ALTER TABLE messages ADD COLUMN target_user_id INTEGER")

        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_chat_target "
            "ON messages(chat_id, target_user_id, id DESC)"
        )

        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS user_chat_summaries ("
            "chat_id INTEGER NOT NULL,"
            "user_id INTEGER NOT NULL,"
            "summary TEXT NOT NULL DEFAULT '',"
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')),"
            "PRIMARY KEY (chat_id, user_id)"
            ")"
        )
        cur = await self.conn.execute("PRAGMA table_info(user_facts)")
        user_fact_columns = {row["name"] for row in await cur.fetchall()}
        if "evidence" not in user_fact_columns:
            await self.conn.execute(
                "ALTER TABLE user_facts ADD COLUMN evidence TEXT NOT NULL DEFAULT ''"
            )
        if "source" not in user_fact_columns:
            await self.conn.execute(
                "ALTER TABLE user_facts ADD COLUMN source TEXT NOT NULL DEFAULT ''"
            )
        if "created_at" not in user_fact_columns:
            await self.conn.execute(
                "ALTER TABLE user_facts ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"
            )
            await self.conn.execute(
                "UPDATE user_facts SET created_at=datetime('now') WHERE created_at=''"
            )
        if "updated_at" not in user_fact_columns:
            await self.conn.execute(
                "ALTER TABLE user_facts ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
            )
            await self.conn.execute(
                "UPDATE user_facts SET updated_at=datetime('now') WHERE updated_at=''"
            )
        if "last_seen_at" not in user_fact_columns:
            await self.conn.execute(
                "ALTER TABLE user_facts ADD COLUMN last_seen_at TEXT NOT NULL DEFAULT ''"
            )
            await self.conn.execute(
                "UPDATE user_facts SET last_seen_at=datetime('now') WHERE last_seen_at=''"
            )

        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_facts_chat_user "
            "ON user_facts(chat_id, user_id)"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_facts_lookup "
            "ON user_facts(chat_id, user_id, fact_type, normalized_value)"
        )
        await self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_facts_unique "
            "ON user_facts(chat_id, user_id, fact_type, normalized_value)"
        )
        cur = await self.conn.execute("PRAGMA table_info(user_profiles)")
        user_profile_columns = {row["name"] for row in await cur.fetchall()}
        if "approved_display_name" not in user_profile_columns:
            await self.conn.execute(
                "ALTER TABLE user_profiles ADD COLUMN approved_display_name TEXT"
            )
        if "extra_fields" not in user_profile_columns:
            await self.conn.execute(
                "ALTER TABLE user_profiles ADD COLUMN extra_fields TEXT NOT NULL DEFAULT '{}'"
            )

        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_name_approvals ("
            "chat_id INTEGER NOT NULL,"
            "user_id INTEGER NOT NULL,"
            "requested_name TEXT NOT NULL,"
            "request_message_id INTEGER,"
            "approval_message_id INTEGER UNIQUE,"
            "requested_by INTEGER NOT NULL,"
            "status TEXT NOT NULL DEFAULT 'pending',"
            "created_at TEXT NOT NULL DEFAULT (datetime('now')),"
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')),"
            "PRIMARY KEY (chat_id, user_id)"
            ")"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_name_approvals_chat_status "
            "ON pending_name_approvals(chat_id, status, updated_at DESC)"
        )

        # Alta internal state tables
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS alta_chat_state ("
            "chat_id INTEGER PRIMARY KEY,"
            "energy REAL NOT NULL DEFAULT 0.7,"
            "irritation REAL NOT NULL DEFAULT 0.0,"
            "playfulness REAL NOT NULL DEFAULT 0.5,"
            "social_battery REAL NOT NULL DEFAULT 0.8,"
            "updated_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS alta_user_relation ("
            "chat_id INTEGER NOT NULL,"
            "user_id INTEGER NOT NULL,"
            "trust REAL NOT NULL DEFAULT 0.5,"
            "warmth REAL NOT NULL DEFAULT 0.5,"
            "caution REAL NOT NULL DEFAULT 0.0,"
            "annoyance REAL NOT NULL DEFAULT 0.0,"
            "interaction_count INTEGER NOT NULL DEFAULT 0,"
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')),"
            "PRIMARY KEY (chat_id, user_id)"
            ")"
        )
        await self.conn.execute(
            "CREATE TABLE IF NOT EXISTS alta_reactions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "chat_id INTEGER NOT NULL,"
            "reaction_kind TEXT NOT NULL,"
            "reaction_strength REAL NOT NULL DEFAULT 0.5,"
            "ttl_messages INTEGER NOT NULL DEFAULT 2,"
            "target_user_id INTEGER,"
            "created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )
        await self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alta_reactions_chat "
            "ON alta_reactions(chat_id, ttl_messages DESC)"
        )

        await self._migrate_legacy_memories_to_user_facts()

    async def _migrate_legacy_memories_to_user_facts(self):
        assert self.conn
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO user_facts(
              chat_id,user_id,fact_type,value,normalized_value,confidence,importance,evidence,source,created_at,updated_at,last_seen_at
            )
            SELECT
              chat_id,
              user_id,
              CASE
                WHEN key IN ('interests', 'hobby') THEN 'interest'
                WHEN key IN ('job', 'work') THEN 'work'
                WHEN key = 'study' THEN 'study'
                WHEN key = 'location' THEN 'location'
                WHEN key = 'goal' THEN 'goal'
                WHEN key = 'relationship_notes' THEN 'relationship'
                WHEN key = 'preference' THEN 'preference'
                ELSE 'fact'
              END,
              value,
              trim(lower(value)),
              confidence,
              importance,
              key,
              CASE
                WHEN coalesce(source, '') = '' THEN 'legacy_memory'
                ELSE 'legacy_memory:' || source
              END,
              updated_at,
              updated_at,
              updated_at
            FROM memories
            WHERE trim(value) <> ''
              AND key NOT LIKE '\\_%' ESCAPE '\\'
              AND key NOT IN (
                'name',
                'preferred_name',
                'gender',
                'pronouns',
                'nickname',
                'communication_style',
                'response_preference'
              )
            """
        )

    async def _cleanup_identity_noise(self):
        assert self.conn
        identity_placeholders = ",".join("?" for _ in INVALID_IDENTITY_VALUES)
        gender_placeholders = ",".join("?" for _ in INVALID_GENDER_VALUES)

        await self.conn.execute(
            f"DELETE FROM memories "
            f"WHERE key IN ('name','preferred_name','nickname') "
            f"AND lower(trim(value)) IN ({identity_placeholders})",
            INVALID_IDENTITY_VALUES,
        )
        await self.conn.execute(
            f"DELETE FROM memories "
            f"WHERE key='gender' "
            f"AND lower(trim(value)) IN ({gender_placeholders})",
            INVALID_GENDER_VALUES,
        )
        await self.conn.execute(
            "DELETE FROM memories "
            "WHERE key IN ('name','preferred_name','nickname','gender') "
            "AND source='auto_extract' "
            "AND confidence < 0.8"
        )
        await self.conn.execute(
            f"UPDATE user_profiles SET display_name=NULL "
            f"WHERE lower(trim(display_name)) IN ({identity_placeholders})",
            INVALID_IDENTITY_VALUES,
        )
        await self.conn.execute(
            f"UPDATE user_profiles SET nickname=NULL "
            f"WHERE lower(trim(nickname)) IN ({identity_placeholders})",
            INVALID_IDENTITY_VALUES,
        )
        await self.conn.execute(
            f"UPDATE user_profiles SET approved_display_name=NULL "
            f"WHERE lower(trim(approved_display_name)) IN ({identity_placeholders})",
            INVALID_IDENTITY_VALUES,
        )
        await self.conn.execute(
            f"UPDATE user_profiles SET gender=NULL "
            f"WHERE lower(trim(gender)) IN ({gender_placeholders})",
            INVALID_GENDER_VALUES,
        )
    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def ensure_allowed_chat(self, chat_id: int, title: str = ""):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO allowed_chats(chat_id,is_enabled,title) VALUES(?,1,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET is_enabled=1,title=excluded.title",
            (chat_id, title),
        )
        await self.conn.commit()

    async def is_allowed(self, chat_id: int) -> bool:
        assert self.conn
        cur = await self.conn.execute("SELECT is_enabled FROM allowed_chats WHERE chat_id=?", (chat_id,))
        row = await cur.fetchone()
        return bool(row and row["is_enabled"] == 1)

    async def upsert_user(self, user_id: int, username: str | None):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO users(user_id,username) VALUES(?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET username=excluded.username,last_seen=datetime('now')",
            (user_id, username),
        )
        await self.conn.commit()

    async def get_user(self, user_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT user_id,username,first_seen,last_seen FROM users WHERE user_id=?",
            (user_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

    async def get_user_by_username(self, username: str) -> dict:
        assert self.conn
        cleaned = str(username or "").strip().lstrip("@")
        if not cleaned:
            return {}
        cur = await self.conn.execute(
            "SELECT user_id,username,first_seen,last_seen "
            "FROM users WHERE lower(username)=lower(?) "
            "ORDER BY last_seen DESC LIMIT 1",
            (cleaned,),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

    async def add_message(
        self,
        chat_id: int,
        user_id: int,
        role: str,
        text: str,
        target_user_id: int | None = None,
    ):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO messages(chat_id,user_id,target_user_id,role,text) VALUES(?,?,?,?,?)",
            (chat_id, user_id, target_user_id, role, text),
        )
        await self.conn.commit()

    async def get_last_messages(
        self,
        chat_id: int,
        limit: int,
        focus_user_id: int | None = None,
    ) -> list[dict]:
        assert self.conn
        if focus_user_id is None:
            cur = await self.conn.execute(
                "SELECT m.id,m.role,m.text,m.user_id,m.target_user_id,m.ts,u.username "
                "FROM messages m "
                "LEFT JOIN users u ON u.user_id = m.user_id "
                "WHERE m.chat_id=? "
                "ORDER BY m.id DESC LIMIT ?",
                (chat_id, limit),
            )
        else:
            cur = await self.conn.execute(
                "SELECT m.id,m.role,m.text,m.user_id,m.target_user_id,m.ts,u.username "
                "FROM messages m "
                "LEFT JOIN users u ON u.user_id = m.user_id "
                "WHERE m.chat_id=? "
                "AND ("
                "  (m.role='user' AND m.user_id=?) "
                "  OR (m.role='assistant' AND m.target_user_id=?)"
                ") "
                "ORDER BY m.id DESC LIMIT ?",
                (chat_id, focus_user_id, focus_user_id, limit),
            )
        rows = list(reversed(await cur.fetchall()))
        out: list[dict] = []
        for r in rows:
            ts_raw = r["ts"] or ""
            # Format: "2026-03-23 14:31:00" -> "14:31"
            ts_short = ""
            if ts_raw and len(ts_raw) >= 16:
                ts_short = ts_raw[11:16]  # HH:MM
            if r["role"] == "user":
                parts = []
                if ts_short:
                    parts.append(ts_short)
                parts.append(f"id={r['user_id']}")
                if r["username"]:
                    parts.append(f"@{r['username']}")
                content = f"[{' '.join(parts)}] {r['text']}"
            else:
                # Don't add timestamps to assistant messages - the model
                # copies the format and puts [HH:MM] in its own replies
                content = r["text"]
            out.append({"id": int(r["id"]), "role": r["role"], "content": content, "ts": ts_raw})
        return out

    async def get_summary(self, chat_id: int, user_id: int | None = None) -> str:
        assert self.conn
        if user_id is not None:
            cur = await self.conn.execute(
                "SELECT summary FROM user_chat_summaries WHERE chat_id=? AND user_id=?",
                (chat_id, user_id),
            )
            row = await cur.fetchone()
            return row["summary"] if row else ""

        cur = await self.conn.execute("SELECT summary FROM chat_summary WHERE chat_id=?", (chat_id,))
        row = await cur.fetchone()
        return row["summary"] if row else ""

    async def set_summary(self, chat_id: int, summary: str, user_id: int | None = None):
        assert self.conn
        if user_id is not None:
            await self.conn.execute(
                "INSERT INTO user_chat_summaries(chat_id,user_id,summary) VALUES(?,?,?) "
                "ON CONFLICT(chat_id,user_id) DO UPDATE SET "
                "summary=excluded.summary,updated_at=datetime('now')",
                (chat_id, user_id, summary),
            )
            await self.conn.commit()
            return

        await self.conn.execute(
            "INSERT INTO chat_summary(chat_id,summary) VALUES(?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET summary=excluded.summary,updated_at=datetime('now')",
            (chat_id, summary),
        )
        await self.conn.commit()

    async def get_chat_participants(
        self,
        chat_id: int,
        exclude_user_id: int | None = None,
        limit: int = 10,
        recent_message_window: int = 200,
    ) -> list[dict]:
        """Return other recently active participants with profile info (no private summaries).

        Uses a bounded scan of recent messages to find active users efficiently
        even in chats with 50+ participants.
        Each dict contains: user_id, username, display_name, gender, interests.
        """
        assert self.conn
        cur = await self.conn.execute(
            "SELECT u.user_id, u.username, "
            "  p.display_name, p.approved_display_name, p.gender, p.interests "
            "FROM users u "
            "LEFT JOIN user_profiles p ON p.user_id = u.user_id AND p.chat_id = ? "
            "WHERE u.user_id IN ("
            "  SELECT DISTINCT sub.user_id FROM ("
            "    SELECT m.user_id FROM messages m "
            "    WHERE m.chat_id = ? AND m.role = 'user' AND m.user_id != 0 "
            "    ORDER BY m.id DESC LIMIT ?"
            "  ) sub"
            ") "
            "AND (? IS NULL OR u.user_id != ?) "
            "ORDER BY u.last_seen DESC "
            "LIMIT ?",
            (chat_id, chat_id, recent_message_window, exclude_user_id, exclude_user_id, limit),
        )
        rows = await cur.fetchall()
        result = []
        for r in rows:
            name = r["approved_display_name"] or r["display_name"] or ""
            result.append({
                "user_id": r["user_id"],
                "username": r["username"] or "",
                "display_name": name,
                "gender": r["gender"] or "",
                "interests": r["interests"] or "",
            })
        return result

    async def get_user_profile(self, chat_id: int, user_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT display_name,nickname,approved_display_name,gender,interests,communication_style,"
            "response_preference,relationship_notes,profile_confidence,extra_fields,updated_at "
            "FROM user_profiles WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

    async def get_user_profile_merged(self, chat_id: int, user_id: int) -> dict:
        """Return user profile merging group-specific data with private-chat profile.

        Private chat profile is stored under (user_id, user_id) and represents
        explicitly user-set values — they ALWAYS take priority over auto-extracted
        group data.  Group profile fills in what the user hasn't set manually.
        """
        import json

        group_profile = await self.get_user_profile(chat_id, user_id)

        # If already querying private chat profile, no merge needed
        if chat_id == user_id:
            return group_profile

        private_profile = await self.get_user_profile(user_id, user_id)
        if not private_profile:
            return group_profile
        if not group_profile:
            return private_profile

        # Base = group profile (auto-extracted); private profile OVERRIDES it field-by-field.
        # This ensures manually set values always win over LLM-extracted ones.
        merged = dict(group_profile)
        for key, value in private_profile.items():
            if value not in (None, "", "{}"):
                merged[key] = value

        # Merge extra_fields JSON: private overrides group per-key
        try:
            private_extra = json.loads(private_profile.get("extra_fields") or "{}") if isinstance(private_profile.get("extra_fields"), str) else (private_profile.get("extra_fields") or {})
            group_extra = json.loads(group_profile.get("extra_fields") or "{}") if isinstance(group_profile.get("extra_fields"), str) else (group_profile.get("extra_fields") or {})
            merged_extra = {**group_extra, **private_extra}  # private overrides group
            merged["extra_fields"] = json.dumps(merged_extra, ensure_ascii=False) if merged_extra else "{}"
        except Exception:
            pass

        return merged

    async def get_all_group_profiles_merged(self, user_id: int) -> dict:
        """Return a merged profile built from all GROUP chats for this user.

        Group chats are any chat where chat_id != user_id.
        Higher profile_confidence rows take priority field-by-field.
        """
        import json

        assert self.conn
        cur = await self.conn.execute(
            "SELECT display_name,nickname,approved_display_name,gender,interests,communication_style,"
            "response_preference,relationship_notes,profile_confidence,extra_fields "
            "FROM user_profiles WHERE user_id=? AND chat_id!=? "
            "ORDER BY profile_confidence DESC, updated_at DESC",
            (user_id, user_id),
        )
        rows = await cur.fetchall()
        if not rows:
            return {}

        # Merge all rows: first non-empty value per field wins (rows sorted by confidence desc)
        text_fields = (
            "display_name", "nickname", "approved_display_name", "gender",
            "interests", "communication_style", "response_preference", "relationship_notes",
        )
        merged: dict = {}
        merged_extra: dict = {}
        best_confidence = 0.0

        for row in rows:
            r = dict(row)
            best_confidence = max(best_confidence, r.get("profile_confidence") or 0.0)
            for f in text_fields:
                if not merged.get(f) and r.get(f):
                    merged[f] = r[f]
            # Merge extra_fields JSON
            try:
                extra = json.loads(r.get("extra_fields") or "{}") if isinstance(r.get("extra_fields"), str) else {}
                for k, v in extra.items():
                    if k not in merged_extra and v:
                        merged_extra[k] = v
            except Exception:
                pass

        merged["profile_confidence"] = best_confidence
        merged["extra_fields"] = json.dumps(merged_extra, ensure_ascii=False) if merged_extra else "{}"
        return merged

    async def count_user_facts_global(self, user_id: int) -> int:
        """Count all user_facts for this user across all group chats."""
        assert self.conn
        cur = await self.conn.execute(
            "SELECT COUNT(*) FROM user_facts WHERE user_id=? AND chat_id!=?",
            (user_id, user_id),
        )
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def get_user_document(self, chat_id: int, user_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT document_json,document_text,updated_at "
            "FROM user_documents WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

    async def upsert_user_document(
        self,
        chat_id: int,
        user_id: int,
        *,
        document_json: str,
        document_text: str,
    ):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO user_documents(chat_id,user_id,document_json,document_text) VALUES(?,?,?,?) "
            "ON CONFLICT(chat_id,user_id) DO UPDATE SET "
            "document_json=excluded.document_json,"
            "document_text=excluded.document_text,"
            "updated_at=datetime('now')",
            (chat_id, user_id, document_json, document_text),
        )
        await self.conn.commit()

    async def upsert_user_profile(
        self,
        chat_id: int,
        user_id: int,
        *,
        display_name: str | None = None,
        nickname: str | None = None,
        approved_display_name: str | None = None,
        gender: str | None = None,
        interests: str | None = None,
        communication_style: str | None = None,
        response_preference: str | None = None,
        relationship_notes: str | None = None,
        profile_confidence: float | None = None,
        extra_fields: str | None = None,
    ):
        assert self.conn
        current = await self.get_user_profile(chat_id, user_id)
        await self.conn.execute(
            "INSERT INTO user_profiles("
            "chat_id,user_id,display_name,nickname,approved_display_name,gender,interests,communication_style,"
            "response_preference,relationship_notes,profile_confidence,extra_fields"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,user_id) DO UPDATE SET "
            "display_name=excluded.display_name,"
            "nickname=excluded.nickname,"
            "approved_display_name=excluded.approved_display_name,"
            "gender=excluded.gender,"
            "interests=excluded.interests,"
            "communication_style=excluded.communication_style,"
            "response_preference=excluded.response_preference,"
            "relationship_notes=excluded.relationship_notes,"
            "profile_confidence=excluded.profile_confidence,"
            "extra_fields=excluded.extra_fields,"
            "updated_at=datetime('now')",
            (
                chat_id,
                user_id,
                display_name if display_name is not None else current.get("display_name"),
                nickname if nickname is not None else current.get("nickname"),
                approved_display_name
                if approved_display_name is not None
                else current.get("approved_display_name"),
                gender if gender is not None else current.get("gender"),
                interests if interests is not None else current.get("interests", ""),
                communication_style
                if communication_style is not None
                else current.get("communication_style", ""),
                response_preference
                if response_preference is not None
                else current.get("response_preference", ""),
                relationship_notes
                if relationship_notes is not None
                else current.get("relationship_notes", ""),
                profile_confidence
                if profile_confidence is not None
                else current.get("profile_confidence", 0.0),
                extra_fields if extra_fields is not None else current.get("extra_fields", "{}"),
            ),
        )
        await self.conn.commit()

    async def create_pending_name_approval(
        self,
        chat_id: int,
        user_id: int,
        *,
        requested_name: str,
        requested_by: int,
        request_message_id: int | None = None,
    ):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO pending_name_approvals("
            "chat_id,user_id,requested_name,request_message_id,requested_by,status,updated_at"
            ") VALUES(?,?,?,?,?,'pending',datetime('now')) "
            "ON CONFLICT(chat_id,user_id) DO UPDATE SET "
            "requested_name=excluded.requested_name,"
            "request_message_id=excluded.request_message_id,"
            "approval_message_id=NULL,"
            "requested_by=excluded.requested_by,"
            "status='pending',"
            "updated_at=datetime('now')",
            (chat_id, user_id, requested_name, request_message_id, requested_by),
        )
        await self.conn.commit()

    async def set_pending_name_approval_message(
        self,
        chat_id: int,
        user_id: int,
        approval_message_id: int,
    ):
        assert self.conn
        await self.conn.execute(
            "UPDATE pending_name_approvals "
            "SET approval_message_id=?, updated_at=datetime('now') "
            "WHERE chat_id=? AND user_id=? AND status='pending'",
            (approval_message_id, chat_id, user_id),
        )
        await self.conn.commit()

    async def get_pending_name_approval(self, chat_id: int, user_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT chat_id,user_id,requested_name,request_message_id,approval_message_id,"
            "requested_by,status,created_at,updated_at "
            "FROM pending_name_approvals WHERE chat_id=? AND user_id=? AND status='pending'",
            (chat_id, user_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

    async def get_pending_name_approval_by_message(self, chat_id: int, approval_message_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT chat_id,user_id,requested_name,request_message_id,approval_message_id,"
            "requested_by,status,created_at,updated_at "
            "FROM pending_name_approvals "
            "WHERE chat_id=? AND approval_message_id=? AND status='pending'",
            (chat_id, approval_message_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

    async def resolve_pending_name_approval(
        self,
        chat_id: int,
        user_id: int,
        *,
        status: str,
        approved_display_name: str | None = None,
    ):
        assert self.conn
        await self.conn.execute(
            "UPDATE pending_name_approvals "
            "SET status=?, updated_at=datetime('now') "
            "WHERE chat_id=? AND user_id=? AND status='pending'",
            (status, chat_id, user_id),
        )
        if status == "approved":
            await self.upsert_user_profile(
                chat_id,
                user_id,
                approved_display_name=approved_display_name,
            )
            return
        await self.conn.commit()

    async def get_memories(self, chat_id: int, user_id: int, limit: int = 30) -> list[dict]:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT key,value,confidence,importance,source FROM memories WHERE chat_id=? AND user_id=? "
            "ORDER BY importance DESC, confidence DESC, updated_at DESC LIMIT ?",
            (chat_id, user_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_user_facts(
        self,
        chat_id: int,
        user_id: int,
        limit: int = 40,
        min_importance: float = 0.0,
    ) -> list[dict]:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT id,fact_type,value,normalized_value,confidence,importance,evidence,source,updated_at,last_seen_at "
            "FROM user_facts WHERE chat_id=? AND user_id=? AND importance>=? "
            "ORDER BY importance DESC, confidence DESC, last_seen_at DESC, updated_at DESC LIMIT ?",
            (chat_id, user_id, float(min_importance), limit),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def upsert_user_fact(
        self,
        chat_id: int,
        user_id: int,
        fact_type: str,
        value: str,
        normalized_value: str,
        confidence: float = 0.6,
        importance: float = 0.5,
        evidence: str = "",
        source: str = "",
    ):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO user_facts("
            "chat_id,user_id,fact_type,value,normalized_value,confidence,importance,evidence,source"
            ") VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,user_id,fact_type,normalized_value) DO UPDATE SET "
            "value=excluded.value,"
            "confidence=excluded.confidence,"
            "importance=excluded.importance,"
            "evidence=CASE WHEN excluded.evidence<>'' THEN excluded.evidence ELSE user_facts.evidence END,"
            "source=CASE WHEN excluded.source<>'' THEN excluded.source ELSE user_facts.source END,"
            "updated_at=datetime('now'),"
            "last_seen_at=datetime('now')",
            (
                chat_id,
                user_id,
                fact_type,
                value,
                normalized_value,
                confidence,
                importance,
                evidence,
                source,
            ),
        )
        await self.conn.commit()

    async def prune_user_facts(self, chat_id: int, user_id: int, keep: int = 80):
        assert self.conn
        await self.conn.execute(
            """
            DELETE FROM user_facts
            WHERE id IN (
              SELECT id FROM user_facts
              WHERE chat_id=? AND user_id=?
              ORDER BY importance DESC, confidence DESC, last_seen_at DESC, updated_at DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (chat_id, user_id, keep),
        )
        await self.conn.commit()

    async def upsert_memory(
        self,
        chat_id: int,
        user_id: int,
        key: str,
        value: str,
        confidence: float = 0.6,
        importance: float = 0.5,
        source: str = "",
    ):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO memories(chat_id,user_id,key,value,confidence,importance,source) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,user_id,key) DO UPDATE SET "
            "value=excluded.value,confidence=excluded.confidence,importance=excluded.importance,"
            "source=excluded.source,updated_at=datetime('now')",
            (chat_id, user_id, key, value, confidence, importance, source),
        )
        await self.conn.commit()

    async def prune_memories(self, chat_id: int, user_id: int, keep: int = 40):
        assert self.conn
        await self.conn.execute(
            """
            DELETE FROM memories
            WHERE id IN (
              SELECT id FROM memories
              WHERE chat_id=? AND user_id=?
              ORDER BY importance DESC, confidence DESC, updated_at DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (chat_id, user_id, keep),
        )
        await self.conn.commit()

    async def count_messages(self, chat_id: int) -> int:
        assert self.conn
        cur = await self.conn.execute("SELECT COUNT(1) AS cnt FROM messages WHERE chat_id=?", (chat_id,))
        row = await cur.fetchone()
        return int(row["cnt"] if row else 0)

    async def count_user_messages(self, chat_id: int, user_id: int | None = None) -> int:
        assert self.conn
        if user_id is None:
            cur = await self.conn.execute(
                "SELECT COUNT(1) AS cnt FROM messages WHERE chat_id=? AND role='user'",
                (chat_id,),
            )
        else:
            cur = await self.conn.execute(
                "SELECT COUNT(1) AS cnt FROM messages "
                "WHERE chat_id=? AND role='user' AND user_id=?",
                (chat_id, user_id),
            )
        row = await cur.fetchone()
        return int(row["cnt"] if row else 0)

    async def has_memory_key(self, chat_id: int, user_id: int, key: str) -> bool:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT 1 AS ok FROM memories WHERE chat_id=? AND user_id=? AND key=? LIMIT 1",
            (chat_id, user_id, key),
        )
        row = await cur.fetchone()
        return bool(row)

    async def get_cooldown_last_ts(self, chat_id: int, user_id: int, cooldown_key: str) -> float:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT last_ts FROM cooldown_state WHERE chat_id=? AND user_id=? AND cooldown_key=?",
            (chat_id, user_id, cooldown_key),
        )
        row = await cur.fetchone()
        return float(row["last_ts"]) if row else 0.0

    async def touch_cooldown(
        self,
        chat_id: int,
        user_id: int,
        cooldown_key: str,
        ts: float,
    ):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO cooldown_state(chat_id,user_id,cooldown_key,last_ts) VALUES(?,?,?,?) "
            "ON CONFLICT(chat_id,user_id,cooldown_key) DO UPDATE SET last_ts=excluded.last_ts",
            (chat_id, user_id, cooldown_key, ts),
        )
        await self.conn.commit()

    async def list_context_files(self) -> dict[str, dict]:
        assert self.conn
        cur = await self.conn.execute("SELECT id,path,mtime,size FROM context_files")
        rows = await cur.fetchall()
        return {
            str(row["path"]): {
                "id": int(row["id"]),
                "mtime": float(row["mtime"]),
                "size": int(row["size"]),
            }
            for row in rows
        }

    async def replace_context_file(
        self,
        path: str,
        mtime: float,
        size: int,
        chunks: list[dict[str, str | int]],
    ):
        assert self.conn
        async with self._transaction():
            await self.conn.execute(
                "INSERT INTO context_files(path,mtime,size) VALUES(?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "mtime=excluded.mtime,size=excluded.size,updated_at=datetime('now')",
                (path, mtime, size),
            )
            cur = await self.conn.execute("SELECT id FROM context_files WHERE path=?", (path,))
            row = await cur.fetchone()
            if not row:
                raise RuntimeError(f"context file row missing after upsert: {path}")

            file_id = int(row["id"])
            await self.conn.execute("DELETE FROM context_chunks WHERE file_id=?", (file_id,))
            if chunks:
                await self.conn.executemany(
                    "INSERT INTO context_chunks(file_id,chunk_index,content,normalized_text) "
                    "VALUES(?,?,?,?)",
                    [
                        (
                            file_id,
                            int(chunk["chunk_index"]),
                            str(chunk["content"]),
                            str(chunk["normalized_text"]),
                        )
                        for chunk in chunks
                    ],
                )

    async def delete_context_file(self, path: str):
        assert self.conn
        async with self._transaction():
            cur = await self.conn.execute("SELECT id FROM context_files WHERE path=?", (path,))
            row = await cur.fetchone()
            if not row:
                return

            file_id = int(row["id"])
            await self.conn.execute("DELETE FROM context_chunks WHERE file_id=?", (file_id,))
            await self.conn.execute("DELETE FROM context_files WHERE id=?", (file_id,))

    async def get_context_chunks(self, limit: int | None = None) -> list[dict]:
        assert self.conn
        sql = (
            "SELECT c.id,c.file_id,c.chunk_index,c.content,c.normalized_text,f.path "
            "FROM context_chunks c "
            "JOIN context_files f ON f.id = c.file_id "
            "ORDER BY f.path ASC, c.chunk_index ASC"
        )
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        cur = await self.conn.execute(sql, params)
        return [dict(row) for row in await cur.fetchall()]

    async def rotate_messages(self, chat_id: int, keep: int = 2000) -> int:
        """Delete oldest messages in a chat, keeping the most recent *keep*."""
        assert self.conn
        cur = await self.conn.execute(
            "SELECT COUNT(1) AS cnt FROM messages WHERE chat_id=?",
            (chat_id,),
        )
        row = await cur.fetchone()
        total = int(row["cnt"] if row else 0)
        if total <= keep:
            return 0
        await self.conn.execute(
            "DELETE FROM messages WHERE chat_id=? AND id NOT IN ("
            "  SELECT id FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?"
            ")",
            (chat_id, chat_id, keep),
        )
        await self.conn.commit()
        deleted = total - keep
        log.info("rotated messages chat_id=%s deleted=%s kept=%s", chat_id, deleted, keep)
        return deleted

    # ------------------------------------------------------------------
    # Alta internal state: chat mood, user relations, short-lived reactions
    # ------------------------------------------------------------------

    async def get_alta_chat_state(self, chat_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT energy,irritation,playfulness,social_battery,updated_at "
            "FROM alta_chat_state WHERE chat_id=?",
            (chat_id,),
        )
        row = await cur.fetchone()
        if row:
            return dict(row)
        return {
            "energy": 0.7, "irritation": 0.0,
            "playfulness": 0.5, "social_battery": 0.8,
            "updated_at": "",
        }

    async def save_alta_chat_state(
        self, chat_id: int,
        energy: float, irritation: float,
        playfulness: float, social_battery: float,
    ) -> None:
        assert self.conn
        await self.conn.execute(
            "INSERT INTO alta_chat_state(chat_id,energy,irritation,playfulness,social_battery) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET "
            "energy=excluded.energy,irritation=excluded.irritation,"
            "playfulness=excluded.playfulness,social_battery=excluded.social_battery,"
            "updated_at=datetime('now')",
            (chat_id, energy, irritation, playfulness, social_battery),
        )
        await self.conn.commit()

    async def get_alta_user_relation(self, chat_id: int, user_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT trust,warmth,caution,annoyance,interaction_count,updated_at "
            "FROM alta_user_relation WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        )
        row = await cur.fetchone()
        if row:
            return dict(row)
        return {
            "trust": 0.5, "warmth": 0.5, "caution": 0.0,
            "annoyance": 0.0, "interaction_count": 0, "updated_at": "",
        }

    async def save_alta_user_relation(
        self, chat_id: int, user_id: int,
        trust: float, warmth: float, caution: float,
        annoyance: float, interaction_count: int,
    ) -> None:
        assert self.conn
        await self.conn.execute(
            "INSERT INTO alta_user_relation"
            "(chat_id,user_id,trust,warmth,caution,annoyance,interaction_count) "
            "VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,user_id) DO UPDATE SET "
            "trust=excluded.trust,warmth=excluded.warmth,caution=excluded.caution,"
            "annoyance=excluded.annoyance,interaction_count=excluded.interaction_count,"
            "updated_at=datetime('now')",
            (chat_id, user_id, trust, warmth, caution, annoyance, interaction_count),
        )
        await self.conn.commit()

    async def get_active_alta_reaction(
        self, chat_id: int, target_user_id: int | None = None
    ) -> dict | None:
        assert self.conn
        if target_user_id is not None:
            cur = await self.conn.execute(
                "SELECT id,reaction_kind,reaction_strength,ttl_messages,target_user_id "
                "FROM alta_reactions "
                "WHERE chat_id=? AND ttl_messages > 0 "
                "AND (target_user_id=? OR target_user_id IS NULL) "
                "ORDER BY reaction_strength DESC LIMIT 1",
                (chat_id, target_user_id),
            )
        else:
            cur = await self.conn.execute(
                "SELECT id,reaction_kind,reaction_strength,ttl_messages,target_user_id "
                "FROM alta_reactions "
                "WHERE chat_id=? AND ttl_messages > 0 "
                "ORDER BY reaction_strength DESC LIMIT 1",
                (chat_id,),
            )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def add_alta_reaction(
        self, chat_id: int, reaction_kind: str,
        reaction_strength: float, ttl_messages: int,
        target_user_id: int | None = None,
    ) -> None:
        assert self.conn
        await self.conn.execute(
            "INSERT INTO alta_reactions"
            "(chat_id,reaction_kind,reaction_strength,ttl_messages,target_user_id) "
            "VALUES(?,?,?,?,?)",
            (chat_id, reaction_kind, reaction_strength, ttl_messages, target_user_id),
        )
        await self.conn.commit()

    async def tick_alta_reactions(self, chat_id: int) -> None:
        """Decrement TTL of all active reactions, delete expired ones."""
        assert self.conn
        await self.conn.execute(
            "UPDATE alta_reactions SET ttl_messages=ttl_messages-1 "
            "WHERE chat_id=? AND ttl_messages > 0",
            (chat_id,),
        )
        await self.conn.execute(
            "DELETE FROM alta_reactions WHERE chat_id=? AND ttl_messages <= 0",
            (chat_id,),
        )
        await self.conn.commit()

    async def prune_debug_traces(self, keep: int = 500) -> int:
        """Delete oldest debug_traces, keeping the most recent *keep*."""
        assert self.conn
        cur = await self.conn.execute("SELECT COUNT(1) AS cnt FROM debug_traces")
        row = await cur.fetchone()
        total = int(row["cnt"] if row else 0)
        if total <= keep:
            return 0
        await self.conn.execute(
            "DELETE FROM debug_traces WHERE id NOT IN ("
            "  SELECT id FROM debug_traces ORDER BY id DESC LIMIT ?"
            ")",
            (keep,),
        )
        await self.conn.commit()
        deleted = total - keep
        log.info("pruned debug_traces deleted=%s kept=%s", deleted, keep)
        return deleted
