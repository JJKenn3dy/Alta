import aiosqlite
from typing import Optional

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
  role TEXT NOT NULL,  -- 'user' | 'assistant'
  text TEXT NOT NULL,
  ts TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_target ON messages(chat_id, target_user_id, id DESC);

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
  gender TEXT,
  interests TEXT NOT NULL DEFAULT '',
  communication_style TEXT NOT NULL DEFAULT '',
  response_preference TEXT NOT NULL DEFAULT '',
  relationship_notes TEXT NOT NULL DEFAULT '',
  profile_confidence REAL NOT NULL DEFAULT 0.0,
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS user_documents (
  chat_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  document_json TEXT NOT NULL DEFAULT '{}',
  document_text TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (chat_id, user_id)
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
  file_id INTEGER NOT NULL,
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

    async def connect(self):
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.executescript(SCHEMA)
        await self._migrate_schema()
        await self._cleanup_identity_noise()
        await self.conn.commit()

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
            f"UPDATE user_profiles SET gender=NULL "
            f"WHERE lower(trim(gender)) IN ({gender_placeholders})",
            INVALID_GENDER_VALUES,
        )
        await self.conn.execute(
            "UPDATE user_profiles SET "
            "gender=NULL,"
            "interests='',"
            "communication_style='',"
            "response_preference='',"
            "relationship_notes='' "
            "WHERE profile_confidence < 0.75"
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
                "SELECT m.id,m.role,m.text,m.user_id,m.target_user_id,u.username "
                "FROM messages m "
                "LEFT JOIN users u ON u.user_id = m.user_id "
                "WHERE m.chat_id=? "
                "ORDER BY m.id DESC LIMIT ?",
                (chat_id, limit),
            )
        else:
            cur = await self.conn.execute(
                "SELECT m.id,m.role,m.text,m.user_id,m.target_user_id,u.username "
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
            if r["role"] == "user":
                parts = [f"id={r['user_id']}"]
                if r["username"]:
                    parts.append(f"@{r['username']}")
                content = f"[{' '.join(parts)}] {r['text']}"
            else:
                content = r["text"]
            out.append({"id": int(r["id"]), "role": r["role"], "content": content})
        return out

    async def get_summary(self, chat_id: int, user_id: int | None = None) -> str:
        assert self.conn
        if user_id is not None:
            cur = await self.conn.execute(
                "SELECT summary FROM user_chat_summaries WHERE chat_id=? AND user_id=?",
                (chat_id, user_id),
            )
            row = await cur.fetchone()
            if row:
                return row["summary"]

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

    async def get_user_profile(self, chat_id: int, user_id: int) -> dict:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT display_name,nickname,gender,interests,communication_style,"
            "response_preference,relationship_notes,profile_confidence,updated_at "
            "FROM user_profiles WHERE chat_id=? AND user_id=?",
            (chat_id, user_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else {}

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
        gender: str | None = None,
        interests: str | None = None,
        communication_style: str | None = None,
        response_preference: str | None = None,
        relationship_notes: str | None = None,
        profile_confidence: float | None = None,
    ):
        assert self.conn
        current = await self.get_user_profile(chat_id, user_id)
        await self.conn.execute(
            "INSERT INTO user_profiles("
            "chat_id,user_id,display_name,nickname,gender,interests,communication_style,"
            "response_preference,relationship_notes,profile_confidence"
            ") VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id,user_id) DO UPDATE SET "
            "display_name=excluded.display_name,"
            "nickname=excluded.nickname,"
            "gender=excluded.gender,"
            "interests=excluded.interests,"
            "communication_style=excluded.communication_style,"
            "response_preference=excluded.response_preference,"
            "relationship_notes=excluded.relationship_notes,"
            "profile_confidence=excluded.profile_confidence,"
            "updated_at=datetime('now')",
            (
                chat_id,
                user_id,
                display_name if display_name is not None else current.get("display_name"),
                nickname if nickname is not None else current.get("nickname"),
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
            ),
        )
        await self.conn.commit()

    async def get_memories(self, chat_id: int, user_id: int, limit: int = 30) -> list[dict]:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT key,value,confidence,importance,source FROM memories WHERE chat_id=? AND user_id=? "
            "ORDER BY importance DESC, confidence DESC, updated_at DESC LIMIT ?",
            (chat_id, user_id, limit),
        )
        return [dict(r) for r in await cur.fetchall()]

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
        await self.conn.commit()

    async def delete_context_file(self, path: str):
        assert self.conn
        cur = await self.conn.execute("SELECT id FROM context_files WHERE path=?", (path,))
        row = await cur.fetchone()
        if not row:
            return

        file_id = int(row["id"])
        await self.conn.execute("DELETE FROM context_chunks WHERE file_id=?", (file_id,))
        await self.conn.execute("DELETE FROM context_files WHERE id=?", (file_id,))
        await self.conn.commit()

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
