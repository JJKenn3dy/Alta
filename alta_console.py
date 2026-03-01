import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import aiosqlite
from dotenv import load_dotenv

from llm import LLM
from prompts import ALTA_SYSTEM_PROMPT
from runtime_io import configure_stdio

load_dotenv()


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    db_path: str
    history_limit: int
    temperature: float
    top_p: float
    max_tokens: int


def load_config() -> Config:
    return Config(
        base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1").strip(),
        api_key=os.getenv("LLM_API_KEY", "local-llm").strip(),
        model=os.getenv("LLM_MODEL", "saiga_nemo_12b.Q8_0.gguf").strip(),
        db_path=os.getenv("DB_PATH", "./alta_console.db").strip(),
        history_limit=int(os.getenv("HISTORY_LIMIT", "40")),
        temperature=float(os.getenv("TEMPERATURE", "0.85")),
        top_p=float(os.getenv("TOP_P", "0.9")),
        max_tokens=int(os.getenv("MAX_TOKENS", "220")),
    )


# =========================
# DB
# =========================
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  role TEXT NOT NULL,   -- 'user' | 'assistant'
  text TEXT NOT NULL,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id DESC);
"""


class DB:
    def __init__(self, path: str):
        self.path = path
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self):
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.executescript(SCHEMA_SQL)
        await self.conn.commit()

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def create_session(self, name: str = "default") -> int:
        assert self.conn
        await self.conn.execute("INSERT INTO sessions(name) VALUES (?)", (name,))
        await self.conn.commit()
        cur = await self.conn.execute("SELECT last_insert_rowid() AS id")
        row = await cur.fetchone()
        return int(row["id"])

    async def list_sessions(self) -> list[dict]:
        assert self.conn
        cur = await self.conn.execute(
            "SELECT id, name, created_at FROM sessions ORDER BY id DESC LIMIT 20"
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def add_message(self, session_id: int, role: str, text: str):
        assert self.conn
        await self.conn.execute(
            "INSERT INTO messages(session_id, role, text) VALUES (?, ?, ?)",
            (session_id, role, text),
        )
        await self.conn.commit()

    async def get_last_messages(self, session_id: int, limit: int) -> list[dict]:
        assert self.conn
        cur = await self.conn.execute(
            """
            SELECT role, text FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = await cur.fetchall()
        rows = list(reversed(rows))
        return [{"role": r["role"], "content": r["text"]} for r in rows]

    async def clear_session(self, session_id: int):
        assert self.conn
        await self.conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        await self.conn.commit()


# =========================
# CONSOLE APP
# =========================
HELP_TEXT = """
Команды:
  /help            показать команды
  /new [name]      создать новую сессию (диалог)
  /sessions        список последних сессий
  /use <id>        переключиться на сессию по id
  /clear           очистить историю текущей сессии
  /exit            выход

Обычный текст - это сообщение Альте.
""".strip()


async def repl():
    configure_stdio()
    cfg = load_config()
    db = DB(cfg.db_path)
    await db.connect()

    llm = LLM(cfg.base_url, cfg.api_key, cfg.model)

    session_id = await db.create_session("default")

    print("Альта: ну шо, базарим. Пиши. (/help если потерялся)")

    while True:
        try:
            user_in = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nАльта: ладно, исчезай. ✌️")
            break

        if not user_in:
            continue

        # Commands
        if user_in.startswith("/"):
            parts = user_in.split()
            cmd = parts[0].lower()

            if cmd in ("/exit", "/quit"):
                print("Альта: давай, не теряйся.")
                break

            if cmd == "/help":
                print(HELP_TEXT)
                continue

            if cmd == "/new":
                name = " ".join(parts[1:]).strip() or "session"
                session_id = await db.create_session(name)
                print(f"Альта: окей, новая болталка. id={session_id}")
                continue

            if cmd == "/sessions":
                sessions = await db.list_sessions()
                if not sessions:
                    print("Альта: пусто.")
                else:
                    for s in sessions:
                        print(f'[{s["id"]}] {s["name"]} @ {s["created_at"]}')
                continue

            if cmd == "/use":
                if len(parts) < 2:
                    print("Альта: /use <id>")
                    continue
                try:
                    session_id = int(parts[1])
                    print(f"Альта: переключилась на id={session_id}")
                except ValueError:
                    print("Альта: id должен быть числом.")
                continue

            if cmd == "/clear":
                await db.clear_session(session_id)
                print("Альта: стерла историю. Начинай заново.")
                continue

            print("Альта: не поняла команду. /help")
            continue

        # Normal message
        await db.add_message(session_id, "user", user_in)

        history = await db.get_last_messages(session_id, cfg.history_limit)
        try:
            answer = await llm.chat(
                system_prompt=ALTA_SYSTEM_PROMPT,
                messages=history,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
            )
        except Exception as e:
            print(f"Альта: бля, у меня тут сбой: {e}")
            continue

        if not answer:
            answer = "Альта: ..."

        print(f"Альта: {answer}")
        await db.add_message(session_id, "assistant", answer)

    await db.close()


if __name__ == "__main__":
    configure_stdio()
    asyncio.run(repl())
