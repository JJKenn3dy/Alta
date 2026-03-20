import os
import sqlite3
import unittest
import uuid

from db import DB


class DBMigrationTests(unittest.IsolatedAsyncioTestCase):
    def _make_db_path(self, prefix: str) -> str:
        return os.path.join("tests", f"{prefix}_{uuid.uuid4().hex}.sqlite3")

    async def test_connect_migrates_legacy_messages_without_context_flag(self):
        path = self._make_db_path("legacy_db")
        try:
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                CREATE TABLE messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  chat_id INTEGER NOT NULL,
                  user_id INTEGER NOT NULL,
                  telegram_message_id INTEGER,
                  target_user_id INTEGER,
                  role TEXT NOT NULL,
                  text TEXT NOT NULL,
                  ts TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )
            conn.commit()
            conn.close()

            db = DB(path)
            await db.connect()
            try:
                cur = await db.conn.execute("PRAGMA table_info(messages)")
                columns = {row["name"] for row in await cur.fetchall()}
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertIn("exclude_from_context", columns)

    async def test_hidden_messages_are_excluded_from_default_history(self):
        path = self._make_db_path("history_db")
        try:
            db = DB(path)
            await db.connect()
            try:
                message_id = await db.add_message(1, 42, "user", "iz spb")
                hidden = await db.hide_messages_from_context([message_id])
                visible_history = await db.get_last_messages(1, 10)
                hidden_history = await db.get_last_messages(1, 10, include_hidden=True)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(hidden, 1)
        self.assertEqual(visible_history, [])
        self.assertEqual(len(hidden_history), 1)
        self.assertIn("[stored in user memory]", hidden_history[0]["content"])

    async def test_connect_cleans_meta_identity_noise_from_user_facts(self):
        path = self._make_db_path("facts_noise_cleanup")
        try:
            db = DB(path)
            await db.connect()
            try:
                bad_value = "\u043d\u0435 \u0437\u043d\u0430\u0435\u0442, \u043a\u0430\u043a \u0435\u0433\u043e \u0437\u043e\u0432\u0443\u0442"
                bad_evidence = "\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442?"
                good_value = "\u043b\u044e\u0431\u0438\u0442 \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438"
                good_evidence = (
                    "\u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u0447\u0442\u043e "
                    "\u044f \u043b\u044e\u0431\u043b\u044e \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438"
                )
                good_fact_value = "\u0437\u043e\u0432\u0443\u0442 \u0417\u0430\u0445\u0430\u0440"
                good_fact_evidence = "\u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442 \u0417\u0430\u0445\u0430\u0440"
                await db.upsert_user_fact(
                    1,
                    42,
                    "fact",
                    bad_value,
                    bad_value.lower(),
                    confidence=0.9,
                    importance=0.9,
                    evidence=bad_evidence,
                    source="auto_extract",
                )
                await db.upsert_user_fact(
                    1,
                    42,
                    "preference",
                    good_value,
                    good_value.lower(),
                    confidence=0.8,
                    importance=0.8,
                    evidence=good_evidence,
                    source="auto_extract",
                )
                await db.upsert_user_fact(
                    1,
                    42,
                    "fact",
                    good_fact_value,
                    good_fact_value.lower(),
                    confidence=0.8,
                    importance=0.8,
                    evidence=good_fact_evidence,
                    source="auto_extract",
                )
            finally:
                await db.close()

            db = DB(path)
            await db.connect()
            try:
                facts = await db.get_user_facts(1, 42, limit=10)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(len(facts), 2)
        kept_values = {item["value"] for item in facts}
        self.assertIn("\u043b\u044e\u0431\u0438\u0442 \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438", kept_values)
        self.assertIn("\u0437\u043e\u0432\u0443\u0442 \u0417\u0430\u0445\u0430\u0440", kept_values)


if __name__ == "__main__":
    unittest.main()
