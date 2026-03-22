import os
import unittest
import uuid

from db import DB


class DBMigrationTests(unittest.IsolatedAsyncioTestCase):
    def _make_db_path(self, prefix: str) -> str:
        return os.path.join("tests", f"{prefix}_{uuid.uuid4().hex}.sqlite3")

    async def test_connect_creates_schema_from_scratch(self):
        path = self._make_db_path("fresh_db")
        try:
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

        self.assertIn("chat_id", columns)
        self.assertIn("user_id", columns)
        self.assertIn("role", columns)
        self.assertIn("text", columns)

    async def test_upsert_and_get_user_facts(self):
        path = self._make_db_path("facts_db")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user_fact(
                    1, 42, "preference",
                    "любит энергетики",
                    "любит энергетики",
                    confidence=0.8, importance=0.8,
                    evidence="запомни что я люблю энергетики",
                    source="auto_extract",
                )
                await db.upsert_user_fact(
                    1, 42, "fact",
                    "зовут Захар",
                    "зовут захар",
                    confidence=0.8, importance=0.8,
                    evidence="меня зовут Захар",
                    source="auto_extract",
                )
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
        self.assertIn("любит энергетики", kept_values)
        self.assertIn("зовут Захар", kept_values)

    async def test_get_summary_no_fallback_to_general(self):
        """After fix #2: get_summary(chat_id, user_id) should NOT fall back to general summary."""
        path = self._make_db_path("summary_nofallback")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.set_summary(1, "общий summary чата")
                # Request user-specific summary that doesn't exist
                user_summary = await db.get_summary(1, user_id=42)
                # Request general summary
                general_summary = await db.get_summary(1)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(user_summary, "")
        self.assertEqual(general_summary, "общий summary чата")


if __name__ == "__main__":
    unittest.main()
