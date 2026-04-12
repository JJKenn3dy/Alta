import os
import unittest
import uuid

from db import DB
from memory_engine import _build_extraction_text


class MemoryExtractionTextTests(unittest.IsolatedAsyncioTestCase):
    def _make_db_path(self, prefix: str) -> str:
        return os.path.join("tests", f"{prefix}_{uuid.uuid4().hex}.sqlite3")

    async def test_extraction_text_marks_foreign_messages(self):
        path = self._make_db_path("extraction_text")
        chat_id = -1001
        target_user_id = 501
        foreign_user_id = 777
        input_text = "я люблю драм-н-бейс"
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user(target_user_id, "target_user")
                await db.upsert_user(foreign_user_id, "foreign_user")
                await db.add_message(chat_id, target_user_id, "user", "привет")
                await db.add_message(chat_id, foreign_user_id, "user", "я работаю тестировщиком")
                await db.add_message(chat_id, 0, "assistant", "ок")
                await db.add_message(chat_id, target_user_id, "user", input_text)

                extraction_text = await _build_extraction_text(
                    db,
                    chat_id,
                    target_user_id,
                    input_text,
                )
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertIn("Другой участник чата:", extraction_text)
        self.assertIn("id=777", extraction_text)
        self.assertIn("Пользователь:", extraction_text)
        self.assertIn("id=501", extraction_text)
        self.assertIn("Сообщение пользователя (извлекать факты ТОЛЬКО из него):", extraction_text)


if __name__ == "__main__":
    unittest.main()
