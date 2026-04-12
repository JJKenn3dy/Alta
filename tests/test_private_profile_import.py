import os
import unittest
import uuid

from db import DB
from private_profile import import_from_groups


class PrivateProfileImportTests(unittest.IsolatedAsyncioTestCase):
    def _make_db_path(self, prefix: str) -> str:
        return os.path.join("tests", f"{prefix}_{uuid.uuid4().hex}.sqlite3")

    async def test_group_import_skips_relationship_notes(self):
        path = self._make_db_path("private_import_skip_rel_notes")
        user_id = 123456
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user_profile(
                    -100500,
                    user_id,
                    approved_display_name="Тест",
                    display_name="Тест",
                    interests="музыка",
                    relationship_notes="транзиентная заметка из группы",
                    profile_confidence=0.8,
                )
                await db.upsert_user_profile(
                    user_id,
                    user_id,
                    profile_confidence=1.0,
                )

                imported = await import_from_groups(db, user_id)
                private = await db.get_user_profile(user_id, user_id)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertGreater(imported, 0)
        self.assertEqual(private.get("relationship_notes", "") or "", "")
        self.assertEqual(private.get("interests", "") or "", "музыка")


if __name__ == "__main__":
    unittest.main()
