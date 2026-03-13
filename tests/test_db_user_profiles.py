import os
import unittest
import uuid

from db import DB


class DBUserProfileTests(unittest.IsolatedAsyncioTestCase):
    def _make_db_path(self, prefix: str) -> str:
        return os.path.join("tests", f"{prefix}_{uuid.uuid4().hex}.sqlite3")

    async def test_profiles_are_isolated_per_user(self):
        path = self._make_db_path("user_profiles")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.replace_user_profile(
                    1,
                    101,
                    profile={
                        "approved_display_name": "Иван",
                        "display_name": "Иван",
                        "gender": "мужской",
                        "interests": "музыка, код",
                        "extra_fields": {"city": "Москва", "occupation": "backend"},
                        "profile_confidence": 1.0,
                    },
                )
                await db.replace_user_profile(
                    1,
                    202,
                    profile={
                        "approved_display_name": "Анна",
                        "display_name": "Анна",
                        "gender": "женский",
                        "interests": "кино, книги",
                        "extra_fields": {"city": "Казань"},
                        "profile_confidence": 1.0,
                    },
                )

                first = await db.get_user_profile(1, 101)
                second = await db.get_user_profile(1, 202)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(first["approved_display_name"], "Иван")
        self.assertEqual(first["extra_fields"]["city"], "Москва")
        self.assertEqual(first["extra_fields"]["occupation"], "backend")
        self.assertEqual(second["approved_display_name"], "Анна")
        self.assertEqual(second["extra_fields"]["city"], "Казань")
        self.assertNotIn("occupation", second["extra_fields"])

    async def test_clear_user_profile_resets_fields_and_extra_json(self):
        path = self._make_db_path("clear_user_profile")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.replace_user_profile(
                    1,
                    101,
                    profile={
                        "approved_display_name": "Иван",
                        "display_name": "Иван",
                        "gender": "мужской",
                        "interests": "музыка, код",
                        "extra_fields": {"city": "Москва"},
                        "profile_confidence": 1.0,
                    },
                )
                await db.clear_user_profile(1, 101)
                profile = await db.get_user_profile(1, 101)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertIsNone(profile["approved_display_name"])
        self.assertIsNone(profile["display_name"])
        self.assertIsNone(profile["gender"])
        self.assertEqual(profile["interests"], "")
        self.assertEqual(profile["extra_fields"], {})
        self.assertEqual(profile["profile_confidence"], 0.0)

    async def test_find_user_by_username_and_delete_user_messages(self):
        path = self._make_db_path("moderation_db_helpers")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user(101, "Target_User")
                await db.add_message(1, 101, "user", "msg-1", telegram_message_id=111)
                await db.add_message(1, 101, "user", "msg-2", telegram_message_id=222)
                await db.add_message(1, 999, "assistant", "bot-msg", telegram_message_id=333, target_user_id=101)

                found_user_id = await db.find_user_id_by_username("@target_user")
                message_ids = await db.list_user_telegram_message_ids(1, 101)
                deleted_count = await db.delete_user_messages(1, 101)
                remaining_user_messages = await db.count_user_messages(1, 101)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(found_user_id, 101)
        self.assertEqual(message_ids, [222, 111])
        self.assertEqual(deleted_count, 2)
        self.assertEqual(remaining_user_messages, 0)


if __name__ == "__main__":
    unittest.main()
