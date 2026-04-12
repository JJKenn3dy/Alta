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
                await db.upsert_user_profile(
                    1, 101,
                    approved_display_name="Иван",
                    display_name="Иван",
                    gender="мужской",
                    interests="музыка, код",
                    profile_confidence=1.0,
                )
                await db.upsert_user_profile(
                    1, 202,
                    approved_display_name="Анна",
                    display_name="Анна",
                    gender="женский",
                    interests="кино, книги",
                    profile_confidence=1.0,
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
        self.assertEqual(first["gender"], "мужской")
        self.assertEqual(second["approved_display_name"], "Анна")
        self.assertEqual(second["gender"], "женский")

    async def test_upsert_overwrites_profile_fields(self):
        path = self._make_db_path("upsert_profile")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user_profile(
                    1, 101,
                    display_name="Иван",
                    gender="мужской",
                )
                await db.upsert_user_profile(
                    1, 101,
                    display_name="Ваня",
                )
                profile = await db.get_user_profile(1, 101)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(profile["display_name"], "Ваня")
        self.assertEqual(profile["gender"], "мужской")

    async def test_add_message_and_count(self):
        path = self._make_db_path("messages_count")
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user(101, "Target_User")
                await db.add_message(1, 101, "user", "msg-1")
                await db.add_message(1, 101, "user", "msg-2")
                await db.add_message(1, 999, "assistant", "bot-msg", target_user_id=101)

                count = await db.count_user_messages(1, 101)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(count, 2)

    async def test_private_profile_overrides_group_profile_in_merged_view(self):
        path = self._make_db_path("profile_merge_priority")
        user_id = 9001
        chat_id = -100700
        try:
            db = DB(path)
            await db.connect()
            try:
                await db.upsert_user_profile(
                    chat_id,
                    user_id,
                    approved_display_name="АвтоИмя",
                    display_name="АвтоИмя",
                    interests="игры",
                    communication_style="коротко",
                    profile_confidence=0.4,
                )
                await db.upsert_user_profile(
                    user_id,
                    user_id,
                    approved_display_name="РучноеИмя",
                    display_name="РучноеИмя",
                    communication_style="развёрнуто",
                    profile_confidence=1.0,
                )
                merged = await db.get_user_profile_merged(chat_id, user_id)
            finally:
                await db.close()
        finally:
            for suffix in ("", "-wal", "-shm"):
                candidate = f"{path}{suffix}"
                if os.path.exists(candidate):
                    os.remove(candidate)

        self.assertEqual(merged.get("approved_display_name"), "РучноеИмя")
        self.assertEqual(merged.get("display_name"), "РучноеИмя")
        self.assertEqual(merged.get("communication_style"), "развёрнуто")
        # Group-only field should still be visible when private profile doesn't set it.
        self.assertEqual(merged.get("interests"), "игры")


if __name__ == "__main__":
    unittest.main()
