import unittest

try:
    from main import _join_verification_ban_kwargs
except Exception as exc:  # pragma: no cover - import guard for optional deps
    _join_verification_ban_kwargs = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    _join_verification_ban_kwargs is None,
    f"failed to import main join verification helpers: {IMPORT_ERROR}",
)
class MainJoinVerificationTests(unittest.TestCase):
    def test_join_verification_uses_permanent_ban_kwargs(self):
        kwargs = _join_verification_ban_kwargs(chat_id=-100123, user_id=42)
        self.assertEqual(kwargs["chat_id"], -100123)
        self.assertEqual(kwargs["user_id"], 42)
        self.assertTrue(kwargs["revoke_messages"])
        self.assertNotIn("until_date", kwargs)


if __name__ == "__main__":
    unittest.main()
