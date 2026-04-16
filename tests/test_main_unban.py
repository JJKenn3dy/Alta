import unittest

try:
    from moderation import UNBAN_COMMAND_RE, _can_restrict_members, _parse_unban_target_token
except Exception as exc:  # pragma: no cover - import guard for optional deps
    UNBAN_COMMAND_RE = None
    _can_restrict_members = None
    _parse_unban_target_token = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _Status:
    def __init__(self, value: str):
        self.value = value


class _Member:
    def __init__(self, *, status, can_restrict_members: bool = False):
        self.status = status
        self.can_restrict_members = can_restrict_members


@unittest.skipIf(
    UNBAN_COMMAND_RE is None or _can_restrict_members is None or _parse_unban_target_token is None,
    f"failed to import unban helpers: {IMPORT_ERROR}",
)
class MainUnbanTests(unittest.TestCase):
    def test_unban_regex_accepts_plain_command(self):
        match = UNBAN_COMMAND_RE.match("/unban 12345")
        self.assertIsNotNone(match)
        self.assertEqual(match.group("target"), "12345")

    def test_unban_regex_accepts_command_with_bot_suffix(self):
        match = UNBAN_COMMAND_RE.match("/unban@TestBot @TargetUser")
        self.assertIsNotNone(match)
        self.assertEqual(match.group("target"), "@TargetUser")

    def test_parse_unban_target_token_with_id(self):
        user_id, username = _parse_unban_target_token("-12345")
        self.assertEqual(user_id, -12345)
        self.assertIsNone(username)

    def test_parse_unban_target_token_with_username(self):
        user_id, username = _parse_unban_target_token("@Target_User")
        self.assertIsNone(user_id)
        self.assertEqual(username, "Target_User")

    def test_parse_unban_target_token_rejects_invalid(self):
        user_id, username = _parse_unban_target_token("bad*name")
        self.assertIsNone(user_id)
        self.assertIsNone(username)

    def test_can_restrict_members_for_creator(self):
        member = _Member(status="creator")
        self.assertTrue(_can_restrict_members(member))

    def test_can_restrict_members_for_admin_with_flag(self):
        member = _Member(status=_Status("administrator"), can_restrict_members=True)
        self.assertTrue(_can_restrict_members(member))

    def test_can_restrict_members_for_admin_without_flag(self):
        member = _Member(status=_Status("administrator"), can_restrict_members=False)
        self.assertFalse(_can_restrict_members(member))


if __name__ == "__main__":
    unittest.main()
