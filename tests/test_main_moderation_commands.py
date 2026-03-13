import unittest

try:
    from main import (
        _can_delete_messages,
        _can_restrict_members,
        parse_moderation_command,
    )
except Exception as exc:  # pragma: no cover - import guard for optional deps
    _can_delete_messages = None
    _can_restrict_members = None
    parse_moderation_command = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _Status:
    def __init__(self, value: str):
        self.value = value


class _Member:
    def __init__(
        self,
        *,
        status: str | _Status,
        can_restrict_members: bool = False,
        can_delete_messages: bool = False,
    ):
        self.status = status
        self.can_restrict_members = can_restrict_members
        self.can_delete_messages = can_delete_messages


@unittest.skipIf(parse_moderation_command is None, f"failed to import moderation helpers: {IMPORT_ERROR}")
class MainModerationCommandTests(unittest.TestCase):
    def test_parse_ban_command_with_mention_prefix(self):
        command = parse_moderation_command("@Target_User бан")
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "ban")
        self.assertEqual(command["target_username"], "Target_User")

    def test_parse_kick_command_with_mention_suffix(self):
        command = parse_moderation_command("пожалуйста кикнуть @Target_User")
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "kick")
        self.assertEqual(command["target_username"], "Target_User")

    def test_parse_purge_command_with_tail(self):
        command = parse_moderation_command("почистить @Target_User все сообщения")
        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "purge")
        self.assertEqual(command["target_username"], "Target_User")

    def test_parse_returns_none_without_target_mention(self):
        self.assertIsNone(parse_moderation_command("забанить его"))

    def test_parse_returns_none_for_non_command_phrase(self):
        self.assertIsNone(parse_moderation_command("не бань @Target_User"))

    def test_owner_has_all_permissions(self):
        member = _Member(status="creator")
        self.assertTrue(_can_restrict_members(member))
        self.assertTrue(_can_delete_messages(member))

    def test_admin_permissions_respect_flags(self):
        member = _Member(
            status=_Status("administrator"),
            can_restrict_members=True,
            can_delete_messages=False,
        )
        self.assertTrue(_can_restrict_members(member))
        self.assertFalse(_can_delete_messages(member))


if __name__ == "__main__":
    unittest.main()
