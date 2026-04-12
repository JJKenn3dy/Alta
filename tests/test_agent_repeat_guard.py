import unittest

from agent import _is_repeated_recent_reply, _recent_assistant_replies, _reply_repeat_key


class AgentRepeatGuardTests(unittest.TestCase):
    def test_reply_repeat_key_strips_prefix_and_punctuation(self):
        source = "[to current speaker id=42 @alex] \u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e."
        self.assertEqual(_reply_repeat_key(source), "\u043d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e")

    def test_repeat_guard_flags_direct_repeat_of_last_reply(self):
        recent = ["\u041d\u0435 \u0437\u043d\u0430\u044e.", "\u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e."]
        self.assertTrue(
            _is_repeated_recent_reply(
                "\u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e",
                recent,
            )
        )

    def test_repeat_guard_flags_phrase_seen_twice_in_recent_window(self):
        recent = [
            "\u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e.",
            "\u041e\u043a.",
            "\u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e",
            "\u0414\u0430\u043b\u044c\u0448\u0435?",
        ]
        self.assertTrue(
            _is_repeated_recent_reply(
                "\u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e!",
                recent,
            )
        )

    def test_repeat_guard_allows_non_repeated_reply(self):
        recent = ["\u041d\u0443 \u0438 \u043b\u0430\u0434\u043d\u043e.", "\u041e\u043a."]
        self.assertFalse(
            _is_repeated_recent_reply(
                "\u041a\u0430\u043a \u0443 \u0442\u0435\u0431\u044f \u0434\u0435\u043b\u0430?",
                recent,
            )
        )

    def test_repeat_guard_flags_short_exact_repeat(self):
        recent = ["\u041b\u0430\u0434\u043d\u043e.", "\u041d\u0435\u0442."]
        self.assertTrue(_is_repeated_recent_reply("\u041d\u0435\u0442", recent))

    def test_repeat_guard_flags_short_exact_phrase_seen_twice(self):
        recent = ["\u041e\u043a.", "\u041d\u0443?", "\u041e\u043a", "\u0414\u0430\u043b\u044c\u0448\u0435"]
        self.assertTrue(_is_repeated_recent_reply("\u043e\u043a", recent))

    def test_repeat_guard_allows_short_non_repeated_reply(self):
        recent = ["\u041d\u0435\u0442.", "\u041b\u0430\u0434\u043d\u043e."]
        self.assertFalse(_is_repeated_recent_reply("\u0434\u0430", recent))

    def test_recent_assistant_replies_uses_only_assistant_and_limit(self):
        history = [
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "ignored"},
            {"role": "assistant", "content": "two"},
            {"role": "assistant", "content": "three"},
        ]
        self.assertEqual(_recent_assistant_replies(history, limit=2), ["two", "three"])


if __name__ == "__main__":
    unittest.main()
