"""Tests for per-user flirt policy prompt block."""
import unittest

from context_builder import build_flirt_policy_block


class TestFlirtPolicyBlock(unittest.TestCase):

    def test_regular_user_gets_restriction(self):
        block = build_flirt_policy_block("some_user")
        self.assertIn("запрещены", block)
        self.assertNotIn("ИСКЛЮЧЕНИЕ", block)

    def test_missing_username_gets_restriction(self):
        block = build_flirt_policy_block(None)
        self.assertIn("запрещены", block)

    def test_exception_username_allows_flirt(self):
        block = build_flirt_policy_block("@JJKenn9DY")
        self.assertIn("ИСКЛЮЧЕНИЕ", block)
        self.assertIn("допустим", block)


if __name__ == "__main__":
    unittest.main()
