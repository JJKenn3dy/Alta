import unittest

from agent import CONTEXT_HISTORY_LIMIT, _summary_update_interval


class AgentSummaryIntervalTests(unittest.TestCase):
    def test_interval_is_clamped_to_effective_history_window(self):
        self.assertEqual(_summary_update_interval(80), CONTEXT_HISTORY_LIMIT)

    def test_interval_uses_smaller_history_limit(self):
        self.assertEqual(_summary_update_interval(10), 10)

    def test_interval_has_safe_minimum(self):
        self.assertEqual(_summary_update_interval(0), 1)

    def test_interval_handles_invalid_input(self):
        self.assertEqual(_summary_update_interval("oops"), CONTEXT_HISTORY_LIMIT)


if __name__ == "__main__":
    unittest.main()
