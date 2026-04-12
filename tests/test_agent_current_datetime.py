import re
import unittest
from datetime import datetime

from agent import _current_datetime_system_block, _deterministic_now_answer


class AgentCurrentDatetimeTests(unittest.TestCase):
    def test_current_datetime_system_block_contains_timestamp_and_rule(self):
        block = _current_datetime_system_block()

        self.assertRegex(block, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertRegex(block, r"UTC[+-]\d{2}:\d{2}")
        self.assertIn("На любой вопрос о дате, времени, годе", block)

    def test_current_datetime_system_block_has_current_year(self):
        block = _current_datetime_system_block()
        self.assertIn(str(datetime.now().year), block)

    def test_deterministic_now_answer_for_year_and_date(self):
        answer = _deterministic_now_answer("\u043a\u0430\u043a\u043e\u0439 \u0441\u0435\u0439\u0447\u0430\u0441 \u0433\u043e\u0434 \u0438 \u0434\u0430\u0442\u0430?")
        self.assertIsNotNone(answer)
        self.assertIn(str(datetime.now().year), answer)
        self.assertIn(datetime.now().strftime("%d.%m.%Y"), answer)

    def test_deterministic_now_answer_is_not_triggered_without_now_reference(self):
        answer = _deterministic_now_answer("\u043a\u0430\u043a\u043e\u0439 \u0431\u044b\u043b \u0433\u043e\u0434 \u0440\u0435\u043b\u0438\u0437\u0430 MiSide?")
        self.assertIsNone(answer)

    def test_deterministic_now_answer_for_time(self):
        answer = _deterministic_now_answer("\u043a\u043e\u0442\u043e\u0440\u044b\u0439 \u0441\u0435\u0439\u0447\u0430\u0441 \u0447\u0430\u0441?")
        self.assertIsNotNone(answer)
        self.assertRegex(answer, r"\d{2}:\d{2}")


if __name__ == "__main__":
    unittest.main()
