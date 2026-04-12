import unittest

from agent import (
    _answer_defers_web_lookup,
    _extract_search_query,
    _pick_web_search_query,
    _should_force_web_fallback,
)


class AgentWebSearchSafetyTests(unittest.TestCase):
    def test_extract_search_query_from_any_line(self):
        answer = "\u041e\u043a, \u0441\u0435\u0439\u0447\u0430\u0441.\n[SEARCH: miside developer]"
        self.assertEqual(_extract_search_query(answer), "miside developer")

    def test_safety_net_triggers_on_new_latin_proper_noun(self):
        history = [
            {"role": "user", "content": "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043f\u0440\u043e MiSide"},
            {"role": "assistant", "content": "\u042d\u0442\u043e \u0438\u0433\u0440\u0430."},
        ]
        query = _pick_web_search_query(
            "\u043a\u0430\u043a\u0438\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438 \u043f\u043e \u044d\u0442\u043e\u0439 \u0438\u0433\u0440\u0435?",
            "\u0415\u0451 \u0441\u0434\u0435\u043b\u0430\u043b Oleg Tishchenko.",
            context_history=history,
        )
        self.assertIsNotNone(query)
        self.assertIn("miside", query.lower())
        self.assertIn("oleg", query.lower())

    def test_safety_net_triggers_on_new_cyrillic_proper_noun(self):
        history = [
            {"role": "user", "content": "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043f\u0440\u043e MiSide"},
            {"role": "assistant", "content": "\u042d\u0442\u043e \u0438\u0433\u0440\u0430."},
        ]
        query = _pick_web_search_query(
            "\u043a\u0430\u043a\u0438\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438 \u043f\u043e \u044d\u0442\u043e\u0439 \u0438\u0433\u0440\u0435?",
            "\u0415\u0451 \u0441\u0434\u0435\u043b\u0430\u043b \u041e\u043b\u0435\u0433 \u0422\u0438\u0449\u0435\u043d\u043a\u043e.",
            context_history=history,
        )
        self.assertIsNotNone(query)
        self.assertIn("miside", query.lower())
        self.assertIn("\u043e\u043b\u0435\u0433", query.lower())

    def test_safety_net_ignores_known_proper_noun(self):
        history = [
            {"role": "user", "content": "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043f\u0440\u043e MiSide \u0438 Oleg Tishchenko"},
            {"role": "assistant", "content": "\u041e\u043a."},
        ]
        query = _pick_web_search_query(
            "\u043a\u0430\u043a\u0438\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438 \u043f\u043e \u044d\u0442\u043e\u0439 \u0438\u0433\u0440\u0435?",
            "\u0415\u0451 \u0441\u0434\u0435\u043b\u0430\u043b Oleg Tishchenko.",
            context_history=history,
        )
        self.assertIsNone(query)

    def test_force_web_fallback_when_answer_defers_search(self):
        self.assertTrue(_answer_defers_web_lookup("\u0425\u043e\u0440\u043e\u0448\u043e, \u0438\u0434\u0443 \u0433\u0443\u0433\u043b\u0438\u0442\u044c. \u0421\u0435\u043a\u0443\u043d\u0434\u0443."))
        self.assertTrue(_should_force_web_fallback("\u0425\u043e\u0440\u043e\u0448\u043e, \u0438\u0434\u0443 \u0433\u0443\u0433\u043b\u0438\u0442\u044c. \u0421\u0435\u043a\u0443\u043d\u0434\u0443."))
        self.assertFalse(_should_force_web_fallback("\u041f\u043e \u0434\u0430\u043d\u043d\u044b\u043c \u043f\u043e\u0438\u0441\u043a\u0430 \u044d\u0442\u043e \u0440\u0430\u0437\u0440\u0430\u0431\u043e\u0442\u0447\u0438\u043a AIHASTO."))

    def test_explicit_latin_search_query_is_ignored_for_cyrillic_input(self):
        query = _pick_web_search_query(
            "\u043a\u0442\u043e \u0441\u0434\u0435\u043b\u0430\u043b miside?",
            "[SEARCH: aihasto list]",
            context_history=[],
        )
        self.assertEqual(query, "\u043a\u0442\u043e \u0441\u0434\u0435\u043b\u0430\u043b miside?")

    def test_username_like_entity_is_not_added_to_claim_query(self):
        history = [
            {"role": "user", "content": "[id=5460037622 @JJKenn9dy] \u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438 \u043f\u0440\u043e MiSide"},
            {"role": "assistant", "content": "\u041e\u043a."},
        ]
        query = _pick_web_search_query(
            "\u043a\u0430\u043a\u0438\u0435 \u043d\u043e\u0432\u043e\u0441\u0442\u0438 \u043f\u043e \u044d\u0442\u043e\u0439 \u0438\u0433\u0440\u0435?",
            "\u0415\u0451 \u0441\u0434\u0435\u043b\u0430\u043b \u041e\u043b\u0435\u0433 \u0422\u0438\u0449\u0435\u043d\u043a\u043e.",
            context_history=history,
        )
        self.assertIsNotNone(query)
        self.assertNotIn("jjkenn9dy", query.lower())


if __name__ == "__main__":
    unittest.main()
