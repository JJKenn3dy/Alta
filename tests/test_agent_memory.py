import unittest

from agent import (
    _infer_dialogue_fact_from_pair,
    _is_meta_identity_fact_candidate,
    _recent_dialogue_fact_candidates,
    _select_user_facts,
    render_user_document_text,
    should_extract_user_facts,
)


class AgentMemoryTests(unittest.TestCase):
    def test_select_user_facts_prefers_specific_types_over_generic_fact(self):
        facts = [
            {
                "fact_type": "fact",
                "value": "lyubit kosmos",
                "normalized_value": "lyubit kosmos",
                "importance": 0.95,
                "confidence": 0.95,
            },
            {
                "fact_type": "work",
                "value": "backend razrabotchik",
                "normalized_value": "backend razrabotchik",
                "importance": 0.7,
                "confidence": 0.8,
            },
        ]

        selected = _select_user_facts(facts, max_items=1, min_importance=0.45)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["fact_type"], "work")

    def test_render_user_document_text_respects_line_limit(self):
        payload = {
            "identity": {
                "display_name": "Zahar",
                "nickname": "@zahar",
                "gender": "male",
            },
            "interests": "music, games",
            "preferences": {
                "communication_style": "short",
                "response_preference": "po delu",
                "relationship_notes": "normalno",
            },
            "facts": [{"label": "\u0424\u0430\u043a\u0442", "value": f"value {idx}"} for idx in range(30)],
        }

        text = render_user_document_text(payload, limit=5000)

        self.assertLessEqual(len(text.splitlines()), 23)

    def test_infer_dialogue_fact_from_pair_for_location(self):
        fact = _infer_dialogue_fact_from_pair("\u0422\u044b \u043e\u0442\u043a\u0443\u0434\u0430?", "\u0418\u0437 \u0421\u041f\u0431")

        self.assertIsNotNone(fact)
        self.assertEqual(fact["fact_type"], "location")
        self.assertEqual(fact["value"], "\u0418\u0437 \u0421\u041f\u0431")

    def test_recent_dialogue_fact_candidates_uses_last_question_answer_pair(self):
        history = [
            {
                "role": "assistant",
                "content": "\u0422\u044b \u043e\u0442\u043a\u0443\u0434\u0430?",
                "target_user_id": 42,
            },
            {
                "role": "user",
                "content": "[id=42 @alex] \u0418\u0437 \u0421\u041f\u0431",
                "user_id": 42,
            },
            {
                "role": "assistant",
                "content": "\u041e\u043a",
                "target_user_id": 42,
            },
        ]

        candidates = _recent_dialogue_fact_candidates(history, current_user_id=42)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["fact_type"], "location")
        self.assertEqual(candidates[0]["value"], "\u0418\u0437 \u0421\u041f\u0431")

    def test_should_extract_user_facts_skips_meta_identity_queries(self):
        self.assertFalse(should_extract_user_facts("\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442?"))
        self.assertFalse(should_extract_user_facts("\u043a\u0442\u043e \u044f \u0442\u0430\u043a\u043e\u0439?"))
        self.assertTrue(
            should_extract_user_facts(
                "\u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u0447\u0442\u043e \u044f \u043b\u044e\u0431\u043b\u044e \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438"
            )
        )

    def test_meta_identity_fact_candidate_filter(self):
        self.assertTrue(
            _is_meta_identity_fact_candidate(
                "fact",
                "\u043d\u0435 \u0437\u043d\u0430\u0435\u0442, \u043a\u0430\u043a \u0435\u0433\u043e \u0437\u043e\u0432\u0443\u0442",
                "\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442?",
            )
        )
        self.assertFalse(
            _is_meta_identity_fact_candidate(
                "preference",
                "\u043b\u044e\u0431\u0438\u0442 \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438",
                "\u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u0447\u0442\u043e \u044f \u043b\u044e\u0431\u043b\u044e \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438",
            )
        )

    def test_select_user_facts_drops_existing_meta_identity_noise(self):
        facts = [
            {
                "fact_type": "fact",
                "value": "\u043d\u0435 \u0437\u043d\u0430\u0435\u0442, \u043a\u0430\u043a \u0435\u0433\u043e \u0437\u043e\u0432\u0443\u0442",
                "normalized_value": "\u043d\u0435 \u0437\u043d\u0430\u0435\u0442, \u043a\u0430\u043a \u0435\u0433\u043e \u0437\u043e\u0432\u0443\u0442",
                "importance": 0.9,
                "confidence": 0.9,
                "evidence": "\u043a\u0430\u043a \u043c\u0435\u043d\u044f \u0437\u043e\u0432\u0443\u0442?",
            },
            {
                "fact_type": "preference",
                "value": "\u043b\u044e\u0431\u0438\u0442 \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438",
                "normalized_value": "\u043b\u044e\u0431\u0438\u0442 \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438",
                "importance": 0.7,
                "confidence": 0.8,
                "evidence": "\u0437\u0430\u043f\u043e\u043c\u043d\u0438 \u0447\u0442\u043e \u044f \u043b\u044e\u0431\u043b\u044e \u044d\u043d\u0435\u0440\u0433\u0435\u0442\u0438\u043a\u0438",
            },
        ]

        selected = _select_user_facts(facts, max_items=3, min_importance=0.45)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["fact_type"], "preference")


if __name__ == "__main__":
    unittest.main()
