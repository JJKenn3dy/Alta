import unittest
from datetime import datetime, timezone

from agent import (
    ReplyContext,
    _effective_reply_max_tokens,
    _authoritative_time_answer,
    _inject_reply_context_history,
    _is_repeated_recent_reply,
    _prompt_history_content,
    _reply_repeat_key,
    _should_repair_unsuitable_reply,
    _summary_history_content,
    _take_history_with_budget,
)


class AgentHistoryTests(unittest.TestCase):
    def test_history_budget_keeps_assistant_target_metadata(self):
        history = [
            {
                "id": 1,
                "role": "user",
                "content": "[id=42 @alex] privet",
                "user_id": 42,
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "otvet",
                "user_id": 8152077529,
                "target_user_id": 42,
                "target_username": "alex",
            },
        ]

        selected = _take_history_with_budget(history, max_messages=10, max_chars=1_000)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[1]["target_user_id"], 42)
        self.assertEqual(selected[1]["target_username"], "alex")

    def test_prompt_and_summary_keep_addressee_markers_after_shrink(self):
        history = [
            {
                "id": 1,
                "role": "assistant",
                "content": "u menya net",
                "user_id": 8152077529,
                "target_user_id": 42,
                "target_username": "alex",
            }
        ]

        selected = _take_history_with_budget(history, max_messages=10, max_chars=1_000)
        assistant_message = selected[0]

        self.assertEqual(
            _prompt_history_content(assistant_message, 42),
            "[to current speaker id=42 @alex] u menya net",
        )
        self.assertEqual(
            _summary_history_content(assistant_message),
            "[to id=42 @alex] u menya net",
        )

    def test_reply_context_assistant_message_is_injected_before_current_user_message(self):
        history = [
            {
                "id": 10,
                "role": "user",
                "content": "[id=42 @alex] Eto vopros",
                "user_id": 42,
            }
        ]
        reply_context = ReplyContext(
            user_id=8152077529,
            username="alta_bot",
            first_name="Alta",
            text="U tebya net sis'?",
            role="assistant",
        )

        injected = _inject_reply_context_history(history, reply_context, current_user_id=42)

        self.assertEqual(len(injected), 2)
        self.assertEqual(injected[0]["role"], "assistant")
        self.assertEqual(injected[0]["content"], "U tebya net sis'?")
        self.assertEqual(injected[1]["content"], "[id=42 @alex] Eto vopros")

    def test_reply_context_user_message_is_injected_with_author_marker(self):
        history = [
            {
                "id": 10,
                "role": "user",
                "content": "[id=42 @alex] Eto vopros",
                "user_id": 42,
            }
        ]
        reply_context = ReplyContext(
            user_id=7,
            username="zahar",
            first_name="Zahar",
            text="Ty seryozno?",
            role="user",
        )

        injected = _inject_reply_context_history(history, reply_context, current_user_id=42)

        self.assertEqual(len(injected), 2)
        self.assertEqual(injected[0]["role"], "user")
        self.assertEqual(injected[0]["content"], "[id=7 @zahar] Ty seryozno?")
        self.assertEqual(injected[1]["content"], "[id=42 @alex] Eto vopros")

    def test_reply_context_is_not_duplicated_if_message_is_already_in_history(self):
        history = [
            {
                "id": 9,
                "role": "assistant",
                "content": "U tebya net sis'?",
            },
            {
                "id": 10,
                "role": "user",
                "content": "[id=42 @alex] Eto vopros",
                "user_id": 42,
            },
        ]
        reply_context = ReplyContext(
            user_id=8152077529,
            username="alta_bot",
            first_name="Alta",
            text="U tebya net sis'?",
            role="assistant",
        )

        injected = _inject_reply_context_history(history, reply_context, current_user_id=42)

        self.assertEqual(injected, history)

    def test_authoritative_time_answer_defaults_to_msk(self):
        now = datetime(2026, 3, 4, 10, 23, tzinfo=timezone.utc)

        self.assertEqual(
            _authoritative_time_answer("сколько времени", now=now),
            "Сейчас 13:23 по МСК",
        )
        self.assertEqual(
            _authoritative_time_answer("ну так сколько время по мск", now=now),
            "Сейчас 13:23 по МСК",
        )

    def test_authoritative_time_answer_does_not_trigger_for_rhetorical_phrase(self):
        now = datetime(2026, 3, 4, 10, 23, tzinfo=timezone.utc)

        self.assertIsNone(
            _authoritative_time_answer("а ты видишь в сообщении сколько щас времени?", now=now)
        )

    def test_repair_guard_flags_rude_or_emptyish_replies_for_neutral_input(self):
        self.assertTrue(_should_repair_unsuitable_reply("Да", "Ну и?"))
        self.assertTrue(_should_repair_unsuitable_reply("тест", "Иди нахуй"))
        self.assertTrue(_should_repair_unsuitable_reply("ладно", "Ну и ладно"))
        self.assertFalse(_should_repair_unsuitable_reply("как дела", "Нормально"))
        self.assertFalse(_should_repair_unsuitable_reply("ты тупая", "Иди нахуй"))

    def test_repair_guard_flags_low_signal_clarifying_questions(self):
        self.assertTrue(
            _should_repair_unsuitable_reply(
                "\u0445\u043e\u0440\u043e\u0448\u0438\u0439 \u0434\u0435\u043d\u044c",
                "\u0427\u0435\u0433\u043e \u0440\u0430\u0434\u0443\u0435\u043c\u0441\u044f-\u0442\u043e?",
            )
        )
        self.assertTrue(
            _should_repair_unsuitable_reply(
                "\u043b\u0430\u0434\u043d\u043e, \u044f \u0432\u0441\u0435 \u043f\u043e\u043d\u044f\u043b",
                "\u041d\u0443 \u0438 \u0447\u0442\u043e \u0442\u044b \u0445\u043e\u0442\u0435\u043b \u0443\u0441\u043b\u044b\u0448\u0430\u0442\u044c?",
            )
        )

    def test_effective_reply_max_tokens_expands_for_complex_or_detailed_input(self):
        self.assertEqual(_effective_reply_max_tokens("ok", 90), 90)
        self.assertGreaterEqual(
            _effective_reply_max_tokens(
                "\u041e\u0431\u044a\u044f\u0441\u043d\u0438 \u043f\u043e\u0434\u0440\u043e\u0431\u043d\u043e, \u043a\u0430\u043a \u0438\u043c\u0435\u043d\u043d\u043e "
                "\u044d\u0442\u043e \u0440\u0430\u0431\u043e\u0442\u0430\u0435\u0442 \u0438 \u043f\u043e\u0447\u0435\u043c\u0443.",
                90,
            ),
            170,
        )

    def test_repeat_key_ignores_target_metadata_prefix(self):
        self.assertEqual(
            _reply_repeat_key("[to current speaker id=42 @alex] privet"),
            _reply_repeat_key("privet"),
        )

    def test_repeat_guard_detects_repeated_reply_with_target_marker_in_history(self):
        messages = [
            {
                "role": "assistant",
                "content": "[to current speaker id=42 @alex] thought you were on a diet",
            }
        ]
        self.assertTrue(_is_repeated_recent_reply("thought you were on a diet", messages))



if __name__ == "__main__":
    unittest.main()
