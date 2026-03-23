"""Tests for the bug fixes applied to Alta."""
import unittest

from agent import (
    _looks_like_context_dependent_reply,
    _remove_forbidden_self_reference,
    build_implicit_reply_context_block,
    should_extract_user_facts,
    REPEAT_REWRITE_PROMPT,
    MAX_EDITOR_REPLY_CHARS,
    MAX_EDITOR_SENTENCES,
)
from llm import CHARS_PER_TOKEN_ESTIMATE, _estimate_text_tokens
from telemetry import CHARS_PER_TOKEN_HEURISTIC


class TokenEstimateTests(unittest.TestCase):
    def test_chars_per_token_is_consistent(self):
        """Fix #1: llm.py and telemetry.py must use the same constant."""
        self.assertEqual(CHARS_PER_TOKEN_ESTIMATE, CHARS_PER_TOKEN_HEURISTIC)

    def test_chars_per_token_is_reasonable_for_russian(self):
        """Fix #1: Value should be ~3.0-3.5 for Russian text."""
        self.assertGreaterEqual(CHARS_PER_TOKEN_ESTIMATE, 2.5)
        self.assertLessEqual(CHARS_PER_TOKEN_ESTIMATE, 4.0)

    def test_russian_text_estimate_not_too_aggressive(self):
        """Fix #1: 1000 chars of Russian should not estimate as 500 tokens."""
        text = "Привет, это тестовый текст для проверки оценки токенов. " * 20
        tokens = _estimate_text_tokens(text)
        self.assertLess(tokens, len(text) / 2.0)


class ContextDependentReplyTests(unittest.TestCase):
    def test_short_question_is_context_dependent(self):
        """Fix #4: Short questions like 'Почему?' should be context-dependent."""
        self.assertTrue(_looks_like_context_dependent_reply("Почему?"))
        self.assertTrue(_looks_like_context_dependent_reply("А зачем?"))
        self.assertTrue(_looks_like_context_dependent_reply("Что?"))

    def test_long_text_is_not_context_dependent(self):
        self.assertFalse(
            _looks_like_context_dependent_reply(
                "Это длинное сообщение которое точно не зависит от контекста предыдущего"
            )
        )

    def test_empty_is_not_context_dependent(self):
        self.assertFalse(_looks_like_context_dependent_reply(""))


class ImplicitReplyContextTests(unittest.TestCase):
    def test_implicit_block_includes_prev_user_message(self):
        """The implicit context should include the user's PREVIOUS message."""
        history = [
            {"role": "user", "content": "Кто такой шеф"},
            {"role": "assistant", "content": "Может, ты про кого-то конкретного?"},
            {"role": "user", "content": "Ну да"},
        ]
        block = build_implicit_reply_context_block(history, "Ну да")
        self.assertIn("Кто такой шеф", block)
        self.assertIn("Может, ты про кого-то", block)

    def test_implicit_block_short_reply_includes_prev_user(self):
        """Short context-dependent replies should also show prev user message."""
        history = [
            {"role": "user", "content": "Где ты живёшь?"},
            {"role": "assistant", "content": "А ты откуда?"},
            {"role": "user", "content": "Из Москвы"},
        ]
        block = build_implicit_reply_context_block(history, "Из Москвы")
        self.assertIn("Где ты живёшь", block)
        self.assertIn("А ты откуда", block)


class FactExtractionTests(unittest.TestCase):
    def test_first_person_only_triggers_extraction(self):
        """Fix #9: First person markers alone should trigger extraction."""
        self.assertTrue(should_extract_user_facts("Я программист из Москвы"))
        self.assertTrue(should_extract_user_facts("Мне 25 лет сейчас"))

    def test_fact_marker_only_triggers_extraction(self):
        """Fix #9: Fact markers alone should trigger extraction."""
        self.assertTrue(should_extract_user_facts("Живу в Москве уже пять лет"))
        self.assertTrue(should_extract_user_facts("Работаю в IT компании"))

    def test_short_text_skipped(self):
        self.assertFalse(should_extract_user_facts("ок"))
        self.assertFalse(should_extract_user_facts(""))


class SelfReferenceTests(unittest.TestCase):
    def test_removes_words_not_sentences(self):
        """Fix #6: Should replace forbidden words, not delete entire sentences."""
        text = "Этот бот работает хорошо. Мне нравится."
        result = _remove_forbidden_self_reference(text)
        self.assertIn("нравится", result)
        self.assertIn("работает", result)

    def test_removes_ai_references(self):
        text = "Как ИИ, я могу помочь."
        result = _remove_forbidden_self_reference(text)
        self.assertNotIn("ИИ", result)


class PostprocessLimitsTests(unittest.TestCase):
    def test_reply_char_limit_increased(self):
        """Fix #6: MAX_EDITOR_REPLY_CHARS should be >= 600."""
        self.assertGreaterEqual(MAX_EDITOR_REPLY_CHARS, 600)

    def test_sentence_limit_increased(self):
        """Fix #6: MAX_EDITOR_SENTENCES should be >= 7."""
        self.assertGreaterEqual(MAX_EDITOR_SENTENCES, 7)


class RepeatRewritePromptTests(unittest.TestCase):
    def test_prompt_is_in_russian(self):
        """Fix #7: REPEAT_REWRITE_PROMPT should be in Russian."""
        self.assertIn("переписываешь", REPEAT_REWRITE_PROMPT.lower())
        self.assertNotIn("rewrite", REPEAT_REWRITE_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
