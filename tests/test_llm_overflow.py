import unittest

from llm import (
    _looks_like_context_overflow,
    _looks_like_max_completion_tokens_not_supported,
    _looks_like_max_tokens_not_supported,
)


class _DummyError(Exception):
    pass


class LLMOverflowDetectionTests(unittest.TestCase):
    def test_detects_context_size_exceeded_server_message(self):
        err = _DummyError(
            "Error code: 500 - {'error': {'code': 500, 'message': 'Context size has been exceeded.', "
            "'type': 'server_error'}}"
        )
        self.assertTrue(_looks_like_context_overflow(err))

    def test_detects_generic_context_exceed_text(self):
        err = _DummyError("request rejected: context budget exceed limit")
        self.assertTrue(_looks_like_context_overflow(err))

    def test_ignores_unrelated_errors(self):
        err = _DummyError("Error code: 500 - backend temporarily unavailable")
        self.assertFalse(_looks_like_context_overflow(err))


class LLMParamCompatibilityTests(unittest.TestCase):
    def test_detects_max_tokens_not_supported_message(self):
        err = _DummyError(
            "Unsupported parameter: 'max_tokens' is not supported with this model. "
            "Use 'max_completion_tokens' instead."
        )
        self.assertTrue(_looks_like_max_tokens_not_supported(err))

    def test_detects_max_completion_tokens_not_supported_message(self):
        err = _DummyError(
            "Unsupported parameter: 'max_completion_tokens' is not supported with this model."
        )
        self.assertTrue(_looks_like_max_completion_tokens_not_supported(err))


if __name__ == "__main__":
    unittest.main()
