import unittest

try:
    from sanitization import sanitize_outgoing_text
except Exception as exc:  # pragma: no cover - import guard for optional deps
    sanitize_outgoing_text = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(sanitize_outgoing_text is None, f"failed to import sanitization: {IMPORT_ERROR}")
class MainSanitizeTests(unittest.TestCase):
    def test_strips_leading_target_marker(self):
        self.assertEqual(
            sanitize_outgoing_text("[to current speaker id=912276883 @Alex_Author] Тест пройден"),
            "Тест пройден",
        )

    def test_strips_leading_target_marker_after_user_marker(self):
        self.assertEqual(
            sanitize_outgoing_text("[id=912276883 @Alex_Author] [to id=912276883 @Alex_Author] Принято"),
            "Принято",
        )


if __name__ == "__main__":
    unittest.main()
