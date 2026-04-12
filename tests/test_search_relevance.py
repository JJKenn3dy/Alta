import unittest

from search import _extract_latin_tokens, _filter_results


class SearchRelevanceTests(unittest.TestCase):
    def test_filter_results_returns_empty_when_no_required_token_matches(self):
        required_tokens = _extract_latin_tokens("who is aihasto miside developer")
        results = [
            {
                "title": "Air fryers - Cheap Air fryer Deals",
                "body": "Best air fryers in UK",
                "href": "https://example.com/air-fryers",
            },
            {
                "title": "Kitchen appliances",
                "body": "Compare modern air fryers",
                "href": "https://example.com/kitchen",
            },
        ]
        filtered = _filter_results(results, required_tokens)
        self.assertEqual(filtered, [])

    def test_filter_results_keeps_matching_entity_result(self):
        required_tokens = _extract_latin_tokens("who is aihasto miside developer")
        results = [
            {
                "title": "MiSide by AIHASTO",
                "body": "Developer AIHASTO released MiSide",
                "href": "https://example.com/miside",
            },
            {
                "title": "Air fryers",
                "body": "Best appliances",
                "href": "https://example.com/air",
            },
        ]
        filtered = _filter_results(results, required_tokens)
        self.assertEqual(len(filtered), 1)
        self.assertIn("MiSide", filtered[0]["title"])


if __name__ == "__main__":
    unittest.main()
