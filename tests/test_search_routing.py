import time
import unittest
from unittest.mock import AsyncMock, patch

from search import (
    CACHE_TTL,
    _cache,
    _cache_key,
    _filter_results,
    _get_cached,
    _get_yandex_search_type,
    _set_cache,
    _should_use_yandex,
    web_search,
)


class SearchRoutingTests(unittest.TestCase):
    def tearDown(self):
        _cache.clear()

    def test_should_use_yandex_for_russian_query(self):
        self.assertTrue(_should_use_yandex("кто сделал MiSide"))

    def test_should_not_use_yandex_for_short_query(self):
        self.assertFalse(_should_use_yandex("ок"))

    def test_should_use_yandex_for_mixed_query_with_cyrillic(self):
        self.assertTrue(_should_use_yandex("кто developed miside aihasto"))

    def test_get_yandex_search_type(self):
        self.assertEqual(_get_yandex_search_type("кто сделал MiSide"), "SEARCH_TYPE_RU")
        self.assertEqual(_get_yandex_search_type("who developed miside"), "SEARCH_TYPE_COM")

    def test_filter_results_scores_and_sorts(self):
        required = ["miside", "aihasto"]
        results = [
            {"title": "MiSide by AIHASTO", "body": "horror game", "href": "https://example.com/1"},
            {"title": "MiSide game", "body": "overview", "href": "https://example.com/2"},
            {"title": "Another title", "body": "none", "href": "https://example.com/3"},
        ]
        filtered = _filter_results(results, required)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["href"], "https://example.com/1")
        self.assertEqual(filtered[1]["href"], "https://example.com/2")

    def test_cache_hit_and_expire(self):
        query = "кто сделал miside"
        data = [{"title": "x", "body": "y", "href": "z"}]

        _set_cache(query, data)
        self.assertEqual(_get_cached(query), data)

        key = _cache_key(query)
        _cache[key] = (time.time() - CACHE_TTL - 1, data)
        self.assertIsNone(_get_cached(query))


class SearchCachingBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        _cache.clear()

    @patch("search._run_search_with_timeout", new_callable=AsyncMock)
    async def test_web_search_does_not_cache_empty_results(self, mock_run_search):
        mock_run_search.return_value = []

        result = await web_search("who developed miside")

        self.assertIsNotNone(result)
        self.assertEqual(result.snippets_found, 0)
        self.assertEqual(_cache, {})


if __name__ == "__main__":
    unittest.main()
