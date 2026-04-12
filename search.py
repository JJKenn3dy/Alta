import asyncio
import base64
import html as html_lib
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Callable, NamedTuple
from urllib.parse import quote_plus

log = logging.getLogger("alta.search")

WEB_RESULT_CHAR_LIMIT = 1_000
WEB_RESULT_MAX_SNIPPETS = 3
WEB_SEARCH_TIMEOUT = 8.0
WEB_SEARCH_MAX_RESULTS = 5
CACHE_TTL = 300
YANDEX_SEARCH_URL = "https://searchapi.api.cloud.yandex.net/v2/web/search"
YANDEX_SEARCH_TIMEOUT = 7

_YANDEX_API_KEY_ENV_NAMES = ("YANDEX_SEARCH_API_KEY", "YANDEX_API_KEY")
_YANDEX_FOLDER_ID_ENV_NAMES = ("YANDEX_FOLDER_ID",)

_LATIN_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_-]{2,}\b", re.IGNORECASE)
_LATIN_GENERIC_WORDS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "the",
    "this",
    "that",
    "about",
    "game",
    "games",
    "news",
    "price",
    "latest",
    "today",
    "now",
    "developer",
    "developers",
    "developed",
    "studio",
    "studios",
    "publisher",
    "publishers",
    "engine",
}

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_TITLE_RE = re.compile(
    r'<a[^>]*href="([^"]*)"[^>]*>\s*'
    r'<div[^>]*class="title search-snippet-title[^"]*"[^>]*>(.*?)</div>\s*</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<div[^>]*class="content[^"]*line-clamp-dynamic[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
_cache: dict[str, tuple[float, list[dict]]] = {}


class WebResult(NamedTuple):
    query: str
    text: str
    snippets_found: int


def _clean_html(s: str) -> str:
    cleaned = html_lib.unescape(_TAG_RE.sub("", s or ""))
    return _SPACE_RE.sub(" ", cleaned).strip()


def _format_web_results(query: str, results: list[dict]) -> str:
    lines = [f"Результаты поиска по запросу «{query}»:"]
    for i, result in enumerate(results[:WEB_RESULT_MAX_SNIPPETS], 1):
        title = (result.get("title") or "").strip()
        body = (result.get("body") or "").strip()
        url = (result.get("href") or "").strip()
        if not body and not title:
            continue
        snippet = (body or title)[:300].rstrip()
        lines.append(f"{i}. {title + ': ' if title and body else ''}{snippet}")
        if url:
            lines.append(f"   Источник: {url}")
    joined = "\n".join(lines)
    if len(joined) > WEB_RESULT_CHAR_LIMIT:
        joined = joined[: WEB_RESULT_CHAR_LIMIT - 1].rstrip() + "…"
    return joined


def _extract_latin_tokens(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _LATIN_TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        if token in _LATIN_GENERIC_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _filter_results(results: list[dict], required_tokens: list[str]) -> list[dict]:
    if not required_tokens:
        return results

    scored: list[tuple[int, dict]] = []
    for result in results:
        haystack = " ".join(
            (
                str(result.get("title") or ""),
                str(result.get("body") or ""),
                str(result.get("href") or ""),
            )
        ).lower()
        score = sum(token in haystack for token in required_tokens)
        if score > 0:
            scored.append((score, result))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [result for _, result in scored]


def _parse_brave_results(page_html: str) -> list[dict]:
    """Parse Brave Search HTML into list of {title, body, href}."""
    results: list[dict] = []
    titles = list(_TITLE_RE.finditer(page_html))
    snippets = list(_SNIPPET_RE.finditer(page_html))

    snippet_idx = 0
    for title_match in titles:
        href = title_match.group(1)
        title = _clean_html(title_match.group(2))
        title_end = title_match.end()

        body = ""
        while snippet_idx < len(snippets):
            if snippets[snippet_idx].start() > title_end:
                body = _clean_html(snippets[snippet_idx].group(1))
                snippet_idx += 1
                break
            snippet_idx += 1

        if not title and not body:
            continue
        results.append({"title": title, "body": body, "href": href})

    return results


def _decode_raw_data(raw: str) -> str:
    return base64.b64decode(raw).decode("utf-8", errors="ignore")


def _parse_yandex_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    results: list[dict] = []

    for doc in root.findall(".//doc"):
        title = (doc.findtext("title") or "").strip()
        url = (doc.findtext("url") or "").strip()

        passages = doc.findall(".//passage")
        snippet_parts = ["".join(passage.itertext()) for passage in passages]
        snippet = _clean_html(" ".join(part for part in snippet_parts if part))
        if not snippet:
            snippet = title

        if not title and not snippet:
            continue
        results.append({"title": title, "body": snippet, "href": url})

    return results


def _get_env_value(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _get_yandex_search_type(query: str) -> str:
    return "SEARCH_TYPE_RU" if _is_russian(query) else "SEARCH_TYPE_COM"


def _yandex_search_sync(query: str, max_results: int) -> list[dict]:
    import requests

    api_key = _get_env_value(_YANDEX_API_KEY_ENV_NAMES)
    folder_id = _get_env_value(_YANDEX_FOLDER_ID_ENV_NAMES)
    if not api_key or not folder_id:
        log.info("yandex search disabled: set %s and %s", _YANDEX_API_KEY_ENV_NAMES, _YANDEX_FOLDER_ID_ENV_NAMES)
        return []

    headers = {"Authorization": f"Api-Key {api_key}"}
    data = {
        "query": {"searchType": _get_yandex_search_type(query), "queryText": query},
        "groupSpec": {
            "groupMode": "GROUP_MODE_FLAT",
            "groupsOnPage": max_results,
            "docsInGroup": 1,
        },
        "folderId": folder_id,
        "responseFormat": "FORMAT_XML",
    }

    resp = requests.post(
        YANDEX_SEARCH_URL,
        json=data,
        headers=headers,
        timeout=YANDEX_SEARCH_TIMEOUT,
    )
    if resp.status_code != 200:
        body_preview = (resp.text or "").replace("\n", " ")[:200]
        log.warning("yandex error %s: %s", resp.status_code, body_preview)
    resp.raise_for_status()
    raw_data = (resp.json() or {}).get("rawData")
    if not raw_data:
        return []

    xml_text = _decode_raw_data(raw_data)
    return _parse_yandex_xml(xml_text)[:max_results]


def _brave_search_sync(query: str, max_results: int) -> list[dict]:
    """Search via Brave Search (scraping, no API key needed)."""
    import primp

    client = primp.Client(impersonate="random", timeout=7)
    url = f"https://search.brave.com/search?q={quote_plus(query)}&source=web"
    resp = client.get(url)
    resp.raise_for_status()
    return _parse_brave_results(resp.text)[:max_results]


def _is_russian(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


def _should_use_yandex(query: str) -> bool:
    query = (query or "").strip()
    if len(query) <= 3:
        return False
    return _is_russian(query)


def _cache_key(query: str) -> str:
    return _SPACE_RE.sub(" ", (query or "").strip().lower())


def _get_cached(query: str) -> list[dict] | None:
    key = _cache_key(query)
    item = _cache.get(key)
    if not item:
        return None
    ts, data = item
    if time.time() - ts > CACHE_TTL:
        _cache.pop(key, None)
        return None
    return data


def _set_cache(query: str, data: list[dict]) -> None:
    _cache[_cache_key(query)] = (time.time(), data)


async def _run_search_with_timeout(
    loop: asyncio.AbstractEventLoop,
    fn: Callable[[str, int], list[dict]],
    query: str,
) -> list[dict]:
    return await asyncio.wait_for(
        loop.run_in_executor(None, fn, query, WEB_SEARCH_MAX_RESULTS),
        timeout=WEB_SEARCH_TIMEOUT,
    )


async def web_search(query: str) -> WebResult | None:
    query = (query or "").strip()
    if not query:
        return None

    loop = asyncio.get_running_loop()

    use_yandex = _should_use_yandex(query)
    results = _get_cached(query)

    if results is None:
        results = []

        if use_yandex:
            try:
                results = await _run_search_with_timeout(loop, _yandex_search_sync, query)
            except asyncio.TimeoutError:
                log.warning("web_search: timeout from yandex for %r", query)
            except Exception:
                log.warning("web_search: yandex failed for %r", query, exc_info=True)

            if not results:
                try:
                    brave_results = await _run_search_with_timeout(loop, _brave_search_sync, query)
                    if brave_results:
                        results = brave_results
                except asyncio.TimeoutError:
                    log.warning("web_search: timeout from brave for %r", query)
                    if not results:
                        return None
                except Exception:
                    log.warning("web_search: brave failed for %r", query, exc_info=True)
                    if not results:
                        return None
        else:
            try:
                results = await _run_search_with_timeout(loop, _brave_search_sync, query)
            except asyncio.TimeoutError:
                log.warning("web_search: timeout from brave for %r", query)
                return None
            except Exception:
                log.warning("web_search: brave failed for %r", query, exc_info=True)
                return None

        if results:
            _set_cache(query, results)

    required_tokens = _extract_latin_tokens(query)
    filtered = _filter_results(results, required_tokens)
    if filtered:
        results = filtered

    if not results:
        log.info("web_search: no results for %r", query)
        return WebResult(query=query, text="", snippets_found=0)

    return WebResult(
        query=query,
        text=_format_web_results(query, results),
        snippets_found=len(results),
    )
