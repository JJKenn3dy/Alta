import asyncio
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("alta.file_context")

SYNC_INTERVAL_SEC = 15.0
MAX_TEXT_BYTES_DEFAULT = 262_144
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_QUERY_CHUNKS = 3
DEFAULT_QUERY_CHARS = 1_800
MAX_CHUNKS_TO_SCORE = 4_000

TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".log",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".sql",
    ".html",
    ".css",
    ".xml",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "for",
    "from",
    "how",
    "that",
    "the",
    "this",
    "with",
    "you",
    "это",
    "как",
    "что",
    "чтобы",
    "если",
    "или",
    "для",
    "про",
    "она",
    "они",
    "оно",
    "мне",
    "меня",
    "тебя",
    "тебе",
    "его",
    "еще",
    "надо",
    "есть",
    "где",
    "когда",
    "почему",
    "зачем",
    "просто",
    "бот",
    "агент",
    "файл",
    "файлы",
}

_LAST_SYNC_TS: dict[str, float] = {}
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё_]{2,}")


def _normalize_text(text: str) -> str:
    tokens = _WORD_RE.findall((text or "").lower().replace("ё", "е"))
    return " " + " ".join(tokens) + " " if tokens else ""


def _query_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _WORD_RE.findall((text or "").lower().replace("ё", "е")):
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _read_text(path: Path, max_bytes: int) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    if len(raw) > max_bytes:
        return ""
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _split_large_piece(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    data = text.strip()
    if not data:
        return pieces

    start = 0
    length = len(data)
    while start < length:
        end = min(length, start + chunk_size)
        piece = data[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    parts = re.split(r"\n\s*\n", (text or "").strip())
    chunks: list[str] = []
    current = ""

    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        candidate = cleaned if not current else f"{current}\n\n{cleaned}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            current = ""

        if len(cleaned) <= chunk_size:
            current = cleaned
            continue

        chunks.extend(_split_large_piece(cleaned, chunk_size, overlap))

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _scan_and_read(
    root: Path,
    max_files: int,
    max_file_bytes: int,
) -> list[tuple[str, float, int, str]]:
    """Blocking I/O: scan directory & read files. Returns list of (rel_path, mtime, size, text)."""
    root.mkdir(parents=True, exist_ok=True)
    candidates = []
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
                continue
            candidates.append(path)
    except OSError:
        log.warning("failed to scan knowledge dir %s", root, exc_info=True)
        return []

    candidates.sort()
    results: list[tuple[str, float, int, str]] = []
    for path in candidates[:max_files]:
        rel_path = path.relative_to(root).as_posix()
        try:
            stat = path.stat()
        except OSError:
            continue
        text = _read_text(path, max_file_bytes)
        results.append((rel_path, float(stat.st_mtime), int(stat.st_size), text))
    return results


async def sync_context_dir(
    db,
    knowledge_dir: str,
    *,
    max_files: int,
    max_file_bytes: int,
):
    root = Path(knowledge_dir)
    root_key = str(root.resolve()).lower()
    now = time.monotonic()
    last_sync = _LAST_SYNC_TS.get(root_key, 0.0)
    if now - last_sync < SYNC_INTERVAL_SEC:
        return
    _LAST_SYNC_TS[root_key] = now

    existing = await db.list_context_files()
    seen_paths: set[str] = set()

    file_entries = await asyncio.to_thread(_scan_and_read, root, max_files, max_file_bytes)

    for rel_path, mtime, size, text in file_entries:
        seen_paths.add(rel_path)

        current = existing.get(rel_path)
        if current:
            same_mtime = abs(float(current["mtime"]) - mtime) < 0.001
            if same_mtime and int(current["size"]) == size:
                continue

        if not text.strip():
            await db.delete_context_file(rel_path)
            continue

        chunks = chunk_text(text)
        if not chunks:
            await db.delete_context_file(rel_path)
            continue

        await db.replace_context_file(
            rel_path,
            mtime,
            size,
            [
                {
                    "chunk_index": index,
                    "content": chunk,
                    "normalized_text": _normalize_text(chunk),
                }
                for index, chunk in enumerate(chunks)
            ],
        )

    for rel_path in set(existing) - seen_paths:
        await db.delete_context_file(rel_path)


def _score_chunk(path: str, normalized_text: str, query: str, tokens: list[str]) -> float:
    score = 0.0
    normalized_path = _normalize_text(path)

    if query and query in normalized_text:
        score += 4.0
    if query and query in normalized_path:
        score += 5.0

    for token in tokens:
        marker = f" {token} "
        if marker in normalized_path:
            score += 3.0
        if marker in normalized_text:
            score += 1.25

    return score


async def retrieve_context_snippets(
    db,
    query_text: str,
    *,
    max_chunks: int = DEFAULT_QUERY_CHUNKS,
    max_chars: int = DEFAULT_QUERY_CHARS,
) -> list[dict]:
    tokens = _query_tokens(query_text)
    if not tokens:
        return []

    normalized_query = _normalize_text(" ".join(tokens))
    chunks = await db.get_context_chunks(limit=MAX_CHUNKS_TO_SCORE)
    scored: list[tuple[float, dict]] = []

    for chunk in chunks:
        normalized_text = str(chunk.get("normalized_text") or "")
        score = _score_chunk(str(chunk.get("path") or ""), normalized_text, normalized_query, tokens)
        if score <= 0:
            continue
        scored.append((score, chunk))

    scored.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("path") or ""),
            int(item[1].get("chunk_index") or 0),
        )
    )

    selected: list[dict] = []
    chars_used = 0
    file_hits: dict[str, int] = {}
    seen_chunks: set[tuple[str, int]] = set()

    for score, chunk in scored:
        path = str(chunk.get("path") or "")
        chunk_index = int(chunk.get("chunk_index") or 0)
        key = (path, chunk_index)
        if key in seen_chunks:
            continue
        if file_hits.get(path, 0) >= 2:
            continue

        content = str(chunk.get("content") or "").strip()
        if not content:
            continue

        payload_size = len(path) + len(content)
        if selected and chars_used + payload_size > max_chars:
            continue

        selected.append({"path": path, "content": content, "score": score})
        seen_chunks.add(key)
        file_hits[path] = file_hits.get(path, 0) + 1
        chars_used += payload_size
        if len(selected) >= max_chunks:
            break

    return selected


def build_file_context_block(snippets: list[dict]) -> str:
    if not snippets:
        return ""

    parts = [
        "Local file context. Use it only when it is relevant to the user's request.",
        "Do not pretend to know anything outside these snippets.",
    ]
    for snippet in snippets:
        path = str(snippet.get("path") or "").strip()
        content = str(snippet.get("content") or "").strip()
        if not path or not content:
            continue
        parts.append(f"[{path}]\n{content}")

    return "\n\n".join(parts).strip()
