import logging
from pathlib import Path

log = logging.getLogger("alta.lore")

_cached_lore: str | None = None
_cached_mtime: float = 0.0


def load_lore(path: str, max_chars: int = 800) -> str:
    """Load lore text from file. Caches by mtime to avoid re-reading on every turn."""
    global _cached_lore, _cached_mtime
    p = Path(path)
    if not p.exists():
        log.debug("lore file not found: %s", path)
        return ""
    try:
        mtime = p.stat().st_mtime
        if _cached_lore is not None and abs(mtime - _cached_mtime) < 0.001:
            return _cached_lore
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            _cached_lore = ""
            _cached_mtime = mtime
            return ""
        text = raw[:max_chars]
        _cached_lore = text
        _cached_mtime = mtime
        log.info("lore loaded from %s (%d chars)", path, len(text))
        return text
    except Exception:
        log.warning("failed to load lore from %s", path, exc_info=True)
        return ""


def lore_to_system_block(lore_text: str) -> str:
    """Convert lore text to a system block string."""
    if not lore_text:
        return ""
    return (
        "Твой внутренний бэкграунд (не цитируй буквально, используй как фон):\n"
        + lore_text
    )
