import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

from aiogram.enums import ParseMode

log = logging.getLogger("alta.media")

MEDIA_PACK_SET_NAMES = [
    "kockrlandemojipack",
    "kockrlandemojipack2",
    "kockrlandemojipack3",
    "kockrlandemojipack4",
    "kockrlandemojipack5",
    "kockrlandemojipack6",
    "kockrlandemojipack7",
    "kockrlandemojipack8",
    "useless_emotes_by_fStikBot",
    "roflmoji",
    "Roflanlyk",
    "JJKennedyMini",
    "Alpha1918",
]

MEDIA_SELECTION_PROMPT = """
Ты решаешь, уместно ли после ответа бота отправить стикер или кастомный эмодзи.
Верни строго JSON-объект:
{"send": true, "mode": "sticker|custom_emoji|any", "emoji": "😂", "intensity": 0.0}

Правила:
- send=true только если это реально усилит реплику.
- Не отправляй медиа на каждый ответ.
- emoji должен быть из списка доступных опорных эмодзи.
- intensity от 0.0 до 1.0 показывает, насколько уместно медиа.
- Если медиа не нужно, верни {"send": false, "mode": "any", "emoji": "", "intensity": 0.0}
""".strip()

EMOJI_REPLY_PROMPT = """
Ты отвечаешь на сообщение, состоящее в основном из эмодзи.
Верни строго JSON-объект:
{"send": true, "mode": "custom_emoji|sticker|any", "emoji": "🙂", "intensity": 0.0}

Правила:
- Почти всегда отвечай тоже медиа, а не текстом.
- Подбирай эмодзи по вайбу входного сообщения.
- Если сообщение странное, можно вернуть нейтральный или ироничный вариант.
- intensity от 0.0 до 1.0.
""".strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def load_media_catalog(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        log.warning("failed to load media catalog from %s", path, exc_info=True)
        return {}


def save_media_catalog(path: str, catalog: dict[str, Any]):
    Path(path).write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


async def build_media_catalog(bot, set_names: list[str] | None = None) -> dict[str, Any]:
    names = set_names or MEDIA_PACK_SET_NAMES
    items: list[dict[str, Any]] = []
    failed_sets: list[str] = []

    for name in names:
        try:
            sticker_set = await bot.get_sticker_set(name)
        except Exception:
            log.warning("failed to fetch sticker set %r", name, exc_info=True)
            failed_sets.append(name)
            continue

        set_type = getattr(sticker_set, "sticker_type", "") or ""
        for sticker in sticker_set.stickers:
            emoji = getattr(sticker, "emoji", "") or ""
            kind = "custom_emoji" if set_type == "custom_emoji" else "sticker"
            items.append(
                {
                    "set_name": name,
                    "kind": kind,
                    "emoji": emoji,
                    "file_id": getattr(sticker, "file_id", None),
                    "custom_emoji_id": getattr(sticker, "custom_emoji_id", None),
                }
            )

    counts = Counter(i["emoji"] for i in items if i.get("emoji"))
    anchors = [emoji for emoji, _ in counts.most_common(48)]
    return {"items": items, "anchors": anchors, "failed_sets": failed_sets}


def ensure_media_catalog_shape(catalog: dict[str, Any]) -> dict[str, Any]:
    items = catalog.get("items", [])
    anchors = catalog.get("anchors", [])
    failed_sets = catalog.get("failed_sets", [])
    if not isinstance(items, list):
        items = []
    if not isinstance(anchors, list):
        anchors = []
    if not isinstance(failed_sets, list):
        failed_sets = []
    return {"items": items, "anchors": anchors, "failed_sets": failed_sets}


async def maybe_refresh_media_catalog(bot, path: str) -> dict[str, Any]:
    catalog = load_media_catalog(path)
    catalog = ensure_media_catalog_shape(catalog)
    if catalog["items"]:
        return catalog

    catalog = await build_media_catalog(bot)
    save_media_catalog(path, catalog)
    return catalog


async def choose_media_intent(
    llm,
    input_text: str,
    answer: str,
    catalog: dict[str, Any],
    turn_id: str | None = None,
    turn_metrics=None,
) -> dict[str, Any]:
    anchors = catalog.get("anchors", [])[:32]
    if not anchors:
        return {"send": False, "mode": "any", "emoji": "", "intensity": 0.0}

    payload = (
        f"Сообщение пользователя: {input_text}\n"
        f"Ответ бота: {answer}\n"
        f"Доступные опорные эмодзи: {' '.join(anchors)}"
    )
    try:
        raw = await llm.chat(
            MEDIA_SELECTION_PROMPT,
            [{"role": "user", "content": payload}],
            temperature=0.3,
            top_p=1.0,
            max_tokens=90,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("media intent LLM call failed", exc_info=True)
        return {"send": False, "mode": "any", "emoji": "", "intensity": 0.0}

    data = _extract_json_object(raw)
    if not data:
        return {"send": False, "mode": "any", "emoji": "", "intensity": 0.0}
    return data


async def choose_emoji_reply_intent(
    llm,
    input_text: str,
    catalog: dict[str, Any],
    turn_id: str | None = None,
    turn_metrics=None,
) -> dict[str, Any]:
    anchors = catalog.get("anchors", [])[:32]
    if not anchors:
        return {"send": False, "mode": "any", "emoji": "", "intensity": 0.0}

    payload = (
        f"Входное сообщение: {input_text}\n"
        f"Доступные опорные эмодзи: {' '.join(anchors)}"
    )
    try:
        raw = await llm.chat(
            EMOJI_REPLY_PROMPT,
            [{"role": "user", "content": payload}],
            temperature=0.45,
            top_p=1.0,
            max_tokens=70,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("emoji reply intent LLM call failed", exc_info=True)
        return {"send": False, "mode": "any", "emoji": "", "intensity": 0.0}

    data = _extract_json_object(raw)
    if not data:
        return {"send": False, "mode": "any", "emoji": "", "intensity": 0.0}
    return data


def pick_media_item(catalog: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any] | None:
    items = catalog.get("items", [])
    if not items:
        return None

    send = bool(intent.get("send"))
    if not send:
        return None

    mode = str(intent.get("mode", "any")).strip() or "any"
    emoji = str(intent.get("emoji", "")).strip()
    try:
        intensity = float(intent.get("intensity", 0.0))
    except (TypeError, ValueError):
        intensity = 0.0
    if intensity < 0.35:
        return None

    candidates = items
    if emoji:
        exact = [i for i in candidates if i.get("emoji") == emoji]
        if exact:
            candidates = exact
    if mode in {"sticker", "custom_emoji"}:
        typed = [i for i in candidates if i.get("kind") == mode]
        if typed:
            candidates = typed

    return random.choice(candidates) if candidates else None


async def send_media_item(bot, chat_id: int, reply_to_message_id: int, item: dict[str, Any]) -> bool:
    kind = item.get("kind")
    if kind == "sticker" and item.get("file_id"):
        try:
            await bot.send_sticker(chat_id, item["file_id"], reply_to_message_id=reply_to_message_id)
            return True
        except Exception:
            log.debug("failed to send sticker file_id=%s", item.get("file_id"), exc_info=True)
            return False

    if kind == "custom_emoji":
        custom_emoji_id = item.get("custom_emoji_id")
        emoji = item.get("emoji") or "🙂"
        if custom_emoji_id:
            try:
                await bot.send_message(
                    chat_id,
                    f'<tg-emoji emoji-id="{custom_emoji_id}">{emoji}</tg-emoji>',
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=reply_to_message_id,
                )
                return True
            except Exception:
                log.debug("failed to send custom_emoji id=%s, falling back to sticker", custom_emoji_id, exc_info=True)
        if item.get("file_id"):
            try:
                await bot.send_sticker(chat_id, item["file_id"], reply_to_message_id=reply_to_message_id)
                return True
            except Exception:
                log.debug("failed to send fallback sticker file_id=%s", item.get("file_id"), exc_info=True)
                return False

    return False
