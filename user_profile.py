from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

PROFILE_COMMAND_RE = re.compile(
    r"^/(?:profile|user_profile)(?:@[A-Za-z0-9_]{3,64})?(?:\s+(?P<args>.*))?$",
    re.IGNORECASE,
)
PROFILE_ASSIGNMENT_RE = re.compile(
    r"^(?P<field>[^\s:=]+)\s*(?:[:=]\s*|\s+)(?P<value>.+)$",
    re.IGNORECASE,
)

PROFILE_FIELD_META: dict[str, dict[str, str]] = {
    "approved_display_name": {"kind": "column", "label": "Имя"},
    "nickname": {"kind": "column", "label": "Ник"},
    "gender": {"kind": "column", "label": "Пол"},
    "interests": {"kind": "column", "label": "Интересы"},
    "communication_style": {"kind": "column", "label": "Стиль общения"},
    "response_preference": {"kind": "column", "label": "Как отвечать"},
    "relationship_notes": {"kind": "column", "label": "Заметки"},
    "about": {"kind": "extra", "label": "О себе"},
    "age": {"kind": "extra", "label": "Возраст"},
    "city": {"kind": "extra", "label": "Город"},
    "location": {"kind": "extra", "label": "Локация"},
    "occupation": {"kind": "extra", "label": "Занятие"},
}

PROFILE_FIELD_ALIASES = {
    "name": "approved_display_name",
    "display_name": "approved_display_name",
    "preferred_name": "approved_display_name",
    "имя": "approved_display_name",
    "nickname": "nickname",
    "nick": "nickname",
    "username": "nickname",
    "ник": "nickname",
    "никнейм": "nickname",
    "gender": "gender",
    "sex": "gender",
    "пол": "gender",
    "interests": "interests",
    "interest": "interests",
    "hobby": "interests",
    "hobbies": "interests",
    "увлечение": "interests",
    "увлечения": "interests",
    "хобби": "interests",
    "communication_style": "communication_style",
    "communication": "communication_style",
    "style": "communication_style",
    "tone": "communication_style",
    "стиль": "communication_style",
    "стиль_общения": "communication_style",
    "общение": "communication_style",
    "response_preference": "response_preference",
    "reply_style": "response_preference",
    "answer_style": "response_preference",
    "reply": "response_preference",
    "ответ": "response_preference",
    "ответы": "response_preference",
    "формат_ответа": "response_preference",
    "relationship_notes": "relationship_notes",
    "relationship": "relationship_notes",
    "notes": "relationship_notes",
    "note": "relationship_notes",
    "заметки": "relationship_notes",
    "заметка": "relationship_notes",
    "about": "about",
    "bio": "about",
    "about_me": "about",
    "описание": "about",
    "о_себе": "about",
    "age": "age",
    "возраст": "age",
    "city": "city",
    "город": "city",
    "location": "location",
    "place": "location",
    "локация": "location",
    "место": "location",
    "местоположение": "location",
    "occupation": "occupation",
    "job": "occupation",
    "work": "occupation",
    "profession": "occupation",
    "занятие": "occupation",
    "работа": "occupation",
    "профессия": "occupation",
    "должность": "occupation",
}

PROFILE_SHOW_ALIASES = {"show", "view", "get", "list", "показать", "посмотреть"}
PROFILE_HELP_ALIASES = {"help", "?", "команды", "помощь"}
PROFILE_CLEAR_ALIASES = {"clear", "reset", "wipe", "очистить", "сбросить"}
PROFILE_SET_ALIASES = {"set", "update", "edit", "save", "изменить", "обновить", "установить"}
PROFILE_PATCH_ALIASES = {"patch", "bulk", "массово", "пакет"}
PROFILE_DELETE_ALIASES = {"delete", "remove", "unset", "del", "удалить", "снести"}

PROFILE_EXTRA_FIELD_ORDER = ("about", "age", "city", "location", "occupation")
PROFILE_VIEW_FIELD_ORDER = (
    "approved_display_name",
    "nickname",
    "gender",
    "interests",
    "communication_style",
    "response_preference",
    "relationship_notes",
)

PROFILE_HELP_TEXT = (
    "Команды профиля:\n"
    "/profile\n"
    "/profile help\n"
    "/profile set name Иван\n"
    "/profile set interests музыка, код, аниме\n"
    "/profile set gender мужской\n"
    "/profile set city Москва\n"
    "/profile patch name=Иван; gender=мужской; interests=музыка, код\n"
    "/profile delete city\n"
    "/profile clear"
)


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_token(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip().lower())
    text = text.replace("ё", "е")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^0-9a-zа-я_]+", "", text)
    return text.strip("_")


def normalize_profile_value(value: Any, limit: int = 240) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip())
    text = re.sub(r"\s+", " ", text)
    return text[:limit].strip()


def normalize_profile_field(value: Any) -> dict[str, str] | None:
    token = _normalize_token(value)
    canonical = PROFILE_FIELD_ALIASES.get(token, token)
    meta = PROFILE_FIELD_META.get(canonical)
    if not meta:
        return None
    return {"field": canonical, "kind": meta["kind"], "label": meta["label"]}


def parse_profile_extra_fields(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        source = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            source = {}
        else:
            source = parsed if isinstance(parsed, dict) else {}
    else:
        source = {}

    extra_fields: dict[str, str] = {}
    for key, value in source.items():
        normalized_key = _normalize_token(key)
        if not normalized_key:
            continue
        canonical = PROFILE_FIELD_ALIASES.get(normalized_key, normalized_key)
        normalized_value = normalize_profile_value(value)
        if not normalized_value:
            continue
        extra_fields[canonical] = normalized_value
    return extra_fields


def serialize_profile_extra_fields(raw: Any) -> str:
    return json.dumps(parse_profile_extra_fields(raw), ensure_ascii=False, sort_keys=True)


def iter_profile_extra_items(profile: dict[str, Any]) -> list[tuple[str, str]]:
    extra_fields = parse_profile_extra_fields(
        profile.get("extra_fields") if isinstance(profile, dict) else None
    )
    if not extra_fields and isinstance(profile, dict):
        extra_fields = parse_profile_extra_fields(profile.get("extra_json"))

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in PROFILE_EXTRA_FIELD_ORDER:
        value = extra_fields.get(key, "")
        if not value:
            continue
        seen.add(key)
        items.append((PROFILE_FIELD_META.get(key, {}).get("label", key), value))

    for key in sorted(extra_fields):
        if key in seen:
            continue
        label = PROFILE_FIELD_META.get(key, {}).get("label") or key.replace("_", " ").title()
        items.append((label, extra_fields[key]))
    return items


def format_profile_view(profile: dict[str, Any]) -> str:
    name = normalize_profile_value(
        profile.get("approved_display_name") or profile.get("display_name"),
        limit=80,
    )
    lines = ["Профиль пользователя:"]

    field_values = {
        "approved_display_name": name,
        "nickname": normalize_profile_value(profile.get("nickname"), limit=80),
        "gender": normalize_profile_value(profile.get("gender"), limit=40),
        "interests": normalize_profile_value(profile.get("interests"), limit=180),
        "communication_style": normalize_profile_value(profile.get("communication_style"), limit=180),
        "response_preference": normalize_profile_value(profile.get("response_preference"), limit=180),
        "relationship_notes": normalize_profile_value(profile.get("relationship_notes"), limit=180),
    }

    for field in PROFILE_VIEW_FIELD_ORDER:
        value = field_values.get(field, "")
        if not value:
            continue
        lines.append(f"- {PROFILE_FIELD_META[field]['label']}: {value}")

    for label, value in iter_profile_extra_items(profile):
        lines.append(f"- {label}: {value}")

    if len(lines) == 1:
        return "Профиль пока пуст. Используй /profile help, чтобы заполнить его."
    return "\n".join(lines)


def apply_profile_updates(profile: dict[str, Any], updates: list[dict[str, str]]) -> dict[str, Any]:
    next_profile = dict(profile or {})
    extra_fields = parse_profile_extra_fields(next_profile.get("extra_fields") or next_profile.get("extra_json"))

    for update in updates:
        field = str(update.get("field") or "").strip()
        value = normalize_profile_value(update.get("value"))
        if not field or not value:
            continue
        meta = PROFILE_FIELD_META.get(field)
        if not meta:
            continue
        if meta["kind"] == "column":
            if field == "approved_display_name":
                next_profile["approved_display_name"] = value
                next_profile["display_name"] = value
            else:
                next_profile[field] = value
        else:
            extra_fields[field] = value

    next_profile["extra_fields"] = extra_fields
    next_profile["profile_confidence"] = 1.0
    return next_profile


def delete_profile_field(profile: dict[str, Any], field: str) -> dict[str, Any]:
    next_profile = dict(profile or {})
    canonical_field = str(field or "").strip()
    meta = PROFILE_FIELD_META.get(canonical_field)
    if not meta:
        return next_profile

    if meta["kind"] == "column":
        if canonical_field == "approved_display_name":
            next_profile["approved_display_name"] = None
            next_profile["display_name"] = None
        elif canonical_field in {"nickname", "gender"}:
            next_profile[canonical_field] = None
        else:
            next_profile[canonical_field] = ""
    else:
        extra_fields = parse_profile_extra_fields(next_profile.get("extra_fields") or next_profile.get("extra_json"))
        extra_fields.pop(canonical_field, None)
        next_profile["extra_fields"] = extra_fields

    next_profile["profile_confidence"] = 1.0
    return next_profile


def cleared_profile() -> dict[str, Any]:
    return {
        "display_name": None,
        "nickname": None,
        "approved_display_name": None,
        "gender": None,
        "interests": "",
        "communication_style": "",
        "response_preference": "",
        "relationship_notes": "",
        "profile_confidence": 0.0,
        "extra_fields": {},
    }


def parse_profile_command(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    match = PROFILE_COMMAND_RE.match(raw)
    if not match:
        return None

    args = _normalize_space(match.group("args") or "")
    if not args:
        return {"action": "show"}

    head, _, tail = args.partition(" ")
    action_token = _normalize_token(head)
    rest = tail.strip()

    if action_token in PROFILE_SHOW_ALIASES:
        return {"action": "show"}
    if action_token in PROFILE_HELP_ALIASES:
        return {"action": "help"}
    if action_token in PROFILE_CLEAR_ALIASES and not rest:
        return {"action": "clear"}
    if action_token in PROFILE_SET_ALIASES:
        update = _parse_profile_assignment(rest)
        if not update:
            return {"action": "invalid", "error": "Не понял команду set.", "help": PROFILE_HELP_TEXT}
        return {"action": "patch", "updates": [update]}
    if action_token in PROFILE_PATCH_ALIASES:
        updates = _parse_profile_patch(rest)
        if not updates:
            return {"action": "invalid", "error": "Не понял команду patch.", "help": PROFILE_HELP_TEXT}
        return {"action": "patch", "updates": updates}
    if action_token in PROFILE_DELETE_ALIASES:
        target = normalize_profile_field(rest)
        if not target:
            return {"action": "invalid", "error": "Неизвестное поле профиля.", "help": PROFILE_HELP_TEXT}
        return {"action": "delete", "field": target["field"], "label": target["label"], "kind": target["kind"]}

    return {"action": "invalid", "error": "Неизвестная команда профиля.", "help": PROFILE_HELP_TEXT}


def _parse_profile_assignment(raw: str) -> dict[str, str] | None:
    text = str(raw or "").strip()
    if not text:
        return None

    match = PROFILE_ASSIGNMENT_RE.match(text)
    if not match:
        return None

    field = normalize_profile_field(match.group("field"))
    if not field:
        return None

    value = normalize_profile_value(match.group("value"))
    if not value:
        return None

    return {
        "field": field["field"],
        "kind": field["kind"],
        "label": field["label"],
        "value": value,
    }


def _parse_profile_patch(raw: str) -> list[dict[str, str]]:
    text = str(raw or "").strip()
    if not text:
        return []

    updates: list[dict[str, str]] = []
    for chunk in re.split(r"\s*;\s*", text):
        if not chunk:
            continue
        parsed = _parse_profile_assignment(chunk)
        if not parsed:
            return []
        updates.append(parsed)
    return updates
