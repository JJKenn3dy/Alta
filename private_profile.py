"""
private_profile.py — private-chat profile settings UI for Alta bot.

No LLM is involved here. Pure deterministic menu + validation flow.
Users can manage their own profile via inline keyboards in private chat.
Profile saved under (user_id, user_id) as a "global" profile that gets
merged with chat-specific profiles in groups.
"""
from __future__ import annotations

import re
import time
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"https?://|t\.me/|tg://|bit\.ly/|goo\.gl/|vk\.cc/",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]{0,40}>")
_SLASH_CMD_RE = re.compile(r"^/")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

GENDER_OPTIONS = ("мужской", "женский", "небинарный", "не указан")

FIELD_LABELS: dict[str, str] = {
    "approved_display_name": "Имя",
    "nickname": "Ник",
    "gender": "Пол",
    "age": "Возраст",
    "city": "Город",
    "occupation": "Занятие",
    "interests": "Интересы",
    "communication_style": "Стиль общения",
    "response_preference": "Формат ответов",
    "about": "О себе",
}

FIELD_PROMPTS: dict[str, str] = {
    "approved_display_name": (
        "Введи своё имя — как тебя называть.\n"
        "Макс. 40 символов, без ссылок.\n\n"
        "Или /cancel для отмены."
    ),
    "nickname": (
        "Введи ник — как ещё тебя можно называть.\n"
        "Макс. 40 символов, без ссылок.\n\n"
        "Или /cancel для отмены."
    ),
    "age": (
        "Введи свой возраст (число от 10 до 120).\n\n"
        "Или /cancel для отмены."
    ),
    "city": (
        "Введи свой город.\n"
        "Макс. 60 символов, без ссылок.\n\n"
        "Или /cancel для отмены."
    ),
    "occupation": (
        "Введи род занятий или профессию.\n"
        "Макс. 100 символов, без ссылок.\n\n"
        "Или /cancel для отмены."
    ),
    "interests": (
        "Введи интересы и хобби через запятую.\n"
        "Макс. 200 символов.\n\n"
        "Или /cancel для отмены."
    ),
    "communication_style": (
        "Введи предпочтительный стиль общения.\n"
        "Например: «неформально», «по-дружески», «с юмором».\n"
        "Макс. 200 символов.\n\n"
        "Или /cancel для отмены."
    ),
    "response_preference": (
        "Введи пожелание по формату ответов.\n"
        "Например: «коротко», «развёрнуто», «без советов».\n"
        "Макс. 200 символов.\n\n"
        "Или /cancel для отмены."
    ),
    "about": (
        "Напиши немного о себе.\n"
        "Макс. 300 символов, без ссылок.\n\n"
        "Или /cancel для отмены."
    ),
}

# Fields stored in extra_fields JSON column
EXTRA_FIELDS = frozenset({"age", "city", "occupation", "about"})
# Fields stored as direct columns in user_profiles
COLUMN_FIELDS = frozenset(
    {"approved_display_name", "nickname", "gender", "interests", "communication_style", "response_preference"}
)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_RATE_WINDOW_SEC = 3600.0  # 1 hour
_RATE_MAX_SAVES = 30       # max saves per hour per user

_rate_log: dict[int, list[float]] = {}


def rate_limit_ok(user_id: int) -> bool:
    """Return True if user is within rate limit, and record a new event."""
    now = time.monotonic()
    log = _rate_log.get(user_id, [])
    log = [t for t in log if now - t < _RATE_WINDOW_SEC]
    if len(log) >= _RATE_MAX_SAVES:
        return False
    log.append(now)
    _rate_log[user_id] = log
    return True


# ---------------------------------------------------------------------------
# In-memory state management
# ---------------------------------------------------------------------------

# user_id -> {"field": str, "menu_message_id": int | None}
_pm_state: dict[int, dict[str, Any]] = {}


def get_pm_state(user_id: int) -> dict[str, Any] | None:
    return _pm_state.get(user_id)


def set_pm_state(user_id: int, field: str, menu_message_id: int | None = None) -> None:
    _pm_state[user_id] = {"field": field, "menu_message_id": menu_message_id}


def clear_pm_state(user_id: int) -> None:
    _pm_state.pop(user_id, None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _clean(value: str) -> str:
    """Strip HTML tags and control characters."""
    v = _HTML_TAG_RE.sub("", value)
    v = _CONTROL_CHARS_RE.sub("", v)
    return " ".join(v.split()).strip()


def validate_field(field: str, raw: str) -> tuple[bool, str, str]:
    """
    Returns (ok, error_message, cleaned_value).
    error_message is empty if ok=True.
    """
    v = _clean(raw)

    if not v:
        return False, "Значение не может быть пустым.", ""

    if _SLASH_CMD_RE.match(v):
        return False, "Значение не может начинаться со слэша.", ""

    if _URL_RE.search(v):
        return False, "Ссылки в профиле не разрешены.", ""

    if field == "approved_display_name":
        if len(v) > 40:
            return False, "Слишком длинно. Максимум 40 символов.", ""
        return True, "", v

    if field == "nickname":
        if len(v) > 40:
            return False, "Слишком длинно. Максимум 40 символов.", ""
        return True, "", v

    if field == "gender":
        if len(v) > 30:
            return False, "Слишком длинно. Максимум 30 символов.", ""
        return True, "", v

    if field == "age":
        digits = re.sub(r"\D", "", v)
        if not digits:
            return False, "Возраст должен быть числом.", ""
        age = int(digits)
        if not (10 <= age <= 120):
            return False, "Возраст должен быть от 10 до 120.", ""
        return True, "", str(age)

    if field == "city":
        if len(v) > 60:
            return False, "Слишком длинно. Максимум 60 символов.", ""
        return True, "", v

    if field == "occupation":
        if len(v) > 100:
            return False, "Слишком длинно. Максимум 100 символов.", ""
        return True, "", v

    if field == "interests":
        if len(v) > 200:
            return False, "Слишком длинно. Максимум 200 символов.", ""
        return True, "", v

    if field == "communication_style":
        if len(v) > 200:
            return False, "Слишком длинно. Максимум 200 символов.", ""
        return True, "", v

    if field == "response_preference":
        if len(v) > 200:
            return False, "Слишком длинно. Максимум 200 символов.", ""
        return True, "", v

    if field == "about":
        if len(v) > 300:
            return False, "Слишком длинно. Максимум 300 символов.", ""
        return True, "", v

    return False, "Неизвестное поле.", ""


# ---------------------------------------------------------------------------
# Inline keyboards
# ---------------------------------------------------------------------------

def build_main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👤 Мой профиль", callback_data="pm:show")],
        [
            InlineKeyboardButton(text="✏️ Редактировать", callback_data="pm:edit"),
            InlineKeyboardButton(text="🗑 Очистить всё", callback_data="pm:clear_confirm"),
        ],
        [InlineKeyboardButton(text="📥 Что обо мне знает Альта", callback_data="pm:import_preview")],
    ])


def build_edit_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Имя", callback_data="pm:field:approved_display_name"),
            InlineKeyboardButton(text="Ник", callback_data="pm:field:nickname"),
        ],
        [
            InlineKeyboardButton(text="Пол", callback_data="pm:field:gender"),
            InlineKeyboardButton(text="Возраст", callback_data="pm:field:age"),
        ],
        [
            InlineKeyboardButton(text="Город", callback_data="pm:field:city"),
            InlineKeyboardButton(text="Занятие", callback_data="pm:field:occupation"),
        ],
        [InlineKeyboardButton(text="Интересы", callback_data="pm:field:interests")],
        [InlineKeyboardButton(text="Стиль общения", callback_data="pm:field:communication_style")],
        [InlineKeyboardButton(text="Формат ответов", callback_data="pm:field:response_preference")],
        [InlineKeyboardButton(text="О себе", callback_data="pm:field:about")],
        [InlineKeyboardButton(text="← Назад", callback_data="pm:menu")],
    ])


def build_gender_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Мужской", callback_data="pm:gender:мужской"),
            InlineKeyboardButton(text="Женский", callback_data="pm:gender:женский"),
        ],
        [
            InlineKeyboardButton(text="Небинарный", callback_data="pm:gender:небинарный"),
            InlineKeyboardButton(text="Не указан", callback_data="pm:gender:не указан"),
        ],
        [InlineKeyboardButton(text="Другое (ввести текст)", callback_data="pm:field:gender_custom")],
        [InlineKeyboardButton(text="← Назад", callback_data="pm:edit")],
    ])


def build_cancel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✗ Отмена", callback_data="pm:cancel")],
    ])


def build_back_to_edit_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✏️ Ещё изменить", callback_data="pm:edit"),
            InlineKeyboardButton(text="🏠 Главное меню", callback_data="pm:menu"),
        ],
    ])


def build_clear_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✓ Да, очистить", callback_data="pm:clear_yes"),
            InlineKeyboardButton(text="✗ Нет, отмена", callback_data="pm:menu"),
        ],
    ])


def build_import_preview_keyboard(has_data: bool) -> InlineKeyboardMarkup:
    rows = []
    if has_data:
        rows.append([InlineKeyboardButton(text="✓ Перенести в мой профиль", callback_data="pm:import_confirm")])
    rows.append([InlineKeyboardButton(text="← Назад", callback_data="pm:menu")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_import_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✓ Да, перенести", callback_data="pm:import_do"),
            InlineKeyboardButton(text="✗ Отмена", callback_data="pm:menu"),
        ],
    ])


# ---------------------------------------------------------------------------
# Profile display helper
# ---------------------------------------------------------------------------

def format_profile_private(profile: dict[str, Any]) -> str:
    """Format profile for display in private chat."""
    from user_profile import format_profile_view, parse_profile_extra_fields

    # Inject extra_fields from separate key if present
    if "extra_fields" in profile and isinstance(profile["extra_fields"], str):
        try:
            import json
            parsed_extra = json.loads(profile["extra_fields"])
            profile = dict(profile)
            profile["extra_fields"] = parsed_extra
        except Exception:
            pass

    return format_profile_view(profile)


# ---------------------------------------------------------------------------
# Profile write helpers  (called from main.py with a db instance)
# ---------------------------------------------------------------------------

async def save_field(db: Any, user_id: int, field: str, value: str) -> None:
    """Save a single profile field for the user's private/global profile."""
    import json

    current = await db.get_user_profile(user_id, user_id)

    if field in COLUMN_FIELDS:
        kwargs: dict[str, Any] = {}
        if field == "approved_display_name":
            kwargs["approved_display_name"] = value
            kwargs["display_name"] = value
        else:
            kwargs[field] = value
        await db.upsert_user_profile(user_id, user_id, profile_confidence=1.0, **kwargs)

    elif field in EXTRA_FIELDS:
        raw_extra = current.get("extra_fields") or "{}"
        try:
            extra = json.loads(raw_extra) if isinstance(raw_extra, str) else dict(raw_extra or {})
        except Exception:
            extra = {}
        extra[field] = value
        await db.upsert_user_profile(
            user_id, user_id,
            profile_confidence=1.0,
            extra_fields=json.dumps(extra, ensure_ascii=False),
        )


async def clear_all_fields(db: Any, user_id: int) -> None:
    """Wipe the private/global profile for this user."""
    await db.upsert_user_profile(
        user_id, user_id,
        display_name=None,
        nickname=None,
        approved_display_name=None,
        gender=None,
        interests="",
        communication_style="",
        response_preference="",
        relationship_notes="",
        profile_confidence=0.0,
        extra_fields="{}",
    )


# ---------------------------------------------------------------------------
# Import-from-chats helpers
# ---------------------------------------------------------------------------

_IMPORT_FIELD_LABELS: list[tuple[str, str]] = [
    ("approved_display_name", "Имя"),
    ("nickname", "Ник"),
    ("gender", "Пол"),
    ("interests", "Интересы"),
    ("communication_style", "Стиль общения"),
    ("response_preference", "Формат ответов"),
    # relationship_notes intentionally excluded — auto-extracted values are
    # often transient conversational states, not stable user attributes.
]
_IMPORT_EXTRA_LABELS: list[tuple[str, str]] = [
    ("age", "Возраст"),
    ("city", "Город"),
    ("occupation", "Занятие"),
    ("about", "О себе"),
]


def format_import_preview(group_profile: dict[str, Any], private_profile: dict[str, Any], facts_count: int) -> tuple[str, bool]:
    """
    Format the import preview message.
    Returns (text, has_importable_data).
    has_importable_data = True if there's anything worth importing.
    """
    import json

    # Parse extra_fields
    try:
        group_extra = json.loads(group_profile.get("extra_fields") or "{}") if isinstance(group_profile.get("extra_fields"), str) else {}
    except Exception:
        group_extra = {}
    try:
        private_extra = json.loads(private_profile.get("extra_fields") or "{}") if isinstance(private_profile.get("extra_fields"), str) else {}
    except Exception:
        private_extra = {}

    lines = ["Вот что Альта собрала о тебе из групповых чатов:\n"]
    has_data = False
    has_importable = False  # fields not already in private profile

    for field, label in _IMPORT_FIELD_LABELS:
        value = (group_profile.get(field) or "").strip()
        if not value:
            continue
        has_data = True
        private_value = (private_profile.get(field) or "").strip()
        if private_value:
            marker = "✓"  # already set
        else:
            marker = "→"  # will be imported
            has_importable = True
        lines.append(f"{marker} {label}: {value}")

    for field, label in _IMPORT_EXTRA_LABELS:
        value = (group_extra.get(field) or "").strip()
        if not value:
            continue
        has_data = True
        private_value = (private_extra.get(field) or "").strip()
        if private_value:
            marker = "✓"
        else:
            marker = "→"
            has_importable = True
        lines.append(f"{marker} {label}: {value}")

    if facts_count > 0:
        has_data = True
        lines.append(f"\nФактов в памяти: {facts_count}")

    if not has_data:
        return "Альта пока ничего не собрала о тебе из групп. Напиши что-нибудь в чате!", False

    lines.append("\n✓ — уже в твоём профиле\n→ — будет добавлено при импорте")

    if not has_importable:
        lines.append("\nВсё уже перенесено в твой профиль.")

    return "\n".join(lines), has_importable


async def import_from_groups(db: Any, user_id: int) -> int:
    """
    Copy non-empty fields from all group profiles to the private/global profile.
    Existing private profile values are NOT overwritten (private = user-set = higher priority).
    Returns the number of fields imported.
    """
    import json

    group_profile = await db.get_all_group_profiles_merged(user_id)
    if not group_profile:
        return 0

    private_profile = await db.get_user_profile(user_id, user_id)

    try:
        group_extra = json.loads(group_profile.get("extra_fields") or "{}") if isinstance(group_profile.get("extra_fields"), str) else {}
    except Exception:
        group_extra = {}
    try:
        private_extra = json.loads(private_profile.get("extra_fields") or "{}") if isinstance(private_profile.get("extra_fields"), str) else {}
    except Exception:
        private_extra = {}

    imported = 0
    kwargs: dict[str, Any] = {}

    # Column fields — only set if private profile doesn't already have a value
    for field in ("approved_display_name", "display_name", "nickname", "gender",
                  "interests", "communication_style", "response_preference", "relationship_notes"):
        group_val = (group_profile.get(field) or "").strip()
        private_val = (private_profile.get(field) or "").strip()
        if group_val and not private_val:
            kwargs[field] = group_val
            imported += 1

    # Extra fields
    merged_extra = dict(private_extra)
    for field, _ in _IMPORT_EXTRA_LABELS:
        group_val = (group_extra.get(field) or "").strip()
        if group_val and field not in private_extra:
            merged_extra[field] = group_val
            imported += 1

    if not kwargs and not (set(merged_extra) - set(private_extra)):
        return 0

    # Preserve existing private confidence (1.0 means user-set)
    current_confidence = private_profile.get("profile_confidence") or 0.0
    new_confidence = max(current_confidence, group_profile.get("profile_confidence") or 0.0)

    kwargs["extra_fields"] = json.dumps(merged_extra, ensure_ascii=False)
    kwargs["profile_confidence"] = new_confidence
    await db.upsert_user_profile(user_id, user_id, **kwargs)
    return imported
