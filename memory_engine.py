"""Memory engine: facts, user profiles, summaries, background extraction.

Handles all persistent knowledge about users — storing, updating, and
retrieving facts/profiles from the DB, including the async background
extraction pipeline that runs after each bot reply.
"""
import asyncio
import json
import logging
import re
from typing import Any

log = logging.getLogger("alta.memory")

from postprocessing import _fit_text, _is_nonsemantic_input, _clip_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MESSAGE_ROTATE_KEEP = 2_000
MESSAGE_ROTATE_TRIGGER_DELTA = 120
SUMMARY_HISTORY_WINDOW = 80
MAX_FACT_ITEMS_PER_TURN = 6
INTRO_MARKER_KEY = "_intro_requested"

IDENTITY_KEYS = {"name", "preferred_name", "gender", "pronouns", "nickname"}
STYLE_KEYS = {"communication_style", "response_preference"}
HIGH_PRIORITY_KEYS = IDENTITY_KEYS | STYLE_KEYS

MEMORY_STORE_MIN_IMPORTANCE = 0.3
FACT_KEEP_PER_USER = 80
# Порог показа факта в контексте = порогу записи. Раньше показ был 0.45, и факты
# с importance 0.3–0.44 записывались в БД, но Альта их не видела ("записал, но не
# помнит"). Порядок по importance DESC + лимит строк и так ставят важное вперёд.
FACT_CONTEXT_MIN_IMPORTANCE = MEMORY_STORE_MIN_IMPORTANCE
MAX_FACT_LINES_IN_CONTEXT = 16

SUMMARY_INPUT_CHAR_LIMIT = 6_000
MEMORY_EXTRACT_INPUT_CHAR_LIMIT = 1_800
PROFILE_EXTRACT_INPUT_CHAR_LIMIT = 2_200
PROFILE_FIELD_CHAR_LIMIT = 200
RELATIONSHIP_NOTES_CHAR_LIMIT = 250
USER_DOCUMENT_CHAR_LIMIT = 900
USER_DOCUMENT_FACT_LIMIT = 20

# Mirror of CONTEXT_HISTORY_LIMIT from context_builder — used in summary cadence.
_CONTEXT_HISTORY_LIMIT = 15

# Стоп-паттерны для значений фактов: ролевые игры, эротика, БДСМ-терминология
_FACT_VALUE_STOP_PATTERNS = (
    re.compile(
        r"\b(slave|master|rabb|рабын|господин|госпожа|bdsm|bddsm|фетиш|fetish"
        r"|striptiz|стриптиз|проститут|escort|порно|porno|секс\s*клуб|sex\s*club"
        r"|leather\s*(club|head)|онлифанс|onlyfans|интим|erotic|эротик"
        r"|флирт\s*клуб|roleplay|ролевая\s*игра)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(sexual|intimacy|finger|penis|vagina|анал|оральн|cunni|fellat"
        r"|gay\s*(website|club|bar)|genitalia)\b",
        re.IGNORECASE,
    ),
)


def _is_toxic_fact_value(value: str) -> bool:
    """Return True if fact value contains roleplay/erotic/BDSM garbage."""
    for pattern in _FACT_VALUE_STOP_PATTERNS:
        if pattern.search(value):
            return True
    return False


INVALID_IDENTITY_VALUES = {
    "ты", "тебя", "тебе", "твой", "твоя", "он", "она", "они",
    "неизвестно", "неизвестен", "неизвестна", "неопределено", "не определено",
    "unknown", "none", "null", "никак",
}
INVALID_GENDER_VALUES = INVALID_IDENTITY_VALUES | {
    "не знаю", "не указан", "не указано",
}

FACT_TYPE_LABELS = {
    "interest": "Интерес",
    "preference": "Предпочтение",
    "work": "Работа",
    "study": "Учеба",
    "location": "Локация",
    "goal": "Цель",
    "relationship": "По общению",
    "bio": "О себе",
    "fact": "Факт",
}

EXPLICIT_PROFILE_PATTERNS = (
    re.compile(r"\bменя зовут\b", re.IGNORECASE),
    re.compile(r"\bмой ник\b", re.IGNORECASE),
    re.compile(r"\bможно (звать|называть)\b", re.IGNORECASE),
    re.compile(r"\bобращайся ко мне\b", re.IGNORECASE),
    re.compile(r"\bя (девушка|женщина|парень|мужчина)\b", re.IGNORECASE),
    re.compile(r"\bя люблю\b", re.IGNORECASE),
    re.compile(r"\bмне нравится\b", re.IGNORECASE),
    re.compile(r"\bя увлекаюсь\b", re.IGNORECASE),
    re.compile(r"\bя занимаюсь\b", re.IGNORECASE),
    re.compile(r"\bя работаю\b", re.IGNORECASE),
    re.compile(r"\bя учусь\b", re.IGNORECASE),
    re.compile(r"\bпредпочитаю\b", re.IGNORECASE),
    re.compile(r"\bмои интересы\b", re.IGNORECASE),
    re.compile(r"\bя из\b", re.IGNORECASE),
)

_HISTORY_PREFIX_USER_ID_RE = re.compile(r"\bid=(\d+)\b")

# ---------------------------------------------------------------------------
# LLM prompt strings
# ---------------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """
Ты модуль сжатия истории чата.
Сделай короткое и полезное резюме текущего диалога на русском.
Включай только факты, договоренности, планы, важные предпочтения и контекст.
Важно: это может быть групповой чат с несколькими участниками.
Никогда не смешивай людей между собой.
Если записываешь факт или предпочтение конкретного человека, обязательно явно укажи его метку из истории, например @username или id=123.
Не пиши безличные пункты вроде "любит пиццу", если неясно кто именно.
Не добавляй выдумки. Формат: 5-10 коротких пунктов.
""".strip()

USER_SUMMARY_SYSTEM_PROMPT = """
Ты модуль сжатия истории чата.
Сделай короткое резюме взаимодействия КОНКРЕТНОГО пользователя с ботом.

Правила:
- Записывай ТОЛЬКО действия, высказывания и факты про указанного пользователя.
- НЕ включай информацию о других участниках чата, даже если они упоминаются в истории.
- Формат: 3-7 коротких пунктов только про этого пользователя.
- Если пользователь мало участвовал, резюме может быть 1-2 пункта.
- Не добавляй заголовки вроде "### Обновленное резюме" и маркдаун-форматирование.
- Не добавляй выдумки.
""".strip()

MEMORY_EXTRACT_PROMPT = """
Извлеки только устойчивые факты о пользователе из его сообщения.
Верни строго JSON-массив объектов формата:
[{"fact_type":"...", "value":"...", "confidence":0.0, "importance":0.0, "evidence":"..."}]

Правила:
- Извлекай ТОЛЬКО факты про самого автора сообщения (раздел "Сообщение пользователя").
- Если в контексте есть "Другой участник чата" — его факты НЕЛЬЗЯ приписывать нашему пользователю.
- Если пользователь говорит о ком-то другом ("мой друг любит...", "она занимается...") — это НЕ его факт.
- Если пользователь говорит о боте/тебе ("ты красивая", "я тебя люблю", "ты моя роботка") — это НЕ факт о пользователе, не записывай.
- СТРОГО ЗАПРЕЩЕНО извлекать факты из: ролевых игр (roleplay), флирта, провокационных или эротических заявлений, шуток про секс/БДСМ/рабство, троллинга, мемов. Если сообщение содержит подобное — верни [].
- Только реальные, серьёзные личные факты из первого лица. Если непонятно — серьёзное это или шутка, не записывай.
- Не извлекай имя, ник, пол, местоимения, стиль ответа и прочие identity-поля: они обрабатываются отдельно.
- Используй только fact_type из набора: interest, preference, work, study, location, goal, bio, fact.
- evidence: короткая цитата или опора из сообщения, максимум 120 символов.
- importance от 0.0 до 1.0: 1.0 — факт стоит помнить очень долго; 0.0 — почти мусор.
- Не сохраняй временные мелочи ("сейчас ем", "сегодня устал"), сиюминутные эмоции, мемные вбросы.
- Максимум 4 объекта.
- confidence от 0.0 до 1.0.
- Если извлекать нечего, верни [].
""".strip()

PROFILE_EXTRACT_PROMPT = """
Ты обновляешь устойчивый профиль человека по его сообщению.
Верни строго JSON-объект формата:
{
  "display_name": "string or empty",
  "nickname": "string or empty",
  "gender": "string or empty",
  "interests": ["..."],
  "communication_style": "string or empty",
  "response_preference": "string or empty",
  "profile_confidence": 0.0
}

Правила:
- Обновляй ТОЛЬКО на основе раздела "Сообщение пользователя", не на основе "Другой участник чата".
- Если пользователь упомянул кого-то другого ("мой друг любит аниме") — это НЕ его интерес.
- Если поле неясно из этого конкретного сообщения, оставь пустую строку или пустой массив.
- interests: 0-3 коротких интереса, только явно заявленные самим пользователем. Не угадывать.
- communication_style: как человек обычно пишет или общается. Только если очевидно из текста.
- response_preference: пожелания к ответу ("коротко", "с юмором"). Только если явно сказано.
- Не записывай мусорные значения вроде "ты", "он", "она", "неизвестно", "неопределено".
- Никаких домыслов и выдумок. Лучше пустое поле, чем неверное.
""".strip()

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


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


# ---------------------------------------------------------------------------
# Normalization / validation helpers
# ---------------------------------------------------------------------------

def _normalize_compact_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_invalid_identity_value(value: Any) -> bool:
    return _normalize_compact_text(value) in INVALID_IDENTITY_VALUES


def _is_invalid_gender_value(value: Any) -> bool:
    return _normalize_compact_text(value) in INVALID_GENDER_VALUES


def _normalize_fact_type(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    aliases = {
        "interests": "interest",
        "hobby": "interest",
        "hobbies": "interest",
        "likes": "interest",
        "pref": "preference",
        "preferences": "preference",
        "workplace": "work",
        "job": "work",
        "occupation": "work",
        "education": "study",
        "city": "location",
        "place": "location",
        "plan": "goal",
        "plans": "goal",
        "relationship_notes": "relationship",
        "communication": "relationship",
    }
    text = aliases.get(text, text)
    return text if text in FACT_TYPE_LABELS else "fact"


def _normalize_fact_value(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_fact_evidence(value: Any) -> str:
    return _fit_text(" ".join(str(value or "").strip().split()), 120)


def _fact_signature(fact_type: str, normalized_value: str) -> str:
    return f"{fact_type}::{normalized_value}"


def _clean_fact_value(value: Any) -> str | None:
    text = _normalize_fact_value(value)
    if not text:
        return None
    if len(text) < 2:
        return None
    lowered = text.lower()
    blocked = {"да", "нет", "ага", "ок", "норм", "ничего", "не знаю", "хз", "лол"}
    if lowered in blocked:
        return None
    return text[:240]


def _clean_memory_value(key: str, value: Any) -> str | None:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    if key in {"name", "preferred_name", "nickname"} and _is_invalid_identity_value(text):
        return None
    if key == "gender" and _is_invalid_gender_value(text):
        return None
    if key in {"name", "preferred_name"} and text.startswith("@"):
        return None
    return text[:400]


def _clean_profile_text(value: Any, limit: int = 240) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:limit]


def _clean_identity_profile_text(value: Any, *, limit: int, field: str) -> str | None:
    text = _clean_profile_text(value, limit)
    if not text:
        return None
    if field in {"display_name", "nickname"} and _is_invalid_identity_value(text):
        return None
    if field == "gender" and _is_invalid_gender_value(text):
        return None
    if field == "display_name" and text.startswith("@"):
        return None
    return text


def _join_list(value: Any, limit: int = 4) -> str:
    if not isinstance(value, list):
        return ""
    items = []
    for x in value[:limit]:
        s = str(x).strip()
        if s:
            items.append(s)
    return ", ".join(items)


def _merge_interests(current: Any, new_value: Any, max_items: int = 8) -> str:
    """Накапливает интересы: старые + новые, дедуп без учёта регистра, порядок
    сохраняется (старые первыми). Нужно потому, что LLM-извлечение возвращает
    интересы ТОЛЬКО из текущего сообщения — без слияния новый интерес затирал бы
    накопленные. Возвращает строку через запятую (как хранится в БД)."""
    merged: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any):
        for part in (raw if isinstance(raw, list) else str(raw or "").split(",")):
            s = str(part).strip()
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                merged.append(s)

    _add(current)
    _add(new_value)
    return ", ".join(merged[:max_items])


def _profile_text_or_list(value: Any, limit: int = 4) -> str:
    if isinstance(value, list):
        return _join_list(value, limit=limit)
    return _clip_text(value, 180)


def _normalize_importance(key: str, importance: Any, confidence: float) -> float:
    try:
        score = float(importance)
    except (TypeError, ValueError):
        score = confidence
    score = max(0.0, min(1.0, score))
    if key in HIGH_PRIORITY_KEYS:
        score = max(score, 0.88)
    elif key in {
        "interests", "job", "work", "hobby", "relationship_notes",
        "interest", "preference", "study", "location", "goal", "relationship", "bio", "fact",
    }:
        score = max(score, 0.65)
    return score


# ---------------------------------------------------------------------------
# Record maps and change detection
# ---------------------------------------------------------------------------

def _memory_map(memories: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in memories:
        key = str(m.get("key", "")).strip()
        value = str(m.get("value", "")).strip()
        if key and value and key not in out:
            out[key] = value
    return out


def _memory_record_map(memories: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for memory in memories:
        key = str(memory.get("key", "")).strip()
        if key and key not in out:
            out[key] = dict(memory)
    return out


def _fact_record_map(facts: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fact in facts:
        fact_type = _normalize_fact_type(fact.get("fact_type"))
        normalized_value = _normalize_fact_value(fact.get("normalized_value")) or _normalize_fact_value(
            fact.get("value")
        ).lower()
        if not normalized_value:
            continue
        signature = _fact_signature(fact_type, normalized_value)
        if signature not in out:
            out[signature] = dict(fact)
    return out


def _would_change_memory(
    current: dict[str, Any] | None,
    *,
    value: str,
    confidence: float,
    importance: float,
    source: str,
) -> bool:
    if not current:
        return True
    return any(
        (
            str(current.get("value", "")).strip() != str(value).strip(),
            float(current.get("confidence", 0.0) or 0.0) != float(confidence),
            float(current.get("importance", 0.0) or 0.0) != float(importance),
            str(current.get("source", "")).strip() != str(source).strip(),
        )
    )


def _would_change_fact(
    current: dict[str, Any] | None,
    *,
    confidence: float,
    importance: float,
    evidence: str,
    source: str,
) -> bool:
    if not current:
        return True
    return any(
        (
            float(current.get("confidence", 0.0) or 0.0) < float(confidence),
            float(current.get("importance", 0.0) or 0.0) < float(importance),
            str(current.get("evidence", "")).strip() != str(evidence).strip() and bool(evidence),
            str(current.get("source", "")).strip() != str(source).strip() and bool(source),
        )
    )


def _compute_profile_changes(current: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, list[str]]:
    changed_fields: list[str] = []
    for key, new_value in candidate.items():
        old_value = current.get(key)
        if key == "profile_confidence":
            if float(old_value or 0.0) != float(new_value or 0.0):
                changed_fields.append(key)
            continue
        old_norm = "" if old_value is None else str(old_value).strip()
        new_norm = "" if new_value is None else str(new_value).strip()
        if old_norm != new_norm:
            changed_fields.append(key)
    return len(changed_fields), changed_fields


# ---------------------------------------------------------------------------
# Fact formatting for context
# ---------------------------------------------------------------------------

def format_user_facts(facts: list[dict], max_lines: int = MAX_FACT_LINES_IN_CONTEXT) -> str:
    if not facts:
        return ""
    lines = []
    for fact in facts:
        fact_type = _normalize_fact_type(fact.get("fact_type"))
        value = _normalize_fact_value(fact.get("value"))
        if not value:
            continue
        try:
            importance = float(fact.get("importance", 0.0))
        except (TypeError, ValueError):
            importance = 0.0
        if importance < FACT_CONTEXT_MIN_IMPORTANCE:
            continue
        label = FACT_TYPE_LABELS.get(fact_type, "Факт")
        lines.append(f"- {label}: {value}")
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User document (structured fact summary used for context injection)
# ---------------------------------------------------------------------------

def _trusted_sender_identity(
    user_profile: dict[str, Any],
    sender_username: str | None,
    sender_first_name: str | None,
) -> tuple[str | None, str | None]:
    approved_display_name = _clean_identity_profile_text(
        user_profile.get("approved_display_name"), limit=80, field="display_name",
    )
    display_name = approved_display_name or _clean_identity_profile_text(
        sender_first_name, limit=80, field="display_name",
    )
    nickname = None
    if sender_username:
        nickname = _clean_identity_profile_text(
            f"@{sender_username.lstrip('@')}", limit=80, field="nickname",
        )
    return display_name, nickname


def build_user_document_payload(
    user_facts: list[dict],
    user_profile: dict,
    sender_username: str | None,
    sender_first_name: str | None,
) -> dict[str, Any]:
    identity: dict[str, str] = {}
    display_name, nickname = _trusted_sender_identity(user_profile, sender_username, sender_first_name)

    if display_name:
        identity["display_name"] = str(display_name).strip()
    if nickname:
        identity["nickname"] = str(nickname).strip()

    preferences: dict[str, str] = {}
    for key, field in (
        ("communication_style", "communication_style"),
        ("response_preference", "response_preference"),
        ("relationship_notes", "relationship_notes"),
    ):
        value = _clean_profile_text(user_profile.get(field), RELATIONSHIP_NOTES_CHAR_LIMIT)
        if value:
            preferences[key] = value

    interests = _profile_text_or_list(user_profile.get("interests"))
    facts: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for fact in user_facts:
        fact_type = _normalize_fact_type(fact.get("fact_type"))
        value = _normalize_fact_value(fact.get("value"))
        normalized_value = _normalize_fact_value(fact.get("normalized_value")) or value.lower()
        if not value:
            continue
        signature = _fact_signature(fact_type, normalized_value)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        facts.append(
            {
                "fact_type": fact_type,
                "label": FACT_TYPE_LABELS.get(fact_type, "Факт"),
                "value": value[:180],
                "confidence": float(fact.get("confidence", 0.0) or 0.0),
                "importance": float(fact.get("importance", 0.0) or 0.0),
            }
        )
        if len(facts) >= USER_DOCUMENT_FACT_LIMIT:
            break

    return {
        "identity": identity,
        "interests": interests,
        "preferences": preferences,
        "facts": facts,
    }


def render_user_document_text(payload: dict[str, Any], limit: int = USER_DOCUMENT_CHAR_LIMIT) -> str:
    identity = payload.get("identity") or {}
    preferences = payload.get("preferences") or {}
    facts = payload.get("facts") or []
    interests = str(payload.get("interests") or "").strip()

    lines = ["Досье собеседника:"]
    if identity.get("display_name"):
        lines.append(f"- Имя: {identity['display_name']}")
    if identity.get("nickname"):
        lines.append(f"- Ник: {identity['nickname']}")
    if identity.get("gender"):
        lines.append(f"- Пол: {identity['gender']}")
    if interests:
        lines.append(f"- Интересы: {interests}")
    if preferences.get("communication_style"):
        lines.append(f"- Стиль общения: {preferences['communication_style']}")
    if preferences.get("response_preference"):
        lines.append(f"- Предпочтения к ответу: {preferences['response_preference']}")
    if preferences.get("relationship_notes"):
        lines.append(f"- Заметка по общению: {preferences['relationship_notes']}")

    if facts:
        lines.append("- Доп. факты:")
        for fact in facts[:USER_DOCUMENT_FACT_LIMIT]:
            lines.append(f"  - {fact['label']}: {fact['value']}")

    text = "\n".join(lines).strip()
    return _fit_text(text, limit)


def should_include_user_document(input_text: str, document_text: str) -> bool:
    from context_builder import USER_DOCUMENT_TRIGGER_PATTERNS
    if not document_text.strip():
        return False
    normalized = " ".join((input_text or "").lower().split())
    return any(pattern.search(normalized) for pattern in USER_DOCUMENT_TRIGGER_PATTERNS)


def should_extract_user_facts(input_text: str) -> bool:
    normalized = " ".join((input_text or "").lower().split())
    if len(normalized) < 6:
        return False
    if any(pattern.search(normalized) for pattern in EXPLICIT_PROFILE_PATTERNS):
        return True

    first_person_markers = (
        "я ", "мне ", "меня ", "у меня", "мой ", "моя ", "мои ", "моё ", "мою ",
        "для меня", "со мной", "обо мне", "по мне",
    )
    fact_markers = (
        # вкусы / занятия
        "люблю", "нравится", "предпочитаю", "увлекаюсь", "занимаюсь",
        "работаю", "учусь", "живу", "изучаю", "интересуюсь", "хочу",
        "не люблю", "не нравится", "зовут", "родом из", "ненавижу",
        "играю в", "слушаю", "смотрю", "читаю", "пишу на", "кодю на",
        "программирую", "увлекаюсь",
        # возраст / место
        "лет", "года", "год", "из города", "живу в", "переехал", "переехала",
        "переезжаю", "родился", "родилась", "вырос", "выросла",
        # жизненные события (оба рода — пользователь может быть мужчиной)
        "сдал", "сдала", "купил", "купила", "продал", "продала",
        "поступил", "поступила", "закончил", "закончила", "уволился",
        "уволилась", "устроился", "устроилась", "женился", "вышла замуж",
        "развёлся", "развелся", "завёл", "завела", "болею", "болел", "болела",
        "получил", "получила", "выиграл", "выиграла", "потерял", "потеряла",
        "начал", "начала", "бросил", "бросила", "встречаюсь", "расстался",
        "рассталась",
    )
    # Временной якорь + любое утверждение часто = личный апдейт ("вчера сдал экзамен").
    temporal_markers = (
        "вчера", "сегодня", "завтра", "недавно", "на днях", "на выходных",
        "в субботу", "в воскресенье", "только что", "позавчера",
    )
    has_first_person = any(marker in normalized for marker in first_person_markers)
    has_fact_marker = any(marker in normalized for marker in fact_markers)
    has_temporal = any(marker in normalized for marker in temporal_markers)
    is_question = "?" in normalized
    # Временной якорь засчитываем только для не-вопросов (вопрос «что сегодня делаем?»
    # — не факт о пользователе).
    return has_first_person or has_fact_marker or (has_temporal and not is_question)


# ---------------------------------------------------------------------------
# Summary interval helper
# ---------------------------------------------------------------------------

def _summary_update_interval(history_limit: int) -> int:
    try:
        parsed = int(history_limit)
    except (TypeError, ValueError):
        parsed = _CONTEXT_HISTORY_LIMIT
    return max(1, min(parsed, _CONTEXT_HISTORY_LIMIT))


# ---------------------------------------------------------------------------
# Async DB operations: summaries, memories, profiles
# ---------------------------------------------------------------------------

async def maybe_update_summary(
    llm,
    db,
    chat_id: int,
    user_id: int,
    history_limit: int,
    turn_id: str | None = None,
    turn_metrics=None,
) -> bool:
    from context_builder import _history_as_text
    updated = False
    message_count = await db.count_messages(chat_id)
    user_msg_count = await db.count_user_messages(chat_id, user_id)
    interval = _summary_update_interval(history_limit)
    should_update_chat_summary = message_count >= interval and message_count % interval == 0
    should_update_user_summary = user_msg_count >= interval and user_msg_count % interval == 0
    if not should_update_chat_summary and not should_update_user_summary:
        return False

    current_summary = await db.get_summary(chat_id)
    recent_history = await db.get_last_messages(
        chat_id, max(history_limit, SUMMARY_HISTORY_WINDOW), focus_user_id=None,
    )
    history_text = _history_as_text(recent_history, SUMMARY_INPUT_CHAR_LIMIT)
    summary_input = (
        f"Текущее резюме:\n{current_summary or '(пусто)'}\n\n"
        f"Новые сообщения:\n{history_text}\n\n"
        f"Обнови резюме."
    )
    try:
        new_summary = await llm.chat(
            SUMMARY_SYSTEM_PROMPT,
            [{"role": "user", "content": summary_input}],
            temperature=0.2,
            top_p=1.0,
            max_tokens=320,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
        normalized_summary = new_summary.strip()
        if normalized_summary and normalized_summary != (current_summary or "").strip():
            await db.set_summary(chat_id, normalized_summary)
            updated = True
    except Exception:
        log.warning("failed to update chat summary chat=%s", chat_id, exc_info=True)

    if should_update_user_summary:
        current_user_summary = await db.get_summary(chat_id, user_id)
        recent_user_history = await db.get_last_messages(
            chat_id, max(history_limit, SUMMARY_HISTORY_WINDOW), focus_user_id=user_id,
        )
        user_only_messages = [m for m in recent_user_history if m.get("role") == "user"]
        user_history_text = _history_as_text(user_only_messages, SUMMARY_INPUT_CHAR_LIMIT)
        user_row = await db.get_user(user_id)
        user_label = f"@{user_row['username']}" if user_row and user_row.get("username") else f"id={user_id}"
        user_summary_input = (
            f"Целевой пользователь: {user_label} (id={user_id})\n\n"
            f"Текущее резюме:\n{current_user_summary or '(пусто)'}\n\n"
            f"Новые сообщения:\n{user_history_text}\n\n"
            f"Обнови резюме ТОЛЬКО про {user_label}. Игнорируй действия других участников."
        )
        try:
            new_user_summary = await llm.chat(
                USER_SUMMARY_SYSTEM_PROMPT,
                [{"role": "user", "content": user_summary_input}],
                temperature=0.2,
                top_p=1.0,
                max_tokens=320,
                turn_id=turn_id,
                turn_metrics=turn_metrics,
            )
            normalized_user_summary = new_user_summary.strip()
            if normalized_user_summary and normalized_user_summary != (current_user_summary or "").strip():
                await db.set_summary(chat_id, normalized_user_summary, user_id=user_id)
                updated = True
        except Exception:
            log.warning("failed to update user summary chat=%s user=%s", chat_id, user_id, exc_info=True)

    if updated and message_count > (MESSAGE_ROTATE_KEEP + MESSAGE_ROTATE_TRIGGER_DELTA):
        try:
            await db.rotate_messages(chat_id, keep=MESSAGE_ROTATE_KEEP)
        except Exception:
            log.warning("failed to rotate messages chat=%s", chat_id, exc_info=True)

    return updated


async def maybe_store_memories(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    turn_id: str | None = None,
    turn_metrics=None,
) -> dict[str, int]:
    text = _fit_text(input_text, MEMORY_EXTRACT_INPUT_CHAR_LIMIT)
    if _is_nonsemantic_input(text):
        return {"attempted": 0, "changed": 0}
    if not should_extract_user_facts(text):
        return {"attempted": 0, "changed": 0}

    try:
        raw = await llm.chat(
            MEMORY_EXTRACT_PROMPT,
            [{"role": "user", "content": text}],
            temperature=0.1,
            top_p=1.0,
            max_tokens=220,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("memory extraction LLM call failed", exc_info=True)
        return {"attempted": 0, "changed": 0}

    items = _extract_json_array(raw)[:MAX_FACT_ITEMS_PER_TURN]
    current_facts = _fact_record_map(await db.get_user_facts(chat_id, user_id, limit=128))
    attempted_count = 0
    changed_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        fact_type = _normalize_fact_type(item.get("fact_type"))
        value = _clean_fact_value(item.get("value"))
        if not value:
            continue
        if _is_toxic_fact_value(value):
            log.debug("fact value blocked by stop-pattern: %r", value[:80])
            continue
        normalized_value = _normalize_fact_value(value).lower()
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))
        evidence = _normalize_fact_evidence(item.get("evidence"))
        importance = _normalize_importance(fact_type, item.get("importance"), confidence)
        if importance < MEMORY_STORE_MIN_IMPORTANCE:
            continue
        signature = _fact_signature(fact_type, normalized_value)
        attempted_count += 1
        if _would_change_fact(
            current_facts.get(signature),
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            source="auto_extract",
        ):
            changed_count += 1
        await db.upsert_user_fact(
            chat_id, user_id, fact_type, value, normalized_value,
            confidence, importance, evidence, source="auto_extract",
        )
        current_facts[signature] = {
            "fact_type": fact_type, "value": value, "normalized_value": normalized_value,
            "confidence": confidence, "importance": importance, "evidence": evidence,
            "source": "auto_extract",
        }
    await db.prune_user_facts(chat_id, user_id, keep=FACT_KEEP_PER_USER)
    return {"attempted": attempted_count, "changed": changed_count}


async def maybe_store_user_profile(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    sender_username: str | None,
    sender_first_name: str | None,
    turn_id: str | None = None,
    turn_metrics=None,
) -> dict[str, int]:
    text = _fit_text(input_text, PROFILE_EXTRACT_INPUT_CHAR_LIMIT)
    if _is_nonsemantic_input(text):
        return {"attempted": 0, "changed": 0}
    if not should_extract_user_facts(text):
        return {"attempted": 0, "changed": 0}

    current_profile = await db.get_user_profile_merged(chat_id, user_id)
    current_profile_json = json.dumps(current_profile, ensure_ascii=False)
    payload = (
        f"Текущий профиль:\n{current_profile_json}\n\n"
        f"Telegram username: {sender_username or ''}\n"
        f"Telegram first_name: {sender_first_name or ''}\n\n"
        f"Сообщение пользователя:\n{text}"
    )

    try:
        raw = await llm.chat(
            PROFILE_EXTRACT_PROMPT,
            [{"role": "user", "content": payload}],
            temperature=0.1,
            top_p=1.0,
            max_tokens=260,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("profile extraction LLM call failed", exc_info=True)
        return {"attempted": 0, "changed": 0}

    profile = _extract_json_object(raw)
    if not profile:
        return {"attempted": 0, "changed": 0}

    try:
        confidence = float(profile.get("profile_confidence", current_profile.get("profile_confidence", 0.0)))
    except (TypeError, ValueError):
        confidence = current_profile.get("profile_confidence", 0.0)
    confidence = max(0.0, min(1.0, confidence))

    # ВАЖНО: авто-извлечение НЕ должно обнулять накопленные поля. Если в этом
    # сообщении интересов/стиля нет, LLM вернёт пусто — превращаем "" в None,
    # чтобы upsert сохранил текущее значение, а не затёр его пустотой.
    # (Ручная очистка профиля идёт через /profile и private_profile — другой путь.)
    # Интересы НАКАПЛИВАЕМ (старые + новые), а не заменяем — LLM возвращает только
    # интересы из текущего сообщения. Если новых нет, merged == current → None,
    # чтобы upsert не трогал поле.
    merged_interests = _merge_interests(current_profile.get("interests"), profile.get("interests"))
    current_interests_norm = str(current_profile.get("interests") or "").strip()
    interests = merged_interests if merged_interests and merged_interests != current_interests_norm else None
    communication_style = _clean_profile_text(profile.get("communication_style"), PROFILE_FIELD_CHAR_LIMIT) or None
    response_preference = _clean_profile_text(profile.get("response_preference"), PROFILE_FIELD_CHAR_LIMIT) or None

    await db.upsert_user_profile(
        chat_id, user_id,
        interests=interests,
        communication_style=communication_style,
        response_preference=response_preference,
        profile_confidence=confidence,
    )
    attempted_profile = {
        "interests": interests,
        "communication_style": communication_style,
        "response_preference": response_preference,
        "profile_confidence": confidence,
    }
    attempted_count = sum(1 for key, value in attempted_profile.items() if key == "profile_confidence" or value)
    changed_count, _ = _compute_profile_changes(current_profile, attempted_profile)
    return {"attempted": attempted_count, "changed": changed_count}


async def maybe_store_identity_from_telegram(
    db,
    chat_id: int,
    user_id: int,
    sender_username: str | None,
    sender_first_name: str | None,
):
    current_memories = _memory_record_map(await db.get_memories(chat_id, user_id, limit=64))
    current_profile = await db.get_user_profile_merged(chat_id, user_id)
    nickname = None
    display_name = None
    memory_items_attempted = 0
    memory_items_changed = 0

    if sender_username:
        nickname = f"@{sender_username.lstrip('@')}"
        memory_items_attempted += 1
        if _would_change_memory(
            current_memories.get("nickname"),
            value=nickname, confidence=0.35, importance=0.88, source="telegram_profile",
        ):
            memory_items_changed += 1
        await db.upsert_memory(
            chat_id, user_id, "nickname", nickname,
            confidence=0.35, importance=0.88, source="telegram_profile",
        )
        current_memories["nickname"] = {
            "key": "nickname", "value": nickname,
            "confidence": 0.35, "importance": 0.88, "source": "telegram_profile",
        }

    if sender_first_name:
        display_name = sender_first_name.strip()
        memory_items_attempted += 1
        if _would_change_memory(
            current_memories.get("preferred_name"),
            value=display_name, confidence=0.25, importance=0.9, source="telegram_profile",
        ):
            memory_items_changed += 1
        await db.upsert_memory(
            chat_id, user_id, "preferred_name", display_name,
            confidence=0.25, importance=0.9, source="telegram_profile",
        )
        current_memories["preferred_name"] = {
            "key": "preferred_name", "value": display_name,
            "confidence": 0.25, "importance": 0.9, "source": "telegram_profile",
        }

    profile_fields_attempted = 0
    profile_fields_changed = 0
    if nickname or display_name:
        candidate_profile = {
            "display_name": display_name if display_name is not None else current_profile.get("display_name"),
            "nickname": nickname if nickname is not None else current_profile.get("nickname"),
            "gender": current_profile.get("gender"),
            "interests": current_profile.get("interests", ""),
            "communication_style": current_profile.get("communication_style", ""),
            "response_preference": current_profile.get("response_preference", ""),
            "relationship_notes": current_profile.get("relationship_notes", ""),
            "profile_confidence": 0.2,
        }
        profile_fields_attempted = sum(
            1 for key, value in candidate_profile.items() if key == "profile_confidence" or value
        )
        profile_fields_changed, _ = _compute_profile_changes(current_profile, candidate_profile)
        await db.upsert_user_profile(
            chat_id, user_id, display_name=display_name, nickname=nickname, profile_confidence=0.2,
        )

    return {
        "memory_attempted": memory_items_attempted,
        "memory_changed": memory_items_changed,
        "profile_attempted": profile_fields_attempted,
        "profile_changed": profile_fields_changed,
    }


async def sync_user_document(
    db,
    chat_id: int,
    user_id: int,
    sender_username: str | None,
    sender_first_name: str | None,
):
    user_facts = await db.get_user_facts(chat_id, user_id, limit=FACT_KEEP_PER_USER)
    user_profile = await db.get_user_profile_merged(chat_id, user_id)
    payload = build_user_document_payload(user_facts, user_profile, sender_username, sender_first_name)
    await db.upsert_user_document(
        chat_id, user_id,
        document_json=json.dumps(payload, ensure_ascii=False),
        document_text=render_user_document_text(payload),
    )


async def should_ask_intro(
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    *,
    user_profile: dict | None = None,
    user_document: dict | None = None,
) -> bool:
    text = (input_text or "").strip()
    if _is_nonsemantic_input(text):
        return False

    profile = user_profile if user_profile is not None else await db.get_user_profile_merged(chat_id, user_id)
    has_display_name = bool(
        (profile.get("approved_display_name") or profile.get("display_name") or "").strip()
    )
    has_nickname = bool((profile.get("nickname") or "").strip())
    if has_display_name or has_nickname:
        return False

    doc = user_document if user_document is not None else await db.get_user_document(chat_id, user_id)
    if (doc.get("document_text") or "").strip():
        return False

    if await db.has_memory_key(chat_id, user_id, INTRO_MARKER_KEY):
        return False

    return True


# ---------------------------------------------------------------------------
# Background extraction pipeline
# ---------------------------------------------------------------------------

async def _build_extraction_text(db, chat_id: int, user_id: int, input_text: str) -> str:
    """Build extraction input with recent dialogue context.

    IMPORTANT: messages from OTHER users are clearly labeled to prevent
    cross-user fact contamination.
    """
    try:
        recent = await db.get_last_messages(chat_id, 8, focus_user_id=None)
    except Exception:
        return input_text

    context_lines = []
    for msg in reversed(recent):  # oldest first
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        msg_user_id = msg.get("user_id")
        if msg_user_id is None and role == "user":
            match = _HISTORY_PREFIX_USER_ID_RE.search(content)
            if match:
                try:
                    msg_user_id = int(match.group(1))
                except (TypeError, ValueError):
                    msg_user_id = None
        if role == "user" and content.endswith(input_text):
            continue
        if role == "assistant":
            prefix = "Альта"
        elif msg_user_id == user_id:
            prefix = "Пользователь"
        else:
            prefix = "Другой участник чата"
        context_lines.append(f"{prefix}: {_fit_text(content, 180)}")

    context_lines = context_lines[-5:]
    if not context_lines:
        return input_text

    context_block = "\n".join(context_lines)
    return (
        f"Контекст диалога (для понимания темы):\n{context_block}\n\n"
        f"Сообщение пользователя (извлекать факты ТОЛЬКО из него):\n{input_text}"
    )


async def _background_extraction(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    history_limit: int,
    sender_username: str | None,
    sender_first_name: str | None,
    sync_stats: dict,
    turn_id: str | None = None,
    turn_metrics=None,
):
    """Run memory/profile extraction and summary update in background after answer is sent."""
    try:
        extraction_text = await _build_extraction_text(db, chat_id, user_id, input_text)
        extracted_memory_stats, extracted_profile_stats = await asyncio.gather(
            maybe_store_memories(
                llm, db, chat_id, user_id, extraction_text,
                turn_id=turn_id, turn_metrics=turn_metrics,
            ),
            maybe_store_user_profile(
                llm, db, chat_id, user_id, extraction_text,
                sender_username, sender_first_name,
                turn_id=turn_id, turn_metrics=turn_metrics,
            ),
        )
        summary_updated = await maybe_update_summary(
            llm, db, chat_id, user_id, history_limit,
            turn_id=turn_id, turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.memory_items_written += int(sync_stats.get("memory_attempted", 0)) + int(
                extracted_memory_stats.get("attempted", 0)
            )
            turn_metrics.memory_items_changed += int(sync_stats.get("memory_changed", 0)) + int(
                extracted_memory_stats.get("changed", 0)
            )
            turn_metrics.memory_sync_items_changed += int(sync_stats.get("memory_changed", 0))
            turn_metrics.memory_extract_items_changed += int(extracted_memory_stats.get("changed", 0))
            turn_metrics.profile_fields_written += int(sync_stats.get("profile_attempted", 0)) + int(
                extracted_profile_stats.get("attempted", 0)
            )
            turn_metrics.profile_fields_changed += int(sync_stats.get("profile_changed", 0)) + int(
                extracted_profile_stats.get("changed", 0)
            )
            turn_metrics.profile_sync_fields_changed += int(sync_stats.get("profile_changed", 0))
            turn_metrics.profile_extract_fields_changed += int(extracted_profile_stats.get("changed", 0))
            turn_metrics.memory_updated = (
                turn_metrics.memory_updated or int(extracted_memory_stats.get("changed", 0)) > 0
            )
            turn_metrics.profile_updated = (
                turn_metrics.profile_updated or int(extracted_profile_stats.get("changed", 0)) > 0
            )
            turn_metrics.summary_updated = turn_metrics.summary_updated or summary_updated
            if summary_updated:
                turn_metrics.summary_write_count += 1
    except Exception:
        log.warning("background extraction failed chat=%s user=%s", chat_id, user_id, exc_info=True)
