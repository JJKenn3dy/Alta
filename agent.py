import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

log = logging.getLogger("alta.agent")

from file_context import build_file_context_block, retrieve_context_snippets, sync_context_dir
from llm import ContextOverflowError
from profile_context import load_profile, profile_to_system_block
from prompts import ALTA_SYSTEM_PROMPT
from telemetry import approx_messages_tokens, approx_text_tokens

SUMMARY_EVERY_N_MESSAGES = 10
SUMMARY_HISTORY_WINDOW = 80
MAX_FACT_ITEMS_PER_TURN = 6
INTRO_MARKER_KEY = "_intro_requested"
IDENTITY_KEYS = {"name", "preferred_name", "gender", "pronouns", "nickname"}
STYLE_KEYS = {"communication_style", "response_preference"}
HIGH_PRIORITY_KEYS = IDENTITY_KEYS | STYLE_KEYS
MEMORY_STORE_MIN_IMPORTANCE = 0.45
FACT_KEEP_PER_USER = 80
FACT_CONTEXT_MIN_IMPORTANCE = 0.45
CONTEXT_HISTORY_LIMIT = 30
SUMMARY_CONTEXT_CHAR_LIMIT = 700
PROFILE_FIELD_CHAR_LIMIT = 200
RELATIONSHIP_NOTES_CHAR_LIMIT = 250
MAX_FACT_LINES_IN_CONTEXT = 12
SUMMARY_INPUT_CHAR_LIMIT = 6_000
MEMORY_EXTRACT_INPUT_CHAR_LIMIT = 1_800
PROFILE_EXTRACT_INPUT_CHAR_LIMIT = 2_200
HISTORY_MESSAGE_CHAR_LIMIT = 900
DEFAULT_CONTEXT_CHAR_BUDGET = 12_000
COMPACT_CONTEXT_CHAR_BUDGET = 7_000
MINIMAL_CONTEXT_CHAR_BUDGET = 3_500
EMERGENCY_CONTEXT_CHAR_BUDGET = 1_500
DEFAULT_FILE_CONTEXT_CHUNKS = 3
COMPACT_FILE_CONTEXT_CHUNKS = 2
MINIMAL_FILE_CONTEXT_CHUNKS = 1
DEFAULT_FILE_CONTEXT_CHARS = 1_800
COMPACT_FILE_CONTEXT_CHARS = 1_000
MINIMAL_FILE_CONTEXT_CHARS = 500
USER_DOCUMENT_CHAR_LIMIT = 900
USER_DOCUMENT_FACT_LIMIT = 20
MAX_CHAT_PARTICIPANTS = 8
REPEAT_GUARD_WINDOW = 6
REPEAT_GUARD_MAX_CHARS = 140
REPEAT_GUARD_MIN_KEY_CHARS = 3
REPEAT_GUARD_SHORT_EXACT_MIN_KEY_CHARS = 2
REPEAT_GUARD_SUBSTRING_MAX_CHARS = 28

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

USER_DOCUMENT_TRIGGER_PATTERNS = (
    re.compile(r"\b(кто я|что ты обо мне|что знаешь обо мне|что помнишь обо мне)\b", re.IGNORECASE),
    re.compile(r"\b(как меня зовут|как ко мне обращаться|обращайся ко мне)\b", re.IGNORECASE),
    re.compile(r"\b(мой|моя|мне|обо мне|про меня|меня)\b", re.IGNORECASE),
    re.compile(r"\b(интерес|люблю|предпочитаю|нравит)\b", re.IGNORECASE),
)
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


@dataclass(frozen=True)
class ContextPlan:
    name: str
    history_messages: int
    history_chars: int
    summary_chars: int
    memory_lines: int
    file_chunks: int
    file_chars: int
    max_tokens: int
    include_creator_profile: bool
    include_participants: bool = True


@dataclass(frozen=True)
class ReplyContext:
    user_id: int | None
    username: str | None
    first_name: str | None
    text: str


@dataclass(frozen=True)
class RouteTurn:
    text_clean: str
    has_file: bool

    @property
    def chars(self) -> int:
        return len(self.text_clean or "")


def route_mode(turn: RouteTurn) -> str:
    if turn.chars <= 50 and not turn.has_file:
        return "DIRECT"
    if turn.chars < 1000:
        return "NORMAL"
    return "HEAVY"


def route_reason(turn: RouteTurn) -> str:
    if turn.chars <= 50 and not turn.has_file:
        return "chars<=50 and no_file"
    if turn.chars <= 50:
        return "chars<=50 but has_file"
    if turn.chars < 1000:
        return "chars<1000"
    return "chars>=1000"

INTRO_GENERATION_PROMPT = """
Сгенерируй 1-2 коротких естественных предложения знакомства на русском для чата.
Цель: мягко попросить человека рассказать о себе: имя или ник, как к нему обращаться, пол, интересы.
Пиши живо, не шаблонно, без эмодзи и без канцелярита.
Верни только сам текст.
""".strip()

REPEAT_REWRITE_PROMPT = """
Ты переписываешь ответ чат-бота, чтобы сломать цикл повторений.
Сохрани язык, смысл и тон черновика.
Не используй запрещённые фразы дословно.
Верни только один короткий ответ (1-2 предложения), без эмодзи, без списков.
""".strip()

LEGACY_STYLE_PROMPT_ARCHIVE = """
Ты приводишь черновой ответ к стабильному голосу Альты.
Альта — женщина. Все формы строго в женском роде (рада, поняла, написала, была, пришла и т.д.).

Правила:
- Сохрани смысл и длину черновика, но перепиши живее и жестче.
- Длина ответа должна соответствовать сложности вопроса: на простое - коротко, на развёрнутое - развёрнуто.
- Если в черновике 3-4 предложения и они по делу, оставь 3-4 предложения.
- Никаких эмодзи.
- Не называй себя ИИ, ботом, моделью или ассистентом.
- Говори сухо, прямо, современно, по-человечески.
- Допустимы ирония, подкол, грубоватость и мат, если это уместно по тону сообщения.
- Не придумывай абсурдных сравнений, мемных метафор и случайных панчлайнов без опоры на сообщение.
- Если смысл неясен, лучше коротко уточни, чем фантазируй.
- Не смягчай ответ без причины и не делай его "сервисным".
- Не добавляй новых фактов.
- Если черновик уже нормальный, верни его как есть.
- НЕ сокращай ответ если он информативный и по делу.

Верни только финальную реплику.
""".strip()


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
Извлеки только устойчивые факты о пользователе из сообщения.
Верни строго JSON-массив объектов формата:
[{"fact_type":"...", "value":"...", "confidence":0.0, "importance":0.0, "evidence":"..."}]

Правила:
- Только личные факты и предпочтения, которые пригодятся позже.
- Сообщение может быть из группового чата: извлекай факты только про автора текущего сообщения, а не про людей, которых он упомянул.
- Не извлекай имя, ник, пол, местоимения, стиль ответа и прочие identity-поля: они обрабатываются отдельно.
- Используй только fact_type из набора: interest, preference, work, study, location, goal, relationship, bio, fact.
- evidence: короткая цитата или опора из сообщения, максимум 120 символов.
- importance от 0.0 до 1.0: 1.0 это факт, который стоит помнить очень долго; 0.0 это почти мусор.
- Не сохраняй как важное временные мелочи вроде "сейчас ем", "сегодня устал", если это не ключ к личности.
- Не сохраняй случайный шум, сиюминутные эмоции, мемные вбросы без устойчивого смысла и факты о третьих лицах.
- Не добавляй временный шум.
- Максимум 6 объектов.
- confidence от 0.0 до 1.0.
- Если извлекать нечего, верни [].
""".strip()

PROFILE_EXTRACT_PROMPT = """
Ты обновляешь устойчивый профиль человека по одному сообщению.
Верни строго JSON-объект формата:
{
  "display_name": "string or empty",
  "nickname": "string or empty",
  "gender": "string or empty",
  "interests": ["..."],
  "communication_style": "string or empty",
  "response_preference": "string or empty",
  "relationship_notes": "string or empty",
  "profile_confidence": 0.0
}

Правила:
- Обновляй только то, что действительно можно вывести из сообщения и уже имеющегося профиля.
- Сообщение может быть из группового чата: обновляй профиль только автора текущего сообщения, не людей, которых он упомянул.
- Если поле неясно, оставь пустую строку или пустой массив.
- interests: 0-4 коротких интереса.
- communication_style: как человек обычно пишет или как хочет, чтобы с ним говорили. Очень коротко.
- response_preference: например "коротко", "по делу", "с юмором", "технически". Очень коротко.
- relationship_notes: только полезные заметки о динамике общения. Максимум 1 короткая фраза.
- Не записывай мусорные значения вроде "ты", "он", "она", "неизвестно", "неопределено".
- Никаких выдумок.
""".strip()

CREATOR_PROFILE_PATTERNS = (
    re.compile(r"\bкто тебя (создал|сделал)\b"),
    re.compile(r"\bкто твой (создатель|владелец|хозяин|owner|creator)\b"),
    re.compile(r"\bтвой (создатель|владелец|хозяин|owner|creator)\b"),
    re.compile(r"\bзахар\b.*\b(создал|сделал|владелец|хозяин)\b"),
    re.compile(r"\b(создал|сделал|владелец|хозяин)\b.*\bзахар\b"),
)
INVALID_IDENTITY_VALUES = {
    "ты",
    "тебя",
    "тебе",
    "твой",
    "твоя",
    "он",
    "она",
    "они",
    "неизвестно",
    "неизвестен",
    "неизвестна",
    "неопределено",
    "не определено",
    "unknown",
    "none",
    "null",
    "никак",
}
INVALID_GENDER_VALUES = INVALID_IDENTITY_VALUES | {
    "не знаю",
    "не указан",
    "не указано",
}

MIN_EDITOR_DRAFT_CHARS = 15
MAX_EDITOR_REPLY_CHARS = 700
MAX_EDITOR_SENTENCES = 8
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAD6"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]",
    re.UNICODE,
)
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
REPLY_REPEAT_PREFIX_PATTERN = re.compile(r"^\s*\[(?:to[^]]+|id=\d+[^]]*)\]\s*", re.IGNORECASE)
REPLY_REPEAT_TOKEN_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
FORBIDDEN_SELF_REFERENCE_PATTERNS = (
    # Only match self-identification: "я бот", "я ИИ", "я модель", "я ассистент"
    re.compile(r"\bя\s+(?:ии|ai|бот|модель|ассистент)\b", re.IGNORECASE),
    re.compile(r"\b(?:как|будучи)\s+(?:ии|ai|бот(?:ом)?|модель(?:ю)?|ассистент(?:ом)?)\b", re.IGNORECASE),
    re.compile(r"\bя\s+(?:языковая|нейро)\s*(?:модель|сеть)\b", re.IGNORECASE),
)
# Masculine → feminine self-reference fixes.
# Only matches first-person context: after "я" or at the very start of the reply
# (sentence-initial = Alta speaking about herself).
# _P prefix: pattern after "я ", _S prefix: sentence-start pattern.
_GENDER_FIX_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "я <masc>" → "я <fem>"
    (re.compile(r"\bя\s+рад\b", re.IGNORECASE), "я рада"),
    (re.compile(r"\bя\s+готов\b", re.IGNORECASE), "я готова"),
    (re.compile(r"\bя\s+понял\b", re.IGNORECASE), "я поняла"),
    (re.compile(r"\bя\s+сделал\b", re.IGNORECASE), "я сделала"),
    (re.compile(r"\bя\s+написал\b", re.IGNORECASE), "я написала"),
    (re.compile(r"\bя\s+пришёл\b", re.IGNORECASE), "я пришла"),
    (re.compile(r"\bя\s+пошёл\b", re.IGNORECASE), "я пошла"),
    (re.compile(r"\bя\s+был\b", re.IGNORECASE), "я была"),
    (re.compile(r"\bя\s+сказал\b", re.IGNORECASE), "я сказала"),
    (re.compile(r"\bя\s+увидел\b", re.IGNORECASE), "я увидела"),
    (re.compile(r"\bя\s+решил\b", re.IGNORECASE), "я решила"),
    (re.compile(r"\bя\s+хотел\b", re.IGNORECASE), "я хотела"),
    (re.compile(r"\bя\s+знал\b", re.IGNORECASE), "я знала"),
    (re.compile(r"\bя\s+думал\b", re.IGNORECASE), "я думала"),
    (re.compile(r"\bя\s+забыл\b", re.IGNORECASE), "я забыла"),
    (re.compile(r"\bя\s+узнал\b", re.IGNORECASE), "я узнала"),
    (re.compile(r"\bя\s+нашёл\b", re.IGNORECASE), "я нашла"),
    (re.compile(r"\bя\s+успел\b", re.IGNORECASE), "я успела"),
    # Sentence-start (Alta about herself): "Рад, что..." → "Рада, что..."
    (re.compile(r"(?:^|(?<=[.!?]\s))Рад\b"), "Рада"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Готов\b"), "Готова"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Понял\b"), "Поняла"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Сделал\b"), "Сделала"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Написал\b"), "Написала"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Решил\b"), "Решила"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Думал\b"), "Думала"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Забыл\b"), "Забыла"),
    (re.compile(r"(?:^|(?<=[.!?]\s))Узнал\b"), "Узнала"),
]


def _fix_gender(text: str) -> str:
    """Replace masculine self-reference verb forms with feminine (first-person context only)."""
    for pattern, replacement in _GENDER_FIX_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


OVER_PERSONA_STRONG_PATTERNS = (
    re.compile(r"[!?]{3,}"),
    re.compile(r"\*[^*]{1,80}\*"),
    re.compile(r"([а-яё])\1{3,}", re.IGNORECASE),
)
OVER_PERSONA_SOFT_PATTERNS = (
    re.compile(r"\b(?:с удовольствием|буду рада помочь|всегда к услугам)\b", re.IGNORECASE),
    re.compile(r"\b(?:чем.{0,5}помочь|обращайся|не стесняйся)\b", re.IGNORECASE),
)

def _build_context_plans(max_tokens: int, history_limit: int) -> list[ContextPlan]:
    safe_max_tokens = max(48, int(max_tokens))
    return [
        ContextPlan(
            name="full",
            history_messages=min(history_limit, CONTEXT_HISTORY_LIMIT),
            history_chars=DEFAULT_CONTEXT_CHAR_BUDGET,
            summary_chars=SUMMARY_CONTEXT_CHAR_LIMIT,
            memory_lines=MAX_FACT_LINES_IN_CONTEXT,
            file_chunks=DEFAULT_FILE_CONTEXT_CHUNKS,
            file_chars=DEFAULT_FILE_CONTEXT_CHARS,
            max_tokens=safe_max_tokens,
            include_creator_profile=True,
        ),
        ContextPlan(
            name="compact",
            history_messages=min(history_limit, 20),
            history_chars=COMPACT_CONTEXT_CHAR_BUDGET,
            summary_chars=min(450, SUMMARY_CONTEXT_CHAR_LIMIT),
            memory_lines=4,
            file_chunks=COMPACT_FILE_CONTEXT_CHUNKS,
            file_chars=COMPACT_FILE_CONTEXT_CHARS,
            max_tokens=min(safe_max_tokens, 280),
            include_creator_profile=True,
        ),
        ContextPlan(
            name="minimal",
            history_messages=min(history_limit, 10),
            history_chars=MINIMAL_CONTEXT_CHAR_BUDGET,
            summary_chars=min(300, SUMMARY_CONTEXT_CHAR_LIMIT),
            memory_lines=2,
            file_chunks=MINIMAL_FILE_CONTEXT_CHUNKS,
            file_chars=MINIMAL_FILE_CONTEXT_CHARS,
            max_tokens=min(safe_max_tokens, 180),
            include_creator_profile=True,
            include_participants=False,
        ),
        ContextPlan(
            name="emergency",
            history_messages=min(history_limit, 2),
            history_chars=EMERGENCY_CONTEXT_CHAR_BUDGET,
            summary_chars=120,
            memory_lines=0,
            file_chunks=0,
            file_chars=0,
            max_tokens=min(safe_max_tokens, 120),
            include_creator_profile=False,
            include_participants=False,
        ),
    ]


def _history_line(role: str, content: str) -> str:
    return f"{role}: {content}"


def _is_nonsemantic_input(text: Any) -> bool:
    normalized = str(text or "").strip().lower()
    return normalized in {"", "ping"}


TIME_GAP_THRESHOLD_MINUTES = 30


def _parse_ts(ts_str: str) -> datetime | None:
    """Parse a timestamp string from the DB."""
    if not ts_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def _insert_time_gaps(messages: list[dict]) -> list[dict]:
    """Insert gap markers when there's a significant pause between messages."""
    if len(messages) < 2:
        return messages

    result: list[dict] = [messages[0]]
    for i in range(1, len(messages)):
        prev_ts = _parse_ts(messages[i - 1].get("ts", ""))
        curr_ts = _parse_ts(messages[i].get("ts", ""))
        if prev_ts and curr_ts:
            gap_minutes = (curr_ts - prev_ts).total_seconds() / 60
            if gap_minutes >= TIME_GAP_THRESHOLD_MINUTES:
                if gap_minutes >= 1440:  # 24+ hours
                    gap_text = f"[... прошло {int(gap_minutes // 1440)} дн. ...]"
                elif gap_minutes >= 60:
                    gap_text = f"[... прошло {int(gap_minutes // 60)} ч. ...]"
                else:
                    gap_text = f"[... прошло {int(gap_minutes)} мин. ...]"
                result.append({"role": "system", "content": gap_text})
        result.append(messages[i])
    return result


def _merge_histories(primary: list[dict], secondary: list[dict], limit: int, char_budget: int) -> list[dict]:
    merged: list[dict] = []
    seen_ids: set[int] = set()

    for item in primary + secondary:
        message_id = int(item.get("id") or 0)
        if message_id and message_id in seen_ids:
            continue
        if message_id:
            seen_ids.add(message_id)
        merged.append(item)

    merged.sort(key=lambda item: int(item.get("id") or 0))
    trimmed = _take_history_with_budget(merged, limit, char_budget)
    return _insert_time_gaps(trimmed)


def _format_reply_author(reply_context: ReplyContext) -> str:
    if reply_context.username:
        return f"@{reply_context.username}"
    if reply_context.first_name:
        return reply_context.first_name
    if reply_context.user_id is not None:
        return f"id={reply_context.user_id}"
    return "unknown"


def build_reply_context_block(reply_context: ReplyContext | None) -> str:
    if not reply_context:
        return ""

    reply_text = _fit_text(reply_context.text, 400)
    if not reply_text:
        return ""

    return (
        "Текущее сообщение является ответом на конкретную реплику в чате.\n"
        f"Автор исходной реплики: {_format_reply_author(reply_context)}\n"
        f"Текст исходной реплики: {reply_text}\n"
        "Учитывай это при интерпретации коротких и зависимых от контекста ответов."
    ).strip()


def _looks_like_context_dependent_reply(text: str) -> bool:
    value = " ".join(str(text or "").strip().lower().split())
    if not value:
        return False
    if len(value) > 48:
        return False
    words = value.split()
    if len(words) > 6:
        return False
    return True


def build_implicit_reply_context_block(history: list[dict], input_text: str) -> str:
    if not _looks_like_context_dependent_reply(input_text):
        return ""
    if len(history) < 2:
        return ""

    last_message = history[-1]
    prev_message = history[-2]
    if last_message.get("role") != "user":
        return ""
    if prev_message.get("role") != "assistant":
        return ""

    prev_bot_text = _fit_text(prev_message.get("content", ""), 300)
    if not prev_bot_text:
        return ""

    # Find the user's own previous message (before the bot reply)
    prev_user_text = ""
    for i in range(len(history) - 3, -1, -1):
        if history[i].get("role") == "user":
            prev_user_text = _fit_text(history[i].get("content", ""), 200)
            break

    parts = [
        "Текущее сообщение короткое и зависит от контекста предыдущих реплик.",
    ]
    if prev_user_text:
        parts.append(f"Предыдущее сообщение пользователя: {prev_user_text}")
    parts.append(f"Предыдущая реплика бота: {prev_bot_text}")
    parts.append(
        "Считай текущее сообщение прежде всего ответом, уточнением или продолжением этого диалога, "
        "а не новой отдельной темой."
    )
    return "\n".join(parts)


def build_mention_only_block(mention_only: bool) -> str:
    if not mention_only:
        return ""
    return (
        "Пользователь обратился к тебе по имени, но не написал содержательный текст.\n"
        "Ответь коротко и естественно, как на оклик или привлечение внимания.\n"
        "Не воспринимай это как слово 'ping' и не отвечай 'понг'."
    ).strip()


def _fit_text(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3].rstrip() + "..."


def _shrink_reply_text(text: str, limit: int = 400, max_sentences: int = 5) -> str:
    value = " ".join((text or "").strip().split())
    if not value:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", value)
    compact = " ".join(part.strip() for part in parts[:max_sentences] if part.strip())
    if compact:
        value = compact

    return _fit_text(value, limit)


def _normalize_reply_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _reply_repeat_key(text: str) -> str:
    value = _normalize_reply_text(text).lower()
    if not value:
        return ""
    value = REPLY_REPEAT_PREFIX_PATTERN.sub("", value)
    value = REPLY_REPEAT_TOKEN_PATTERN.sub(" ", value)
    return " ".join(value.split())


def _looks_like_same_reply_key(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter = left if len(left) <= len(right) else right
    longer = right if len(left) <= len(right) else left
    if len(shorter) < REPEAT_GUARD_MIN_KEY_CHARS:
        return False
    if len(shorter) > REPEAT_GUARD_SUBSTRING_MAX_CHARS:
        return False
    return shorter in longer


def _recent_assistant_replies(messages: list[dict], limit: int = REPEAT_GUARD_WINDOW) -> list[str]:
    replies: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content", "")).strip()
        if text:
            replies.append(text)
    if limit <= 0:
        return replies
    return replies[-limit:]


def _is_repeated_recent_reply(answer: str, recent_assistant_replies: list[str]) -> bool:
    candidate = _normalize_reply_text(answer)
    if not candidate:
        return False
    if len(candidate) > REPEAT_GUARD_MAX_CHARS:
        return False

    candidate_key = _reply_repeat_key(candidate)
    recent_keys: list[str] = []
    for item in recent_assistant_replies:
        key = _reply_repeat_key(item)
        if len(key) >= REPEAT_GUARD_SHORT_EXACT_MIN_KEY_CHARS:
            recent_keys.append(key)
    if not recent_keys:
        return False

    # Handle ultra-short replies with exact matching only ("нет", "да", "ок").
    if len(candidate_key) < REPEAT_GUARD_MIN_KEY_CHARS:
        if len(candidate_key) < REPEAT_GUARD_SHORT_EXACT_MIN_KEY_CHARS:
            return False
        if candidate_key == recent_keys[-1]:
            return True
        repeats = sum(1 for key in recent_keys if key == candidate_key)
        return repeats >= 2

    if _looks_like_same_reply_key(candidate_key, recent_keys[-1]):
        return True

    repeats = sum(1 for key in recent_keys if _looks_like_same_reply_key(candidate_key, key))
    return repeats >= 2


def _repeat_guard_hard_fallback(input_text: str) -> str:
    normalized = _normalize_reply_text(input_text)
    if not normalized:
        return "Скажи конкретнее, что нужно."
    if "?" in normalized:
        return "Уточни вопрос чуть конкретнее."
    return "Давай по делу: что именно нужно?"


def _sentence_count(text: str) -> int:
    value = _normalize_reply_text(text)
    if not value:
        return 0
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]
    return max(1, len(parts))


def _contains_emoji(text: str) -> bool:
    return bool(EMOJI_PATTERN.search(text or ""))


def _has_forbidden_self_reference(text: str) -> bool:
    value = _normalize_reply_text(text)
    if not value:
        return False
    return any(pattern.search(value) for pattern in FORBIDDEN_SELF_REFERENCE_PATTERNS)


def _has_format_violation(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    value = _normalize_reply_text(raw)
    if _contains_emoji(value):
        return True
    if _has_forbidden_self_reference(value):
        return True
    if len(value) > MAX_EDITOR_REPLY_CHARS:
        return True
    if _sentence_count(value) > MAX_EDITOR_SENTENCES:
        return True
    if LIST_PREFIX_PATTERN.search(raw) is not None:
        return True
    return False


def _has_over_persona(text: str) -> bool:
    value = _normalize_reply_text(text)
    if not value or len(value) < MIN_EDITOR_DRAFT_CHARS:
        return False
    if any(pattern.search(value) for pattern in OVER_PERSONA_STRONG_PATTERNS):
        return True
    soft_hits = sum(1 for pattern in OVER_PERSONA_SOFT_PATTERNS if pattern.search(value))
    return soft_hits >= 2


def _editor_triggers_for(draft_answer: str) -> list[str]:
    raw = str(draft_answer or "").strip()
    if not raw:
        return ["empty_draft"]

    value = _normalize_reply_text(raw)
    triggers: list[str] = []
    if len(value) < MIN_EDITOR_DRAFT_CHARS:
        triggers.append("short_draft")
    if _has_format_violation(raw):
        triggers.append("format_violation")
    if _has_over_persona(raw):
        triggers.append("over_persona")
    return triggers


def _editor_safe_output(draft_answer: str, candidate_answer: str) -> str:
    draft = _normalize_reply_text(draft_answer)
    if not draft:
        return ""

    candidate = _normalize_reply_text(candidate_answer)
    if not candidate:
        return _fix_gender(draft)
    max_candidate_len = int(len(draft) * 1.3)
    if len(draft) <= 24:
        # Allow short drafts to expand into a meaningful 1-2 sentence reply.
        max_candidate_len = max(max_candidate_len, len(draft) + 80)
    if len(candidate) > max_candidate_len:
        return _fix_gender(draft)
    if _contains_emoji(candidate) or _has_forbidden_self_reference(candidate):
        return _fix_gender(draft)
    if _sentence_count(candidate) > max(MAX_EDITOR_SENTENCES, _sentence_count(draft) + 1):
        return _fix_gender(draft)
    return _fix_gender(candidate)


def _split_reply_sentences(text: str) -> list[str]:
    value = _normalize_reply_text(text)
    if not value:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]


def _strip_list_prefixes(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = LIST_PREFIX_PATTERN.sub("", raw_line).strip()
        if line:
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)


def _remove_forbidden_self_reference(text: str) -> str:
    value = text
    value = re.sub(r"\bкак\s+(?:ии|ai|бот(?:ом)?|модель(?:ю)?|ассистент(?:ом)?)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bбудучи\s+(?:ии|ai|бот(?:ом)?|модель(?:ю)?|ассистент(?:ом)?)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bя\s+(?:ии|ai|бот|модель|ассистент)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\bя\s+(?:языковая|нейро)\s*(?:модель|сеть)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value if value else text


def _deterministic_postprocess_answer(draft_answer: str) -> str:
    raw = str(draft_answer or "").strip()
    if not raw:
        return ""

    value = _strip_list_prefixes(raw) if "\n" in raw or LIST_PREFIX_PATTERN.search(raw) else raw
    value = EMOJI_PATTERN.sub("", value)
    # Replace em-dash / en-dash with a regular hyphen surrounded by spaces
    value = re.sub(r"\s*[—–]\s*", " - ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*([,.:;!?])\s*", r"\1 ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"([!?])\1{2,}", r"\1\1", value)
    value = re.sub(r"\.{4,}", "...", value)

    if _has_forbidden_self_reference(value):
        value = _remove_forbidden_self_reference(value)

    if _has_over_persona(value):
        value = re.sub(r"\*[^*]{1,80}\*", "", value)
        value = re.sub(r"([А-Яа-яЁё])\1{3,}", r"\1\1", value, flags=re.IGNORECASE)
        value = re.sub(r"[!?]{3,}", "!!", value)
        value = _normalize_reply_text(value)

    value = _fix_gender(value)
    return _shrink_reply_text(value, limit=MAX_EDITOR_REPLY_CHARS, max_sentences=MAX_EDITOR_SENTENCES)


def _shrink_message(message: dict, limit: int = HISTORY_MESSAGE_CHAR_LIMIT) -> dict:
    return {
        "role": message["role"],
        "content": _fit_text(message.get("content", ""), limit),
    }


def _take_history_with_budget(history: list[dict], max_messages: int, max_chars: int) -> list[dict]:
    if not history or max_messages <= 0 or max_chars <= 0:
        return []

    selected: list[dict] = []
    skipped_count = 0
    chars_used = 0
    recent = list(reversed(history[-max_messages:]))

    for index, message in enumerate(recent):
        remaining = max_chars - chars_used
        if remaining <= 80:
            skipped_count += len(recent) - index
            break

        per_message_limit = min(HISTORY_MESSAGE_CHAR_LIMIT, max(120, remaining - 24))
        shrunk = _shrink_message(message, per_message_limit)
        cost = len(shrunk["content"]) + 24
        if chars_used + cost > max_chars:
            skipped_count += 1
            continue

        selected.append(shrunk)
        chars_used += cost

    result = list(reversed(selected))

    if skipped_count > 0 and result:
        gap_note = {
            "role": "system",
            "content": f"[... пропущено {skipped_count} сообщений из-за лимита контекста ...]",
        }
        result.insert(0, gap_note)

    return result


def _history_as_text(history: list[dict], max_chars: int) -> str:
    selected = _take_history_with_budget(history, len(history), max_chars)
    return "\n".join(_history_line(item["role"], item["content"]) for item in selected)


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


def _trusted_sender_identity(
    user_profile: dict[str, Any],
    sender_username: str | None,
    sender_first_name: str | None,
) -> tuple[str | None, str | None]:
    approved_display_name = _clean_identity_profile_text(
        user_profile.get("approved_display_name"),
        limit=80,
        field="display_name",
    )
    display_name = approved_display_name or _clean_identity_profile_text(
        sender_first_name,
        limit=80,
        field="display_name",
    )
    nickname = None
    if sender_username:
        nickname = _clean_identity_profile_text(
            f"@{sender_username.lstrip('@')}",
            limit=80,
            field="nickname",
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
    if not document_text.strip():
        return False
    normalized = " ".join((input_text or "").lower().split())
    return any(pattern.search(normalized) for pattern in USER_DOCUMENT_TRIGGER_PATTERNS)


def should_extract_user_facts(input_text: str) -> bool:
    normalized = " ".join((input_text or "").lower().split())
    if len(normalized) < 10:
        return False
    if any(pattern.search(normalized) for pattern in EXPLICIT_PROFILE_PATTERNS):
        return True
    if "?" in normalized:
        return False

    first_person_markers = (
        "я ",
        "я ",
        "мне ",
        "у меня",
        "мой ",
        "моя ",
        "мои ",
        "для меня",
    )
    fact_markers = (
        "люблю",
        "нравится",
        "предпочитаю",
        "увлекаюсь",
        "занимаюсь",
        "работаю",
        "учусь",
        "живу",
        "изучаю",
        "интересуюсь",
        "хочу",
        "не люблю",
        "не нравится",
    )
    has_first_person = any(marker in normalized for marker in first_person_markers)
    has_fact_marker = any(marker in normalized for marker in fact_markers)
    return has_first_person or has_fact_marker


def _normalize_compact_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_invalid_identity_value(value: Any) -> bool:
    return _normalize_compact_text(value) in INVALID_IDENTITY_VALUES


def _is_invalid_gender_value(value: Any) -> bool:
    return _normalize_compact_text(value) in INVALID_GENDER_VALUES


def _clean_fact_value(value: Any) -> str | None:
    text = _normalize_fact_value(value)
    if not text:
        return None
    if len(text) < 2:
        return None
    lowered = text.lower()
    blocked = {
        "да",
        "нет",
        "ага",
        "ок",
        "норм",
        "ничего",
        "не знаю",
        "хз",
        "лол",
    }
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


def _clean_identity_profile_text(
    value: Any,
    *,
    limit: int,
    field: str,
) -> str | None:
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


def _is_creator_profile_query(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    return any(pattern.search(normalized) for pattern in CREATOR_PROFILE_PATTERNS)


def build_identity_block(
    user_profile: dict,
    sender_username: str | None,
    sender_first_name: str | None,
) -> str:
    preferred_name, nickname = _trusted_sender_identity(user_profile, sender_username, sender_first_name)
    gender = _clip_text(user_profile.get("gender"), PROFILE_FIELD_CHAR_LIMIT)
    interests = _clip_text(user_profile.get("interests"), PROFILE_FIELD_CHAR_LIMIT)
    communication_style = _clip_text(
        user_profile.get("communication_style"),
        PROFILE_FIELD_CHAR_LIMIT,
    )
    response_preference = _clip_text(
        user_profile.get("response_preference"),
        PROFILE_FIELD_CHAR_LIMIT,
    )
    relationship_notes = _clip_text(
        user_profile.get("relationship_notes"),
        RELATIONSHIP_NOTES_CHAR_LIMIT,
    )

    lines = []
    # Skip preferred_name if it's just the telegram first_name (already in current_speaker block)
    if preferred_name and preferred_name.strip() != (sender_first_name or "").strip():
        lines.append(f"- Предпочтительное обращение: {preferred_name}")
    # Skip nickname — already shown as @username in current_speaker block
    if gender:
        lines.append(f"- Пол: {gender}")
    if interests:
        lines.append(f"- Интересы: {interests}")
    if communication_style:
        lines.append(f"- Как человек общается: {communication_style}")
    if response_preference:
        lines.append(f"- Как ему лучше отвечать: {response_preference}")
    if relationship_notes:
        lines.append(f"- Заметка о динамике общения: {relationship_notes}")

    if not lines:
        return ""

    return (
        "Профиль текущего собеседника:\n"
        + "\n".join(lines)
        + "\n"
        + "Учитывай это в обращениях, интонации и длине ответа. Не проговаривай этот блок пользователю."
    )


def build_current_speaker_block(
    user_id: int,
    sender_username: str | None,
    sender_first_name: str | None,
) -> str:
    lines = [
        "Текущий собеседник для этого ответа:",
        f"- user_id: {user_id}",
    ]
    if sender_username:
        lines.append(f"- username: @{sender_username.lstrip('@')}")
    if sender_first_name:
        lines.append(f"- telegram_first_name: {sender_first_name.strip()}")
    lines.extend(
        [
            "- Отвечай именно этому человеку.",
            "- Никогда не переноси имя, ник, пол, интересы и факты другого участника на текущего собеседника.",
            "- Если есть сомнение в имени или обращении, используй нейтральный ответ без имени.",
        ]
    )
    return "\n".join(lines)


def build_chat_participants_block(
    participants: list[dict],
    current_user_id: int,
) -> str:
    """Build a system block listing other recently active chat participants.

    Only includes public profile data (name, gender, interests).
    Private conversation summaries are never shared between users.
    """
    others = [p for p in participants if p.get("user_id") != current_user_id]
    if not others:
        return ""

    lines = [
        "Другие недавно активные участники этого чата:"
    ]
    for p in others:
        name_parts = []
        display = (p.get("display_name") or "").strip()
        username = (p.get("username") or "").strip()
        if display:
            name_parts.append(display)
        if username:
            name_parts.append(f"@{username}")
        identifier = " / ".join(name_parts) if name_parts else f"id={p['user_id']}"

        detail_parts = []
        gender = (p.get("gender") or "").strip()
        if gender:
            detail_parts.append(f"пол: {gender}")
        interests = (p.get("interests") or "").strip()
        if interests:
            detail_parts.append(f"интересы: {_clip_text(interests, 80)}")

        detail = "; ".join(detail_parts)
        if detail:
            lines.append(f"- {identifier} ({detail})")
        else:
            lines.append(f"- {identifier}")

    lines.append(
        "Не путай факты этих людей с фактами текущего собеседника."
    )
    return "\n".join(lines)


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


def should_include_profile(input_text: str, user_id: int, creator_user_id: int | None) -> bool:
    if creator_user_id is not None and user_id == creator_user_id:
        return True

    return _is_creator_profile_query(input_text)


def _clean_profile_text(value: Any, limit: int = 240) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:limit]


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _join_list(value: Any, limit: int = 4) -> str:
    if not isinstance(value, list):
        return ""
    items = []
    for x in value[:limit]:
        s = str(x).strip()
        if s:
            items.append(s)
    return ", ".join(items)


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
        "interests",
        "job",
        "work",
        "hobby",
        "relationship_notes",
        "interest",
        "preference",
        "study",
        "location",
        "goal",
        "relationship",
        "bio",
        "fact",
    }:
        score = max(score, 0.65)
    return score


async def maybe_update_summary(
    llm,
    db,
    chat_id: int,
    user_id: int,
    history_limit: int,
    turn_id: str | None = None,
    turn_metrics=None,
) -> bool:
    updated = False
    message_count = await db.count_messages(chat_id)
    user_msg_count = await db.count_user_messages(chat_id, user_id)
    should_update_chat_summary = (
        message_count >= SUMMARY_EVERY_N_MESSAGES and message_count % SUMMARY_EVERY_N_MESSAGES == 0
    )
    should_update_user_summary = (
        user_msg_count >= SUMMARY_EVERY_N_MESSAGES and user_msg_count % SUMMARY_EVERY_N_MESSAGES == 0
    )
    if not should_update_chat_summary and not should_update_user_summary:
        return False

    current_summary = await db.get_summary(chat_id)
    recent_history = await db.get_last_messages(
        chat_id,
        max(history_limit, SUMMARY_HISTORY_WINDOW),
        focus_user_id=None,
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
            chat_id,
            max(history_limit, SUMMARY_HISTORY_WINDOW),
            focus_user_id=user_id,
        )
        user_history_text = _history_as_text(recent_user_history, SUMMARY_INPUT_CHAR_LIMIT)
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

    if updated and message_count > 2000:
        try:
            await db.rotate_messages(chat_id, keep=2000)
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
            chat_id,
            user_id,
            fact_type,
            value,
            normalized_value,
            confidence,
            importance,
            evidence,
            source="auto_extract",
        )
        current_facts[signature] = {
            "fact_type": fact_type,
            "value": value,
            "normalized_value": normalized_value,
            "confidence": confidence,
            "importance": importance,
            "evidence": evidence,
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

    current_profile = await db.get_user_profile(chat_id, user_id)
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

    interests = _join_list(profile.get("interests"))
    communication_style = _clean_profile_text(
        profile.get("communication_style"),
        PROFILE_FIELD_CHAR_LIMIT,
    )
    response_preference = _clean_profile_text(
        profile.get("response_preference"),
        PROFILE_FIELD_CHAR_LIMIT,
    )
    relationship_notes = _clean_profile_text(
        profile.get("relationship_notes"),
        RELATIONSHIP_NOTES_CHAR_LIMIT,
    )
    await db.upsert_user_profile(
        chat_id,
        user_id,
        interests=interests,
        communication_style=communication_style,
        response_preference=response_preference,
        relationship_notes=relationship_notes,
        profile_confidence=confidence,
    )
    attempted_profile = {
        "interests": interests,
        "communication_style": communication_style,
        "response_preference": response_preference,
        "relationship_notes": relationship_notes,
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
    current_profile = await db.get_user_profile(chat_id, user_id)
    nickname = None
    display_name = None
    memory_items_attempted = 0
    memory_items_changed = 0

    if sender_username:
        nickname = f"@{sender_username.lstrip('@')}"
        memory_items_attempted += 1
        if _would_change_memory(
            current_memories.get("nickname"),
            value=nickname,
            confidence=0.35,
            importance=0.88,
            source="telegram_profile",
        ):
            memory_items_changed += 1
        await db.upsert_memory(
            chat_id,
            user_id,
            "nickname",
            nickname,
            confidence=0.35,
            importance=0.88,
            source="telegram_profile",
        )
        current_memories["nickname"] = {
            "key": "nickname",
            "value": nickname,
            "confidence": 0.35,
            "importance": 0.88,
            "source": "telegram_profile",
        }

    if sender_first_name:
        display_name = sender_first_name.strip()
        memory_items_attempted += 1
        if _would_change_memory(
            current_memories.get("preferred_name"),
            value=display_name,
            confidence=0.25,
            importance=0.9,
            source="telegram_profile",
        ):
            memory_items_changed += 1
        await db.upsert_memory(
            chat_id,
            user_id,
            "preferred_name",
            display_name,
            confidence=0.25,
            importance=0.9,
            source="telegram_profile",
        )
        current_memories["preferred_name"] = {
            "key": "preferred_name",
            "value": display_name,
            "confidence": 0.25,
            "importance": 0.9,
            "source": "telegram_profile",
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
        profile_fields_attempted = sum(1 for key, value in candidate_profile.items() if key == "profile_confidence" or value)
        profile_fields_changed, _ = _compute_profile_changes(current_profile, candidate_profile)
        await db.upsert_user_profile(
            chat_id,
            user_id,
            display_name=display_name,
            nickname=nickname,
            profile_confidence=0.2,
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
    user_profile = await db.get_user_profile(chat_id, user_id)
    payload = build_user_document_payload(
        user_facts,
        user_profile,
        sender_username,
        sender_first_name,
    )
    await db.upsert_user_document(
        chat_id,
        user_id,
        document_json=json.dumps(payload, ensure_ascii=False),
        document_text=render_user_document_text(payload),
    )


async def should_ask_intro(db, chat_id: int, user_id: int, input_text: str) -> bool:
    text = (input_text or "").strip()
    if _is_nonsemantic_input(text):
        return False

    profile = await db.get_user_profile(chat_id, user_id)

    has_display_name = bool(
        (profile.get("approved_display_name") or profile.get("display_name") or "").strip()
    )
    has_nickname = bool((profile.get("nickname") or "").strip())

    if has_display_name or has_nickname:
        return False

    user_document = await db.get_user_document(chat_id, user_id)
    if (user_document.get("document_text") or "").strip():
        return False

    if await db.has_memory_key(chat_id, user_id, INTRO_MARKER_KEY):
        return False

    return True


async def build_context(
    db,
    chat_id: int,
    user_id: int,
    history_limit: int,
    sender_username: str | None,
    sender_first_name: str | None,
    plan: ContextPlan,
    knowledge_dir: str,
    knowledge_max_files: int,
    knowledge_max_file_bytes: int,
    input_text: str,
    mention_only: bool = False,
    reply_context: ReplyContext | None = None,
) -> tuple[list[dict[str, str]], list[dict], dict[str, int]]:
    summary = await db.get_summary(chat_id)
    speaker_summary = await db.get_summary(chat_id, user_id)
    user_facts = await db.get_user_facts(chat_id, user_id, limit=FACT_KEEP_PER_USER)
    user_profile = await db.get_user_profile(chat_id, user_id)
    user_document = await db.get_user_document(chat_id, user_id)
    chat_participants = (
        await db.get_chat_participants(
            chat_id, exclude_user_id=user_id, limit=MAX_CHAT_PARTICIPANTS,
        )
        if plan.include_participants
        else []
    )
    focused_history = await db.get_last_messages(
        chat_id,
        min(history_limit, max(plan.history_messages * 2, 12)),
        focus_user_id=user_id,
    )
    shared_history = await db.get_last_messages(
        chat_id,
        min(history_limit, max(plan.history_messages + 2, 10)),
        focus_user_id=None,
    )
    history = _merge_histories(focused_history, shared_history, plan.history_messages, plan.history_chars)
    prompt_history = [{"role": item["role"], "content": item["content"]} for item in history]
    implicit_reply_block = build_implicit_reply_context_block(prompt_history, input_text)

    # Inject implicit context directly into the last user message so the model
    # sees it right next to the short message, not buried in the system prompt.
    if implicit_reply_block and prompt_history:
        for i in range(len(prompt_history) - 1, -1, -1):
            if prompt_history[i].get("role") == "user":
                original_content = prompt_history[i]["content"]
                prompt_history[i] = {
                    "role": "user",
                    "content": f"[контекст: {implicit_reply_block}]\n{original_content}",
                }
                break

    system_blocks = []

    system_blocks.extend([
        {
            "content": (
                "В истории пользовательские реплики помечены автором в формате [id=123 @username]. "
                "Строго различай авторов и не приписывай слова одного человека другому."
            ),
            "bucket": "system",
            "label": "speaker_guard",
        },
        {
            "content": build_current_speaker_block(user_id, sender_username, sender_first_name),
            "bucket": "system",
            "label": "current_speaker",
        }
    ])
    if implicit_reply_block:
        system_blocks.append(
            {
                "content": implicit_reply_block,
                "bucket": "system",
                "label": "implicit_reply_context",
            }
        )
    mention_only_block = build_mention_only_block(mention_only)
    if mention_only_block:
        system_blocks.append(
            {
                "content": mention_only_block,
                "bucket": "system",
                "label": "mention_only",
            }
        )
    reply_block = build_reply_context_block(reply_context)
    if reply_block:
        system_blocks.append(
            {
                "content": reply_block,
                "bucket": "system",
                "label": "reply_context",
            }
        )
    speaker_summary_budget = max(plan.summary_chars * 3 // 5, 200)
    general_summary_budget = max(plan.summary_chars - speaker_summary_budget, 120)
    clipped_speaker_summary = _clip_text(speaker_summary, speaker_summary_budget)
    if clipped_speaker_summary:
        system_blocks.append(
            {
                "bucket": "summary",
                "label": "speaker_summary",
                "content": (
                    "Краткое резюме диалога с текущим собеседником. "
                    "Это важнее общего summary, если нужно понять, на что он отвечает.\n"
                    f"{clipped_speaker_summary}"
                ).strip(),
            }
        )

    clipped_summary = _clip_text(summary, general_summary_budget)
    if clipped_summary:
        system_blocks.append(
            {
                "bucket": "summary",
                "label": "summary",
                "content": (
                    "Краткое резюме общего чата. В нем могут быть разные участники. "
                    "Не переноси чужие факты на текущего собеседника.\n"
                    f"{clipped_summary}"
                ).strip(),
            }
        )

    identity_block = build_identity_block(user_profile, sender_username, sender_first_name)
    if identity_block:
        system_blocks.append({"content": identity_block, "bucket": "system", "label": "identity"})

    if plan.include_participants:
        participants_block = build_chat_participants_block(chat_participants, user_id)
        if participants_block:
            system_blocks.append({
                "content": participants_block,
                "bucket": "system",
                "label": "participants",
            })

    mem_text = format_user_facts(user_facts, max_lines=plan.memory_lines)
    if mem_text:
        system_blocks.append(
            {
                "content": f"Заметки о собеседнике:\n{mem_text}".strip(),
                "bucket": "memory",
                "label": "memory",
            }
        )

    document_text = _fit_text(user_document.get("document_text", ""), USER_DOCUMENT_CHAR_LIMIT)
    if should_include_user_document(input_text, document_text):
        system_blocks.append(
            {
                "bucket": "system",
                "label": "user_document",
                "content": (
                    f"{document_text}\n"
                    "Используй это только как внутренний контекст. Не цитируй досье буквально."
                ).strip(),
            }
        )

    if plan.file_chunks > 0 and plan.file_chars > 0:
        await sync_context_dir(
            db,
            knowledge_dir,
            max_files=knowledge_max_files,
            max_file_bytes=knowledge_max_file_bytes,
        )
        snippets = await retrieve_context_snippets(
            db,
            input_text,
            max_chunks=plan.file_chunks,
            max_chars=plan.file_chars,
        )
        file_block = build_file_context_block(snippets)
        if file_block:
            system_blocks.append({"content": file_block, "bucket": "rag", "label": "rag"})

    context_metrics = {
        "history_tokens": approx_messages_tokens(prompt_history),
        "summary_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("bucket") == "summary"
        ),
        "memory_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("bucket") == "memory"
        ),
        "rag_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("bucket") == "rag"
        ),
        "identity_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "identity"
        ),
        "user_document_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "user_document"
        ),
        "current_speaker_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "current_speaker"
        ),
        "reply_context_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "reply_context"
        ),
        "participants_tokens": sum(
            approx_text_tokens(item["content"]) for item in system_blocks if item.get("label") == "participants"
        ),
    }

    return [item for item in system_blocks if item.get("content")], prompt_history, context_metrics


def _compose_system_prompt(
    input_text: str,
    user_id: int,
    creator_user_id: int | None,
    creator_profile_path: str,
    system_blocks: list[dict[str, str]],
    include_creator_profile: bool,
) -> tuple[str, dict[str, int]]:
    system_prompt = ALTA_SYSTEM_PROMPT
    if system_blocks:
        system_prompt = (
            system_prompt
            + "\n\n---\nКОНТЕКСТ (используй как справку, не меняй свой стиль):\n"
            + "\n\n".join(item["content"] for item in system_blocks)
        )

    creator_profile_tokens = 0
    if include_creator_profile and should_include_profile(input_text, user_id, creator_user_id):
        profile = load_profile(creator_profile_path)
        profile_block = profile_to_system_block(profile)
        if profile_block:
            system_prompt = system_prompt + "\n\n" + profile_block
            creator_profile_tokens = approx_text_tokens(profile_block)

    # Reminder at the end to keep personality dominant over context blocks
    system_prompt += "\n\n---\nПомни: ты Альта. Отвечай коротко, живо, по-человечески. Контекст выше - справка, не шаблон."

    system_tokens = approx_text_tokens(ALTA_SYSTEM_PROMPT) + sum(
        approx_text_tokens(item["content"])
        for item in system_blocks
        if item.get("bucket") == "system"
    ) + creator_profile_tokens

    return system_prompt, {
        "system_tokens": system_tokens,
        "creator_profile_tokens": creator_profile_tokens,
    }


def _measure_context_snapshot(
    system_prompt: str,
    system_blocks: list[dict[str, str]],
    history: list[dict],
) -> dict[str, int]:
    summary_tokens = 0
    memory_tokens = 0
    rag_tokens = 0
    identity_tokens = 0
    user_document_tokens = 0
    current_speaker_tokens = 0
    reply_context_tokens = 0
    participants_tokens = 0

    for block in system_blocks:
        text = str((block or {}).get("content") or "")
        if not text:
            continue
        tokens = approx_text_tokens(text)
        bucket = str((block or {}).get("bucket") or "")
        label = str((block or {}).get("label") or "")
        if bucket == "summary":
            summary_tokens += tokens
            continue
        if bucket == "memory":
            memory_tokens += tokens
            continue
        if bucket == "rag":
            rag_tokens += tokens
            continue
        if label == "identity":
            identity_tokens += tokens
        if label == "current_speaker":
            current_speaker_tokens += tokens
        if label == "reply_context":
            reply_context_tokens += tokens
        if label == "user_document":
            user_document_tokens += tokens
        if label == "participants":
            participants_tokens += tokens
        if label in {"identity", "current_speaker", "reply_context", "user_document", "participants"}:
            continue
        if "Краткое резюме" in text:
            summary_tokens += tokens
        elif "Заметки о собеседнике:" in text:
            memory_tokens += tokens
        elif "Контекст из файлов" in text:
            rag_tokens += tokens
        else:
            if "Профиль текущего собеседника:" in text: 
                identity_tokens += tokens
            if "Текущий собеседник для этого ответа:" in text:
                current_speaker_tokens += tokens
            if "Текущее сообщение является ответом на конкретную реплику в чате." in text:
                reply_context_tokens += tokens
            if "Используй это только как внутренний контекст." in text:
                user_document_tokens += tokens

    history_tokens = approx_messages_tokens(history)
    system_tokens = max(
        0,
        approx_text_tokens(system_prompt) - summary_tokens - memory_tokens - rag_tokens,
    )
    return {
        "history_tokens": history_tokens,
        "summary_tokens": summary_tokens,
        "memory_tokens": memory_tokens,
        "rag_tokens": rag_tokens,
        "system_tokens": system_tokens,
        "input_tokens": history_tokens + summary_tokens + memory_tokens + rag_tokens + system_tokens,
        "identity_tokens": identity_tokens,
        "user_document_tokens": user_document_tokens,
        "current_speaker_tokens": current_speaker_tokens,
        "reply_context_tokens": reply_context_tokens,
        "participants_tokens": participants_tokens,
        "creator_profile_tokens": 0,
    }


def _merge_context_metrics(
    context_metrics: dict[str, int],
    prompt_metrics: dict[str, int],
) -> dict[str, int]:
    merged = dict(context_metrics)
    merged["system_tokens"] = int(prompt_metrics.get("system_tokens", 0))
    merged["creator_profile_tokens"] = int(prompt_metrics.get("creator_profile_tokens", 0))
    merged["input_tokens"] = (
        int(merged.get("history_tokens", 0))
        + int(merged.get("summary_tokens", 0))
        + int(merged.get("memory_tokens", 0))
        + int(merged.get("rag_tokens", 0))
        + int(merged.get("system_tokens", 0))
    )
    return merged


async def _generate_answer_with_fallback(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    history_limit: int,
    creator_user_id: int | None,
    creator_profile_path: str,
    sender_username: str | None,
    sender_first_name: str | None,
    knowledge_dir: str,
    knowledge_max_files: int,
    knowledge_max_file_bytes: int,
    mention_only: bool = False,
    reply_context: ReplyContext | None = None,
    turn_id: str | None = None,
    turn_metrics=None,
) -> tuple[str, str, list[dict], list[dict], bool]:
    last_overflow: Exception | None = None

    for plan in _build_context_plans(max_tokens, history_limit):
        system_blocks, filtered, context_metrics = await build_context(
            db,
            chat_id,
            user_id,
            history_limit,
            sender_username,
            sender_first_name,
            plan,
            knowledge_dir,
            knowledge_max_files,
            knowledge_max_file_bytes,
            input_text,
            mention_only,
            reply_context,
        )
        system_prompt, prompt_metrics = _compose_system_prompt(
            input_text,
            user_id,
            creator_user_id,
            creator_profile_path,
            system_blocks,
            plan.include_creator_profile,
        )
        context_snapshot = _merge_context_metrics(context_metrics, prompt_metrics)
        if turn_metrics is not None:
            turn_metrics.set_context_metrics(context_snapshot)

        try:
            answer = await llm.chat(
                system_prompt,
                filtered,
                temperature,
                top_p,
                plan.max_tokens,
                turn_id=turn_id,
                turn_metrics=turn_metrics,
            )
            return answer, system_prompt, filtered, system_blocks, False
        except ContextOverflowError as e:
            last_overflow = e
            log.warning("context overflow on plan=%s, retrying", plan.name)

    fallback_input = _fit_text(input_text, 420) if not mention_only else ""
    fallback_messages = [{"role": "user", "content": fallback_input}]
    try:
        fallback_answer = await llm.chat(
            ALTA_SYSTEM_PROMPT,
            fallback_messages,
            temperature,
            top_p,
            min(max(48, int(max_tokens)), 64),
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.set_context_metrics(
                {
                    "history_tokens": approx_messages_tokens(fallback_messages),
                    "summary_tokens": 0,
                    "memory_tokens": 0,
                    "rag_tokens": 0,
                    "system_tokens": approx_text_tokens(ALTA_SYSTEM_PROMPT),
                    "input_tokens": approx_messages_tokens(fallback_messages) + approx_text_tokens(ALTA_SYSTEM_PROMPT),
                    "identity_tokens": 0,
                    "user_document_tokens": 0,
                    "current_speaker_tokens": 0,
                    "reply_context_tokens": 0,
                    "creator_profile_tokens": 0,
                }
            )
        return fallback_answer, ALTA_SYSTEM_PROMPT, fallback_messages, [], True
    except ContextOverflowError as e:
        last_overflow = e

    raise RuntimeError("LLM context budget exhausted") from last_overflow


async def edit_answer(
    llm,
    input_text: str,
    draft_answer: str,
    temperature: float,
    top_p: float,
    turn_id: str | None = None,
    turn_metrics=None,
) -> str:
    draft = _normalize_reply_text(draft_answer)
    if not draft:
        return ""

    prompt = (
        f"Сообщение пользователя:\n{_fit_text(input_text, 500)}\n\n"
        f"Черновик ответа:\n{_fit_text(draft, 400)}"
    )
    try:
        edited = await llm.chat(
            LEGACY_STYLE_PROMPT_ARCHIVE,
            [{"role": "user", "content": prompt}],
            temperature=min(max(temperature, 0.2), 0.45),
            top_p=top_p,
            max_tokens=200,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("edit_answer LLM call failed, using deterministic fallback", exc_info=True)
        return _deterministic_postprocess_answer(draft)

    return _editor_safe_output(draft, edited)


async def _rewrite_repeated_reply(
    llm,
    input_text: str,
    draft_answer: str,
    recent_assistant_replies: list[str],
    temperature: float,
    top_p: float,
    turn_id: str | None = None,
    turn_metrics=None,
) -> str:
    draft = _normalize_reply_text(draft_answer)
    if not draft:
        return ""

    draft_key = _reply_repeat_key(draft)
    if len(draft_key) < REPEAT_GUARD_MIN_KEY_CHARS:
        return ""

    forbidden: list[str] = []
    for recent in reversed(recent_assistant_replies):
        recent_text = _normalize_reply_text(recent)
        recent_key = _reply_repeat_key(recent_text)
        if len(recent_key) < REPEAT_GUARD_MIN_KEY_CHARS:
            continue
        if not _looks_like_same_reply_key(draft_key, recent_key):
            continue
        if recent_text and recent_text not in forbidden:
            forbidden.append(recent_text)
        if len(forbidden) >= 3:
            break

    if not forbidden:
        forbidden.append(draft)

    forbidden_lines = "\n".join(f"- {_fit_text(item, 120)}" for item in forbidden)
    rewrite_payload = (
        f"User message:\n{_fit_text(input_text, 420)}\n\n"
        f"Draft answer:\n{_fit_text(draft, 240)}\n\n"
        f"Forbidden exact phrases:\n{forbidden_lines}"
    )
    try:
        rewritten = await llm.chat(
            REPEAT_REWRITE_PROMPT,
            [{"role": "user", "content": rewrite_payload}],
            temperature=min(max(float(temperature), 0.25), 0.55),
            top_p=top_p,
            max_tokens=120,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("repeat guard rewrite failed", exc_info=True)
        return ""

    return _deterministic_postprocess_answer(rewritten)


async def generate_simple_reply_fallback(
    llm,
    input_text: str,
    temperature: float,
    top_p: float,
    mention_only: bool = False,
    turn_id: str | None = None,
    turn_metrics=None,
) -> str:
    if mention_only or _is_nonsemantic_input(input_text):
        return "а?"

    text = _fit_text(input_text, 300)
    if not text:
        return "а?"

    fallback_system = ALTA_SYSTEM_PROMPT + "\n\nОтвечай коротко, одно-два предложения."

    try:
        answer = await llm.chat(
            fallback_system,
            [{"role": "user", "content": text}],
            temperature=min(max(temperature, 0.35), 0.6),
            top_p=top_p,
            max_tokens=80,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
    except Exception:
        log.debug("simple reply fallback LLM call failed", exc_info=True)
        return ""

    return _shrink_reply_text(answer, limit=180)


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
        extracted_memory_stats, extracted_profile_stats = await asyncio.gather(
            maybe_store_memories(
                llm, db, chat_id, user_id, input_text,
                turn_id=turn_id, turn_metrics=turn_metrics,
            ),
            maybe_store_user_profile(
                llm, db, chat_id, user_id, input_text,
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
            turn_metrics.memory_updated = turn_metrics.memory_updated or int(extracted_memory_stats.get("changed", 0)) > 0
            turn_metrics.profile_updated = turn_metrics.profile_updated or int(extracted_profile_stats.get("changed", 0)) > 0
            turn_metrics.summary_updated = turn_metrics.summary_updated or summary_updated
            if summary_updated:
                turn_metrics.summary_write_count += 1
    except Exception:
        log.warning("background extraction failed chat=%s user=%s", chat_id, user_id, exc_info=True)


# Global set to prevent background tasks from being garbage-collected
_background_tasks: set[asyncio.Task] = set()


async def run_agent(
    llm,
    db,
    chat_id: int,
    user_id: int,
    input_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    history_limit: int,
    has_attachments: bool = False,
    debug_traces: bool = False,
    creator_user_id: int | None = None,
    creator_profile_path: str = "./creator_profile.json",
    sender_username: str | None = None,
    sender_first_name: str | None = None,
    knowledge_dir: str = "./knowledge",
    knowledge_max_files: int = 64,
    knowledge_max_file_bytes: int = 262_144,
    mention_only: bool = False,
    reply_context: ReplyContext | None = None,
    turn_id: str | None = None,
    turn_metrics=None,
):
    turn = RouteTurn(text_clean=input_text or "", has_file=has_attachments)
    mode = route_mode(turn)
    routed_reason = route_reason(turn)
    if turn_metrics is not None:
        turn_metrics.mode = mode
        turn_metrics.route_reason = routed_reason

    # Fast DB-only ops before answer generation
    sync_stats = await maybe_store_identity_from_telegram(
        db,
        chat_id,
        user_id,
        sender_username,
        sender_first_name,
    )
    await sync_user_document(
        db,
        chat_id,
        user_id,
        sender_username,
        sender_first_name,
    )
    ask_intro = await should_ask_intro(db, chat_id, user_id, input_text)

    # Generate answer FIRST — no LLM extraction blocking the response
    answer, system_prompt, filtered, system_blocks, fallback_used = await _generate_answer_with_fallback(
        llm,
        db,
        chat_id,
        user_id,
        input_text,
        temperature,
        top_p,
        max_tokens,
        history_limit,
        creator_user_id,
        creator_profile_path,
        sender_username,
        sender_first_name,
        knowledge_dir,
        knowledge_max_files,
        knowledge_max_file_bytes,
        mention_only,
        reply_context,
        turn_id=turn_id,
        turn_metrics=turn_metrics,
    )
    if turn_metrics is not None:
        turn_metrics.fallback_used = turn_metrics.fallback_used or fallback_used

    draft_answer = (answer or "").strip()
    editor_triggers = _editor_triggers_for(draft_answer)
    if editor_triggers:
        if turn_metrics is not None:
            turn_metrics.editor_attempted = True
            turn_metrics.editor_trigger = ",".join(editor_triggers)
        edited_answer = draft_answer
        if "short_draft" in editor_triggers:
            edited_answer = await edit_answer(
                llm,
                input_text,
                draft_answer,
                temperature,
                top_p,
                turn_id=turn_id,
                turn_metrics=turn_metrics,
            )
        answer = _deterministic_postprocess_answer(edited_answer or draft_answer)
        if turn_metrics is not None and (answer or "").strip() != draft_answer:
            turn_metrics.editor_changed_output = True
    if not (answer or "").strip():
        answer = await generate_simple_reply_fallback(
            llm,
            input_text,
            temperature,
            top_p,
            mention_only=mention_only,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.fallback_used = True
        fallback_used = True

    recent_assistant_replies = _recent_assistant_replies(filtered)
    if _is_repeated_recent_reply(answer, recent_assistant_replies):
        log.info("repeat guard triggered chat=%s user=%s", chat_id, user_id)
        rewritten = await _rewrite_repeated_reply(
            llm,
            input_text,
            answer,
            recent_assistant_replies,
            temperature,
            top_p,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
        if rewritten:
            answer = rewritten.strip()

        if _is_repeated_recent_reply(answer, recent_assistant_replies):
            answer = await generate_simple_reply_fallback(
                llm,
                input_text,
                temperature,
                top_p,
                mention_only=mention_only,
                turn_id=turn_id,
                turn_metrics=turn_metrics,
            )
            fallback_used = True
            if turn_metrics is not None:
                turn_metrics.fallback_used = True

        if _is_repeated_recent_reply(answer, recent_assistant_replies):
            answer = _repeat_guard_hard_fallback(input_text)

    if ask_intro:
        intro_line = ""
        try:
            intro_line = (
                await llm.chat(
                    INTRO_GENERATION_PROMPT,
                    [{"role": "user", "content": _fit_text(input_text or "привет", 600)}],
                    temperature=max(temperature, 0.8),
                    top_p=top_p,
                    max_tokens=80,
                    turn_id=turn_id,
                    turn_metrics=turn_metrics,
                )
            ).strip()
        except Exception:
            log.debug("intro generation LLM call failed", exc_info=True)
            intro_line = ""

        answer = (answer or "").strip()
        if intro_line:
            answer = f"{answer}\n\n{intro_line}" if answer else intro_line

        if intro_line:
            await db.upsert_memory(
                chat_id,
                user_id,
                INTRO_MARKER_KEY,
                "asked",
                confidence=1.0,
                importance=1.0,
                source="system_intro",
            )
            if turn_metrics is not None:
                turn_metrics.memory_items_written += 1
                turn_metrics.memory_items_changed += 1
                turn_metrics.memory_intro_items_changed += 1
                turn_metrics.memory_updated = True

    if not (answer or "").strip():
        answer = await generate_simple_reply_fallback(
            llm,
            input_text,
            temperature,
            top_p,
            mention_only=mention_only,
            turn_id=turn_id,
            turn_metrics=turn_metrics,
        )
        if turn_metrics is not None:
            turn_metrics.fallback_used = True
        fallback_used = True

    if debug_traces and hasattr(db, "conn") and db.conn:
        reply_context_payload = None
        if reply_context is not None:
            reply_context_payload = {
                "user_id": reply_context.user_id,
                "username": reply_context.username,
                "first_name": reply_context.first_name,
                "text": reply_context.text,
            }

        llm_calls_payload: list[dict[str, Any]] = []
        metrics_payload: dict[str, Any] = {}
        if turn_metrics is not None:
            llm_calls_payload = [dict(vars(call)) for call in getattr(turn_metrics, "llm_calls", [])]
            metrics_payload = {
                "history_tokens": turn_metrics.history_tokens,
                "summary_tokens": turn_metrics.summary_tokens,
                "memory_tokens": turn_metrics.memory_tokens,
                "rag_tokens": turn_metrics.rag_tokens,
                "system_tokens": turn_metrics.system_tokens,
                "input_tokens": turn_metrics.input_tokens,
                "identity_tokens": turn_metrics.identity_tokens,
                "user_document_tokens": turn_metrics.user_document_tokens,
                "creator_profile_tokens": turn_metrics.creator_profile_tokens,
                "current_speaker_tokens": turn_metrics.current_speaker_tokens,
                "reply_context_tokens": turn_metrics.reply_context_tokens,
                "total_llm_calls": turn_metrics.total_llm_calls,
                "total_tokens": turn_metrics.total_tokens,
                "total_latency_ms": turn_metrics.total_latency_ms,
                "fallback_used": turn_metrics.fallback_used,
                "editor_attempted": turn_metrics.editor_attempted,
                "editor_trigger": turn_metrics.editor_trigger,
                "editor_changed_output": turn_metrics.editor_changed_output,
                "summary_updated": turn_metrics.summary_updated,
                "memory_updated": turn_metrics.memory_updated,
                "profile_updated": turn_metrics.profile_updated,
                "llm_calls": llm_calls_payload,
            }

        trace_payload = {
            "turn_id": turn_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "mode": mode,
            "route_reason": routed_reason,
            "has_attachments": bool(has_attachments),
            "mention_only": bool(mention_only),
            "reply_context": reply_context_payload,
            "generation": {
                "model": getattr(llm, "model", None),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "history_limit": int(history_limit),
            },
            "context": {
                "system": system_prompt,
                "messages": filtered,
                "system_blocks": [
                    {
                        "label": (block or {}).get("label"),
                        "bucket": (block or {}).get("bucket"),
                        "content": (block or {}).get("content"),
                    }
                    for block in system_blocks
                ],
            },
            "result": {
                "answer": answer,
                "fallback_used": bool(fallback_used),
            },
            "metrics": metrics_payload,
        }

        await db.conn.execute(
            "INSERT INTO debug_traces(chat_id,input_text,context_json,output_text) VALUES(?,?,?,?)",
            (
                chat_id,
                input_text,
                json.dumps(trace_payload, ensure_ascii=False),
                answer,
            ),
        )
        await db.conn.commit()

    # Schedule heavy extraction (memory, profile, summary) in background
    task = asyncio.create_task(
        _background_extraction(
            llm, db, chat_id, user_id, input_text, history_limit,
            sender_username, sender_first_name, sync_stats,
            turn_id=turn_id, turn_metrics=turn_metrics,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return answer
