# CLAUDE.md — Контекст для AI-агентов

## Что это за проект

Alta — Telegram-бот с характером (AI-персонаж ~20 лет, женский пол). Работает в групповых чатах. Не ассистент — ведет себя как живой участник. Русскоязычный.

## Технологии

- Python 3.11+, aiogram 3.4, aiosqlite, openai (для локальной LLM), python-dotenv
- LLM: OpenAI-compatible API (по умолчанию llama.cpp с Mistral Small)
- DB: SQLite (WAL mode)
- TTS: ElevenLabs API (опционально)
- Платформа: Windows (используется WindowsSelectorEventLoopPolicy)

## Структура файлов

```
main.py           — точка входа, aiogram Dispatcher, все Telegram handlers
agent.py          — ЯДРО: context building, LLM generation, memory/profile extraction, postprocessing
llm.py            — OpenAI API wrapper (retry, overflow detection, telemetry)
db.py             — SQLite: schema, migrations, все CRUD операции
config.py         — загрузка .env в Config dataclass
prompts.py        — системный промпт персонажа (ALTA_SYSTEM_PROMPT)
voice.py          — ElevenLabs TTS + распознавание voice requests в русском тексте
media.py          — стикеры/кастомные эмодзи (каталог, LLM-выбор, отправка)
file_context.py   — RAG-lite: индексация knowledge/, chunk scoring, retrieval
lore_context.py   — загрузка файла с бэкграундом персонажа
profile_context.py — загрузка creator_profile.json
user_profile.py   — /profile команда: парсинг, валидация, CRUD полей профиля
private_profile.py — личный кабинет в ЛС бота: inline-меню, валидация, импорт из групп
router.py         — legacy роутер (mention/reply detection)
telemetry.py      — JSONL логирование метрик в orchestration.log
runtime_io.py     — UTF-8 reconfigure для Windows
knowledge/        — текстовые файлы лора и контекста (автоиндексируются)
tests/            — pytest тесты
```

## Ключевые архитектурные решения

1. **agent.py — самый важный файл** (~2700 строк). Содержит всю логику генерации ответа, сборки контекста, памяти, постобработки. При изменении логики ответов — редактировать здесь.

2. **Context overflow fallback**: 4 уровня контекста (full → compact → minimal → emergency). Если LLM не влезает в контекст, система автоматически уменьшает историю, память и RAG.

3. **Фоновая extraction**: после отправки ответа пользователю, в фоне запускается `_background_extraction()` — LLM извлекает факты и обновляет профиль. Это НЕ блокирует ответ.

4. **Промпт собирается из блоков**: system_blocks — список dict'ов с полями `content`, `bucket`, `label`. Bucket'ы: system, summary, memory, rag. Это позволяет мерить токены по категориям.

5. **Repeat guard**: если ответ совпадает с недавними — перегенерация через специальный промпт, затем fallback.

6. **Постобработка ответа**: `_deterministic_postprocess_answer()` — удаление эмодзи, списков, "я бот", fix пола (мужской→женский).

7. **Санитизация**: `sanitize_incoming_text()` и `sanitize_outgoing_text()` защищают от prompt injection (удаление `[id=... @user]` тегов, `system:` / `assistant:` префиксов).

8. **Двухуровневый профиль пользователя**:
   - Групповой профиль `(chat_id, user_id)` — авто-собирается LLM из сообщений в группе
   - Приватный профиль `(user_id, user_id)` — задаётся пользователем вручную через ЛС бота
   - `get_user_profile_merged()` всегда возвращает слияние: приватный профиль имеет приоритет

9. **Защита от загрязнения памяти**: в `_build_extraction_text()` сообщения других участников чата явно помечаются как "Другой участник чата" — LLM не может приписать их факты целевому пользователю.

## Как запускать

```bash
pip install -r requirements.txt
# Нужен .env с BOT_TOKEN и запущенный LLM сервер
python main.py
```

## Как запускать тесты

```bash
pytest tests/
```

## Важные константы (agent.py)

- `SUMMARY_EVERY_N_MESSAGES = 10` — частота обновления summary
- `FACT_KEEP_PER_USER = 80` — макс. фактов на пользователя
- `MEMORY_STORE_MIN_IMPORTANCE = 0.45` — минимальный порог для сохранения факта
- `MAX_EDITOR_REPLY_CHARS = 400` — макс. длина ответа
- `REPEAT_GUARD_WINDOW = 6` — окно для проверки повторений

## База данных (основные таблицы)

- `messages` — история (chat_id, user_id, role, text, target_user_id)
- `user_facts` — факты о пользователях (fact_type, value, confidence, importance)
- `user_profiles` — профили (display_name, gender, interests, communication_style, extra_fields)
- `chat_summary` / `user_chat_summaries` — сжатая история
- `context_files` / `context_chunks` — индекс knowledge-файлов
- `memories` — legacy память (мигрируется в user_facts)

Профиль `user_profiles` хранит `extra_fields TEXT` — JSON с дополнительными полями (age, city, occupation, about).
Приватный профиль пользователя хранится под ключом `chat_id = user_id` (ЛС бота).

## Система профилей пользователя

### Команда /profile (в групповых чатах)
Команды: show, set, patch (bulk), delete, clear. Алиасы полей на русском и английском.

### Личный кабинет в ЛС бота (private_profile.py)
Пользователь пишет боту в личку → получает inline-меню без какого-либо LLM.
- Поля: Имя, Ник, Пол, Возраст, Город, Занятие, Интересы, Стиль общения, Формат ответов, О себе
- Пол: быстрый выбор кнопками (Мужской / Женский / Небинарный / Не указан / Своё)
- Кнопка "📥 Что обо мне знает Альта" — показывает данные собранные из групп с разметкой ✓/→
- Импорт не перезаписывает уже заданные вручную значения

### Валидация в ЛС
- Имя/Ник: макс. 40 симв., нет ссылок, нет HTML, нет /команд
- Возраст: целое 10–120
- Город/Занятие: макс. 60/100 симв., нет ссылок
- Интересы/Стили: макс. 200 симв.
- О себе: макс. 300 симв.
- Rate limit: 30 сохранений/час на пользователя

### Слияние профилей (get_user_profile_merged)
Приватный профиль (user-set) всегда побеждает авто-собранный групповой.
Групповой профиль заполняет только пустые поля приватного.

## Что НЕ трогать без понимания

- `sanitize_incoming_text()` / `sanitize_outgoing_text()` — prompt injection protection
- `_migrate_schema()` в db.py — автомиграции, добавлять новые в конец
- `_fix_gender()` в agent.py — regex'ы для gender fix, ломаются при неаккуратном изменении
- `_pending_join_verification` в main.py — in-memory state верификации, race conditions
- `get_user_profile_merged()` в db.py — логика приоритета приватного профиля над групповым
- `_build_extraction_text()` в agent.py — разметка сообщений по owner'у, нельзя смешивать

## Известные ограничения / TODO

- `relationship_notes` не извлекается автоматически (только вручную через /profile) — слишком склонно к записи транзиентных состояний
- `_pending_join_verification` хранится только в памяти (теряется при рестарте)
- `route_mode()` пока не влияет на контекстный бюджет
