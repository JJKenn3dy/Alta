# Alta — AI-персонаж для Telegram

Alta (Альта) — Telegram-бот с характером. Это не ассистент и не чат-бот в обычном понимании: Alta — самостоятельный персонаж женского пола (~20 лет), который живет в групповых чатах, запоминает собеседников, отвечает голосом, шлет стикеры и ведет себя как живой участник.

Бот работает через локальную LLM (по умолчанию Mistral через llama.cpp / OpenAI-compatible API), хранит все данные в SQLite и не зависит от внешних облачных AI-сервисов.

---

## Архитектура (высокоуровневая)

```
Telegram (aiogram)
       |
   main.py          — точка входа, Dispatcher, обработка сообщений/реакций/верификации
       |
   agent.py          — ядро: сборка контекста, генерация ответа, память, профили, repeat guard
       |
   +-- llm.py        — обертка над OpenAI-compatible API (retry, overflow protection)
   +-- db.py          — SQLite через aiosqlite (сообщения, память, профили, факты, контекст-файлы)
   +-- prompts.py     — системный промпт персонажа
   +-- voice.py       — TTS через ElevenLabs API
   +-- media.py       — стикеры/кастомные эмодзи: каталог, LLM-выбор, отправка
   +-- file_context.py — RAG-lite: индексация knowledge-файлов, chunk scoring, retrieval
   +-- lore_context.py — загрузка лора персонажа из файла
   +-- profile_context.py — загрузка JSON-профиля создателя
   +-- user_profile.py — команда /profile: парсинг, валидация, CRUD полей профиля
   +-- router.py       — простой роутер: mention/reply detection + cooldown
   +-- config.py       — загрузка конфига из .env
   +-- telemetry.py    — JSONL-логирование метрик каждого turn'а
   +-- runtime_io.py   — UTF-8 reconfigure для Windows stdout/stderr

knowledge/           — текстовые файлы лора и контекста (автоиндексация)
scripts/             — утилиты (init_db, build_media_catalog, analyze logs, debug dashboard)
tests/               — unit-тесты (pytest)
```

---

## Поток обработки сообщения (пошагово)

```
1. Telegram message приходит в on_message (main.py)
2. Проверки: allowed chat? not private? not bot? not pending verification?
3. is_trigger() — бот реагирует на @mention, reply на свое сообщение, или имя "Альта"/"альт"/etc.
4. cooldown_ok() — per-user rate limit
5. sanitize_incoming_text() — удаление injection-попыток (системные теги, role-префиксы)
6. detect_voice_request() — распознавание запроса на голосовой ответ
7. extract_requested_display_name() — распознавание "зови меня X"
8. run_agent() вызывается (agent.py):
   a. maybe_store_identity_from_telegram() — синхронизация username/first_name в память
   b. sync_user_document() — пересборка досье пользователя
   c. should_ask_intro() — проверка, нужно ли спросить "кто ты?"
   d. _generate_answer_with_fallback() — основной цикл генерации:
      - build_context() собирает system prompt из блоков:
        * ALTA_SYSTEM_PROMPT (характер)
        * current_speaker (кто пишет)
        * identity (профиль собеседника)
        * participants (другие участники чата)
        * speaker_summary + chat_summary (сжатая история)
        * user_facts/memory (факты о пользователе)
        * user_document (досье)
        * reply_context (если ответ на чье-то сообщение)
        * file_context/RAG (релевантные куски из knowledge/)
        * lore (бэкграунд персонажа)
        * creator_profile (профиль создателя)
      - _compose_system_prompt() — финальная склейка
      - llm.chat() — вызов LLM
      - 4 уровня fallback при context overflow: full -> compact -> minimal -> emergency
   e. _deterministic_postprocess_answer() — постобработка:
      * удаление эмодзи, списков, self-reference ("я бот")
      * fix_gender (мужской род -> женский для "я")
      * shrink длинных ответов
   f. Repeat guard — если ответ повторяет недавние, перегенерация
   g. Intro generation — для новых пользователей
   h. _background_extraction() — в фоне после ответа:
      * maybe_store_memories() — извлечение фактов из сообщения через LLM
      * maybe_store_user_profile() — обновление профиля через LLM
      * maybe_update_summary() — обновление summary чата каждые N сообщений
9. Постобработка ответа: sanitize_outgoing_text(), strip mentions
10. Отправка: текст или голос (ElevenLabs TTS, случайно или по запросу)
11. Опционально: отправка стикера/кастомного эмодзи (LLM выбирает)
12. turn_metrics.finish() — запись телеметрии
```

---

## Модули (подробно)

### `main.py` (~1430 строк)
Точка входа. Инициализация бота (aiogram), LLM, DB, TTS, media catalog. Содержит:

- **`CooldownManager`** — per-key rate limiter с TTL-eviction
- **`LockRegistry`** — per-chat asyncio.Lock для сериализации turn'ов
- **`is_trigger()`** — определяет, нужно ли боту реагировать (mention, reply, alias)
- **`strip_mention()`** — убирает @botname и алиасы из начала сообщения
- **`sanitize_incoming_text()`** — защита от prompt injection: удаляет `[id=123 @user]` теги, `system:` / `assistant:` префиксы, мета-строки профилей
- **`sanitize_outgoing_text()`** — очистка ответа бота: удаление тегов, эмодзи, role-префиксов
- **`on_new_chat_members()`** — антибот верификация: mute + код из 4 цифр + timeout + автобан
- **`on_message()`** — главный handler: маршрутизация, профили, генерация, голос, медиа
- **`on_message_reaction()`** — реакция на реакции пользователей (с вероятностью)
- **Join verification система** — полный flow: генерация кода, таймаут, бан ботов, `/unban`
- **Name approval система** — запрос имени -> одобрение создателем -> `/approve_name` / `/deny_name`
- **`/profile` команда** — делегируется в `user_profile.py`

### `agent.py` (~2680 строк)
Ядро интеллекта. Самый большой и важный файл:

- **Context planning** — `ContextPlan` dataclass определяет 4 плана (full/compact/minimal/emergency) с разными бюджетами на историю, память, RAG, токены
- **`build_context()`** — собирает все блоки системного промпта:
  - `build_current_speaker_block()` — кто сейчас пишет (user_id, username, first_name)
  - `build_identity_block()` — профиль собеседника (имя, пол, интересы, стиль)
  - `build_chat_participants_block()` — другие участники (чтобы не путать людей)
  - `build_reply_context_block()` — контекст reply-сообщения
  - `build_implicit_reply_context_block()` — угадывание контекста для коротких ответов
  - `build_mention_only_block()` — когда упомянули без текста
  - `format_user_facts()` — факты из памяти
  - `build_file_context_block()` — RAG-контекст из knowledge/
- **`_generate_answer_with_fallback()`** — попытка сгенерировать ответ с постепенным уменьшением контекста при overflow
- **`_deterministic_postprocess_answer()`** — детерминистическая постобработка (эмодзи, списки, self-reference, gender fix, длина)
- **`_fix_gender()`** — замена мужского рода на женский ("я понял" -> "я поняла")
- **Repeat guard** — `_is_repeated_recent_reply()`, `_rewrite_repeated_reply()`, `_repeat_guard_hard_fallback()`
- **Memory extraction** — `maybe_store_memories()`: LLM извлекает факты из сообщений, сохраняет в `user_facts`
- **Profile extraction** — `maybe_store_user_profile()`: LLM обновляет профиль пользователя
- **Summary update** — `maybe_update_summary()`: периодическое сжатие истории (каждые N сообщений)
- **User document** — `build_user_document_payload()`, `render_user_document_text()`: досье пользователя из фактов + профиля
- **`run_agent()`** — главная функция, оркестрирует весь pipeline

### `llm.py` (~197 строк)
OpenAI-compatible API клиент:

- **`LLM`** — класс-обертка над `AsyncOpenAI`
- **`chat()`** — основной метод: retry с exponential backoff (до 7 попыток), context overflow detection, телеметрия
- **`_ensure_context_budget()`** — pre-flight проверка размера контекста перед вызовом API
- **`ContextOverflowError`** — кастомное исключение для переполнения контекста
- **`estimate_chat_tokens()`** — эвристическая оценка токенов (chars / 2.0)

### `db.py` (~1068 строк)
SQLite storage через aiosqlite. Таблицы:

| Таблица | Назначение |
|---------|-----------|
| `allowed_chats` | Whitelist чатов |
| `users` | Telegram-пользователи (user_id, username, timestamps) |
| `messages` | История сообщений (chat_id, user_id, role, text, target_user_id) |
| `chat_summary` | Сжатое резюме чата |
| `user_chat_summaries` | Per-user сжатое резюме диалога |
| `user_profiles` | Профили пользователей (имя, пол, интересы, стиль общения) |
| `pending_name_approvals` | Запросы на смену имени (ожидание одобрения создателем) |
| `user_documents` | Досье пользователей (JSON + текст) |
| `user_facts` | Извлеченные факты о пользователях (type, value, confidence, importance) |
| `memories` | Legacy key-value память (мигрируется в user_facts) |
| `debug_traces` | Debug-трейсы turn'ов |
| `cooldown_state` | Персистентные cooldown'ы |
| `context_files` + `context_chunks` | Индекс knowledge-файлов и их чанков |

Ключевые методы:
- `get_last_messages()` — история с опциональной фокусировкой на конкретном пользователе
- `get_chat_participants()` — активные участники через bounded scan
- `upsert_user_fact()` — upsert с MAX(confidence, importance) при конфликте
- `prune_user_facts()` — обрезка до лимита по importance/confidence
- `rotate_messages()` — ротация старых сообщений (keep=2000)
- Auto-migration: `_migrate_schema()` добавляет новые колонки/таблицы автоматически

### `voice.py` (~279 строк)
Text-to-speech через ElevenLabs:

- **`detect_voice_request()`** — распознавание запросов на голос в русском тексте ("запиши голосом", "скажи голосом", "озвучь", "войсом" и т.д.)
- **`ElevenLabsTTS`** — синтез речи через ElevenLabs REST API (urllib, не SDK)
- Поддержка direct text ("озвучь: текст") и reply mode ("ответь голосом")
- Random voice replies: с заданной вероятностью бот отвечает голосом

### `media.py` (~282 строки)
Стикеры и кастомные эмодзи:

- **`build_media_catalog()`** — сканирование стикер-паков через Telegram API
- **`choose_media_intent()`** — LLM решает, нужен ли стикер после ответа
- **`choose_emoji_reply_intent()`** — LLM отвечает на эмодзи-сообщения
- **`pick_media_item()`** — выбор конкретного стикера по intent'у (emoji match, kind filter)
- **`send_media_item()`** — отправка стикера или custom emoji в чат

### `file_context.py` (~427 строк)
RAG-lite система для knowledge-файлов:

- **`sync_context_dir()`** — сканирует `knowledge/` каждые 15 сек, индексирует файлы в SQLite
- **`chunk_text()`** — разбивка текста на чанки (900 chars, 120 overlap)
- **`retrieve_context_snippets()`** — keyword-based retrieval с scoring:
  - Exact query match: +4.0
  - Path match: +5.0
  - Token match: +1.25 (text) / +3.0 (path)
  - Lore boost: +2.0 для файлов с префиксом `lore_`
- **`build_file_context_block()`** — форматирование snippets для промпта (lore vs обычные файлы)

### `lore_context.py` (~44 строки)
Загрузка текстового файла с бэкграундом персонажа (кэширование по mtime).

### `profile_context.py` (~59 строк)
Загрузка JSON-профиля создателя (Захара) и форматирование в системный блок.

### `user_profile.py` (~386 строк)
Команда `/profile` — полный CRUD профиля пользователя:

- Субкоманды: show, help, set, patch (bulk), delete, clear
- Алиасы полей на русском и английском ("имя" -> "approved_display_name")
- Валидация и нормализация значений
- Extra fields в JSON (about, age, city, location, occupation)

### `router.py` (~65 строк)
Простой роутер (legacy, частично заменен логикой в main.py):
- Mention detection
- Reply-to-bot detection
- Per-user cooldown

### `config.py` (~203 строки)
Загрузка конфигурации из `.env`:

| Переменная | Описание | Default |
|-----------|----------|---------|
| `BOT_TOKEN` | Telegram bot token | (required) |
| `ALLOWED_CHAT_IDS` | Whitelist chat IDs | "" |
| `LLM_BASE_URL` | URL LLM API | `http://127.0.0.1:8080/v1` |
| `LLM_API_KEY` | API key для LLM | `local-llm` |
| `LLM_MODEL` | Модель LLM | `Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf` |
| `LLM_CONTEXT_WINDOW` | Размер контекстного окна | 12000 |
| `DB_PATH` | Путь к SQLite | `./alta_tg_test.db` |
| `HISTORY_LIMIT` | Лимит сообщений в истории | 40 |
| `TEMPERATURE` | Температура LLM | 0.35 |
| `MAX_TOKENS` | Макс. токенов в ответе | 200 |
| `USER_COOLDOWN_SEC` | Cooldown между ответами | 15 |
| `CREATOR_USER_ID` | Telegram ID создателя | — |
| `ELEVENLABS_API_KEY` | API key для TTS | — |
| `ELEVENLABS_VOICE_ID` | Voice ID для TTS | — |
| `VOICE_REPLY_PROB` | Вероятность голосового ответа | 0.18 |
| `MEDIA_REPLY_PROB` | Вероятность стикера после ответа | 0.22 |
| `KNOWLEDGE_DIR` | Директория с knowledge-файлами | `./knowledge` |

### `telemetry.py` (~280 строк)
JSONL-логирование в `orchestration.log`:

- **`TurnMetrics`** — dataclass со всеми метриками turn'а (LLM calls, токены, латентность, обновления памяти)
- **`JSONLLogger`** — thread-safe append-only JSONL writer
- События: `turn_start`, `context_snapshot`, `llm_call`, `turn_end`

### `runtime_io.py` (~14 строк)
Reconfigure stdout/stderr в UTF-8 для Windows.

---

## Knowledge-система

Директория `knowledge/` содержит текстовые файлы, которые автоматически индексируются и используются как контекст:

- **`lore_*.txt`** — файлы с префиксом `lore_` получают boost в scoring и подаются как внутренний бэкграунд персонажа:
  - `lore_core.txt` — кто такая Альта, связь с каналом JJKennedy
  - `lore_appearance.txt` — внешность
  - `lore_speech.txt` — стиль речи
  - `lore_music.txt` — музыкальные вкусы
  - `lore_food.txt` — еда
  - `lore_opinions.txt` — мнения
  - `lore_tech.txt` — отношение к технологиям
- Обычные файлы (`.txt`, `.md`, `.json`, `.py` и т.д.) — используются как справочный контекст

---

## Система памяти

Бот запоминает информацию о каждом пользователе на двух уровнях:

1. **User Facts** (`user_facts` таблица) — структурированные факты с типами:
   - `interest`, `preference`, `work`, `study`, `location`, `goal`, `relationship`, `bio`, `fact`
   - Каждый факт имеет `confidence` (0-1) и `importance` (0-1)
   - Извлекаются автоматически через LLM из сообщений (в фоне после ответа)
   - Pruning: хранится до 80 фактов на пользователя

2. **User Profiles** (`user_profiles` таблица) — агрегированный профиль:
   - Имя, ник, пол, интересы, стиль общения, предпочтения к ответам
   - Обновляется через LLM-extraction и команду `/profile`
   - `approved_display_name` — имя, одобренное создателем

3. **User Documents** (`user_documents` таблица) — досье пользователя:
   - Автоматически пересобирается из фактов + профиля
   - Инжектится в промпт при релевантных запросах

4. **Summaries** — сжатие истории:
   - `chat_summary` — общее резюме чата (обновляется каждые N сообщений)
   - `user_chat_summaries` — per-user резюме (не шарится между пользователями)

---

## Anti-spam / Верификация

При входе нового пользователя в чат:
1. Бот удаляет сервисное сообщение о входе
2. Mute пользователя (только текст разрешен)
3. Генерирует 4-значный код
4. Пользователь должен отправить код за N секунд
5. Если не прошел — бан. Если отправил медиа/ссылки — мгновенный бан
6. Бот-аккаунты банятся сразу
7. `/unban <user_id|@username>` — разбан (только админы)

---

## Голосовые ответы

- Распознавание запросов на русском: "запиши голосом", "скажи войсом", "озвучь"
- Direct mode: "озвучь: конкретный текст" — озвучивает указанный текст
- Reply mode: "ответь голосом" — генерирует ответ и озвучивает
- Random voice: с вероятностью `VOICE_REPLY_PROB` бот отвечает голосом
- TTS: ElevenLabs API (настраивается через env vars)

---

## Стек технологий

- **Python 3.11+**
- **aiogram 3.4** — Telegram Bot API
- **aiosqlite** — async SQLite
- **openai** — OpenAI-compatible API client (для локальной LLM)
- **python-dotenv** — загрузка .env
- **ElevenLabs API** — TTS (опционально)
- **LLM**: любая OpenAI-compatible модель (по умолчанию Mistral Small через llama.cpp)

---

## Запуск

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env (см. config.py для всех переменных)
cp .env.example .env
# Заполнить BOT_TOKEN, ALLOWED_CHAT_IDS, etc.

# 3. Запустить LLM сервер (llama.cpp или совместимый)
# llama-server -m model.gguf --port 8080

# 4. Запустить бота
python main.py
```

---

## Тесты

```bash
pytest tests/
```

Тесты покрывают:
- `test_agent_history.py` — сборка и обрезка истории
- `test_agent_memory.py` — извлечение и хранение фактов
- `test_agent_repeat_guard.py` — защита от повторений
- `test_db_migration.py` — миграции схемы
- `test_db_user_profiles.py` — профили пользователей
- `test_llm_overflow.py` — обработка context overflow
- `test_main_daily_promo.py` — промо-функционал
- `test_main_join_verification.py` — антибот верификация
- `test_main_moderation_commands.py` — команды модерации
- `test_main_sanitize.py` — санитизация текста
- `test_main_unban.py` — разбан
- `test_user_profile_commands.py` — команда /profile
