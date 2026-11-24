# Alta — AI-персонаж для Telegram

Alta (Альта) — Telegram-бот с характером. Это не ассистент и не чат-бот в обычном понимании: Alta — самостоятельный персонаж женского пола (~20 лет), который живет в групповых чатах, запоминает собеседников, отвечает голосом, шлет стикеры и ведет себя как живой участник.

Бот работает через локальную LLM (по умолчанию Mistral через llama.cpp / OpenAI-compatible API), хранит все данные в SQLite и не зависит от внешних облачных AI-сервисов.

---

## Архитектура (высокоуровневая)

```
Telegram (aiogram)
       |
   main.py              — точка входа, Dispatcher, обработка сообщений/реакций/верификации/ЛС
       |
   agent.py             — оркестратор: re-export из суб-модулей, run_agent() точка входа
       |
   +-- context_builder.py     — сборка системного промпта, история, mood state Альты
   +-- generation_pipeline.py — LLM генерация, web search, repeat rewrite, edit_answer
   +-- memory_engine.py       — факты, профили, summary, background extraction
   +-- postprocessing.py      — очистка ответа, repeat guard, gender fix, shrink
   +-- llm.py                 — обертка над OpenAI-compatible API (retry, overflow protection)
   +-- db.py                  — SQLite через aiosqlite (все таблицы и CRUD)
   +-- prompts.py             — системный промпт персонажа
   +-- voice.py               — TTS через ElevenLabs API
   +-- media.py               — стикеры/кастомные эмодзи: каталог, LLM-выбор, отправка
   +-- file_context.py        — RAG-lite: индексация knowledge-файлов, chunk scoring, retrieval
   +-- lore_context.py        — helper для загрузки лора из файла (legacy)
   +-- profile_context.py     — загрузка JSON-профиля создателя
   +-- user_profile.py        — команда /profile: парсинг, валидация, CRUD полей профиля
   +-- private_profile.py     — личный кабинет в ЛС: inline-меню, валидация, импорт из групп
   +-- cooldowns.py           — CooldownManager, LockRegistry, per-chat locks
   +-- moderation.py          — JoinVerification, join/unban логика, spam detection
   +-- sanitization.py        — sanitize_incoming/outgoing_text, prompt injection protection
   +-- search.py              — веб-поиск (Yandex API + Brave fallback), result parsing, cache
   +-- router.py              — legacy роутер: mention/reply detection
   +-- config.py              — загрузка конфига из .env
   +-- telemetry.py           — JSONL-логирование метрик каждого turn'а
   +-- runtime_io.py          — UTF-8 reconfigure для Windows stdout/stderr

launcher.py          — TUI control panel (Textual): запуск бота, туннель, llama-server, stats
dashboard.py         — HTTP dashboard: статистика из orchestration.log и SQLite
alta_console.py      — консольный интерфейс для прямого общения с Альтой без Telegram

knowledge/           — текстовые файлы лора и контекста (автоиндексация)
scripts/             — утилиты: init_db, build_media_catalog, analyze_log, debug_dashboard, simulate_chat
tests/               — unit-тесты (pytest)
```

---

## Поток обработки сообщения (пошагово)

```
1. Telegram message приходит в on_message (main.py)
2. Проверки: allowed chat? not private? not bot? not pending verification?
3. /profile команда? → обрабатывается без LLM, выход
4. is_trigger() — бот реагирует на @mention, reply на своё сообщение, или имя "Альта"/"альт"/etc.
5. cooldown_ok() — per-user rate limit (cooldowns.py)
6. sanitize_incoming_text() — удаление injection-попыток (sanitization.py)
7. detect_voice_request() — распознавание запроса на голосовой ответ (voice.py)
8. extract_requested_display_name() — распознавание "зови меня X"
9. run_agent() вызывается (agent.py):
   a. maybe_store_identity_from_telegram() — синхронизация username/first_name в память
   b. sync_user_document() — пересборка досье пользователя
   c. should_ask_intro() — проверка, нужно ли спросить "кто ты?"
   d. _generate_answer_with_fallback() — основной цикл генерации:
      - build_context() собирает system prompt из блоков:
        * ALTA_SYSTEM_PROMPT (характер)
        * current_speaker (кто пишет)
        * identity (профиль собеседника — merged из приватного + группового)
        * participants (другие участники чата)
        * speaker_summary + chat_summary (сжатая история)
        * user_facts/memory (факты о пользователе)
        * user_document (досье)
        * reply_context (если ответ на чьё-то сообщение)
        * file_context/RAG (релевантные куски из knowledge/)
        * lore (бэкграунд персонажа)
        * creator_profile (профиль создателя)
        * alta_state (внутреннее состояние — тон, настроение, реакция)
      - build_context() также обновляет mood state (детерминированно, без LLM):
        * читает alta_chat_state + alta_user_relation из DB
        * применяет временное затухание
        * вычисляет новые значения по сигналам из входящего текста
        * сохраняет обратно в DB, добавляет краткоживущую реакцию если нужно
      - _compose_system_prompt() — финальная склейка
      - llm.chat() — вызов LLM
      - 4 уровня fallback при context overflow: full → compact → minimal → emergency
   e. _deterministic_postprocess_answer() — постобработка:
      * удаление эмодзи, списков, self-reference ("я бот")
      * fix_gender (мужской род → женский для "я")
      * shrink длинных ответов
   f. Repeat guard — если ответ повторяет недавние, перегенерация
   g. Intro generation — для новых пользователей
   h. _background_extraction() — в фоне после ответа:
      * maybe_store_memories() — извлечение фактов из сообщения через LLM
      * maybe_store_user_profile() — обновление профиля через LLM
      * maybe_update_summary() — обновление summary чата каждые N сообщений
10. Постобработка ответа: sanitize_outgoing_text(), strip mentions
11. Отправка: текст или голос (ElevenLabs TTS, случайно или по запросу)
12. Опционально: отправка стикера/кастомного эмодзи (LLM выбирает)
13. turn_metrics.finish() — запись телеметрии

Приватный чат (ЛС бота):
1. on_private_message / on_pm_callback — обработка без LLM
2. Inline-меню профиля: просмотр, редактирование, импорт из групп
3. Никакого вызова LLM или run_agent()
```

---

## Модули (подробно)

### `main.py`
Точка входа. Инициализация бота (aiogram), LLM, DB, TTS, media catalog. Содержит:

- **`on_new_chat_members()`** — антибот верификация: mute + код из 4 цифр + timeout + автобан
- **`on_message()`** — главный handler: маршрутизация, /profile, генерация, голос, медиа; включает `_keep_typing()` — фоновая задача обновляет typing-индикатор каждые 4 с пока агент работает
- **`on_private_message()`** — handler для ЛС: профиль через inline-меню, без LLM
- **`on_pm_callback()`** — обработка inline-кнопок личного кабинета
- **`on_message_reaction()`** — реакция на реакции пользователей (с вероятностью)
- **Name approval система** — запрос имени → одобрение создателем → `/approve_name` / `/deny_name`
- **`/profile` команда** — делегируется в `user_profile.py`

### `agent.py`
Тонкий оркестратор. Содержит только `run_agent()` и re-export всех публичных символов из суб-модулей — для обратной совместимости с `main.py`.

### `context_builder.py`
Сборка контекста для LLM. Ключевые функции:

- **`build_context()`** — главная функция: собирает все system_blocks, историю, обновляет mood state; ответы Альты другим участникам группы помечаются `[->id=X]` — модель знает, кому был адресован каждый её ответ
- **`_build_context_plans()`** — 4 плана (full/compact/minimal/emergency) с разными бюджетами
- **`build_identity_block()`** — блок профиля собеседника
- **`build_current_speaker_block()`** — блок текущего отправителя (speaker guard)
- **`build_chat_participants_block()`** — блок других участников чата
- **`build_reply_context_block()`** / **`build_implicit_reply_context_block()`** — контекст ответа на сообщение
- **Mood system** (детерминированная, без LLM):
  - `compute_state_updates()` — вычисляет изменения состояния по сигналам из текста
  - `apply_time_decay_chat()` / `apply_time_decay_relation()` — временное затухание
  - `build_alta_state_block()` — форматирует состояние в компактный блок для промпта

### `generation_pipeline.py`
LLM-генерация ответа:

- **`_generate_answer_with_fallback()`** — главный цикл: попытка с 4 уровнями контекста при overflow
- **`_compose_system_prompt()`** — склейка system_blocks в финальный промпт
- **`edit_answer()`** — постредактирование через LLM при нарушениях стиля
- **`generate_simple_reply_fallback()`** — аварийная генерация без контекста
- **`_rewrite_repeated_reply()`** — перегенерация при повторе ответа
- Web search делегируется в `search.py`; `_answer_defers_web_lookup()`, `_pick_web_search_query()`, `_should_force_web_fallback()` — маршрутизация поиска

### `search.py`
Веб-поиск, выделен из generation_pipeline в отдельный модуль:

- **Yandex Search API** — основной бэкенд (YC API key)
- **Brave Search** — fallback (primp HTTP client)
- **Кэш результатов** — TTL 300 секунд, per-query
- **Авто-детект языка запроса** — по доле кириллицы, переключает entre Yandex / Brave
- **Result parsing** — извлечение сниппетов из XML (Yandex) и JSON (Brave)
- Константы: `WEB_RESULT_MAX_SNIPPETS = 3`, `WEB_RESULT_CHAR_LIMIT = 1000`, `WEB_SEARCH_TIMEOUT = 8.0`

### `memory_engine.py`
Факты, профили, summary:

- **`maybe_store_memories()`** — LLM извлекает факты, сохраняет в `user_facts`
- **`maybe_store_user_profile()`** — LLM обновляет профиль (interests, style). `relationship_notes` **не** обновляется автоматически
- **`maybe_update_summary()`** — периодическое сжатие истории (каждые N сообщений)
- **`_build_extraction_text()`** — формирует текст для экстракции с разметкой по владельцу (`Пользователь` / `Другой участник чата` / `Альта`)
- **`format_user_facts()`** — форматирование фактов для промпта
- **`sync_user_document()`** / **`build_user_document_payload()`** — досье пользователя

### `postprocessing.py`
Постобработка ответа:

- **`_deterministic_postprocess_answer()`** — главная функция: эмодзи, списки, self-reference, gender fix, длина
- **`_fix_gender()`** — замена мужского рода на женский: покрывает формы с "я" ("я понял" → "я поняла") и без — глаголы в начале ответа или после запятой
- **`_is_repeated_recent_reply()`** / **`_repeat_guard_hard_fallback()`** — защита от повторов
- **`_fit_text()`** / **`_clip_text()`** — обрезка текста с учётом предложений

### `llm.py`
OpenAI-compatible API клиент:

- **`LLM`** — класс-обертка над `AsyncOpenAI`
- **`chat()`** — основной метод: retry с exponential backoff (до 7 попыток), context overflow detection, телеметрия
- **`_ensure_context_budget()`** — pre-flight проверка размера контекста перед вызовом API
- **`ContextOverflowError`** — кастомное исключение для переполнения контекста
- **`estimate_chat_tokens()`** — эвристическая оценка токенов (chars / 3.2)

### `cooldowns.py`
Rate limiting и per-chat блокировки. Выделено из `main.py`:

- **`CooldownManager`** — per-key rate limiter с TTL-eviction, max 5000 записей
- **`LockRegistry`** — per-chat `asyncio.Lock` для сериализации turn'ов (один ответ за раз на чат)
- Используется в `main.py` для `USER_COOLDOWN_SEC` и предотвращения параллельных turn'ов

### `moderation.py`
Логика модерации, выделено из `main.py`:

- **`JoinVerification`** — dataclass состояния ожидающей верификации (user_id, chat_id, код, дедлайн)
- **`_pending_join_verification`** — in-memory dict `{(chat_id, user_id): JoinVerification}`; сбрасывается при рестарте
- **`generate_verification_code()`** — 4-значный числовой код
- **`check_verification_code()`** — проверка кода из текста сообщения
- **`is_spam_message()`** — детект ссылок/инвайтов в сообщениях новых участников
- **`parse_unban_target()`** — парсинг `/unban <user_id|@username>`

### `sanitization.py`
Защита от prompt injection. Выделено из `main.py`:

- **`sanitize_incoming_text()`** — удаление `[id=... @user]` тегов, `system:` / `assistant:` префиксов, мета-строк профилей перед подачей в LLM
- **`sanitize_outgoing_text()`** — очистка ответа бота: удаление тегов, role-префиксов
- **`is_trigger()`** — определяет, нужно ли боту реагировать (mention, reply, alias)
- **`strip_mention()`** — убирает @botname и алиасы из начала сообщения

### `db.py`
SQLite storage через aiosqlite. Таблицы:

| Таблица | Назначение |
|---------|-----------|
| `allowed_chats` | Whitelist чатов |
| `users` | Telegram-пользователи (user_id, username, timestamps) |
| `messages` | История сообщений (chat_id, user_id, role, text, target_user_id) |
| `chat_summary` | Сжатое резюме чата |
| `user_chat_summaries` | Per-user сжатое резюме диалога |
| `user_profiles` | Профили пользователей (имя, пол, интересы, стиль, extra_fields JSON) |
| `pending_name_approvals` | Запросы на смену имени (ожидание одобрения создателем) |
| `user_documents` | Досье пользователей (JSON + текст) |
| `user_facts` | Извлечённые факты о пользователях (type, value, confidence, importance) |
| `memories` | Legacy key-value память (мигрируется в user_facts) |
| `debug_traces` | Debug-трейсы turn'ов (retention: 30 строк) |
| `cooldown_state` | Персистентные cooldown'ы |
| `context_files` + `context_chunks` | Индекс knowledge-файлов и их чанков |
| `alta_chat_state` | Состояние Альты в чате (energy, irritation, playfulness, social_battery) |
| `alta_user_relation` | Отношение к пользователю (trust, warmth, caution, annoyance, interaction_count) |
| `alta_reactions` | Краткоживущие реакции с TTL (reaction_kind, reaction_strength, ttl_messages) |

Ключевые методы:
- `get_last_messages()` — история с опциональной фокусировкой на конкретном пользователе; возвращает `user_id` и `target_user_id`
- `get_chat_participants()` — активные участники через bounded scan
- `get_user_profile_merged(chat_id, user_id)` — слияние группового + приватного профиля; приватный (user-set) имеет приоритет
- `get_all_group_profiles_merged(user_id)` — слияние профилей из всех групп (для импорта)
- `count_user_facts_global(user_id)` — количество фактов по всем групповым чатам
- `upsert_user_fact()` — upsert с заменой confidence/importance на новые значения при конфликте
- `prune_user_facts()` — обрезка до лимита по importance/confidence
- `rotate_messages()` — ротация старых сообщений (keep=2000)
- `get_alta_chat_state()` / `save_alta_chat_state()` — mood: фон чата
- `get_alta_user_relation()` / `save_alta_user_relation()` — mood: отношение к пользователю
- `add_alta_reaction()` / `tick_alta_reactions()` / `get_active_alta_reaction()` — краткоживущие реакции
- Auto-migration: `_migrate_schema()` добавляет новые колонки/таблицы автоматически

### `voice.py`
Text-to-speech через ElevenLabs:

- **`detect_voice_request()`** — распознавание запросов на голос в русском тексте ("запиши голосом", "скажи войсом", "озвучь", "войсом" и т.д.)
- **`ElevenLabsTTS`** — синтез речи через ElevenLabs REST API (urllib, не SDK)
- Поддержка direct text ("озвучь: текст") и reply mode ("ответь голосом")
- Random voice replies: с заданной вероятностью бот отвечает голосом

### `media.py`
Стикеры и кастомные эмодзи:

- **`build_media_catalog()`** — сканирование стикер-паков через Telegram API
- **`choose_media_intent()`** — LLM решает, нужен ли стикер после ответа
- **`choose_emoji_reply_intent()`** — LLM отвечает на эмодзи-сообщения
- **`pick_media_item()`** — выбор конкретного стикера по intent'у (emoji match, kind filter)
- **`send_media_item()`** — отправка стикера или custom emoji в чат

### `file_context.py`
RAG-lite система для knowledge-файлов:

- **`sync_context_dir()`** — сканирует `knowledge/` каждые 15 сек, индексирует файлы в SQLite
- **`chunk_text()`** — разбивка текста на чанки (900 chars, 120 overlap)
- **`retrieve_context_snippets()`** — keyword-based retrieval с scoring:
  - Exact query match: +4.0 / Path match: +5.0 / Token match: +1.25 (text) / +3.0 (path) / Lore boost: +2.0
- **`build_file_context_block()`** — форматирование snippets для промпта (lore vs обычные файлы)

### `user_profile.py`
Команда `/profile` — полный CRUD профиля пользователя в групповом чате:

- Субкоманды: show, help, set, patch (bulk), delete, clear
- Алиасы полей на русском и английском ("имя" → "approved_display_name")
- Extra fields в JSON-колонке `extra_fields` (about, age, city, location, occupation)
- `apply_profile_updates()` — bulk update из списка изменений
- `format_profile_view()` — форматирование профиля для вывода пользователю

### `private_profile.py`
Личный кабинет — управление профилем через ЛС бота без LLM:

- **Inline-меню** (aiogram InlineKeyboardMarkup): главное меню, меню редактирования, выбор пола
- **In-memory state** `_pm_state: dict[int, dict]` — отслеживает, какое поле редактирует пользователь
- **Rate limiter** — макс. 30 сохранений/час на пользователя
- **`validate_field(field, value)`** — блокирует ссылки, HTML-теги, управляющие символы, Telegram-команды; лимиты длины по полю
- **`import_from_groups(db, user_id)`** — копирует поля из групповых профилей в приватный, не перезаписывая вручную заданные
- **`format_import_preview()`** — превью данных из групп: ✓ = уже в личном профиле, → = будет добавлено

### `launcher.py`
TUI control panel на Textual. Запускается напрямую (`python launcher.py`):

- **`AltaPanel`** — Textual App с sidebar, логом и панелью управления
- **Статусы сервисов** — llama-server (HTTP health-check), SSH туннель (socket probe), Alta бот (PowerShell process query)
- **Статистика серверов** — SSH-запрос CPU/RAM/GPU/VRAM/Temp с LAN-сервера и VPS каждые 15 сек
- **Model selector** — читает список `.gguf` файлов с сервера через SSH, обновляет `LLM_MODEL` в `.env`
- **"Запустить всё"** — выбор модели → обновление `.env` → SSH туннель → `start_alta.sh` на сервере → ожидание llama-server (до 90 сек) → запуск `main.py`
- **Log streaming** — stdout/stderr бота в реальном времени с цветовой разметкой
- **Прямое SSH** — открывает Windows Terminal / cmd с SSH-сессией на сервер или VPS
- Горячие клавиши: `r` рестарт, `s` запустить всё, `q` выход

### `dashboard.py`
HTTP-дашборд. Запускается отдельно, слушает `127.0.0.1:8765`:

- Читает `orchestration.log` (JSONL) и SQLite напрямую — нет зависимости от запущенного бота
- Статистика turn'ов: LLM latency, token usage, memory updates
- Топ пользователей по активности
- Последние сообщения и debug-трейсы
- Доступен в браузере без авторизации (localhost only)

### `alta_console.py`
Консольный интерфейс для прямого общения с Альтой без Telegram и без запуска бота:

- Подключается к той же БД и тому же LLM-серверу что и основной бот
- Полезен для разработки и отладки промптов
- Поддерживает весь тот же pipeline (context_builder, generation_pipeline, mood, memory)

### `config.py`
Загрузка конфигурации из `.env`:

| Переменная | Описание | Default |
|-----------|----------|---------|
| `BOT_TOKEN` | Telegram bot token | (required) |
| `ALLOWED_CHAT_IDS` | Whitelist chat IDs | "" |
| `LLM_BASE_URL` | URL LLM API | `http://127.0.0.1:8080/v1` |
| `LLM_API_KEY` | API key для LLM | `local-llm` |
| `LLM_MODEL` | Модель LLM | `Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf` |
| `LLM_CONTEXT_WINDOW` | Размер контекстного окна | 12000 |
| `LLM_CONTEXT_SAFETY_MARGIN` | Резерв токенов для защиты от overflow | 512 |
| `DB_PATH` | Путь к SQLite | `./alta_tg_test.db` |
| `HISTORY_LIMIT` | Лимит сообщений в истории | 40 |
| `TEMPERATURE` | Температура LLM | 0.35 |
| `TOP_P` | Top-p sampling | 0.85 |
| `MAX_TOKENS` | Макс. токенов в ответе | 350 |
| `USER_COOLDOWN_SEC` | Cooldown между ответами | 15 |
| `CREATOR_USER_ID` | Telegram ID создателя | — |
| `ELEVENLABS_API_KEY` | API key для TTS | — |
| `ELEVENLABS_VOICE_ID` | Voice ID для TTS | — |
| `VOICE_REPLY_PROB` | Вероятность голосового ответа | 0 |
| `MEDIA_REPLY_PROB` | Вероятность стикера после ответа | 0.22 |
| `KNOWLEDGE_DIR` | Директория с knowledge-файлами | `./knowledge` |
| `YANDEX_SEARCH_API_KEY` | API key для Yandex Search | — |
| `YANDEX_FOLDER_ID` | Folder ID для Yandex Search | — |

### `telemetry.py`
JSONL-логирование в `orchestration.log`:

- **`TurnMetrics`** — dataclass со всеми метриками turn'а (LLM calls, токены, латентность, обновления памяти)
- **`JSONLLogger`** — thread-safe append-only JSONL writer
- События: `turn_start`, `context_snapshot`, `llm_call`, `turn_end`

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

## Система памяти и профилей

### 1. User Facts (`user_facts`)
Структурированные факты с типами: `interest`, `preference`, `work`, `study`, `location`, `goal`, `bio`, `fact`
- Каждый факт имеет `confidence` (0–1) и `importance` (0–1)
- Извлекаются автоматически через LLM из сообщений (в фоне после ответа)
- Pruning: хранится до 80 фактов на пользователя (`FACT_KEEP_PER_USER`)
- **Защита от загрязнения**: `_build_extraction_text()` явно размечает сообщения других участников, LLM не приписывает чужие факты целевому пользователю
- **Постфильтр**: `_FACT_VALUE_STOP_PATTERNS` отсекает мусорные факты (roleplay, транзиентные состояния)

### 2. User Profiles (`user_profiles`)
Агрегированный профиль: имя, ник, пол, интересы, стиль общения, предпочтения, extra_fields JSON
- Групповой профиль `(chat_id, user_id)` — обновляется автоматически через LLM
- Приватный профиль `(user_id, user_id)` — задаётся вручную через ЛС бота
- `get_user_profile_merged()` — итоговый профиль: приватный (user-set) **всегда побеждает** групповой (auto-extracted)
- `relationship_notes` **не обновляется автоматически** — только вручную

### 3. User Documents (`user_documents`)
Досье пользователя — автоматически пересобирается из фактов + профиля, инжектится в промпт при релевантных запросах.

### 4. Summaries
- `chat_summary` — общее резюме чата (обновляется каждые N сообщений)
- `user_chat_summaries` — per-user резюме (не шарится между пользователями)

### Личный кабинет (ЛС бота)
Пользователь пишет боту в личку → inline-меню без LLM:
- Просмотр текущего профиля
- Редактирование любого поля через inline-меню
- Кнопка **"📥 Что обо мне знает Альта"** — показывает данные собранные из групп: ✓ = уже есть, → = можно импортировать
- Импорт не перезаписывает вручную заданные значения

---

## Система внутреннего состояния Альты (mood system)

Детерминированная система без отдельных LLM-вызовов. Реализована в `context_builder.py`, данные в SQLite.

### Три уровня состояния

**Состояние чата** (`alta_chat_state`, per chat_id):
- `energy` — общая энергия (0–1, базовое 0.7)
- `irritation` — раздражённость (0–1)
- `playfulness` — игривость (0–1, базовое 0.5)
- `social_battery` — заряд социальной батареи (0–1, базовое 0.8)

**Отношение к пользователю** (`alta_user_relation`, per chat_id+user_id):
- `trust` — доверие (0–1, базовое 0.5)
- `warmth` — теплота (0–1, базовое 0.5)
- `caution` — осторожность (0–1)
- `annoyance` — раздражение конкретным человеком (0–1)
- `interaction_count` — счётчик взаимодействий

**Краткоживущие реакции** (`alta_reactions`, TTL 1–3 сообщения):
- Виды: `amused`, `curious`, `annoyed`, `concerned`, `embarrassed`
- Привязаны к конкретному пользователю (`target_user_id`)
- Автоматически удаляются по истечении TTL

### Сигналы

| Сигнал | Паттерны | Эффект |
|--------|----------|--------|
| Агрессия | мат, оскорбления, "заткнись" | ↑ irritation, ↑ annoyance, ↓ energy, ↓ trust |
| Благодарность | "спасибо", "thanks" | ↑ warmth, ↑ trust, реакция `amused` |
| Творческий | вопросы, "объясни", "придумай" | ↑ playfulness, реакция `curious` |

### Drain (energy / social_battery)
- energy: −0.003/msg (при агрессии дополнительно −0.03); восстановление +0.08/час
- social_battery: −0.005/msg (при агрессии −0.06); восстановление +0.12/час

### Обновление на каждом ходу
1. Чтение состояния из DB
2. Временное затухание (irritation/annoyance экспоненциально, energy/battery time-based)
3. Вычисление новых значений по сигналам входящего сообщения
4. Сохранение в DB (background task, не блокирует ответ)
5. Добавление краткоживущей реакции (если нужно) + декремент TTL
6. Форматирование в блок `[Твоё состояние — влияй на тон, не упоминай вслух]`

Блок добавляется только если состояние отклоняется от нейтрального. Числа никогда не попадают в промпт.

---

## Anti-spam / Верификация

При входе нового пользователя:
1. Бот удаляет сервисное сообщение о входе
2. Mute пользователя
3. Генерирует 4-значный код
4. Пользователь должен отправить код за N секунд
5. Если не прошёл — бан. Если отправил медиа/ссылки — мгновенный бан
6. Бот-аккаунты банятся сразу
7. `/unban <user_id|@username>` — разбан (только админы)

`_pending_join_verification` хранится только в памяти — сбрасывается при рестарте.

---

## Запуск

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env
cp .env.example .env
# Заполнить BOT_TOKEN, ALLOWED_CHAT_IDS, etc.

# 3. Запустить LLM сервер (llama.cpp или совместимый)
llama-server -m model.gguf --port 8080

# 4. Запустить бота
python main.py
```

**Control panel (рекомендуется):**
```bash
python launcher.py
```
Открывает TUI с управлением llama-server, SSH-туннелем и ботом в одном интерфейсе.

**Dashboard (опционально):**
```bash
python dashboard.py
# → http://127.0.0.1:8765
```

---

## Тесты

```bash
pytest tests/
```

Тесты покрывают:
- `test_agent_current_datetime.py` — детерминированные ответы по текущей дате/времени
- `test_agent_repeat_guard.py` — защита от повторов
- `test_agent_summary_interval.py` — периодичность обновления summary
- `test_agent_web_search_safety.py` — безопасный веб-поиск
- `test_context_flirt_policy.py` — политика флирта/исключения
- `test_db_migration.py` — миграции схемы БД
- `test_db_user_profiles.py` — профильные CRUD/merge кейсы
- `test_fixes.py` — регрессии по критическим фиксам
- `test_llm_overflow.py` — обработка context overflow
- `test_main_daily_promo.py` — промо-функционал
- `test_main_join_verification.py` — anti-spam / join verification
- `test_main_moderation_commands.py` — команды модерации
- `test_main_sanitize.py` — санитизация входа/выхода
- `test_main_unban.py` — разбан пользователей
- `test_memory_extraction_text.py` — разметка extraction-контекста по владельцу сообщений
- `test_mood.py` — mood system (state update, decay, prompt block, DB CRUD)
- `test_private_profile_import.py` — импорт профиля из групп в ЛС-профиль
- `test_search_relevance.py` — релевантность поиска
- `test_search_routing.py` — маршрутизация веб-поиска
- `test_stress.py` — нагрузочные/оценочные сценарии
- `test_user_profile_commands.py` — команда `/profile`

---

## Стек технологий

- **Python 3.11+**
- **aiogram 3.4** — Telegram Bot API
- **aiosqlite** — async SQLite
- **openai** — OpenAI-compatible API client (для локальной LLM)
- **python-dotenv** — загрузка .env
- **textual** — TUI для launcher.py
- **httpx** — async HTTP (launcher.py)
- **requests + primp** — web search backends (Yandex API / Brave fallback)
- **ElevenLabs API** — TTS (опционально)
- **LLM**: любая OpenAI-compatible модель (по умолчанию Mistral Small через llama.cpp)
