# TODO — Фиксы Alta

## КРИТИЧЕСКИЕ 

- [x] **#1 CHARS_PER_TOKEN_ESTIMATE слишком агрессивный** — единая константа 3.2 в telemetry.py, llm.py импортирует оттуда.

- [x] **#2 Summary self-poisoning (двойное summary)** — убран fallback в `get_summary()`. Если нет user-specific summary — возвращает "".

- [x] **#3 Системный промпт перегружен метаданными** — добавлен разделитель "КОНТЕКСТ" между personality и блоками + reminder в конце.

- [x] **#4 `_looks_like_context_dependent_reply()` отсекает вопросы** — убрана проверка `if "?" in value`.

- [x] **#5 MAX_TOKENS по умолчанию 200 → 350** — compact 280, minimal 180, emergency 120.

- [x] **#6 Постпроцессинг слишком агрессивно режет** — MAX_EDITOR_REPLY_CHARS 400→700, MAX_EDITOR_SENTENCES 5→8. Self-reference: замена слов вместо удаления предложений.

## ВЫСОКИЙ ПРИОРИТЕТ

- [x] **#7 REPEAT_REWRITE_PROMPT на английском** — переписан на русском.

- [x] **#8 Lore забивает RAG-слоты** — lore добавляется отдельно, не считается в max_chunks лимите.

- [x] **#9 Факты извлекаются слишком редко** — AND → OR (first_person или fact_marker).

- [x] **#10 upsert_user_fact — garbage never degrades** — MAX(confidence) → excluded.confidence.

- [x] **#11 Нет инструкции "отвечай на русском"** — добавлено в ALTA_SYSTEM_PROMPT.

- [x] **#12 Смешение языков в system_blocks** — переведены английские инструкции в file_context.py.

- [x] **#22 Профиль пользователя нельзя настроить удобно** — реализован личный кабинет в ЛС бота (private_profile.py): inline-меню, все поля, валидация, rate limit.

- [x] **#23 Данные собранные в группе недоступны пользователю** — кнопка "📥 Что обо мне знает Альта" в ЛС: превью собранных данных с разметкой ✓/→, импорт без перезаписи user-set значений.

- [x] **#24 Загрязнение профиля чужими данными из группового чата** — `_build_extraction_text()` теперь явно размечает сообщения: "Пользователь" vs "Другой участник чата". LLM не приписывает чужие факты целевому пользователю.

- [x] **#25 `relationship_notes` пишет транзиентные состояния** — поле убрано из авто-экстракции (`maybe_store_user_profile`). Доступно только для ручного заполнения через /profile. Исключено из импорт-превью и импорта в приватный профиль.

- [x] **#26 Неверный приоритет мержа профилей** — `get_user_profile_merged()` теперь: приватный профиль (user-set) всегда побеждает авто-извлечённый групповой. Ранее было наоборот.

- [x] **#29 Внутреннее состояние персонажа (mood system)** — реализована детерминированная система без отдельных LLM-вызовов:
  - `alta_chat_state` (per chat): energy, irritation, playfulness, social_battery
  - `alta_user_relation` (per chat+user): trust, warmth, caution, annoyance, interaction_count
  - `alta_reactions`: краткоживущие реакции с TTL (amused, curious, annoyed, concerned, embarrassed)
  - Сигналы: агрессия / благодарность / творческий запрос → детерминированные изменения
  - Временное затухание: раздражение падает, энергия восстанавливается
  - Блок `[Твоё состояние]` в промпте (только при noteworthy отклонении от нейтрали)
  - покрытие в `tests/test_mood.py`

## СРЕДНИЙ ПРИОРИТЕТ (оставшиеся)

- [ ] **#32 Лишние DB round-trip'ы замедляют генерацию** — до LLM-вызова выполняется ~10-12 избыточных DB-операций:
  - `get_user_profile_merged()` вызывается 4 раза за ход: в `maybe_store_identity_from_telegram`, `sync_user_document`, `should_ask_intro`, `build_context`
  - `get_user_facts()` — 2 раза (`sync_user_document` + `build_context`)
  - `get_user_document()` — 2 раза (`should_ask_intro` + `build_context`)
  - Mood state: 4 write'а (`save_chat_state`, `save_user_relation`, `add_reaction`, `tick_reactions`) блокируют hot path хотя нужны только для следующего хода
  
  Фиксы (в порядке приоритета):
  1. `sync_user_document()` → background (денормализованный кэш, допустимо быть на 1 ход позади)
  2. `maybe_store_identity_from_telegram()` → background (username/first_name не меняются внутри хода)
  3. Mood state writes → background (читать state на hot path, писать после отправки ответа)
  4. Прочитать `user_profile` / `user_facts` один раз в `run_agent()` и передать параметрами вниз вместо повторных чтений

- [x] **#31 Mood system: aggression-сигнал не учитывает накопленный trust** — при высоком trust "дура"/"дебил" от давнего пользователя дают тот же удар по warmth/annoyance что и от незнакомца. Фикс: домножать эффект aggression на `(1 - trust * 0.5)` в `compute_state_updates()`. Также рассмотреть удаление "дура" и "дебил" из `_SIGNAL_AGGRESSION` как ambiguous-слов, или перенести их в отдельный слабый сигнал.

- [x] **#30 Постпроцессинг душит характер** — `_deterministic_postprocess_answer()` запускается на каждом ответе и содержит несколько агрессивных правил:
  - **Эмодзи удаляются безусловно** (строка 257 postprocessing.py) — молодой персонаж 20 лет органично использует эмодзи. Фикс: заменить полное удаление на лимит (≤2 эмодзи на ответ).
  - **over_persona каскад** — `!!!` или `*текст*` тригерят весь блок, включая "ааааа"→"аа". Живые эмоциональные конструкции убиваются вместе с реальным перебором. Фикс: разделить триггеры — убирать `*текст*` отдельно, не трогать повторяющиеся буквы.
  - **Каскад через format_violation** — один эмодзи тригерит `format_violation` → `_deterministic_postprocess_answer` → удаление эмодзи + возможная очистка over_persona. Фикс: убрать эмодзи из `_has_format_violation`.
  - 8 предложений / 700 символов — не проблема для Telegram.

- [ ] **#13 route_mode() не влияет на контекст** — DIRECT/NORMAL/HEAVY только для телеметрии. Фикс: адаптировать контекстный бюджет.

- [ ] **#14 edit_answer() — упростить редакторный этап** — сейчас `edit_answer()` используется, но LEGACY_STYLE_PROMPT_ARCHIVE перегружен. Нужен более компактный prompt и чёткие триггеры.

- [ ] **#15 _pending_join_verification теряется при рестарте** — In-memory dict. Фикс: хранить в SQLite.

- [ ] **#16 Summary self-poisoning при генерации** — подавать только raw-сообщения при генерации summary.

- [ ] **#17 HISTORY_MESSAGE_CHAR_LIMIT=900 обрезает длинные сообщения** — увеличить до 1200-1500.

- [x] **#18 Fallback cascade слишком быстро деградирует** — compact 160→280, minimal 96→180, emergency 64→120.

- [ ] **#27 /profile в группах не сообщает пользователю про ЛС** — при вводе /profile в группе стоит подсказать "Для удобной настройки напиши мне в личку".

## НИЗКИЙ ПРИОРИТЕТ

- [ ] **#19 USER_DOCUMENT_TRIGGER_PATTERNS слишком широкие** — паттерн "мне/мой" матчит почти всё.

- [ ] **#20 REPEAT_GUARD_MAX_CHARS=140** — длинные повторы не ловятся.

- [ ] **#21 Телеметрия и approx_text_tokens расходятся** — решено в #1, теперь единая константа.

- [ ] **#28 pm_state хранится в памяти** — если бот перезапустится пока пользователь вводит поле, состояние потеряется. Обычно не критично, но стоит учитывать.

## ТЕСТЫ

- [x] **Починить test_llm_overflow** — добавлены маркеры "context size" и "context budget".
- [x] **Починить test_main_sanitize** — regex расширен для `[to current speaker ...]`.
- [x] **Починить test_db_migration** — переписаны на актуальный API.
- [x] **Починить test_db_user_profiles** — переписаны на `upsert_user_profile`.
- [x] **Удалить test_agent_history, test_agent_memory** — импортировали несуществующие функции.
- [x] **Написать test_fixes.py** — тесты для всех критических фиксов.
- [x] **Написать test_mood.py** — тесты для mood system: compute_state_updates, time decay, build_alta_state_block, DB CRUD.
- [ ] **Написать тесты для private_profile.py** — validate_field, rate_limit_ok, format_import_preview.
- [x] **Написать тест для get_user_profile_merged** — проверить приоритет приватного над групповым.
- [x] **Написать тест для _build_extraction_text** — проверить разметку чужих сообщений.
