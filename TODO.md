# TODO — Фиксы Alta

## КРИТИЧЕСКИЕ (прямо влияют на "тупость" бота)

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

## СРЕДНИЙ ПРИОРИТЕТ (оставшиеся)

- [ ] **#13 route_mode() не влияет на контекст** — DIRECT/NORMAL/HEAVY только для телеметрии. Фикс: адаптировать контекстный бюджет.

- [ ] **#14 edit_answer() — мёртвый код** — LEGACY_STYLE_PROMPT_ARCHIVE не используется. Удалить.

- [ ] **#15 _pending_join_verification теряется при рестарте** — In-memory dict. Фикс: хранить в SQLite.

- [ ] **#16 Summary self-poisoning при генерации** — подавать только raw-сообщения при генерации summary.

- [ ] **#17 HISTORY_MESSAGE_CHAR_LIMIT=900 обрезает длинные сообщения** — увеличить до 1200-1500.

- [x] **#18 Fallback cascade слишком быстро деградирует** — compact 160→280, minimal 96→180, emergency 64→120.

## НИЗКИЙ ПРИОРИТЕТ

- [ ] **#19 USER_DOCUMENT_TRIGGER_PATTERNS слишком широкие** — паттерн "мне/мой" матчит почти всё.

- [ ] **#20 REPEAT_GUARD_MAX_CHARS=140** — длинные повторы не ловятся.

- [ ] **#21 Телеметрия и approx_text_tokens расходятся** — решено в #1, теперь единая константа.

## ТЕСТЫ

- [x] **Починить test_llm_overflow** — добавлены маркеры "context size" и "context budget".
- [x] **Починить test_main_sanitize** — regex расширен для `[to current speaker ...]`.
- [x] **Починить test_db_migration** — переписаны на актуальный API.
- [x] **Починить test_db_user_profiles** — переписаны на `upsert_user_profile`.
- [x] **Удалить test_agent_history, test_agent_memory** — импортировали несуществующие функции.
- [x] **Написать test_fixes.py** — тесты для всех критических фиксов.
