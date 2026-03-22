# TODO — Фиксы Alta

## КРИТИЧЕСКИЕ (прямо влияют на "тупость" бота)

- [ ] **#1 CHARS_PER_TOKEN_ESTIMATE слишком агрессивный** — `llm.py:19` = 2.0, `telemetry.py:14` = 4.0. Русский текст ~3.0-3.5. Ложный ContextOverflowError → падение на emergency context. Фикс: единая константа 3.2 в одном месте.

- [ ] **#2 Summary self-poisoning (двойное summary)** — `db.py:516-529`. `get_summary(chat_id, user_id)` фолбэчится на общий summary. В `build_context` оба блока (speaker_summary + summary) могут содержать одинаковый текст. Фикс: убрать fallback, возвращать "" если нет user-specific.

- [ ] **#3 Системный промпт перегружен метаданными** — `agent.py:1989-1999`. К ALTA_SYSTEM_PROMPT (~800 символов) конкатенируются все system_blocks (~5000+ символов). Характер тонет. Фикс: вынести блоки в отдельные system messages или приоритизировать.

- [ ] **#4 `_looks_like_context_dependent_reply()` отсекает вопросы** — `agent.py:456`. `if "?" in value: return False`. Короткие вопросы ("Почему?", "А зачем?") не получают implicit reply context. Фикс: убрать эту проверку.

- [ ] **#5 MAX_TOKENS по умолчанию 200 — мало** — `config.py`. При compact/minimal ещё режется до 160/96. Фикс: базовый 300-400, compact 250, minimal 150.

- [ ] **#6 Постпроцессинг слишком агрессивно режет** — `agent.py:707-730`. MAX_EDITOR_REPLY_CHARS=400, MAX_EDITOR_SENTENCES=5. `_remove_forbidden_self_reference()` удаляет целые предложения. Фикс: увеличить лимиты, заменять слова вместо удаления предложений.

## ВЫСОКИЙ ПРИОРИТЕТ

- [ ] **#7 REPEAT_REWRITE_PROMPT на английском** — `agent.py:145-150`. Переписать на русском.

- [ ] **#8 Lore забивает RAG-слоты** — `file_context.py:321`. Lore всегда получает score >= 2.0, при 3 слотах вытесняет полезный контент. Фикс: отдельный канал для lore, не считать в лимите RAG-чанков.

- [ ] **#9 Факты извлекаются слишком редко** — `agent.py:1014-1050`. Требует AND (first_person + fact_marker). Пропускает "Живу в Москве", "Я программист". Фикс: OR вместо AND.

- [ ] **#10 upsert_user_fact — garbage never degrades** — `db.py:822-825`. MAX(confidence) не позволяет снизить оценку мусорного факта. Фикс: использовать excluded.confidence напрямую.

- [ ] **#11 Нет инструкции "отвечай на русском"** — `prompts.py`. Добавить в ALTA_SYSTEM_PROMPT.

- [ ] **#12 Смешение языков в system_blocks** — `file_context.py:421-423`. Английская инструкция в русском промпте. Перевести.

## СРЕДНИЙ ПРИОРИТЕТ

- [ ] **#13 route_mode() не влияет на контекст** — `agent.py:121-126`. DIRECT/NORMAL/HEAVY только для телеметрии. Фикс: адаптировать контекстный бюджет под тип сообщения.

- [ ] **#14 edit_answer() — мёртвый код** — `agent.py:152`. LEGACY_STYLE_PROMPT_ARCHIVE не используется. Удалить.

- [ ] **#15 _pending_join_verification теряется при рестарте** — `main.py`. In-memory dict. Фикс: хранить в SQLite.

- [ ] **#16 Summary self-poisoning при генерации** — LLM сжимает историю, сжатие становится входом для следующего сжатия. Фикс: подавать только raw-сообщения при генерации summary.

- [ ] **#17 HISTORY_MESSAGE_CHAR_LIMIT=900 обрезает длинные сообщения** — `agent.py:34`. Увеличить до 1200-1500 или не обрезать текущее сообщение.

- [ ] **#18 Fallback cascade слишком быстро деградирует** — `agent.py:349-400`. Compact = 50% от full. Фикс: compact = 80% от full.

## НИЗКИЙ ПРИОРИТЕТ

- [ ] **#19 USER_DOCUMENT_TRIGGER_PATTERNS слишком широкие** — `agent.py:65-69`. Паттерн "мне/мой" матчит почти всё. Сузить.

- [ ] **#20 REPEAT_GUARD_MAX_CHARS=140** — `agent.py:49`. Длинные повторы не ловятся. Увеличить до 300.

- [ ] **#21 Телеметрия и approx_text_tokens расходятся** — `telemetry.py` vs `llm.py`. Единая функция, единый коэффициент.

## ТЕСТЫ (уже сломаны до начала фиксов)

- [ ] **Починить test_llm_overflow** — не ловит "context size exceeded" и "context budget exceed"
- [ ] **Починить test_main_sanitize** — `sanitize_outgoing_text` не чистит `[to current speaker ...]`
- [ ] **Починить test_db_migration** — рассинхрон API (отсутствующие методы/аргументы)
- [ ] **Починить test_db_user_profiles** — `replace_user_profile` не существует
- [ ] **Удалить/переписать test_agent_history, test_agent_memory** — импортируют несуществующие функции
