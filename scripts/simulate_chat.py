"""Симуляция чата с Альтой — тест mood system.

Запуск: python scripts/simulate_chat.py

Прогоняет заготовленный сценарий разговора через полную цепочку состояния
(DB, decay, signals, state block) без реального LLM.
"""
import asyncio
import io
import os
import sys
import uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_builder import (
    apply_time_decay_chat,
    apply_time_decay_relation,
    build_alta_state_block,
    compute_state_updates,
    _hours_since_state,
)
from db import DB

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
GRAY = "\033[90m"


def color(text: str, c: str) -> str:
    return f"{c}{text}{RESET}"


def bar(val: float, width: int = 16) -> str:
    filled = round(val * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val:.2f}"


def print_state(chat_state: dict, user_rel: dict, reaction: dict | None, label: str = ""):
    if label:
        print(color(f"\n  ── {label} ──", DIM))
    print(color("  Чат:     ", GRAY) +
          f"energy {bar(chat_state['energy'], 10)}  "
          f"irritation {bar(chat_state['irritation'], 10)}  "
          f"playfulness {bar(chat_state['playfulness'], 10)}  "
          f"social_battery {bar(chat_state['social_battery'], 10)}")
    print(color("  Юзер:    ", GRAY) +
          f"trust {bar(user_rel['trust'], 10)}  "
          f"warmth {bar(user_rel['warmth'], 10)}  "
          f"annoyance {bar(user_rel['annoyance'], 10)}  "
          f"interactions={user_rel['interaction_count']}")
    if reaction:
        print(color("  Реакция: ", GRAY) +
              color(f"{reaction['reaction_kind']} (strength={reaction['reaction_strength']:.2f}, ttl={reaction['ttl_messages']})", MAGENTA))


def print_state_block(block: str):
    if not block:
        print(color("  [state block: пусто — нейтральное состояние]", DIM))
    else:
        for line in block.split("\n"):
            print(color("  │ ", BLUE) + color(line, CYAN))


SCENARIO = [
    # (username, user_id, message)
    ("Kostya",   101, "Привет альта"),
    ("Kostya",   101, "как дела?"),
    ("Kostya",   101, "слушай объясни мне что такое квантовый компьютер"),
    ("Kostya",   101, "крутая объяснилка, спасибо!"),
    ("Dima",     202, "хей"),
    ("Dima",     202, "иди нафиг"),
    ("Dima",     202, "ты тупая вообще"),
    ("Dima",     202, "заткнись"),
    ("Kostya",   101, "не обращай на него внимание"),
    ("Kostya",   101, "а вот скажи, что если бы у людей была фотографическая память?"),
    ("Dima",     202, "ладно прости"),
    ("Dima",     202, "спасибо что терпишь нас"),
    ("Dima",     202, "расскажи что-нибудь интересное"),
    # Долгая пауза (симулируется через hours_gap)
    ("Kostya",   101, "[после долгой паузы] снова здесь"),
]

# Сколько часов прошло перед каждым сообщением (None = мгновенно)
TIME_GAPS = [
    None, None, None, None,
    None, None, None, None,
    None, None, None, None, None,
    3.5,   # пауза перед последним
]


async def main():
    db_path = os.path.join("tests", f"sim_{uuid.uuid4().hex[:8]}.sqlite3")
    db = DB(db_path)
    await db.connect()

    CHAT_ID = 999
    print(color("\n╔══════════════════════════════════════════════════════╗", BOLD))
    print(color("║     СИМУЛЯЦИЯ ЧАТА — ТЕСТ MOOD SYSTEM АЛЬТЫ         ║", BOLD))
    print(color("╚══════════════════════════════════════════════════════╝\n", BOLD))

    try:
        for i, ((username, user_id, message), time_gap) in enumerate(zip(SCENARIO, TIME_GAPS), 1):
            print(color(f"{'─'*60}", GRAY))
            print(color(f"[{i}/{len(SCENARIO)}] ", DIM) +
                  color(f"@{username}", YELLOW) +
                  ": " +
                  color(message, BOLD))

            # Загрузить текущее состояние
            chat_state_raw = await db.get_alta_chat_state(CHAT_ID)
            user_rel_raw = await db.get_alta_user_relation(CHAT_ID, user_id)

            # Симулировать временной gap
            if time_gap and time_gap > 0:
                print(color(f"\n  ⏱  прошло {time_gap} ч.", YELLOW))
                hours_chat = time_gap
                hours_rel = time_gap
            else:
                hours_chat = _hours_since_state(chat_state_raw.get("updated_at"))
                hours_rel = _hours_since_state(user_rel_raw.get("updated_at"))

            # Decay
            chat_decayed = apply_time_decay_chat(chat_state_raw, hours_chat)
            rel_decayed = apply_time_decay_relation(user_rel_raw, hours_rel)

            # Обновление по сигналам
            new_chat, new_rel, reaction_kind, reaction_strength = compute_state_updates(
                message, chat_decayed, rel_decayed,
            )

            # Сохранить в DB
            await db.save_alta_chat_state(
                CHAT_ID,
                energy=new_chat["energy"],
                irritation=new_chat["irritation"],
                playfulness=new_chat["playfulness"],
                social_battery=new_chat["social_battery"],
            )
            await db.save_alta_user_relation(
                CHAT_ID, user_id,
                trust=new_rel["trust"],
                warmth=new_rel["warmth"],
                caution=new_rel["caution"],
                annoyance=new_rel["annoyance"],
                interaction_count=new_rel["interaction_count"],
            )
            if reaction_kind:
                await db.add_alta_reaction(
                    CHAT_ID, reaction_kind, reaction_strength,
                    ttl_messages=2, target_user_id=user_id,
                )
            await db.tick_alta_reactions(CHAT_ID)
            active_reaction = await db.get_active_alta_reaction(CHAT_ID, user_id)

            # Вывести состояние
            print_state(new_chat, new_rel, active_reaction, label="состояние после")

            # Вывести блок промпта
            state_block = build_alta_state_block(new_chat, new_rel, active_reaction)
            print(color("\n  → блок в промпт:", DIM))
            print_state_block(state_block)
            print()

    finally:
        await db.close()
        for suffix in ("", "-wal", "-shm"):
            p = f"{db_path}{suffix}"
            if os.path.exists(p):
                os.remove(p)

    print(color("╔══════════════════════════════════════════════════════╗", BOLD))
    print(color("║                    ГОТОВО                           ║", BOLD))
    print(color("╚══════════════════════════════════════════════════════╝", BOLD))


if __name__ == "__main__":
    asyncio.run(main())
