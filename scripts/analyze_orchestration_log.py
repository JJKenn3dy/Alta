import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return float(ordered[index])


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("orchestration.log")
    if not path.exists():
        print(f"log not found: {path}")
        return 1

    turn_end_events: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "turn_end":
                turn_end_events.append(event)

    if not turn_end_events:
        print("no turn_end events found")
        return 1

    total_tokens = [int(event.get("total_tokens", 0) or 0) for event in turn_end_events]
    total_calls = [int(event.get("total_llm_calls", 0) or 0) for event in turn_end_events]
    total_latency = [int(event.get("total_latency_ms", 0) or 0) for event in turn_end_events]
    history_tokens = [int(event.get("history_tokens", 0) or 0) for event in turn_end_events]
    summary_tokens = [int(event.get("summary_tokens", 0) or 0) for event in turn_end_events]
    memory_tokens = [int(event.get("memory_tokens", 0) or 0) for event in turn_end_events]
    rag_tokens = [int(event.get("rag_tokens", 0) or 0) for event in turn_end_events]
    system_tokens = [int(event.get("system_tokens", 0) or 0) for event in turn_end_events]

    editor_attempt_turns = sum(
        1
        for event in turn_end_events
        if event.get("editor_attempted", event.get("rewrite_attempted", event.get("rewrite_used")))
    )
    editor_changed_turns = sum(
        1
        for event in turn_end_events
        if event.get("editor_changed_output", event.get("rewrite_changed_output", event.get("rewrite_used")))
    )
    editor_triggers: dict[str, int] = defaultdict(int)
    memory_turns = sum(1 for event in turn_end_events if event.get("memory_updated"))
    summary_turns = sum(1 for event in turn_end_events if event.get("summary_updated"))
    profile_turns = sum(1 for event in turn_end_events if event.get("profile_updated"))
    fallback_turns = sum(1 for event in turn_end_events if event.get("fallback_used"))
    turns_by_mode: dict[str, int] = defaultdict(int)
    llm_calls_by_mode: dict[str, list[int]] = defaultdict(list)

    memory_writes_by_day: dict[str, int] = defaultdict(int)
    memory_changes_by_day: dict[str, int] = defaultdict(int)
    memory_sync_changes_by_day: dict[str, int] = defaultdict(int)
    memory_extract_changes_by_day: dict[str, int] = defaultdict(int)
    memory_intro_changes_by_day: dict[str, int] = defaultdict(int)
    for event in turn_end_events:
        mode = str(event.get("mode", "unknown") or "unknown")
        turns_by_mode[mode] += 1
        llm_calls_by_mode[mode].append(int(event.get("total_llm_calls", 0) or 0))

        trigger_value = str(event.get("editor_trigger", "") or "").strip()
        if trigger_value:
            for trigger in (part.strip() for part in trigger_value.split(",")):
                if trigger:
                    editor_triggers[trigger] += 1

        ts = str(event.get("ts", ""))
        day = ts[:10] if len(ts) >= 10 else "unknown"
        memory_writes_by_day[day] += int(event.get("memory_items_written", 0) or 0)
        memory_changes_by_day[day] += int(event.get("memory_items_changed", event.get("memory_items_written", 0)) or 0)
        memory_sync_changes_by_day[day] += int(event.get("memory_sync_items_changed", 0) or 0)
        memory_extract_changes_by_day[day] += int(event.get("memory_extract_items_changed", 0) or 0)
        memory_intro_changes_by_day[day] += int(event.get("memory_intro_items_changed", 0) or 0)

    ponyal_turns = [
        event for event in turn_end_events if str(event.get("input_preview", "")).strip().lower() == "понял"
    ]
    block_avgs = {
        "history": statistics.fmean(history_tokens) if history_tokens else 0.0,
        "summary": statistics.fmean(summary_tokens) if summary_tokens else 0.0,
        "memory": statistics.fmean(memory_tokens) if memory_tokens else 0.0,
        "rag": statistics.fmean(rag_tokens) if rag_tokens else 0.0,
        "system": statistics.fmean(system_tokens) if system_tokens else 0.0,
    }
    largest_block = max(block_avgs.items(), key=lambda item: item[1])[0]

    print(f"turns: {len(turn_end_events)}")
    print(f"average_tokens_per_turn: {statistics.fmean(total_tokens):.2f}")
    print(f"p95_tokens_per_turn: {p95(total_tokens):.2f}")
    print(f"average_llm_calls_per_turn: {statistics.fmean(total_calls):.2f}")
    print(f"p95_latency_ms: {p95(total_latency):.2f}")
    print(f"editor_attempt_turn_pct: {pct(editor_attempt_turns, len(turn_end_events)):.2f}")
    print(f"editor_changed_output_turn_pct: {pct(editor_changed_turns, len(turn_end_events)):.2f}")
    print(f"memory_update_turn_pct: {pct(memory_turns, len(turn_end_events)):.2f}")
    print(f"summary_update_turn_pct: {pct(summary_turns, len(turn_end_events)):.2f}")
    print(f"profile_update_turn_pct: {pct(profile_turns, len(turn_end_events)):.2f}")
    print(f"fallback_turn_pct: {pct(fallback_turns, len(turn_end_events)):.2f}")
    print(f"average_history_tokens: {block_avgs['history']:.2f}")
    print(f"average_summary_tokens: {block_avgs['summary']:.2f}")
    print(f"average_memory_tokens: {block_avgs['memory']:.2f}")
    print(f"average_rag_tokens: {block_avgs['rag']:.2f}")
    print(f"average_system_tokens: {block_avgs['system']:.2f}")
    print(f"largest_average_context_block: {largest_block}")
    print(f"ponyal_turns: {len(ponyal_turns)}")
    if ponyal_turns:
        ponyal_calls = [int(event.get('total_llm_calls', 0) or 0) for event in ponyal_turns]
        print(f"average_llm_calls_for_ponyal: {statistics.fmean(ponyal_calls):.2f}")
    print("turns_by_mode:")
    for mode in sorted(turns_by_mode):
        print(f"  {mode}: {turns_by_mode[mode]}")
    print("avg_llm_calls_by_mode:")
    for mode in sorted(llm_calls_by_mode):
        print(f"  {mode}: {statistics.fmean(llm_calls_by_mode[mode]):.2f}")
    print("editor_triggers:")
    for trigger in sorted(editor_triggers):
        print(f"  {trigger}: {editor_triggers[trigger]}")

    print("memory_writes_by_day:")
    for day in sorted(memory_writes_by_day):
        print(f"  {day}: {memory_writes_by_day[day]}")
    print("memory_changes_by_day:")
    for day in sorted(memory_changes_by_day):
        print(f"  {day}: {memory_changes_by_day[day]}")
    print("memory_sync_changes_by_day:")
    for day in sorted(memory_sync_changes_by_day):
        print(f"  {day}: {memory_sync_changes_by_day[day]}")
    print("memory_extract_changes_by_day:")
    for day in sorted(memory_extract_changes_by_day):
        print(f"  {day}: {memory_extract_changes_by_day[day]}")
    print("memory_intro_changes_by_day:")
    for day in sorted(memory_intro_changes_by_day):
        print(f"  {day}: {memory_intro_changes_by_day[day]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
