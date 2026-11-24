"""
Stress-тесты для анализа нагрузки на LLM.

Моделируют реальные сценарии и считают точные токены/символы
для каждого блока контекста. Не требуют запущенного LLM.
"""

import math
import sys
import os
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import ALTA_SYSTEM_PROMPT

# ---------- constants (mirrors from codebase) ----------
CHARS_PER_TOKEN = 3.2  # telemetry.py / llm.py unified
LLM_CONTEXT_WINDOW = 12_000  # config default
SAFETY_MARGIN = 512
MAX_TOKENS_OUTPUT = 350

# agent.py constants
DEFAULT_CONTEXT_CHAR_BUDGET = 12_000
COMPACT_CONTEXT_CHAR_BUDGET = 7_000
MINIMAL_CONTEXT_CHAR_BUDGET = 3_500
EMERGENCY_CONTEXT_CHAR_BUDGET = 1_500
SUMMARY_CONTEXT_CHAR_LIMIT = 700
MAX_FACT_LINES_IN_CONTEXT = 12
DEFAULT_FILE_CONTEXT_CHARS = 1_800
COMPACT_FILE_CONTEXT_CHARS = 1_000
MINIMAL_FILE_CONTEXT_CHARS = 500
HISTORY_MESSAGE_CHAR_LIMIT = 900
USER_DOCUMENT_CHAR_LIMIT = 900


def chars_to_tokens(chars: int) -> int:
    return max(1, math.ceil(chars / CHARS_PER_TOKEN))


def tokens_to_chars(tokens: int) -> int:
    return int(tokens * CHARS_PER_TOKEN)


# ============================================================
# Block size estimators
# ============================================================

def estimate_system_prompt_tokens() -> dict:
    """Measure the base system prompt."""
    chars = len(ALTA_SYSTEM_PROMPT)
    return {"label": "ALTA_SYSTEM_PROMPT", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_speaker_guard() -> dict:
    """Fixed speaker-guard block (always present)."""
    text = "СТРОГО: отвечай ТОЛЬКО как Альта."
    chars = len(text)
    return {"label": "speaker_guard", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_current_speaker(username: str = "zahar", first_name: str = "Захар") -> dict:
    text = f"Сейчас говорит: @{username} ({first_name})"
    chars = len(text)
    return {"label": "current_speaker", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_summary(chars_limit: int = SUMMARY_CONTEXT_CHAR_LIMIT) -> dict:
    """Worst case: summary filled to limit."""
    return {"label": "summary", "chars": chars_limit, "tokens": chars_to_tokens(chars_limit)}


def estimate_identity_block() -> dict:
    """Typical user identity block."""
    text = "Пользователь: @zahar (Захар). Пол: мужской. Стиль: прямой."
    chars = len(text)
    return {"label": "identity", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_participants(count: int = 5) -> dict:
    """Participants block with N users."""
    line = "Участник: @user123 (Имя Фамилия), мужской\n"
    chars = len(line) * count
    return {"label": f"participants ({count})", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_memory(lines: int = MAX_FACT_LINES_IN_CONTEXT) -> dict:
    """Memory facts block."""
    avg_fact_chars = 60  # "Любит программирование (confidence: 0.8)"
    chars = avg_fact_chars * lines
    return {"label": f"memory ({lines} facts)", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_document() -> dict:
    """User document block (worst case = limit)."""
    return {"label": "user_document", "chars": USER_DOCUMENT_CHAR_LIMIT, "tokens": chars_to_tokens(USER_DOCUMENT_CHAR_LIMIT)}


def estimate_rag(file_chars: int = DEFAULT_FILE_CONTEXT_CHARS) -> dict:
    """RAG/knowledge chunks."""
    return {"label": "RAG chunks", "chars": file_chars, "tokens": chars_to_tokens(file_chars)}


def estimate_history(msg_count: int = 30, avg_chars_per_msg: int = 120) -> dict:
    """Conversation history."""
    overhead_per_msg = 8  # tokens overhead per message in estimate_chat_tokens
    text_chars = min(avg_chars_per_msg, HISTORY_MESSAGE_CHAR_LIMIT) * msg_count
    text_tokens = chars_to_tokens(text_chars)
    total_tokens = text_tokens + (overhead_per_msg * msg_count)
    return {"label": f"history ({msg_count} msgs)", "chars": text_chars, "tokens": total_tokens}


def estimate_separator_and_reminder() -> dict:
    """The separator + reminder added around system blocks."""
    text = "---\nКОНТЕКСТ (используй как справку, не меняй свой стиль):\n---\nПомни: ты Альта. Отвечай коротко, живо, по-человечески."
    chars = len(text)
    return {"label": "separator+reminder", "chars": chars, "tokens": chars_to_tokens(chars)}


def estimate_creator_profile() -> dict:
    """Creator profile block (typical)."""
    chars = 400  # typical profile JSON rendered
    return {"label": "creator_profile", "chars": chars, "tokens": chars_to_tokens(chars)}


# ============================================================
# Scenario builders
# ============================================================

def build_scenario(name: str, history_msgs: int, avg_msg_len: int,
                   summary_chars: int, memory_lines: int, file_chars: int,
                   participants: int, max_tokens: int,
                   include_document: bool = True,
                   include_creator: bool = True) -> dict:
    """Build a full token budget for a scenario."""
    blocks = [
        estimate_system_prompt_tokens(),
        estimate_separator_and_reminder(),
        estimate_speaker_guard(),
        estimate_current_speaker(),
        estimate_identity_block(),
    ]

    if summary_chars > 0:
        blocks.append(estimate_summary(summary_chars))
    if memory_lines > 0:
        blocks.append(estimate_memory(memory_lines))
    if include_document:
        blocks.append(estimate_document())
    if file_chars > 0:
        blocks.append(estimate_rag(file_chars))
    if participants > 0:
        blocks.append(estimate_participants(participants))
    if include_creator:
        blocks.append(estimate_creator_profile())

    blocks.append(estimate_history(history_msgs, avg_msg_len))

    total_input_tokens = sum(b["tokens"] for b in blocks) + 8  # base overhead
    total_chars = sum(b["chars"] for b in blocks)

    return {
        "name": name,
        "blocks": blocks,
        "total_input_tokens": total_input_tokens,
        "max_output_tokens": max_tokens,
        "total_tokens": total_input_tokens + max_tokens,
        "total_chars": total_chars,
        "context_window": LLM_CONTEXT_WINDOW,
        "usable_window": LLM_CONTEXT_WINDOW - SAFETY_MARGIN,
        "fits": (total_input_tokens + max_tokens) <= (LLM_CONTEXT_WINDOW - SAFETY_MARGIN),
    }


# ============================================================
# TESTS
# ============================================================

class TestTokenBudget:
    """Verify token budgets for all context plans."""

    def test_system_prompt_size(self):
        info = estimate_system_prompt_tokens()
        assert info["chars"] > 0
        # Персонаж намеренно обогащён: тон + повадки + few-shot примеры из реального
        # чата. ~1400 токенов это ~18% от 8000-контекста, под историю/память остаётся
        # ~6500 — приемлемо. 1500 — твёрдый потолок против разрастания.
        assert info["tokens"] < 1500, f"System prompt too large: {info['tokens']} tokens"
        print(f"\n  System prompt: {info['chars']} chars = ~{info['tokens']} tokens")

    def test_full_plan_fits(self):
        """Full plan with typical load should fit in 12K context."""
        s = build_scenario(
            "full_typical", history_msgs=30, avg_msg_len=120,
            summary_chars=700, memory_lines=12, file_chars=1800,
            participants=5, max_tokens=350,
        )
        assert s["fits"], (
            f"Full plan doesn't fit: {s['total_tokens']} > {s['usable_window']}\n"
            f"Blocks: {[(b['label'], b['tokens']) for b in s['blocks']]}"
        )
        headroom = s["usable_window"] - s["total_tokens"]
        print(f"\n  Full plan: {s['total_tokens']}/{s['usable_window']} tokens, headroom={headroom}")

    def test_full_plan_worst_case(self):
        """Full plan with long messages (worst case)."""
        s = build_scenario(
            "full_worst", history_msgs=30, avg_msg_len=400,
            summary_chars=700, memory_lines=12, file_chars=1800,
            participants=8, max_tokens=350,
        )
        print(f"\n  Full worst case: {s['total_tokens']}/{s['usable_window']} tokens, fits={s['fits']}")
        # This MAY not fit - that's what compact fallback is for
        if not s["fits"]:
            overflow = s["total_tokens"] - s["usable_window"]
            print(f"  Overflow by {overflow} tokens - compact fallback needed")

    def test_compact_plan_fits(self):
        """Compact plan should always fit."""
        s = build_scenario(
            "compact", history_msgs=20, avg_msg_len=200,
            summary_chars=450, memory_lines=4, file_chars=1000,
            participants=5, max_tokens=280,
        )
        assert s["fits"], f"Compact doesn't fit: {s['total_tokens']} > {s['usable_window']}"
        headroom = s["usable_window"] - s["total_tokens"]
        print(f"\n  Compact plan: {s['total_tokens']}/{s['usable_window']} tokens, headroom={headroom}")

    def test_minimal_plan_fits(self):
        """Minimal plan should always fit even with long messages."""
        s = build_scenario(
            "minimal", history_msgs=10, avg_msg_len=400,
            summary_chars=300, memory_lines=2, file_chars=500,
            participants=0, max_tokens=180,
            include_creator=True,
        )
        assert s["fits"], f"Minimal doesn't fit: {s['total_tokens']} > {s['usable_window']}"
        headroom = s["usable_window"] - s["total_tokens"]
        print(f"\n  Minimal plan: {s['total_tokens']}/{s['usable_window']} tokens, headroom={headroom}")

    def test_emergency_plan_always_fits(self):
        """Emergency plan must ALWAYS fit no matter what."""
        s = build_scenario(
            "emergency", history_msgs=2, avg_msg_len=900,
            summary_chars=120, memory_lines=0, file_chars=0,
            participants=0, max_tokens=120,
            include_document=False, include_creator=False,
        )
        assert s["fits"], f"Emergency doesn't fit: {s['total_tokens']} > {s['usable_window']}"
        headroom = s["usable_window"] - s["total_tokens"]
        print(f"\n  Emergency plan: {s['total_tokens']}/{s['usable_window']} tokens, headroom={headroom}")


class TestContextDistribution:
    """Analyze what percentage of context is used by each component."""

    def _print_distribution(self, scenario: dict):
        total = scenario["total_input_tokens"]
        print(f"\n  === {scenario['name']} ===")
        print(f"  Total input: {total} tokens | Output: {scenario['max_output_tokens']} tokens")
        print(f"  Window: {scenario['total_tokens']}/{scenario['usable_window']} "
              f"({'OK' if scenario['fits'] else 'OVERFLOW'})")
        print(f"  {'Block':<30} {'Tokens':>7} {'%':>6} {'Chars':>7}")
        print(f"  {'-'*52}")
        for b in sorted(scenario["blocks"], key=lambda x: -x["tokens"]):
            pct = b["tokens"] / total * 100 if total else 0
            print(f"  {b['label']:<30} {b['tokens']:>7} {pct:>5.1f}% {b['chars']:>7}")
        print(f"  {'='*52}")
        print(f"  {'TOTAL INPUT':<30} {total:>7} {'100%':>6} {scenario['total_chars']:>7}")

    def test_typical_chat_distribution(self):
        """Typical active group chat: 30 messages, moderate length."""
        s = build_scenario(
            "Typical group chat", history_msgs=30, avg_msg_len=120,
            summary_chars=500, memory_lines=8, file_chars=1200,
            participants=5, max_tokens=350,
        )
        self._print_distribution(s)

    def test_heavy_chat_distribution(self):
        """Heavy chat: long messages, full memory, full RAG."""
        s = build_scenario(
            "Heavy group chat", history_msgs=30, avg_msg_len=350,
            summary_chars=700, memory_lines=12, file_chars=1800,
            participants=8, max_tokens=350,
        )
        self._print_distribution(s)

    def test_quiet_dm_distribution(self):
        """Quiet DM: few short messages."""
        s = build_scenario(
            "Quiet DM", history_msgs=5, avg_msg_len=60,
            summary_chars=0, memory_lines=3, file_chars=0,
            participants=0, max_tokens=350,
            include_creator=False,
        )
        self._print_distribution(s)

    def test_new_user_distribution(self):
        """Brand new user: no memory, no summary, no document."""
        s = build_scenario(
            "New user (first message)", history_msgs=1, avg_msg_len=80,
            summary_chars=0, memory_lines=0, file_chars=0,
            participants=0, max_tokens=350,
            include_document=False, include_creator=False,
        )
        self._print_distribution(s)


class TestModelSuitability:
    """Evaluate if Mistral Small 24B Q4_K_XL is suitable."""

    MISTRAL_SMALL_CONTEXT = 32_768  # native context window
    LLAMA_CPP_CONTEXT = 12_000      # configured in .env / config.py
    MISTRAL_SMALL_SPEED_TPS = 25    # ~tokens/sec on decent GPU, Q4 quantization
    MISTRAL_SMALL_SPEED_TPS_CPU = 5  # ~tokens/sec CPU-only

    def test_context_utilization(self):
        """How much of Mistral Small's native context do we use?"""
        used = LLAMA_CPP_CONTEXT = 12_000
        native = self.MISTRAL_SMALL_CONTEXT
        pct = used / native * 100
        print(f"\n  Configured window: {used} / {native} native = {pct:.0f}%")
        print(f"  Headroom: {native - used} tokens unused")
        print(f"  VERDICT: {'Safe' if pct < 80 else 'Tight'} - model can handle more if needed")

    def test_response_latency_estimate(self):
        """Estimate response time for different scenarios."""
        scenarios = [
            ("Short reply (50 tokens)", 50),
            ("Medium reply (150 tokens)", 150),
            ("Max reply (350 tokens)", 350),
        ]
        print(f"\n  Response latency estimates:")
        print(f"  {'Scenario':<30} {'GPU (25 t/s)':>14} {'CPU (5 t/s)':>14}")
        print(f"  {'-'*60}")
        for name, tokens in scenarios:
            gpu_sec = tokens / self.MISTRAL_SMALL_SPEED_TPS
            cpu_sec = tokens / self.MISTRAL_SMALL_SPEED_TPS_CPU
            print(f"  {name:<30} {gpu_sec:>11.1f}s {cpu_sec:>13.1f}s")

    def test_concurrent_load(self):
        """Estimate max concurrent requests."""
        # Mistral Small 24B Q4_K_XL ~= 14GB VRAM
        vram_gb = 14
        per_request_overhead_gb = 0.5  # KV cache per concurrent request
        base_model_gb = 13  # model weights
        available_for_kv = vram_gb - base_model_gb
        max_concurrent = max(1, int(available_for_kv / per_request_overhead_gb))
        print(f"\n  Model VRAM: ~{base_model_gb}GB weights + KV cache")
        print(f"  Available for KV: ~{available_for_kv}GB")
        print(f"  Max concurrent requests: ~{max_concurrent}")
        print(f"  NOTE: With 12K context window, each request KV cache = ~{per_request_overhead_gb}GB")

    def test_model_comparison(self):
        """Compare model options for this workload."""
        models = [
            {
                "name": "Mistral Small 3.2 24B Q4",
                "params": "24B",
                "quant": "Q4_K_XL",
                "vram_gb": 14,
                "context": 32768,
                "speed_tps": 25,
                "quality": "Good",
                "russian": "Good (trained on multilingual)",
                "fits_24gb": True,
            },
            {
                "name": "Mistral Nemo 12B Q8",
                "params": "12B",
                "quant": "Q8_0",
                "vram_gb": 13,
                "context": 131072,
                "speed_tps": 40,
                "quality": "Decent",
                "russian": "Good",
                "fits_24gb": True,
            },
            {
                "name": "Qwen2.5 32B Q4",
                "params": "32B",
                "quant": "Q4_K_M",
                "vram_gb": 20,
                "context": 131072,
                "speed_tps": 18,
                "quality": "Very good",
                "russian": "Excellent (strong multilingual)",
                "fits_24gb": True,
            },
            {
                "name": "Gemma 3 27B Q4",
                "params": "27B",
                "quant": "Q4_K_M",
                "vram_gb": 17,
                "context": 131072,
                "speed_tps": 22,
                "quality": "Very good",
                "russian": "Good",
                "fits_24gb": True,
            },
            {
                "name": "Llama 3.1 70B Q3",
                "params": "70B",
                "quant": "Q3_K_M",
                "vram_gb": 36,
                "context": 131072,
                "speed_tps": 8,
                "quality": "Excellent",
                "russian": "Decent (English-centric)",
                "fits_24gb": False,
            },
        ]

        print(f"\n  {'Model':<30} {'VRAM':>6} {'t/s':>5} {'Quality':>10} {'Russian':>12} {'24GB?':>6}")
        print(f"  {'-'*75}")
        for m in models:
            print(f"  {m['name']:<30} {m['vram_gb']:>4}GB {m['speed_tps']:>5} "
                  f"{m['quality']:>10} {m['russian']:>12} {'Yes' if m['fits_24gb'] else 'No':>6}")


class TestBackgroundExtractionLoad:
    """Measure the HIDDEN load from background tasks."""

    def test_extraction_calls_per_message(self):
        """Each user message triggers up to 3 extra LLM calls in background."""
        # From agent.py _background_extraction():
        # 1. maybe_extract_user_facts() - up to MEMORY_EXTRACT_INPUT_CHAR_LIMIT (1800) chars
        # 2. maybe_extract_user_profile() - up to PROFILE_EXTRACT_INPUT_CHAR_LIMIT (2200) chars
        # 3. maybe_update_chat_summary() - up to SUMMARY_INPUT_CHAR_LIMIT (6000) chars (every 10 msgs)

        calls = [
            {"name": "fact_extraction", "input_chars": 1800, "output_tokens": 200,
             "frequency": "every message with facts"},
            {"name": "profile_extraction", "input_chars": 2200, "output_tokens": 150,
             "frequency": "every message with profile hints"},
            {"name": "summary_update", "input_chars": 6000, "output_tokens": 400,
             "frequency": "every 10 messages"},
        ]

        print(f"\n  Background LLM calls per user message:")
        print(f"  {'Task':<25} {'Input':>8} {'Output':>8} {'Frequency'}")
        print(f"  {'-'*70}")
        total_input = 0
        for c in calls:
            input_tok = chars_to_tokens(c["input_chars"])
            print(f"  {c['name']:<25} {input_tok:>6}tk {c['output_tokens']:>6}tk {c['frequency']}")
            total_input += input_tok

        print(f"\n  Worst case (all 3 fire): ~{total_input} input + ~750 output = ~{total_input + 750} tokens")
        print(f"  This is ON TOP of the main response generation!")

    def test_total_tokens_per_message(self):
        """Total LLM tokens consumed per single user message."""
        # Main generation
        main_input = 4200  # typical full plan
        main_output = 200  # average response

        # Background (worst case)
        bg_fact_input = chars_to_tokens(1800)
        bg_fact_output = 200
        bg_profile_input = chars_to_tokens(2200)
        bg_profile_output = 150
        bg_summary_input = chars_to_tokens(6000)  # 1 in 10 messages
        bg_summary_output = 400

        total_always = main_input + main_output + bg_fact_input + bg_fact_output + bg_profile_input + bg_profile_output
        total_with_summary = total_always + bg_summary_input + bg_summary_output

        print(f"\n  Tokens per user message (typical):")
        print(f"    Main generation:     {main_input} in + {main_output} out = {main_input + main_output}")
        print(f"    Fact extraction:     {bg_fact_input} in + {bg_fact_output} out = {bg_fact_input + bg_fact_output}")
        print(f"    Profile extraction:  {bg_profile_input} in + {bg_profile_output} out = {bg_profile_input + bg_profile_output}")
        print(f"    Summary (1/10):      {bg_summary_input} in + {bg_summary_output} out = {bg_summary_input + bg_summary_output}")
        print(f"  ---")
        print(f"    TOTAL (no summary):  ~{total_always} tokens")
        print(f"    TOTAL (with summary): ~{total_with_summary} tokens")
        print(f"    Average per msg:     ~{total_always + (bg_summary_input + bg_summary_output) // 10} tokens")


class TestEdgeCases:
    """Edge cases that might cause problems."""

    def test_single_huge_message(self):
        """User sends a 3000-char message."""
        msg_chars = 3000
        capped = min(msg_chars, HISTORY_MESSAGE_CHAR_LIMIT)
        lost = msg_chars - capped
        print(f"\n  User sends {msg_chars} char message")
        print(f"  Capped to: {capped} chars (HISTORY_MESSAGE_CHAR_LIMIT)")
        print(f"  Lost: {lost} chars ({lost/msg_chars*100:.0f}%)")
        assert capped == HISTORY_MESSAGE_CHAR_LIMIT

    def test_all_users_max_facts(self):
        """DB grows with 80 facts per user."""
        users = 50
        facts_per_user = 80  # FACT_KEEP_PER_USER
        avg_fact_bytes = 120
        total_mb = (users * facts_per_user * avg_fact_bytes) / 1024 / 1024
        print(f"\n  {users} users x {facts_per_user} facts = {users * facts_per_user} rows")
        print(f"  Estimated DB size (facts only): ~{total_mb:.1f} MB")
        print(f"  In context per request: only {MAX_FACT_LINES_IN_CONTEXT} facts shown")

    def test_rapid_fire_messages(self):
        """5 messages in 1 second from same user."""
        cooldown_sec = 15  # default USER_COOLDOWN_SEC
        msgs_blocked = 4  # out of 5
        print(f"\n  5 rapid messages -> {5 - msgs_blocked} processed, {msgs_blocked} blocked by {cooldown_sec}s cooldown")

    def test_empty_history_new_chat(self):
        """First message in a new chat."""
        s = build_scenario(
            "First message ever", history_msgs=0, avg_msg_len=0,
            summary_chars=0, memory_lines=0, file_chars=0,
            participants=0, max_tokens=350,
            include_document=False, include_creator=False,
        )
        print(f"\n  First message: {s['total_tokens']}/{s['usable_window']} tokens")
        print(f"  {s['usable_window'] - s['total_tokens']} tokens headroom")
        assert s["fits"]


# ============================================================
# Main: run as script for detailed report
# ============================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
