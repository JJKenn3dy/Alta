import asyncio
import math
import time

from openai import AsyncOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, RateLimitError
from telemetry import LLMCallMetrics, approx_text_tokens


class ContextOverflowError(RuntimeError):
    pass


DEFAULT_MAX_CONTEXT_TOKENS = 4096
DEFAULT_CONTEXT_SAFETY_MARGIN = 512
CHARS_PER_TOKEN_ESTIMATE = 2.0


def _looks_like_context_overflow(error: Exception) -> bool:
    text = str(error).lower()
    markers = (
        "context length",
        "context_length",
        "too many tokens",
        "prompt is too long",
        "maximum context",
        "requested tokens",
        "reduce the length",
        "out of memory",
        "cuda out of memory",
        "oom",
        "kv cache",
    )
    return any(marker in text for marker in markers)


def _estimate_text_tokens(text: str) -> int:
    normalized = " ".join((text or "").split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / CHARS_PER_TOKEN_ESTIMATE))


def estimate_chat_tokens(system_prompt: str, messages: list[dict], max_tokens: int) -> int:
    total = 8 + _estimate_text_tokens(system_prompt)
    for message in messages:
        total += 8 + _estimate_text_tokens(str(message.get("content", "")))
    return total + max(1, int(max_tokens))


class LLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_retries: int = 7,
        base_delay: float = 0.7,
        timeout: float = 60.0,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        context_safety_margin: int = DEFAULT_CONTEXT_SAFETY_MARGIN,
    ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_context_tokens = max(256, int(max_context_tokens))
        self.context_safety_margin = max(0, int(context_safety_margin))

    def _ensure_context_budget(self, system_prompt: str, messages: list[dict], max_tokens: int):
        allowed_total = max(128, self.max_context_tokens - self.context_safety_margin)
        estimated_total = estimate_chat_tokens(system_prompt, messages, max_tokens)
        if estimated_total > allowed_total:
            raise ContextOverflowError(
                "Estimated context budget exceeded before API call: "
                f"{estimated_total} > {allowed_total} tokens "
                f"(window={self.max_context_tokens}, margin={self.context_safety_margin}, "
                f"max_tokens={max_tokens})"
            )

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        turn_id: str | None = None,
        turn_metrics=None,
    ) -> str:
        last_err = None
        started_at = time.perf_counter()
        call_index = (len(turn_metrics.llm_calls) + 1) if turn_metrics is not None else 1
        estimated_input_tokens = approx_text_tokens(system_prompt) + sum(
            approx_text_tokens(message.get("content", "")) for message in messages
        )

        def _record_failure(error: Exception):
            if turn_metrics is None:
                return
            latency_ms = max(1, int((time.perf_counter() - started_at) * 1000))
            turn_metrics.record_llm_call(
                LLMCallMetrics(
                    llm_call_index=call_index,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    input_tokens=estimated_input_tokens,
                    output_tokens=0,
                    total_tokens=estimated_input_tokens,
                    latency_ms=latency_ms,
                    ok=False,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

        for attempt in range(1, self.max_retries + 1):
            try:
                self._ensure_context_budget(system_prompt, messages, max_tokens)
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stream=False,
                )
                content = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

                if not input_tokens:
                    input_tokens = estimated_input_tokens
                if not output_tokens:
                    output_tokens = approx_text_tokens(content)
                if not total_tokens:
                    total_tokens = input_tokens + output_tokens

                if turn_metrics is not None:
                    latency_ms = max(1, int((time.perf_counter() - started_at) * 1000))
                    turn_metrics.record_llm_call(
                        LLMCallMetrics(
                            llm_call_index=call_index,
                            model=self.model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            latency_ms=latency_ms,
                        )
                    )
                return content

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_err = e
                delay = self.base_delay * attempt
                print(
                    f"[LLM] attempt {attempt}/{self.max_retries} failed: {type(e).__name__}, retry in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

            except (BadRequestError, APIStatusError) as e:
                if _looks_like_context_overflow(e):
                    _record_failure(e)
                    raise ContextOverflowError(str(e)) from e
                _record_failure(e)
                print(f"[LLM] fatal api error: {e}")
                raise

            except Exception as e:
                _record_failure(e)
                print(f"[LLM] fatal error: {e}")
                raise

        if last_err is not None:
            _record_failure(last_err)
        raise RuntimeError(f"LLM unavailable after {self.max_retries} retries") from last_err
