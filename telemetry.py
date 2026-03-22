import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("alta.telemetry")


CHARS_PER_TOKEN_HEURISTIC = 3.2


def approx_text_tokens(text: Any) -> int:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / CHARS_PER_TOKEN_HEURISTIC))


def approx_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += approx_text_tokens(message.get("content", ""))
    return total


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JSONLLogger:
    def __init__(self, path: str, enabled: bool, debug_enabled: bool = False):
        self.path = Path(path)
        self.enabled = enabled
        self.debug_enabled = debug_enabled
        self._lock = threading.Lock()

    def log_event(self, payload: dict[str, Any]):
        if not self.enabled:
            return

        event = dict(payload)
        event.setdefault("ts", _utc_now_iso())
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        if self.debug_enabled:
            log.debug("metrics: %s turn=%s", event.get("event"), event.get("turn_id"))


@dataclass
class LLMCallMetrics:
    llm_call_index: int
    model: str
    temperature: float
    max_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    ok: bool = True
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class TurnMetrics:
    turn_id: str
    chat_id: int
    user_id: int
    mode: str
    input_chars: int
    has_attachments: bool
    route_reason: str | None = None
    logger: JSONLLogger | None = None
    input_preview: str | None = None
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)
    history_tokens: int = 0
    summary_tokens: int = 0
    memory_tokens: int = 0
    rag_tokens: int = 0
    system_tokens: int = 0
    input_tokens: int = 0
    identity_tokens: int = 0
    user_document_tokens: int = 0
    creator_profile_tokens: int = 0
    current_speaker_tokens: int = 0
    reply_context_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    total_llm_calls: int = 0
    fallback_used: bool = False
    editor_attempted: bool = False
    editor_trigger: str | None = None
    editor_changed_output: bool = False
    summary_updated: bool = False
    memory_updated: bool = False
    profile_updated: bool = False
    memory_items_written: int = 0
    memory_items_changed: int = 0
    memory_sync_items_changed: int = 0
    memory_extract_items_changed: int = 0
    memory_intro_items_changed: int = 0
    profile_fields_written: int = 0
    profile_fields_changed: int = 0
    profile_sync_fields_changed: int = 0
    profile_extract_fields_changed: int = 0
    summary_write_count: int = 0
    error_type: str | None = None
    error_message: str | None = None
    _finished: bool = False

    def log_start(self):
        if not self.logger:
            return
        payload = {
            "event": "turn_start",
            "turn_id": self.turn_id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "input_chars": self.input_chars,
            "has_attachments": self.has_attachments,
        }
        if self.route_reason is not None:
            payload["route_reason"] = self.route_reason
        if self.input_preview is not None:
            payload["input_preview"] = self.input_preview
        self.logger.log_event(payload)

    def set_context_metrics(self, metrics: dict[str, int]):
        self.history_tokens = int(metrics.get("history_tokens", 0))
        self.summary_tokens = int(metrics.get("summary_tokens", 0))
        self.memory_tokens = int(metrics.get("memory_tokens", 0))
        self.rag_tokens = int(metrics.get("rag_tokens", 0))
        self.system_tokens = int(metrics.get("system_tokens", 0))
        self.input_tokens = int(metrics.get("input_tokens", 0))
        self.identity_tokens = int(metrics.get("identity_tokens", 0))
        self.user_document_tokens = int(metrics.get("user_document_tokens", 0))
        self.creator_profile_tokens = int(metrics.get("creator_profile_tokens", 0))
        self.current_speaker_tokens = int(metrics.get("current_speaker_tokens", 0))
        self.reply_context_tokens = int(metrics.get("reply_context_tokens", 0))
        if self.logger:
            payload = {
                "event": "context_snapshot",
                "turn_id": self.turn_id,
                "mode": self.mode,
                "history_tokens": self.history_tokens,
                "summary_tokens": self.summary_tokens,
                "memory_tokens": self.memory_tokens,
                "rag_tokens": self.rag_tokens,
                "system_tokens": self.system_tokens,
                "input_tokens": self.input_tokens,
                "identity_tokens": self.identity_tokens,
                "user_document_tokens": self.user_document_tokens,
                "creator_profile_tokens": self.creator_profile_tokens,
                "current_speaker_tokens": self.current_speaker_tokens,
                "reply_context_tokens": self.reply_context_tokens,
            }
            self.logger.log_event(payload)

    def record_llm_call(self, call: LLMCallMetrics):
        self.llm_calls.append(call)
        self.total_llm_calls = len(self.llm_calls)
        self.total_tokens += call.total_tokens
        self.total_latency_ms += call.latency_ms
        if self.logger:
            payload = {"event": "llm_call", "turn_id": self.turn_id, "mode": self.mode}
            payload.update(asdict(call))
            self.logger.log_event(payload)

    def set_error(self, error: Exception):
        self.error_type = type(error).__name__
        self.error_message = str(error)

    def finish(self):
        if self._finished:
            return
        self._finished = True
        if not self.logger:
            return
        payload = {
            "event": "turn_end",
            "turn_id": self.turn_id,
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "route_reason": self.route_reason,
            "input_chars": self.input_chars,
            "has_attachments": self.has_attachments,
            "history_tokens": self.history_tokens,
            "summary_tokens": self.summary_tokens,
            "memory_tokens": self.memory_tokens,
            "rag_tokens": self.rag_tokens,
            "system_tokens": self.system_tokens,
            "input_tokens": self.input_tokens,
            "identity_tokens": self.identity_tokens,
            "user_document_tokens": self.user_document_tokens,
            "creator_profile_tokens": self.creator_profile_tokens,
            "current_speaker_tokens": self.current_speaker_tokens,
            "reply_context_tokens": self.reply_context_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "fallback_used": self.fallback_used,
            "editor_attempted": self.editor_attempted,
            "editor_trigger": self.editor_trigger,
            "editor_changed_output": self.editor_changed_output,
            "summary_updated": self.summary_updated,
            "memory_updated": self.memory_updated,
            "profile_updated": self.profile_updated,
            "memory_items_written": self.memory_items_written,
            "memory_items_changed": self.memory_items_changed,
            "memory_sync_items_changed": self.memory_sync_items_changed,
            "memory_extract_items_changed": self.memory_extract_items_changed,
            "memory_intro_items_changed": self.memory_intro_items_changed,
            "profile_fields_written": self.profile_fields_written,
            "profile_fields_changed": self.profile_fields_changed,
            "profile_sync_fields_changed": self.profile_sync_fields_changed,
            "profile_extract_fields_changed": self.profile_extract_fields_changed,
            "summary_write_count": self.summary_write_count,
            "llm_calls": [asdict(call) for call in self.llm_calls],
        }
        if self.input_preview is not None:
            payload["input_preview"] = self.input_preview
        if self.error_type:
            payload["error_type"] = self.error_type
            payload["error_message"] = self.error_message
        self.logger.log_event(payload)


class Telemetry:
    def __init__(
        self,
        *,
        enabled: bool,
        debug_enabled: bool,
        log_path: str,
        mode: str = "legacy",
    ):
        self.enabled = enabled
        self.debug_enabled = debug_enabled
        self.mode = mode
        self.logger = JSONLLogger(log_path, enabled=enabled, debug_enabled=debug_enabled)

    def start_turn(
        self,
        *,
        turn_id: str,
        chat_id: int,
        user_id: int,
        input_text: str,
        has_attachments: bool,
        mode: str | None = None,
        route_reason: str | None = None,
    ) -> TurnMetrics:
        preview = None
        if self.debug_enabled:
            preview = " ".join((input_text or "").split())[:120]
        metrics = TurnMetrics(
            turn_id=turn_id,
            chat_id=chat_id,
            user_id=user_id,
            mode=mode or self.mode,
            route_reason=route_reason,
            input_chars=len(input_text or ""),
            has_attachments=has_attachments,
            input_preview=preview,
            logger=self.logger if self.enabled else None,
        )
        metrics.log_start()
        return metrics
