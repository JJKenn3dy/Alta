import argparse
import json
import sqlite3
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
DEFAULT_TRACE_LIMIT = 60
MAX_TRACE_LIMIT = 250
DEFAULT_EVENT_LIMIT = 120
LIVE_EVENTS = {"turn_start", "context_snapshot", "llm_call", "turn_end", "postprocess_end"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def short_text(value: Any, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def parse_json_or_none(value: str | None) -> Any:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_messages:
        if isinstance(item, dict):
            out.append(
                {
                    "role": str(item.get("role") or ""),
                    "content": str(item.get("content") or ""),
                }
            )
    return out


def normalize_blocks(raw_blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_blocks, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_blocks:
        if isinstance(item, dict):
            out.append(
                {
                    "label": str(item.get("label") or ""),
                    "bucket": str(item.get("bucket") or ""),
                    "content": str(item.get("content") or ""),
                }
            )
    return out


class DevTraceStore:
    def __init__(self, db_path: str, log_path: str):
        self.db_path = Path(db_path)
        self.log_path = Path(log_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        return row is not None

    def get_overview(self) -> dict[str, Any]:
        db_exists = self.db_path.exists()
        db_size = self.db_path.stat().st_size if db_exists else 0
        log_exists = self.log_path.exists()
        log_size = self.log_path.stat().st_size if log_exists else 0
        log_updated_at = (
            datetime.fromtimestamp(self.log_path.stat().st_mtime, tz=timezone.utc).isoformat()
            if log_exists
            else None
        )

        trace_count = 0
        trace_latest_id = 0
        trace_latest_ts = None
        message_count = 0
        chat_count = 0
        has_debug_traces_table = False

        if db_exists:
            with self._connect() as conn:
                if self._table_exists(conn, "debug_traces"):
                    has_debug_traces_table = True
                    row = conn.execute(
                        "SELECT COUNT(*) AS cnt, MAX(id) AS max_id, MAX(ts) AS max_ts FROM debug_traces"
                    ).fetchone()
                    trace_count = to_int(row["cnt"])
                    trace_latest_id = to_int(row["max_id"])
                    trace_latest_ts = row["max_ts"]
                if self._table_exists(conn, "messages"):
                    row = conn.execute("SELECT COUNT(*) AS cnt, COUNT(DISTINCT chat_id) AS chats FROM messages").fetchone()
                    message_count = to_int(row["cnt"])
                    chat_count = to_int(row["chats"])

        return {
            "generated_at": utc_now_iso(),
            "db": {
                "path": str(self.db_path),
                "exists": db_exists,
                "size_bytes": int(db_size),
                "has_debug_traces_table": has_debug_traces_table,
                "messages_count": message_count,
                "chats_count": chat_count,
            },
            "traces": {
                "count": trace_count,
                "latest_id": trace_latest_id,
                "latest_ts": trace_latest_ts,
                "debug_hint": "Set DEBUG_TRACES=1 in .env to capture full prompt payloads."
                if trace_count == 0
                else "",
            },
            "log": {
                "path": str(self.log_path),
                "exists": log_exists,
                "size_bytes": int(log_size),
                "updated_at": log_updated_at,
            },
        }

    def list_trace_chats(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        with self._connect() as conn:
            if not self._table_exists(conn, "debug_traces"):
                return []
            rows = conn.execute(
                """
                SELECT chat_id, COUNT(*) AS traces_count, MAX(ts) AS last_ts
                FROM debug_traces
                GROUP BY chat_id
                ORDER BY last_ts DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def _row_to_trace(self, row: sqlite3.Row) -> dict[str, Any]:
        context_json = parse_json_or_none(row["context_json"]) or {}
        if not isinstance(context_json, dict):
            context_json = {"raw_context_json": context_json}

        generation = context_json.get("generation")
        if not isinstance(generation, dict):
            generation = {}

        context = context_json.get("context")
        if not isinstance(context, dict):
            context = {}

        result = context_json.get("result")
        if not isinstance(result, dict):
            result = {}

        metrics = context_json.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        system_prompt = str(context.get("system") or context_json.get("system") or "")
        messages = normalize_messages(context.get("messages", context_json.get("messages")))
        system_blocks = normalize_blocks(context.get("system_blocks", context_json.get("system_blocks")))
        llm_calls = metrics.get("llm_calls")
        if not isinstance(llm_calls, list):
            llm_calls = []

        return {
            "id": to_int(row["id"]),
            "ts": row["ts"],
            "chat_id": to_int(row["chat_id"]),
            "input_text": row["input_text"] or "",
            "output_text": row["output_text"] or "",
            "turn_id": context_json.get("turn_id"),
            "mode": context_json.get("mode"),
            "route_reason": context_json.get("route_reason"),
            "has_attachments": bool(context_json.get("has_attachments")),
            "mention_only": bool(context_json.get("mention_only")),
            "reply_context": context_json.get("reply_context"),
            "generation": generation,
            "context": {
                "system": system_prompt,
                "messages": messages,
                "system_blocks": system_blocks,
            },
            "messages_count": len(messages),
            "system_blocks_count": len(system_blocks),
            "metrics": metrics,
            "llm_calls_count": len(llm_calls),
            "result": result,
            "raw_context_json": context_json,
        }

    def get_traces(
        self,
        *,
        chat_id: int | None = None,
        since_id: int = 0,
        limit: int = DEFAULT_TRACE_LIMIT,
    ) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"generated_at": utc_now_iso(), "items": [], "last_id": since_id}

        items: list[dict[str, Any]] = []
        last_id = since_id

        with self._connect() as conn:
            if not self._table_exists(conn, "debug_traces"):
                return {"generated_at": utc_now_iso(), "items": [], "last_id": since_id}

            params: list[Any] = []
            where: list[str] = []
            if since_id > 0:
                where.append("id > ?")
                params.append(since_id)
            if chat_id is not None:
                where.append("chat_id = ?")
                params.append(chat_id)
            where_sql = ("WHERE " + " AND ".join(where)) if where else ""

            if since_id > 0:
                sql = (
                    "SELECT id, ts, chat_id, input_text, context_json, output_text "
                    f"FROM debug_traces {where_sql} ORDER BY id ASC LIMIT ?"
                )
                params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                sql = (
                    "SELECT id, ts, chat_id, input_text, context_json, output_text "
                    f"FROM debug_traces {where_sql} ORDER BY id DESC LIMIT ?"
                )
                params.append(limit)
                rows = list(reversed(conn.execute(sql, tuple(params)).fetchall()))

            for row in rows:
                trace = self._row_to_trace(row)
                items.append(trace)
                last_id = max(last_id, to_int(trace.get("id")))

        return {"generated_at": utc_now_iso(), "items": items, "last_id": last_id}

    def get_log_events(
        self,
        *,
        since_event_id: int = 0,
        limit: int = DEFAULT_EVENT_LIMIT,
        chat_id: int | None = None,
    ) -> dict[str, Any]:
        if not self.log_path.exists():
            return {"generated_at": utc_now_iso(), "items": [], "last_event_id": since_event_id}

        event_id = 0
        items: list[dict[str, Any]] = []

        with self.log_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                event = parse_json_or_none(line)
                if not isinstance(event, dict):
                    continue
                event_id += 1
                if event_id <= since_event_id:
                    continue
                event_type = str(event.get("event") or "")
                if event_type not in LIVE_EVENTS:
                    continue
                event_chat_id = event.get("chat_id")
                if chat_id is not None and to_int(event_chat_id, 0) != chat_id:
                    continue
                items.append(
                    {
                        "event_id": event_id,
                        "event": event_type,
                        "ts": event.get("ts"),
                        "turn_id": event.get("turn_id"),
                        "chat_id": event_chat_id,
                        "user_id": event.get("user_id"),
                        "mode": event.get("mode"),
                        "preview": short_text(event.get("input_preview"), 120),
                        "payload": event,
                    }
                )

        if len(items) > limit:
            items = items[-limit:]
        return {"generated_at": utc_now_iso(), "items": items, "last_event_id": event_id}


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alta Dev Trace Dashboard</title>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #0f1620; color: #e6eef6; }
    .wrap { padding: 12px; display: grid; gap: 12px; }
    .panel { background: #16212d; border: 1px solid #2b3d4f; border-radius: 10px; padding: 10px; }
    .head { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .head strong { margin-right: auto; font-size: 18px; }
    select, button { background: #0f1822; color: #e6eef6; border: 1px solid #2b3d4f; border-radius: 8px; padding: 6px 8px; }
    button { cursor: pointer; }
    .meta { margin-top: 8px; color: #a9bccc; font-size: 13px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .chip { background: #223447; color: #d2e3f2; border-radius: 999px; padding: 3px 8px; font-size: 12px; }
    .grid { display: grid; grid-template-columns: 1.45fr 1fr; gap: 12px; }
    .list { display: grid; gap: 8px; max-height: 76vh; overflow: auto; }
    .card { border: 1px solid #2b3d4f; border-radius: 8px; background: #101a25; padding: 8px; display: grid; gap: 6px; }
    .row { display: flex; flex-wrap: wrap; gap: 6px; color: #abc0d2; font-size: 13px; }
    .kv { color: #abc0d2; font-size: 12px; }
    .mono { font-family: Consolas, "Courier New", monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
    details { border: 1px solid #2a3b4b; border-radius: 6px; background: #0d151f; padding: 5px 7px; }
    summary { cursor: pointer; color: #bed8ef; }
    .empty { border: 1px dashed #355069; border-radius: 8px; padding: 12px; text-align: center; color: #9ab2c6; font-size: 13px; }
    @media (max-width: 1080px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <div class="head">
        <strong>Alta Dev Trace Dashboard</strong>
        <select id="chatSelect"></select>
        <select id="limitSelect">
          <option value="20">20 traces</option>
          <option value="40">40 traces</option>
          <option value="60" selected>60 traces</option>
          <option value="100">100 traces</option>
        </select>
        <select id="intervalSelect">
          <option value="0">live off</option>
          <option value="1000" selected>live 1s</option>
          <option value="2000">live 2s</option>
          <option value="5000">live 5s</option>
        </select>
        <button id="reloadBtn">Reload</button>
      </div>
      <div class="meta" id="metaLine">loading...</div>
      <div class="chips" id="overviewChips"></div>
    </section>
    <div class="grid">
      <section class="panel">
        <div class="row"><strong>Prompt Traces</strong></div>
        <div class="list" id="traceList"></div>
      </section>
      <section class="panel">
        <div class="row"><strong>Live Orchestration Events</strong></div>
        <div class="list" id="eventList"></div>
      </section>
    </div>
  </div>
  <script>
    const MAX_TRACE_LIMIT = 250;
    const $ = (id) => document.getElementById(id);
    const state = { chatId: "", traceLimit: 60, intervalMs: 1000, tracesLastId: 0, eventsLastId: 0, traces: [], events: [], timer: null };

    function esc(value) {
      return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
    }
    function fmtTs(value) {
      if (!value) return "-";
      const d = new Date(value);
      return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString();
    }
    function fmtBytes(value) {
      const n = Number(value || 0);
      if (n < 1024) return `${n} B`;
      if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
      return `${(n / (1024 * 1024)).toFixed(2)} MB`;
    }
    function jsonPretty(value) {
      try { return JSON.stringify(value ?? {}, null, 2); } catch (_e) { return String(value ?? ""); }
    }
    async function getJson(url) {
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    }
    function upsertById(existing, incoming, key) {
      if (!incoming.length) return existing;
      const known = new Set(existing.map((it) => String(it[key])));
      const out = existing.slice();
      for (const it of incoming) {
        const id = String(it[key]);
        if (known.has(id)) continue;
        out.push(it);
        known.add(id);
      }
      return out;
    }
    function capTail(items, maxItems) {
      if (items.length <= maxItems) return items;
      return items.slice(items.length - maxItems);
    }
    function buildTraceCard(trace) {
      const generation = trace.generation || {};
      const metrics = trace.metrics || {};
      const context = trace.context || {};
      const messages = Array.isArray(context.messages) ? context.messages : [];
      const blocks = Array.isArray(context.system_blocks) ? context.system_blocks : [];
      const llmCalls = Array.isArray(metrics.llm_calls) ? metrics.llm_calls : [];
      return `<article class="card">
        <div class="row">
          <strong>#${trace.id}</strong>
          <span>chat ${esc(trace.chat_id)}</span>
          <span>${fmtTs(trace.ts)}</span>
          <span class="chip">${esc(trace.mode || "unknown")}</span>
          ${trace.turn_id ? `<span class="chip">turn ${esc(trace.turn_id)}</span>` : ""}
          ${trace.mention_only ? `<span class="chip">mention-only</span>` : ""}
          ${trace.has_attachments ? `<span class="chip">attachments</span>` : ""}
        </div>
        <div class="kv">model=${esc(generation.model || "-")} | temp=${esc(generation.temperature)} | top_p=${esc(generation.top_p)} | max_tokens=${esc(generation.max_tokens)} | history_limit=${esc(generation.history_limit)}</div>
        <details open><summary>Input / Output</summary><pre class="mono">INPUT:
${esc(trace.input_text || "")}

OUTPUT:
${esc(trace.output_text || "")}</pre></details>
        <details><summary>System Prompt (${(context.system || "").length} chars)</summary><pre class="mono">${esc(context.system || "")}</pre></details>
        <details><summary>Messages (${messages.length})</summary><pre class="mono">${esc(jsonPretty(messages))}</pre></details>
        <details><summary>System Blocks (${blocks.length})</summary><pre class="mono">${esc(jsonPretty(blocks))}</pre></details>
        <details><summary>Metrics / LLM Calls (${llmCalls.length})</summary><pre class="mono">${esc(jsonPretty(metrics))}</pre></details>
        <details><summary>Raw context_json</summary><pre class="mono">${esc(jsonPretty(trace.raw_context_json || {}))}</pre></details>
      </article>`;
    }
    function buildEventCard(item) {
      return `<article class="card">
        <div class="row">
          <strong>ev#${item.event_id}</strong>
          <span>${esc(item.event)}</span>
          <span>${fmtTs(item.ts)}</span>
          ${item.chat_id != null ? `<span class="chip">chat ${esc(item.chat_id)}</span>` : ""}
          ${item.turn_id ? `<span class="chip">turn ${esc(item.turn_id)}</span>` : ""}
          ${item.mode ? `<span class="chip">${esc(item.mode)}</span>` : ""}
        </div>
        ${item.preview ? `<div class="kv">${esc(item.preview)}</div>` : ""}
        <details><summary>Payload</summary><pre class="mono">${esc(jsonPretty(item.payload || {}))}</pre></details>
      </article>`;
    }
    function renderTraces() {
      const node = $("traceList");
      if (!state.traces.length) {
        node.innerHTML = '<div class="empty">No traces yet. Send messages to bot with DEBUG_TRACES=1.</div>';
        return;
      }
      node.innerHTML = state.traces.slice().reverse().map(buildTraceCard).join("");
    }
    function renderEvents() {
      const node = $("eventList");
      if (!state.events.length) {
        node.innerHTML = '<div class="empty">No orchestration events yet.</div>';
        return;
      }
      node.innerHTML = state.events.slice().reverse().map(buildEventCard).join("");
    }
    function renderOverview(overview) {
      const db = overview.db || {};
      const traces = overview.traces || {};
      const log = overview.log || {};
      $("metaLine").textContent = `${fmtTs(overview.generated_at)} | DB ${fmtBytes(db.size_bytes)} | LOG ${fmtBytes(log.size_bytes)} | ${db.path} | ${log.path}`;
      const chips = [];
      chips.push(`<span class="chip">DB: ${db.exists ? "ok" : "missing"}</span>`);
      chips.push(`<span class="chip">LOG: ${log.exists ? "ok" : "missing"}</span>`);
      chips.push(`<span class="chip">debug_traces: ${traces.count || 0}</span>`);
      chips.push(`<span class="chip">messages: ${db.messages_count || 0}</span>`);
      chips.push(`<span class="chip">chats: ${db.chats_count || 0}</span>`);
      if (traces.latest_ts) chips.push(`<span class="chip">latest trace: ${fmtTs(traces.latest_ts)}</span>`);
      if (traces.debug_hint) chips.push(`<span class="chip">${esc(traces.debug_hint)}</span>`);
      $("overviewChips").innerHTML = chips.join("");
    }
    function renderChats(chats) {
      const options = ['<option value="">All chats</option>']
        .concat((chats || []).map((chat) => {
          const selected = String(chat.chat_id) === String(state.chatId) ? " selected" : "";
          return `<option value="${chat.chat_id}"${selected}>chat ${chat.chat_id} (${chat.traces_count})</option>`;
        }));
      $("chatSelect").innerHTML = options.join("");
    }
    async function loadOverviewAndChats() {
      const [overview, chats] = await Promise.all([getJson("/api/overview"), getJson("/api/chats")]);
      renderOverview(overview);
      renderChats(chats);
    }
    async function reloadAll() {
      await loadOverviewAndChats();
      state.tracesLastId = 0;
      state.eventsLastId = 0;
      const traceUrl = `/api/traces?limit=${encodeURIComponent(state.traceLimit)}${state.chatId ? `&chat_id=${encodeURIComponent(state.chatId)}` : ""}`;
      const eventUrl = `/api/events?limit=200${state.chatId ? `&chat_id=${encodeURIComponent(state.chatId)}` : ""}`;
      const [tracesResp, eventsResp] = await Promise.all([getJson(traceUrl), getJson(eventUrl)]);
      state.traces = tracesResp.items || [];
      state.events = eventsResp.items || [];
      state.tracesLastId = Number(tracesResp.last_id || 0);
      state.eventsLastId = Number(eventsResp.last_event_id || 0);
      renderTraces();
      renderEvents();
    }
    async function pollLive() {
      const traceUrl = `/api/traces?since_id=${encodeURIComponent(state.tracesLastId)}&limit=${encodeURIComponent(MAX_TRACE_LIMIT)}${state.chatId ? `&chat_id=${encodeURIComponent(state.chatId)}` : ""}`;
      const eventUrl = `/api/events?since_event_id=${encodeURIComponent(state.eventsLastId)}&limit=200${state.chatId ? `&chat_id=${encodeURIComponent(state.chatId)}` : ""}`;
      const [overview, tracesResp, eventsResp] = await Promise.all([getJson("/api/overview"), getJson(traceUrl), getJson(eventUrl)]);
      renderOverview(overview);
      const newTraces = tracesResp.items || [];
      const newEvents = eventsResp.items || [];
      if (newTraces.length) {
        state.traces = capTail(upsertById(state.traces, newTraces, "id"), 400);
        renderTraces();
      }
      if (newEvents.length) {
        state.events = capTail(upsertById(state.events, newEvents, "event_id"), 500);
        renderEvents();
      }
      state.tracesLastId = Number(tracesResp.last_id || state.tracesLastId);
      state.eventsLastId = Number(eventsResp.last_event_id || state.eventsLastId);
    }
    function syncTimer() {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }
      if (state.intervalMs > 0) {
        state.timer = setInterval(async () => {
          try { await pollLive(); } catch (e) { $("metaLine").textContent = `Live refresh failed: ${e.message}`; }
        }, state.intervalMs);
      }
    }
    $("reloadBtn").onclick = async () => { try { await reloadAll(); } catch (e) { $("metaLine").textContent = `Reload failed: ${e.message}`; } };
    $("chatSelect").onchange = async () => { state.chatId = $("chatSelect").value; try { await reloadAll(); } catch (e) { $("metaLine").textContent = `Chat filter failed: ${e.message}`; } };
    $("limitSelect").onchange = async () => { state.traceLimit = Number($("limitSelect").value || 60); try { await reloadAll(); } catch (e) { $("metaLine").textContent = `Limit update failed: ${e.message}`; } };
    $("intervalSelect").onchange = () => { state.intervalMs = Number($("intervalSelect").value || 0); syncTimer(); };
    (async () => { try { await reloadAll(); } catch (e) { $("metaLine").textContent = `Initial load failed: ${e.message}`; } syncTimer(); })();
  </script>
</body>
</html>
"""


class DevTraceHandler(BaseHTTPRequestHandler):
    store: DevTraceStore

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/overview":
            self._send_json(self.store.get_overview())
            return
        if parsed.path == "/api/chats":
            self._send_json(self.store.list_trace_chats())
            return
        if parsed.path == "/api/traces":
            params = parse_qs(parsed.query)
            chat_id_value = to_int((params.get("chat_id") or [""])[0], 0)
            since_id = max(0, to_int((params.get("since_id") or [0])[0], 0))
            limit = clamp(to_int((params.get("limit") or [DEFAULT_TRACE_LIMIT])[0], DEFAULT_TRACE_LIMIT), 1, MAX_TRACE_LIMIT)
            chat_id = chat_id_value if chat_id_value else None
            self._send_json(self.store.get_traces(chat_id=chat_id, since_id=since_id, limit=limit))
            return
        if parsed.path == "/api/events":
            params = parse_qs(parsed.query)
            chat_id_value = to_int((params.get("chat_id") or [""])[0], 0)
            since_event_id = max(0, to_int((params.get("since_event_id") or [0])[0], 0))
            limit = clamp(to_int((params.get("limit") or [DEFAULT_EVENT_LIMIT])[0], DEFAULT_EVENT_LIMIT), 1, 500)
            chat_id = chat_id_value if chat_id_value else None
            self._send_json(
                self.store.get_log_events(
                    since_event_id=since_event_id,
                    limit=limit,
                    chat_id=chat_id,
                )
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any):
        return

    def _send_json(self, payload: Any):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def build_handler(store: DevTraceStore):
    class Handler(DevTraceHandler):
        pass

    Handler.store = store
    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live dev dashboard for Alta bot trace payloads")
    parser.add_argument("--db", default="alta_tg_test.db", help="Path to SQLite DB")
    parser.add_argument("--log", default="orchestration.log", help="Path to JSONL orchestration log")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = DevTraceStore(db_path=args.db, log_path=args.log)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(store))
    print(f"Dev trace dashboard: http://{args.host}:{args.port}")
    print(f"DB: {Path(args.db).resolve()}")
    print(f"LOG: {Path(args.log).resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
