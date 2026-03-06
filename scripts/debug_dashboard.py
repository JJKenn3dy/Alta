#!/usr/bin/env python3
"""Alta Debug Dashboard — web UI for inspecting LLM debug traces.

Usage:
    python scripts/debug_dashboard.py --db ./alta_tg_test.db --port 8089
"""

import argparse
import http.server
import json
import math
import os
import re
import sqlite3
import sys
import urllib.parse

# ── Constants ──────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4.0

BLOCK_MARKERS = [
    ("В истории пользовательские реплики помечены автором", "speaker_guard", "system"),
    ("Текущий собеседник для этого ответа:", "current_speaker", "system"),
    ("Текущее сообщение короткое и зависит от", "implicit_reply_context", "system"),
    ("Ему бот был только упомянут", "mention_only", "system"),
    ("Текущее сообщение является ответом на конкретную реплику", "reply_context", "system"),
    ("Краткое резюме диалога с текущим собеседником", "speaker_summary", "summary"),
    ("Краткое резюме общего чата", "summary", "summary"),
    ("Профиль текущего собеседника:", "identity", "system"),
    ("Заметки о собеседнике:", "memory", "memory"),
    ("Используй это только как внутренний контекст", "user_document", "system"),
    ("Контекст из файлов", "rag", "rag"),
]

ALTA_SYSTEM_PROMPT_MARKER = "Ты Альта. Женщина около 20 лет."

DB_PATH = ""

# ── DB Layer ───────────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def fetch_traces(limit: int = 50, offset: int = 0, chat_id: str = "", query: str = "") -> dict:
    conn = get_db()
    try:
        where_clauses = []
        params: list = []

        if chat_id:
            where_clauses.append("chat_id = ?")
            params.append(int(chat_id))
        if query:
            where_clauses.append("(input_text LIKE ? OR output_text LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        count_row = conn.execute(f"SELECT COUNT(1) AS cnt FROM debug_traces{where_sql}", params).fetchone()
        total = int(count_row["cnt"]) if count_row else 0

        rows = conn.execute(
            f"SELECT id, ts, chat_id, input_text, output_text "
            f"FROM debug_traces{where_sql} "
            f"ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        traces = []
        for r in rows:
            inp = (r["input_text"] or "")[:120]
            out = (r["output_text"] or "")[:120]
            traces.append({
                "id": r["id"],
                "ts": r["ts"],
                "chat_id": r["chat_id"],
                "input_preview": inp,
                "output_preview": out,
                "input_chars": len(r["input_text"] or ""),
                "output_chars": len(r["output_text"] or ""),
            })

        return {"traces": traces, "total": total, "offset": offset, "limit": limit}
    finally:
        conn.close()


def fetch_trace_detail(trace_id: int) -> dict | None:
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT id, ts, chat_id, input_text, context_json, output_text FROM debug_traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
        if not row:
            return None

        context = {}
        try:
            context = json.loads(row["context_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        system_prompt = context.get("system", "")
        messages = context.get("messages", [])
        system_blocks = context.get("system_blocks", None)

        if system_blocks is None:
            system_blocks = _parse_system_blocks_from_prompt(system_prompt)

        base_prompt, _ = _split_base_prompt(system_prompt)

        system_chars = len(system_prompt)
        messages_chars = sum(len(m.get("content", "")) for m in messages)

        return {
            "id": row["id"],
            "ts": row["ts"],
            "chat_id": row["chat_id"],
            "input_text": row["input_text"] or "",
            "output_text": row["output_text"] or "",
            "base_prompt": base_prompt,
            "system_blocks": system_blocks,
            "messages": messages,
            "meta": {
                "system_prompt_chars": system_chars,
                "system_prompt_approx_tokens": _approx_tokens(system_prompt),
                "messages_count": len(messages),
                "messages_chars": messages_chars,
                "messages_approx_tokens": _approx_tokens_messages(messages),
                "total_approx_tokens": _approx_tokens(system_prompt) + _approx_tokens_messages(messages),
                "block_count": len(system_blocks),
                "input_chars": len(row["input_text"] or ""),
                "output_chars": len(row["output_text"] or ""),
            },
        }
    finally:
        conn.close()


def fetch_stats() -> dict:
    conn = get_db()
    try:
        count_row = conn.execute("SELECT COUNT(1) AS cnt FROM debug_traces").fetchone()
        total = int(count_row["cnt"]) if count_row else 0

        chat_ids_rows = conn.execute(
            "SELECT DISTINCT chat_id FROM debug_traces ORDER BY chat_id"
        ).fetchall()
        chat_ids = [r["chat_id"] for r in chat_ids_rows]

        ts_row = conn.execute(
            "SELECT MIN(ts) AS min_ts, MAX(ts) AS max_ts FROM debug_traces"
        ).fetchone()

        return {
            "total_traces": total,
            "chat_ids": chat_ids,
            "min_ts": ts_row["min_ts"] if ts_row else None,
            "max_ts": ts_row["max_ts"] if ts_row else None,
        }
    finally:
        conn.close()


# ── Block Parsing ──────────────────────────────────────────────────


def _approx_tokens(text: str) -> int:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / CHARS_PER_TOKEN))


def _approx_tokens_messages(messages: list[dict]) -> int:
    return sum(_approx_tokens(m.get("content", "")) for m in messages)


def _split_base_prompt(system_prompt: str) -> tuple[str, str]:
    """Split system prompt into base ALTA prompt and the rest."""
    idx = system_prompt.find(ALTA_SYSTEM_PROMPT_MARKER)
    if idx == -1:
        return "", system_prompt

    # Find where the base prompt ends — look for double newline after known end markers
    end_markers = ["- длинные лекции", "КАК НЕЛЬЗЯ:\n- длинные лекции"]
    rest_start = -1
    for marker in end_markers:
        pos = system_prompt.find(marker)
        if pos != -1:
            rest_start = pos + len(marker)
            break

    if rest_start == -1:
        # Fallback: split at first occurrence of known block marker
        for marker_text, _, _ in BLOCK_MARKERS:
            pos = system_prompt.find(marker_text)
            if pos != -1:
                rest_start = pos
                break

    if rest_start == -1:
        return system_prompt, ""

    base = system_prompt[:rest_start].strip()
    rest = system_prompt[rest_start:].strip()
    return base, rest


def _parse_system_blocks_from_prompt(system_prompt: str) -> list[dict]:
    """Parse the concatenated system prompt back into blocks (fallback for old traces)."""
    _, rest = _split_base_prompt(system_prompt)
    if not rest:
        return []

    # Split by double newline
    segments = re.split(r"\n\n+", rest)
    blocks: list[dict] = []
    current_text = ""
    current_label = "unknown"
    current_bucket = "system"

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        matched_label = None
        matched_bucket = None
        for marker_text, label, bucket in BLOCK_MARKERS:
            if marker_text in segment:
                matched_label = label
                matched_bucket = bucket
                break

        if matched_label:
            if current_text.strip():
                blocks.append({
                    "label": current_label,
                    "bucket": current_bucket,
                    "content": current_text.strip(),
                })
            current_label = matched_label
            current_bucket = matched_bucket
            current_text = segment
        else:
            if current_text:
                current_text += "\n\n" + segment
            else:
                current_text = segment

    if current_text.strip():
        blocks.append({
            "label": current_label,
            "bucket": current_bucket,
            "content": current_text.strip(),
        })

    return blocks


# ── HTML Template ──────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alta Debug Dashboard</title>
<style>
:root {
  --bg: #0f0f17;
  --bg-card: #181825;
  --bg-sidebar: #131320;
  --bg-hover: #1e1e32;
  --bg-selected: #252540;
  --border: #2a2a40;
  --text: #cdd6f4;
  --text-dim: #6c7086;
  --text-bright: #f0f0f8;
  --accent: #89b4fa;
  --accent-dim: #45475a;
  --green: #a6e3a1;
  --blue: #89b4fa;
  --purple: #cba6f7;
  --red: #f38ba8;
  --peach: #fab387;
  --yellow: #f9e2af;
  --teal: #94e2d5;
  --mono: 'Cascadia Code', 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
  --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
/* ── Top bar ── */
.topbar {
  background: var(--bg-sidebar);
  border-bottom: 1px solid var(--border);
  padding: 10px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  height: 52px;
}
.topbar h1 {
  font-size: 15px;
  font-weight: 600;
  color: var(--accent);
  white-space: nowrap;
}
.topbar .stats {
  font-size: 12px;
  color: var(--text-dim);
  white-space: nowrap;
}
.topbar .spacer { flex: 1; }
.topbar input[type="text"] {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  color: var(--text);
  font-size: 13px;
  width: 200px;
  outline: none;
}
.topbar input[type="text"]:focus { border-color: var(--accent); }
.topbar select {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 8px;
  color: var(--text);
  font-size: 13px;
  outline: none;
}
.topbar label {
  font-size: 12px;
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  white-space: nowrap;
}
.btn-refresh {
  background: var(--accent-dim);
  border: none;
  border-radius: 6px;
  padding: 5px 12px;
  color: var(--text);
  font-size: 12px;
  cursor: pointer;
}
.btn-refresh:hover { background: var(--accent); color: var(--bg); }

/* ── Layout ── */
.main-layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 340px;
  min-width: 260px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
.trace-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}
.trace-item {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.15s;
}
.trace-item:hover { background: var(--bg-hover); }
.trace-item.selected { background: var(--bg-selected); border-left: 3px solid var(--accent); }
.trace-item .meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.trace-item .meta .id { font-size: 11px; color: var(--accent); font-weight: 600; }
.trace-item .meta .ts { font-size: 11px; color: var(--text-dim); }
.trace-item .preview {
  font-size: 12px;
  color: var(--text-dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.4;
}
.trace-item .preview.input-preview { color: var(--text); }
.trace-item .chat-id { font-size: 10px; color: var(--accent-dim); }
.sidebar-footer {
  padding: 8px 14px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-dim);
  display: flex;
  justify-content: space-between;
}
.pagination { display: flex; gap: 6px; }
.pagination button {
  background: var(--accent-dim);
  border: none;
  border-radius: 4px;
  padding: 3px 10px;
  color: var(--text);
  font-size: 11px;
  cursor: pointer;
}
.pagination button:disabled { opacity: 0.3; cursor: default; }
.pagination button:hover:not(:disabled) { background: var(--accent); color: var(--bg); }

/* ── Content ── */
.content {
  flex: 1;
  overflow-y: auto;
  padding: 20px 28px;
}
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-dim);
  font-size: 14px;
}

/* ── Trace Detail ── */
.trace-detail h2 {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.trace-detail h2 .badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
  margin-bottom: 20px;
}
.meta-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.meta-item .label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
.meta-item .value { font-size: 14px; font-weight: 600; color: var(--text-bright); margin-top: 2px; }

/* ── Sections ── */
.section {
  margin-bottom: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}
.section-header {
  padding: 10px 14px;
  background: var(--bg-card);
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  user-select: none;
}
.section-header:hover { background: var(--bg-hover); }
.section-header .arrow {
  font-size: 10px;
  color: var(--text-dim);
  transition: transform 0.2s;
  width: 14px;
}
.section-header.open .arrow { transform: rotate(90deg); }
.section-header .title { font-size: 13px; font-weight: 600; }
.section-header .info { font-size: 11px; color: var(--text-dim); margin-left: auto; }
.section-body {
  display: none;
  padding: 12px 14px;
  background: var(--bg);
  max-height: 500px;
  overflow-y: auto;
}
.section-body.open { display: block; }
.section-body pre {
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
}

/* ── Block cards ── */
.block-card {
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
}
.block-card-header {
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 12px;
}
.block-card-header:hover { filter: brightness(1.1); }
.block-card-header .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.block-card-header .block-label { font-weight: 600; }
.block-card-header .block-info { margin-left: auto; color: var(--text-dim); font-size: 11px; }
.block-card-body {
  display: none;
  padding: 10px 12px;
  background: var(--bg);
  border-top: 1px solid var(--border);
}
.block-card-body.open { display: block; }
.block-card-body pre {
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.bucket-system .block-card-header { background: #1e1e2e; }
.bucket-system .dot { background: #585b70; }
.bucket-summary .block-card-header { background: #1a1a30; }
.bucket-summary .dot { background: var(--blue); }
.bucket-memory .block-card-header { background: #1a2a1a; }
.bucket-memory .dot { background: var(--green); }
.bucket-rag .block-card-header { background: #231a30; }
.bucket-rag .dot { background: var(--purple); }

/* ── Messages ── */
.message-list { display: flex; flex-direction: column; gap: 6px; }
.message {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  line-height: 1.5;
  font-family: var(--mono);
  white-space: pre-wrap;
  word-break: break-word;
}
.message.user {
  background: #1a2233;
  border-left: 3px solid var(--blue);
}
.message.assistant {
  background: #1a2a1e;
  border-left: 3px solid var(--green);
}
.message .role-tag {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 4px;
  display: block;
}
.message.user .role-tag { color: var(--blue); }
.message.assistant .role-tag { color: var(--green); }

/* ── IO boxes ── */
.io-box {
  padding: 12px 14px;
  border-radius: 6px;
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
.io-box.input { background: #1a2233; border-left: 3px solid var(--blue); }
.io-box.output { background: #1a2a1e; border-left: 3px solid var(--green); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent-dim); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
</style>
</head>
<body>

<div class="topbar">
  <h1>Alta Debug Dashboard</h1>
  <span class="stats" id="stats-text"></span>
  <div class="spacer"></div>
  <select id="chat-filter"><option value="">All chats</option></select>
  <input type="text" id="search-input" placeholder="Search...">
  <label><input type="checkbox" id="auto-refresh"> Auto</label>
  <button class="btn-refresh" onclick="loadTraces()">Refresh</button>
</div>

<div class="main-layout">
  <div class="sidebar">
    <div class="trace-list" id="trace-list"></div>
    <div class="sidebar-footer">
      <span id="page-info"></span>
      <div class="pagination">
        <button id="btn-prev" onclick="prevPage()" disabled>&lt; Prev</button>
        <button id="btn-next" onclick="nextPage()" disabled>Next &gt;</button>
      </div>
    </div>
  </div>
  <div class="content" id="content">
    <div class="empty-state">Select a trace from the sidebar</div>
  </div>
</div>

<script>
const PAGE_SIZE = 50;
let currentOffset = 0;
let totalTraces = 0;
let selectedId = null;
let autoRefreshTimer = null;

function approxTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.replace(/\s+/g, ' ').trim().length / 4));
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

async function loadStats() {
  try {
    const resp = await fetch('/api/stats');
    const data = await resp.json();
    document.getElementById('stats-text').textContent =
      `${data.total_traces} traces`;
    const sel = document.getElementById('chat-filter');
    const current = sel.value;
    sel.innerHTML = '<option value="">All chats</option>';
    (data.chat_ids || []).forEach(cid => {
      const opt = document.createElement('option');
      opt.value = cid;
      opt.textContent = `Chat ${cid}`;
      sel.appendChild(opt);
    });
    sel.value = current;
  } catch(e) { console.error('loadStats error', e); }
}

async function loadTraces() {
  const chatId = document.getElementById('chat-filter').value;
  const query = document.getElementById('search-input').value;
  try {
    const params = new URLSearchParams({
      limit: PAGE_SIZE, offset: currentOffset,
      chat_id: chatId, q: query,
    });
    const resp = await fetch('/api/traces?' + params);
    const data = await resp.json();
    totalTraces = data.total;

    const list = document.getElementById('trace-list');
    list.innerHTML = '';
    data.traces.forEach(t => {
      const item = document.createElement('div');
      item.className = 'trace-item' + (t.id === selectedId ? ' selected' : '');
      item.onclick = () => loadTraceDetail(t.id);
      item.innerHTML = `
        <div class="meta">
          <span class="id">#${t.id}</span>
          <span class="ts">${t.ts || ''}</span>
        </div>
        <div class="preview input-preview">${escapeHtml(t.input_preview)}</div>
        <div class="preview">${escapeHtml(t.output_preview)}</div>
        <div class="chat-id">chat: ${t.chat_id}</div>
      `;
      list.appendChild(item);
    });

    const total = data.total;
    const pageStart = total > 0 ? currentOffset + 1 : 0;
    const pageEnd = Math.min(currentOffset + PAGE_SIZE, total);
    document.getElementById('page-info').textContent = `${pageStart}-${pageEnd} of ${total}`;
    document.getElementById('btn-prev').disabled = currentOffset <= 0;
    document.getElementById('btn-next').disabled = currentOffset + PAGE_SIZE >= total;

  } catch(e) { console.error('loadTraces error', e); }
}

function prevPage() { currentOffset = Math.max(0, currentOffset - PAGE_SIZE); loadTraces(); }
function nextPage() { currentOffset += PAGE_SIZE; loadTraces(); }

async function loadTraceDetail(id) {
  selectedId = id;
  document.querySelectorAll('.trace-item').forEach(el => {
    el.classList.toggle('selected', el.querySelector('.id').textContent === '#' + id);
  });

  const content = document.getElementById('content');
  content.innerHTML = '<div class="empty-state">Loading...</div>';

  try {
    const resp = await fetch('/api/traces/' + id);
    const t = await resp.json();
    if (!t || t.error) {
      content.innerHTML = '<div class="empty-state">Trace not found</div>';
      return;
    }
    renderTraceDetail(t);
  } catch(e) {
    content.innerHTML = '<div class="empty-state">Error loading trace</div>';
    console.error(e);
  }
}

function toggleSection(header) {
  header.classList.toggle('open');
  const body = header.nextElementSibling;
  body.classList.toggle('open');
}

function toggleBlock(header) {
  const body = header.nextElementSibling;
  body.classList.toggle('open');
}

function renderTraceDetail(t) {
  const content = document.getElementById('content');
  const m = t.meta || {};

  let html = `<div class="trace-detail">`;

  // Header
  html += `<h2>Trace #${t.id}
    <span class="badge" style="background:var(--accent-dim);color:var(--text);">chat ${t.chat_id}</span>
    <span class="badge" style="background:var(--accent-dim);color:var(--text-dim);">${t.ts}</span>
  </h2>`;

  // Meta grid
  html += `<div class="meta-grid">
    <div class="meta-item"><div class="label">Total ~Tokens</div><div class="value">${m.total_approx_tokens || '?'}</div></div>
    <div class="meta-item"><div class="label">System ~Tokens</div><div class="value">${m.system_prompt_approx_tokens || '?'}</div></div>
    <div class="meta-item"><div class="label">Messages</div><div class="value">${m.messages_count || 0}</div></div>
    <div class="meta-item"><div class="label">Msg ~Tokens</div><div class="value">${m.messages_approx_tokens || '?'}</div></div>
    <div class="meta-item"><div class="label">Context Blocks</div><div class="value">${m.block_count || 0}</div></div>
    <div class="meta-item"><div class="label">Input Chars</div><div class="value">${m.input_chars || 0}</div></div>
    <div class="meta-item"><div class="label">Output Chars</div><div class="value">${m.output_chars || 0}</div></div>
  </div>`;

  // Input
  html += `<div class="section">
    <div class="section-header open" onclick="toggleSection(this)">
      <span class="arrow">&#9654;</span>
      <span class="title" style="color:var(--blue)">Input</span>
      <span class="info">${t.input_text.length} chars</span>
    </div>
    <div class="section-body open"><div class="io-box input">${escapeHtml(t.input_text)}</div></div>
  </div>`;

  // Output
  html += `<div class="section">
    <div class="section-header open" onclick="toggleSection(this)">
      <span class="arrow">&#9654;</span>
      <span class="title" style="color:var(--green)">Output</span>
      <span class="info">${t.output_text.length} chars</span>
    </div>
    <div class="section-body open"><div class="io-box output">${escapeHtml(t.output_text)}</div></div>
  </div>`;

  // Context blocks
  const blocks = t.system_blocks || [];
  if (blocks.length > 0) {
    let blocksHtml = '';
    blocks.forEach((b, i) => {
      const bucket = b.bucket || 'system';
      const label = b.label || 'unknown';
      const chars = (b.content || '').length;
      const tokens = approxTokens(b.content || '');
      blocksHtml += `
        <div class="block-card bucket-${bucket}">
          <div class="block-card-header" onclick="toggleBlock(this)">
            <span class="dot"></span>
            <span class="block-label">${escapeHtml(label)}</span>
            <span style="color:var(--text-dim);font-size:11px;">[${bucket}]</span>
            <span class="block-info">${chars} chars / ~${tokens} tok</span>
          </div>
          <div class="block-card-body"><pre>${escapeHtml(b.content || '')}</pre></div>
        </div>
      `;
    });

    const totalBlockTokens = blocks.reduce((s, b) => s + approxTokens(b.content || ''), 0);
    html += `<div class="section">
      <div class="section-header open" onclick="toggleSection(this)">
        <span class="arrow">&#9654;</span>
        <span class="title" style="color:var(--purple)">Context Blocks</span>
        <span class="info">${blocks.length} blocks / ~${totalBlockTokens} tok</span>
      </div>
      <div class="section-body open">${blocksHtml}</div>
    </div>`;
  }

  // Base system prompt
  if (t.base_prompt) {
    html += `<div class="section">
      <div class="section-header" onclick="toggleSection(this)">
        <span class="arrow">&#9654;</span>
        <span class="title" style="color:var(--peach)">Base System Prompt</span>
        <span class="info">${t.base_prompt.length} chars</span>
      </div>
      <div class="section-body"><pre>${escapeHtml(t.base_prompt)}</pre></div>
    </div>`;
  }

  // Message history
  if (t.messages && t.messages.length > 0) {
    let msgsHtml = '<div class="message-list">';
    t.messages.forEach(msg => {
      const role = msg.role || 'user';
      msgsHtml += `<div class="message ${role}">
        <span class="role-tag">${role}</span>${escapeHtml(msg.content || '')}
      </div>`;
    });
    msgsHtml += '</div>';

    html += `<div class="section">
      <div class="section-header open" onclick="toggleSection(this)">
        <span class="arrow">&#9654;</span>
        <span class="title" style="color:var(--yellow)">Message History</span>
        <span class="info">${t.messages.length} messages / ~${m.messages_approx_tokens || '?'} tok</span>
      </div>
      <div class="section-body open">${msgsHtml}</div>
    </div>`;
  }

  html += `</div>`;
  content.innerHTML = html;
}

// Init
document.getElementById('chat-filter').addEventListener('change', () => {
  currentOffset = 0;
  loadTraces();
});

let searchTimeout;
document.getElementById('search-input').addEventListener('input', () => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => { currentOffset = 0; loadTraces(); }, 400);
});

document.getElementById('auto-refresh').addEventListener('change', (e) => {
  if (e.target.checked) {
    autoRefreshTimer = setInterval(() => { loadTraces(); loadStats(); }, 5000);
  } else {
    clearInterval(autoRefreshTimer);
    autoRefreshTimer = null;
  }
});

loadStats();
loadTraces();
</script>

</body>
</html>"""


# ── HTTP Handler ───────────────────────────────────────────────────


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if path == "/":
            self._serve_html()
        elif path == "/api/traces":
            self._serve_api_traces(params)
        elif path == "/api/stats":
            self._serve_api_stats()
        elif path.startswith("/api/traces/"):
            try:
                trace_id = int(path.split("/")[-1])
            except (ValueError, IndexError):
                self._json_response({"error": "invalid trace id"}, 400)
                return
            self._serve_api_trace_detail(trace_id)
        else:
            self.send_error(404)

    def _serve_html(self):
        content = HTML_TEMPLATE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_api_traces(self, params: dict):
        limit = min(int(params.get("limit", "50")), 200)
        offset = max(int(params.get("offset", "0")), 0)
        chat_id = params.get("chat_id", "")
        query = params.get("q", "")
        data = fetch_traces(limit, offset, chat_id, query)
        self._json_response(data)

    def _serve_api_trace_detail(self, trace_id: int):
        data = fetch_trace_detail(trace_id)
        if data is None:
            self._json_response({"error": "not found"}, 404)
        else:
            self._json_response(data)

    def _serve_api_stats(self):
        data = fetch_stats()
        self._json_response(data)

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if "/api/" not in (args[0] if args else ""):
            http.server.BaseHTTPRequestHandler.log_message(self, format, *args)


# ── Main ───────────────────────────────────────────────────────────


def main():
    global DB_PATH

    parser = argparse.ArgumentParser(description="Alta Debug Dashboard")
    parser.add_argument(
        "--db",
        default=os.getenv("DB_PATH", "./alta_tg_test.db"),
        help="Path to SQLite database (default: ./alta_tg_test.db)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DASHBOARD_PORT", "8089")),
        help="Port to serve on (default: 8089)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: database not found: {args.db}")
        sys.exit(1)

    DB_PATH = os.path.abspath(args.db)

    # Verify DB is accessible
    try:
        conn = get_db()
        conn.execute("SELECT 1 FROM debug_traces LIMIT 1")
        conn.close()
    except Exception as e:
        print(f"Error: cannot access debug_traces table: {e}")
        sys.exit(1)

    server = http.server.HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Alta Debug Dashboard")
    print(f"  URL:      http://{args.host}:{args.port}")
    print(f"  Database: {DB_PATH}")
    print(f"  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
