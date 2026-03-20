#!/usr/bin/env python3
"""Alta Debug Dashboard — generates a self-contained HTML file with all debug traces.

Usage:
    python scripts/debug_dashboard.py --db ./alta_tg_test.db
    python scripts/debug_dashboard.py --db ./alta_tg_test.db --out dashboard.html
    python scripts/debug_dashboard.py --db ./alta_tg_test.db --serve 8089
"""

import argparse
import http.server
import json
import math
import os
import re
import sqlite3
import sys
import traceback
import urllib.parse
import webbrowser

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

# ── DB Layer ───────────────────────────────────────────────────────


def get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_all_traces(db_path: str, limit: int = 200) -> list[dict]:
    conn = get_db(db_path)
    try:
        rows = conn.execute(
            "SELECT id, ts, chat_id, input_text, context_json, output_text "
            "FROM debug_traces ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        traces = []
        for r in rows:
            context = {}
            try:
                context = json.loads(r["context_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            system_prompt = context.get("system", "")
            messages = context.get("messages", [])
            system_blocks = context.get("system_blocks", None)

            if system_blocks is None:
                system_blocks = _parse_system_blocks_from_prompt(system_prompt)

            base_prompt, _ = _split_base_prompt(system_prompt)

            traces.append({
                "id": r["id"],
                "ts": r["ts"] or "",
                "chat_id": r["chat_id"],
                "input_text": r["input_text"] or "",
                "output_text": r["output_text"] or "",
                "base_prompt": base_prompt,
                "system_blocks": system_blocks,
                "messages": messages,
                "system_prompt_chars": len(system_prompt),
                "system_prompt_tokens": _approx_tokens(system_prompt),
                "messages_count": len(messages),
                "messages_tokens": sum(_approx_tokens(m.get("content", "")) for m in messages),
                "total_tokens": _approx_tokens(system_prompt) + sum(_approx_tokens(m.get("content", "")) for m in messages),
                "block_count": len(system_blocks) if system_blocks else 0,
            })

        return traces
    finally:
        conn.close()


# ── Block Parsing ──────────────────────────────────────────────────


def _approx_tokens(text: str) -> int:
    normalized = " ".join(str(text or "").split())
    return max(1, math.ceil(len(normalized) / CHARS_PER_TOKEN)) if normalized else 0


def _split_base_prompt(system_prompt: str) -> tuple[str, str]:
    idx = system_prompt.find(ALTA_SYSTEM_PROMPT_MARKER)
    if idx == -1:
        return "", system_prompt
    end_markers = ["- длинные лекции", "КАК НЕЛЬЗЯ:\n- длинные лекции"]
    rest_start = -1
    for marker in end_markers:
        pos = system_prompt.find(marker)
        if pos != -1:
            rest_start = pos + len(marker)
            break
    if rest_start == -1:
        for marker_text, _, _ in BLOCK_MARKERS:
            pos = system_prompt.find(marker_text)
            if pos != -1:
                rest_start = pos
                break
    if rest_start == -1:
        return system_prompt, ""
    return system_prompt[:rest_start].strip(), system_prompt[rest_start:].strip()


def _parse_system_blocks_from_prompt(system_prompt: str) -> list[dict]:
    _, rest = _split_base_prompt(system_prompt)
    if not rest:
        return []
    segments = re.split(r"\n\n+", rest)
    blocks: list[dict] = []
    current_text = ""
    current_label = "unknown"
    current_bucket = "system"
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        matched_label = matched_bucket = None
        for marker_text, label, bucket in BLOCK_MARKERS:
            if marker_text in segment:
                matched_label, matched_bucket = label, bucket
                break
        if matched_label:
            if current_text.strip():
                blocks.append({"label": current_label, "bucket": current_bucket, "content": current_text.strip()})
            current_label, current_bucket, current_text = matched_label, matched_bucket, segment
        else:
            current_text = (current_text + "\n\n" + segment) if current_text else segment
    if current_text.strip():
        blocks.append({"label": current_label, "bucket": current_bucket, "content": current_text.strip()})
    return blocks


# ── HTML Generator ─────────────────────────────────────────────────


def generate_html(traces: list[dict]) -> str:
    """Generate a fully self-contained HTML file with all trace data embedded as JSON."""

    # Serialize traces to JSON for embedding in <script>
    traces_json = json.dumps(traces, ensure_ascii=True, indent=None)

    return """<!DOCTYPE html>
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
  --peach: #fab387;
  --yellow: #f9e2af;
  --mono: 'Cascadia Code','JetBrains Mono','Fira Code','Consolas',monospace;
  --sans: 'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--sans);background:var(--bg);color:var(--text);height:100vh;overflow:hidden;display:flex;flex-direction:column}
.topbar{background:var(--bg-sidebar);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:16px;flex-shrink:0;height:52px}
.topbar h1{font-size:15px;font-weight:600;color:var(--accent);white-space:nowrap}
.topbar .stats{font-size:12px;color:var(--text-dim)}
.topbar .spacer{flex:1}
.topbar input[type="text"]{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:5px 10px;color:var(--text);font-size:13px;width:200px;outline:none}
.topbar input:focus{border-color:var(--accent)}
.topbar select{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:5px 8px;color:var(--text);font-size:13px;outline:none}
.main-layout{display:flex;flex:1;overflow:hidden}
.sidebar{width:340px;min-width:260px;background:var(--bg-sidebar);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
.trace-list{flex:1;overflow-y:auto;padding:4px 0}
.trace-item{padding:10px 14px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .15s}
.trace-item:hover{background:var(--bg-hover)}
.trace-item.selected{background:var(--bg-selected);border-left:3px solid var(--accent)}
.trace-item .meta{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.trace-item .meta .id{font-size:11px;color:var(--accent);font-weight:600}
.trace-item .meta .ts{font-size:11px;color:var(--text-dim)}
.trace-item .preview{font-size:12px;color:var(--text-dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;line-height:1.4}
.trace-item .preview.inp{color:var(--text)}
.trace-item .chat-id{font-size:10px;color:var(--accent-dim)}
.content{flex:1;overflow-y:auto;padding:20px 28px}
.empty-state{display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-dim);font-size:14px}
.trace-detail h2{font-size:16px;font-weight:600;color:var(--text-bright);margin-bottom:16px;display:flex;align-items:center;gap:12px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:500;background:var(--accent-dim);color:var(--text)}
.meta-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin-bottom:20px}
.meta-item{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.meta-item .label{font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.5px}
.meta-item .value{font-size:14px;font-weight:600;color:var(--text-bright);margin-top:2px}
.section{margin-bottom:12px;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.section-header{padding:10px 14px;background:var(--bg-card);cursor:pointer;display:flex;align-items:center;gap:8px;user-select:none}
.section-header:hover{background:var(--bg-hover)}
.section-header .arrow{font-size:10px;color:var(--text-dim);transition:transform .2s;width:14px}
.section-header.open .arrow{transform:rotate(90deg)}
.section-header .title{font-size:13px;font-weight:600}
.section-header .info{font-size:11px;color:var(--text-dim);margin-left:auto}
.section-body{display:none;padding:12px 14px;background:var(--bg);max-height:500px;overflow-y:auto}
.section-body.open{display:block}
.section-body pre{font-family:var(--mono);font-size:12px;line-height:1.6;white-space:pre-wrap;word-break:break-word;color:var(--text)}
.block-card{border:1px solid var(--border);border-radius:6px;margin-bottom:8px;overflow:hidden}
.block-card-header{padding:8px 12px;display:flex;align-items:center;gap:8px;cursor:pointer;font-size:12px}
.block-card-header:hover{filter:brightness(1.1)}
.block-card-header .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.block-card-header .block-label{font-weight:600}
.block-card-header .block-info{margin-left:auto;color:var(--text-dim);font-size:11px}
.block-card-body{display:none;padding:10px 12px;background:var(--bg);border-top:1px solid var(--border)}
.block-card-body.open{display:block}
.block-card-body pre{font-family:var(--mono);font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.bucket-system .block-card-header{background:#1e1e2e}
.bucket-system .dot{background:#585b70}
.bucket-summary .block-card-header{background:#1a1a30}
.bucket-summary .dot{background:var(--blue)}
.bucket-memory .block-card-header{background:#1a2a1a}
.bucket-memory .dot{background:var(--green)}
.bucket-rag .block-card-header{background:#231a30}
.bucket-rag .dot{background:var(--purple)}
.message-list{display:flex;flex-direction:column;gap:6px}
.message{padding:8px 12px;border-radius:6px;font-size:12px;line-height:1.5;font-family:var(--mono);white-space:pre-wrap;word-break:break-word}
.message.user{background:#1a2233;border-left:3px solid var(--blue)}
.message.assistant{background:#1a2a1e;border-left:3px solid var(--green)}
.message .role-tag{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;display:block}
.message.user .role-tag{color:var(--blue)}
.message.assistant .role-tag{color:var(--green)}
.io-box{padding:12px 14px;border-radius:6px;font-family:var(--mono);font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
.io-box.input{background:#1a2233;border-left:3px solid var(--blue)}
.io-box.output{background:#1a2a1e;border-left:3px solid var(--green)}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--accent-dim);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--text-dim)}
</style>
</head>
<body>

<div class="topbar">
  <h1>Alta Debug Dashboard</h1>
  <span class="stats" id="stats"></span>
  <div class="spacer"></div>
  <select id="chat-filter" onchange="filterTraces()"><option value="">All chats</option></select>
  <input type="text" id="search" placeholder="Search..." oninput="filterTraces()">
</div>

<div class="main-layout">
  <div class="sidebar">
    <div class="trace-list" id="trace-list"></div>
  </div>
  <div class="content" id="content">
    <div class="empty-state">Select a trace from the sidebar</div>
  </div>
</div>

<script>
var ALL_TRACES = """ + traces_json + """;

var filteredTraces = ALL_TRACES;
var selectedId = null;

function esc(s) {
  var d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function approxTok(s) {
  if (!s) return 0;
  return Math.max(1, Math.ceil(s.replace(/\\s+/g,' ').trim().length / 4));
}

function filterTraces() {
  var chatId = document.getElementById('chat-filter').value;
  var q = document.getElementById('search').value.toLowerCase();
  filteredTraces = ALL_TRACES.filter(function(t) {
    if (chatId && String(t.chat_id) !== chatId) return false;
    if (q && t.input_text.toLowerCase().indexOf(q) === -1 && t.output_text.toLowerCase().indexOf(q) === -1) return false;
    return true;
  });
  renderList();
}

function renderList() {
  var list = document.getElementById('trace-list');
  list.innerHTML = '';
  document.getElementById('stats').textContent = filteredTraces.length + ' / ' + ALL_TRACES.length + ' traces';
  for (var i = 0; i < filteredTraces.length; i++) {
    var t = filteredTraces[i];
    var item = document.createElement('div');
    item.className = 'trace-item' + (t.id === selectedId ? ' selected' : '');
    item.setAttribute('data-id', t.id);
    item.onclick = (function(id){ return function(){ showTrace(id); }; })(t.id);
    item.innerHTML =
      '<div class="meta"><span class="id">#' + t.id + '</span><span class="ts">' + esc(t.ts) + '</span></div>' +
      '<div class="preview inp">' + esc((t.input_text||'').substring(0,120)) + '</div>' +
      '<div class="preview">' + esc((t.output_text||'').substring(0,120)) + '</div>' +
      '<div class="chat-id">chat: ' + t.chat_id + '</div>';
    list.appendChild(item);
  }
}

function showTrace(id) {
  selectedId = id;
  var t = null;
  for (var i = 0; i < ALL_TRACES.length; i++) {
    if (ALL_TRACES[i].id === id) { t = ALL_TRACES[i]; break; }
  }
  if (!t) return;

  // Update selected state in sidebar
  var items = document.querySelectorAll('.trace-item');
  for (var i = 0; i < items.length; i++) {
    items[i].classList.toggle('selected', parseInt(items[i].getAttribute('data-id')) === id);
  }

  var c = document.getElementById('content');
  var h = '<div class="trace-detail">';

  // Header
  h += '<h2>Trace #' + t.id + ' <span class="badge">chat ' + t.chat_id + '</span> <span class="badge" style="color:var(--text-dim)">' + esc(t.ts) + '</span></h2>';

  // Meta
  h += '<div class="meta-grid">';
  var metas = [
    ['Total ~Tokens', t.total_tokens],
    ['System ~Tokens', t.system_prompt_tokens],
    ['Messages', t.messages_count],
    ['Msg ~Tokens', t.messages_tokens],
    ['Blocks', t.block_count],
    ['Input Chars', t.input_text.length],
    ['Output Chars', t.output_text.length]
  ];
  for (var i = 0; i < metas.length; i++) {
    h += '<div class="meta-item"><div class="label">' + metas[i][0] + '</div><div class="value">' + metas[i][1] + '</div></div>';
  }
  h += '</div>';

  // Input
  h += section('Input', 'var(--blue)', t.input_text.length + ' chars', '<div class="io-box input">' + esc(t.input_text) + '</div>', true);

  // Output
  h += section('Output', 'var(--green)', t.output_text.length + ' chars', '<div class="io-box output">' + esc(t.output_text) + '</div>', true);

  // Context blocks
  var blocks = t.system_blocks || [];
  if (blocks.length > 0) {
    var bh = '';
    var totalBTok = 0;
    for (var i = 0; i < blocks.length; i++) {
      var b = blocks[i];
      var bkt = b.bucket || 'system';
      var tok = approxTok(b.content);
      totalBTok += tok;
      bh += '<div class="block-card bucket-' + esc(bkt) + '">' +
        '<div class="block-card-header" onclick="this.nextElementSibling.classList.toggle(\'open\')">' +
        '<span class="dot"></span>' +
        '<span class="block-label">' + esc(b.label) + '</span>' +
        '<span style="color:var(--text-dim);font-size:11px">[' + esc(bkt) + ']</span>' +
        '<span class="block-info">' + (b.content||'').length + ' chars / ~' + tok + ' tok</span>' +
        '</div>' +
        '<div class="block-card-body"><pre>' + esc(b.content) + '</pre></div></div>';
    }
    h += section('Context Blocks', 'var(--purple)', blocks.length + ' blocks / ~' + totalBTok + ' tok', bh, true);
  }

  // Base system prompt
  if (t.base_prompt) {
    h += section('Base System Prompt', 'var(--peach)', t.base_prompt.length + ' chars', '<pre>' + esc(t.base_prompt) + '</pre>', false);
  }

  // Messages
  var msgs = t.messages || [];
  if (msgs.length > 0) {
    var mh = '<div class="message-list">';
    for (var i = 0; i < msgs.length; i++) {
      var role = msgs[i].role || 'user';
      mh += '<div class="message ' + esc(role) + '"><span class="role-tag">' + esc(role) + '</span>' + esc(msgs[i].content) + '</div>';
    }
    mh += '</div>';
    h += section('Message History', 'var(--yellow)', msgs.length + ' messages / ~' + t.messages_tokens + ' tok', mh, true);
  }

  h += '</div>';
  c.innerHTML = h;
}

function section(title, color, info, body, openByDefault) {
  var oc = openByDefault ? ' open' : '';
  return '<div class="section">' +
    '<div class="section-header' + oc + '" onclick="this.classList.toggle(\'open\');this.nextElementSibling.classList.toggle(\'open\')">' +
    '<span class="arrow">&#9654;</span>' +
    '<span class="title" style="color:' + color + '">' + esc(title) + '</span>' +
    '<span class="info">' + info + '</span>' +
    '</div>' +
    '<div class="section-body' + oc + '">' + body + '</div></div>';
}

// Init: populate chat filter
var chatIds = {};
for (var i = 0; i < ALL_TRACES.length; i++) chatIds[ALL_TRACES[i].chat_id] = true;
var sel = document.getElementById('chat-filter');
for (var cid in chatIds) {
  var opt = document.createElement('option');
  opt.value = cid;
  opt.textContent = 'Chat ' + cid;
  sel.appendChild(opt);
}

renderList();
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Alta Debug Dashboard - generates HTML with all debug traces"
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DB_PATH", "./alta_tg_test.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output HTML file path (default: alta_dashboard.html next to DB)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max traces to include (default: 200)",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=0,
        help="If set, start HTTP server on this port instead of writing a file",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --serve mode (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f"Error: database not found: {db_path}")
        sys.exit(1)

    print(f"Reading traces from: {db_path}")
    traces = fetch_all_traces(db_path, limit=args.limit)
    print(f"Found {len(traces)} traces")

    if not traces:
        print("No debug traces found. Make sure DEBUG_TRACES=1 in .env and the bot has processed messages.")
        sys.exit(0)

    html = generate_html(traces)
    print(f"Generated HTML: {len(html)} bytes")

    if args.serve:
        # Serve mode: serve the generated HTML on a port
        html_bytes = html.encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

            def log_message(self, fmt, *a):
                pass

        server = http.server.HTTPServer((args.host, args.serve), Handler)
        url = f"http://{args.host}:{args.serve}"
        print(f"Serving at: {url}")
        print("Press Ctrl+C to stop")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDone.")
            server.server_close()
    else:
        # File mode: write HTML and open in browser
        out_path = args.out
        if not out_path:
            out_path = os.path.join(os.path.dirname(db_path), "alta_dashboard.html")

        out_path = os.path.abspath(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Written to: {out_path}")
        print("Opening in browser...")
        webbrowser.open("file:///" + out_path.replace("\\", "/"))


if __name__ == "__main__":
    main()
