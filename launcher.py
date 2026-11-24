# -*- coding: utf-8 -*-
"""Alta Control Panel — запуск, мониторинг, управление."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, RichLog, Select, Static

# ─── Конфигурация ────────────────────────────────────────────────────────────

BOT_DIR      = Path(__file__).parent
SERVER_SSH   = os.getenv("ALTA_SERVER_SSH", "user@llm-server")
VPS_SSH      = os.getenv("ALTA_VPS_SSH", "user@vps")
LLAMA_HEALTH = "http://127.0.0.1:8080/health"

_LIST_MODELS_CMD = "ls ~/models/*.gguf 2>/dev/null | xargs -I{} basename {}"

_LAN_STATS_CMD = (
    "awk 'NR==1{idle=$5;t=0;for(i=2;i<=8;i++)t+=$i;"
    "printf \"CPU:%d%% \",int((1-idle/t)*100)}' /proc/stat; "
    "free -m | awk '/Mem:/{printf \"RAM:%.1f/%.1fG \", $3/1024, $2/1024}'; "
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
    "--format=csv,noheader,nounits 2>/dev/null | head -1 | "
    "awk -F', ' '{printf \"GPU:%d%% VRAM:%dM/%dM TEMP:%dC\",$1,$2,$3,$4}'"
)

_VPS_STATS_CMD = (
    "awk 'NR==1{idle=$5;t=0;for(i=2;i<=8;i++)t+=$i;"
    "printf \"CPU:%d%% \",int((1-idle/t)*100)}' /proc/stat; "
    "free -m | awk '/Mem:/{printf \"RAM:%.1f/%.1fG\",$3/1024,$2/1024}'"
)

# ─── Утилиты ─────────────────────────────────────────────────────────────────

def _update_env_model(model_name: str) -> None:
    """Обновляет LLM_MODEL в .env файле."""
    env_path = BOT_DIR / ".env"
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("LLM_MODEL="):
            new_lines.append(f"LLM_MODEL={model_name}")
        else:
            new_lines.append(line)
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _bar(pct: int, width: int = 6) -> str:
    filled = round(pct / 100 * width)
    return "#" * filled + "." * (width - filled)

def _parse_stats(raw: str) -> list[str]:
    lines = []
    for token in raw.split():
        if token.startswith("CPU:"):
            val = token[4:]
            try:
                pct = int(val.rstrip("%"))
            except ValueError:
                pct = 0
            bar   = _bar(pct)
            color = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
            lines.append(f"CPU  [{color}]{bar} {val}[/{color}]")
        elif token.startswith("RAM:"):
            lines.append(f"RAM  [cyan]{token[4:]}[/cyan]")
        elif token.startswith("GPU:"):
            val = token[4:]
            try:
                pct = int(val.rstrip("%"))
            except ValueError:
                pct = 0
            color = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
            lines.append(f"GPU  [{color}]{val}[/{color}]")
        elif token.startswith("VRAM:"):
            lines.append(f"VRAM [cyan]{token[5:]}[/cyan]")
        elif token.startswith("TEMP:"):
            val = token[5:]
            try:
                temp = int(val.rstrip("C"))
            except ValueError:
                temp = 0
            color = "green" if temp < 65 else ("yellow" if temp < 80 else "red")
            lines.append(f"Temp [{color}]{val}[/{color}]")
    return lines if lines else ["[dim]нет данных[/dim]"]


# ─── Виджеты ─────────────────────────────────────────────────────────────────

class ServiceRow(Static):
    status: reactive[str] = reactive("stopped")

    _ICONS = {
        "running":  "[green]+[/green]",
        "stopped":  "[red]-[/red]",
        "starting": "[yellow]~[/yellow]",
        "unknown":  "[dim]?[/dim]",
    }
    _LABELS = {
        "running":  "[green]работает[/green]",
        "stopped":  "[red]остановлен[/red]",
        "starting": "[yellow]запускается[/yellow]",
        "unknown":  "[dim]неизвестно[/dim]",
    }

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self._name = name

    def on_mount(self) -> None:
        self._refresh()

    def watch_status(self, _: str) -> None:
        self._refresh()

    def _refresh(self) -> None:
        icon  = self._ICONS.get(self.status, "[dim]?[/dim]")
        label = self._LABELS.get(self.status, self.status)
        self.update(f"{icon} {self._name:<13}{label}")


class ServerStats(Static):
    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self._title = title

    def on_mount(self) -> None:
        self.update(f"[bold dim]{self._title}[/bold dim]\n[dim]...[/dim]")

    def set_stats(self, lines: list[str]) -> None:
        body = "\n".join(lines)
        self.update(f"[bold dim]{self._title}[/bold dim]\n{body}")

    def set_offline(self) -> None:
        self.update(f"[bold dim]{self._title}[/bold dim]\n[red]недоступен[/red]")


# ─── Приложение ──────────────────────────────────────────────────────────────

class AltaPanel(App):
    TITLE = "Alta Control Panel"

    CSS = """
    Screen { background: $surface; }

    #sidebar {
        width: 36;
        background: $panel;
        border-right: solid $primary-darken-3;
        padding: 1 2;
    }

    .section-label {
        color: $text-muted;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }

    ServiceRow  { height: 1; margin-bottom: 1; }
    ServerStats { height: auto; margin-bottom: 1; }

    #log-panel { border: blank; padding: 0 1; }

    #controls {
        height: 5;
        background: $panel;
        border-top: solid $primary-darken-3;
        align: center middle;
        padding: 0 2;
    }

    Button { margin: 0 1; min-width: 20; }

    #btn-start   { background: #238636; border: tall #2ea043; }
    #btn-restart { background: #1f6feb; border: tall #388bfd; }
    #btn-stop    { background: #6e1c1c; border: tall #b91c1c; }
    #btn-server  { background: #6e40c9; border: tall #8957e5; }
    #btn-vps     { background: #3d444d; border: tall #6e7681; }

    #model-bar {
        height: 3;
        background: $panel;
        border-top: solid $primary-darken-3;
        align: left middle;
        padding: 0 2;
    }
    #model-label { margin-right: 1; width: auto; }
    #model-select { width: 1fr; }
    """

    BINDINGS = [
        Binding("r", "restart_bot", "Рестарт бота"),
        Binding("s", "start_all",   "Запустить"),
        Binding("q", "quit",        "Выход"),
    ]

    def __init__(self):
        super().__init__()
        self._bot_proc: subprocess.Popen | None = None
        self._tunnel_proc: subprocess.Popen | None = None

    # ── Compose ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Сервисы", classes="section-label")
                yield ServiceRow("llama-server", id="st-llama")
                yield ServiceRow("SSH туннель",  id="st-tunnel")
                yield ServiceRow("Alta бот",     id="st-bot")

                yield Label("LAN Сервер", classes="section-label")
                yield ServerStats(SERVER_SSH.split("@")[-1], id="stats-lan")

                yield Label("VPS", classes="section-label")
                yield ServerStats(VPS_SSH.split("@")[-1], id="stats-vps")

                yield Label("Подключения", classes="section-label")
                yield Button(">> Сервер LAN", id="btn-server")
                yield Button(">> VPS",        id="btn-vps")

            yield RichLog(id="log-panel", highlight=True, markup=True, wrap=True)

        with Horizontal(id="model-bar"):
            yield Label("Модель:", id="model-label")
            yield Select(options=[], id="model-select", prompt="загрузка списка...")

        with Horizontal(id="controls"):
            yield Button("> Запустить всё",  id="btn-start")
            yield Button("* Рестарт бота",   id="btn-restart")
            yield Button("x Стоп",           id="btn-stop")

        yield Footer()

    # ── Polling ──────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._poll_services()
        self._poll_stats()
        self._load_model_list()

    @work(thread=True)
    def _poll_services(self) -> None:
        import time
        while True:
            self._check_services_sync()
            time.sleep(3)

    @work(thread=True)
    def _load_model_list(self) -> None:
        """Читает список .gguf файлов с сервера и заполняет Select."""
        try:
            out = subprocess.check_output(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
                 SERVER_SSH, _LIST_MODELS_CMD],
                stderr=subprocess.DEVNULL, timeout=8, text=True,
            ).strip()
            models = [m.strip() for m in out.splitlines() if m.strip().endswith(".gguf")]
        except Exception as e:
            models = []
            err = str(e)
        else:
            err = ""

        # Текущая модель из .env
        current = ""
        env_path = BOT_DIR / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("LLM_MODEL="):
                    current = line.split("=", 1)[1].strip()

        def _apply():
            sel = self.query_one("#model-select", Select)
            log = self.query_one("#log-panel", RichLog)
            if not models:
                log.write(f"[red]Не удалось загрузить список моделей с сервера: {err or 'SSH недоступен'}[/red]")
                # Хотя бы покажем текущую из .env
                if current:
                    sel.set_options([(current, current)])
                    sel.value = current
                return
            options = [(m, m) for m in models]
            sel.set_options(options)
            if current in models:
                sel.value = current
            log.write(f"[dim]Загружено моделей: {len(models)}[/dim]")

        self.call_from_thread(_apply)

    @work(exclusive=True)
    async def _poll_stats(self) -> None:
        while True:
            self._fetch_lan_stats()
            self._fetch_vps_stats()
            await asyncio.sleep(15)

    def _check_services_sync(self) -> None:
        import socket
        import urllib.request, urllib.error

        # llama-server + туннель — один порт, проверяем HTTP /health
        llama = "stopped"
        tunnel = "stopped"
        try:
            with urllib.request.urlopen(LLAMA_HEALTH, timeout=2) as r:
                if r.status == 200:
                    llama = "running"
                    tunnel = "running"
                else:
                    llama = "starting"
                    tunnel = "running"
        except urllib.error.URLError:
            # Порт может быть открыт (туннель есть), но llama ещё не готова
            try:
                s = socket.create_connection(("127.0.0.1", 8080), timeout=1)
                s.close()
                tunnel = "running"
                llama = "starting"
            except OSError:
                pass
        except Exception:
            pass

        # Бот — сначала по нашему процессу, затем PowerShell (wmic deprecated в Win11)
        bot_running = self._bot_proc is not None and self._bot_proc.poll() is None
        if not bot_running:
            try:
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
                     "| Select-Object -ExpandProperty CommandLine"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                )
                bot_running = "main.py" in out
            except Exception:
                pass

        self.call_from_thread(
            self.query_one("#st-llama",  ServiceRow).__setattr__, "status", llama
        )
        self.call_from_thread(
            self.query_one("#st-tunnel", ServiceRow).__setattr__, "status", tunnel
        )
        self.call_from_thread(
            self.query_one("#st-bot", ServiceRow).__setattr__, "status",
            "running" if bot_running else "stopped"
        )

    # ── Статистика серверов ───────────────────────────────────────────────────

    @work(thread=True)
    def _fetch_lan_stats(self) -> None:
        widget = self.query_one("#stats-lan", ServerStats)
        try:
            out = subprocess.check_output(
                ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes",
                 SERVER_SSH, _LAN_STATS_CMD],
                stderr=subprocess.DEVNULL, timeout=8, text=True,
            ).strip()
            self.call_from_thread(widget.set_stats, _parse_stats(out))
        except Exception:
            self.call_from_thread(widget.set_offline)

    @work(thread=True)
    def _fetch_vps_stats(self) -> None:
        widget = self.query_one("#stats-vps", ServerStats)
        try:
            out = subprocess.check_output(
                ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes",
                 VPS_SSH, _VPS_STATS_CMD],
                stderr=subprocess.DEVNULL, timeout=8, text=True,
            ).strip()
            self.call_from_thread(widget.set_stats, _parse_stats(out))
        except Exception:
            self.call_from_thread(widget.set_offline)

    # ── Кнопки ───────────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "btn-start":   self.action_start_all,
            "btn-restart": self.action_restart_bot,
            "btn-stop":    self.action_stop_all,
            "btn-server":  lambda: self._open_ssh(SERVER_SSH, "Сервер LAN"),
            "btn-vps":     lambda: self._open_ssh(VPS_SSH, "VPS"),
        }
        h = handlers.get(event.button.id)
        if h:
            h()

    # ── Действия ─────────────────────────────────────────────────────────────

    def action_start_all(self) -> None:
        sel = self.query_one("#model-select", Select)
        model = str(sel.value) if sel.value and sel.value is not Select.BLANK else ""
        log = self.query_one("#log-panel", RichLog)
        if not model:
            log.write("[red]Сначала выбери модель в строке снизу[/red]")
            return
        self._do_start_all(model)

    @work(thread=True)
    def _do_start_all(self, model: str) -> None:
        def log(msg: str):
            self.call_from_thread(self.query_one("#log-panel", RichLog).write, msg)

        log(f"[bold yellow]> Запускаем с моделью: {model}[/bold yellow]")

        # Обновляем .env
        _update_env_model(model)
        log("  [dim]-> .env обновлён[/dim]")

        # Останавливаем текущий бот
        if self._bot_proc and self._bot_proc.poll() is None:
            self._bot_proc.terminate()
            self._bot_proc.wait()
            log("  [dim]-> старый бот остановлен[/dim]")

        # Поднимаем SSH туннель если нет
        if self._tunnel_proc is None or self._tunnel_proc.poll() is not None:
            self._tunnel_proc = subprocess.Popen(
                ["ssh", "-L", "8080:127.0.0.1:8080", "-N",
                 "-o", "ServerAliveInterval=30", SERVER_SSH],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            log("  [dim]-> SSH туннель поднят[/dim]")

        # Запускаем llama-server на сервере через start_alta.sh
        self.call_from_thread(
            self.query_one("#st-llama", ServiceRow).__setattr__, "status", "starting"
        )
        try:
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
                 SERVER_SSH, f"bash ~/start_alta.sh '{model}'"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                log(f"[red]SSH ошибка (start_alta.sh):\n{result.stderr.strip() or result.stdout.strip()}[/red]")
                return
            log("  [dim]-> start_alta.sh выполнен[/dim]")
        except subprocess.TimeoutExpired:
            log("[red]Таймаут SSH при запуске start_alta.sh[/red]")
            return
        except Exception as e:
            log(f"[red]SSH ошибка: {e}[/red]")
            return

        # Ждём пока llama-server поднимется (до 90 секунд)
        log("  [dim]-> ждём llama-server...[/dim]")
        import time
        import urllib.request, urllib.error
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(LLAMA_HEALTH, timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(2)

        if not ready:
            log("[red]llama-server не поднялся за 90 секунд[/red]")
            return

        log("  [green]-> llama-server готов[/green]")

        # Запускаем бота
        self.call_from_thread(self._launch_bot_from_thread)

    def action_restart_bot(self) -> None:
        log = self.query_one("#log-panel", RichLog)
        log.write("[bold yellow]* Рестарт бота...[/bold yellow]")
        if self._bot_proc and self._bot_proc.poll() is None:
            self._bot_proc.terminate()
            self._bot_proc.wait()
        self._launch_bot(log)

    def action_stop_all(self) -> None:
        self._kill_procs()
        try:
            log = self.query_one("#log-panel", RichLog)
            log.write("[red]x Остановлено[/red]")
        except Exception:
            pass

    def _kill_procs(self) -> None:
        if self._bot_proc and self._bot_proc.poll() is None:
            self._bot_proc.terminate()
        if self._tunnel_proc and self._tunnel_proc.poll() is None:
            self._tunnel_proc.terminate()

    def _launch_bot_from_thread(self) -> None:
        """Вызывается из воркера через call_from_thread."""
        log = self.query_one("#log-panel", RichLog)
        self._launch_bot(log)

    def _launch_bot(self, log: RichLog) -> None:
        if self._bot_proc and self._bot_proc.poll() is None:
            self._bot_proc.terminate()
            self._bot_proc.wait()
        self._bot_proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(BOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            encoding="utf-8", errors="replace",
        )
        log.write("[green][+] Бот запущен[/green]")
        self._stream_logs()

    @work(thread=True)
    def _stream_logs(self) -> None:
        log = self.query_one("#log-panel", RichLog)
        if not self._bot_proc or not self._bot_proc.stdout:
            return
        for raw in self._bot_proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            if "ERROR" in line or "error" in line.lower():
                markup = f"[red]{line}[/red]"
            elif "WARNING" in line or "WARN" in line:
                markup = f"[yellow]{line}[/yellow]"
            elif "[INFO]" in line:
                markup = f"[dim]{line}[/dim]"
            else:
                markup = line
            self.call_from_thread(log.write, markup)
        self.call_from_thread(log.write, "[bold red][-] Бот завершил работу[/bold red]")

    # ── SSH-терминал ─────────────────────────────────────────────────────────

    def _open_ssh(self, host: str, label: str) -> None:
        log = self.query_one("#log-panel", RichLog)
        log.write(f"[dim]-> SSH: {host}[/dim]")
        try:
            subprocess.Popen(
                ["wt", "-w", "0", "new-tab", "--title", label, "--", "ssh", host]
            )
        except FileNotFoundError:
            subprocess.Popen(["start", "cmd", "/k", f"ssh {host}"], shell=True)

    # ── Выход ─────────────────────────────────────────────────────────────────

    def on_unmount(self) -> None:
        self._kill_procs()


if __name__ == "__main__":
    AltaPanel().run()
