#!/usr/bin/env python3
"""
Chronos Menu Bar Controller
─────────────────────────────
A native macOS menu bar app for managing all PlaudBlender services:
  API server, ngrok tunnel, Qdrant, Web UI, and pipeline.

Double-click or run:  venv/bin/python scripts/chronos_menubar.py
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import rumps

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(ROOT, "venv", "bin", "python")
LOG_DIR = os.path.join(ROOT, ".logs")
PID_DIR = LOG_DIR
ENV_FILE = os.path.join(ROOT, ".env")
QDRANT_BIN = os.path.expanduser("~/bin/qdrant")
QDRANT_CFG = os.path.expanduser("~/.config/qdrant/config.yaml")

os.makedirs(LOG_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_pid(name: str) -> int | None:
    try:
        with open(os.path.join(PID_DIR, f"{name}.pid")) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_pid(name: str, pid: int):
    with open(os.path.join(PID_DIR, f"{name}.pid"), "w") as f:
        f.write(str(pid))


def _rm_pid(name: str):
    try:
        os.remove(os.path.join(PID_DIR, f"{name}.pid"))
    except FileNotFoundError:
        pass


def _is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _ngrok_url() -> str | None:
    try:
        data = json.loads(
            urllib.request.urlopen(
                "http://127.0.0.1:4040/api/tunnels", timeout=2
            ).read()
        )
        for t in data.get("tunnels", []):
            if t.get("proto") == "https":
                return t["public_url"]
        tunnels = data.get("tunnels", [])
        if tunnels:
            return tunnels[0]["public_url"]
    except Exception:
        pass
    return None


def _update_env_redirect(ngrok_url: str):
    callback = f"{ngrok_url}/api/v1/auth/notion/callback"
    lines = []
    found = False
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("NOTION_REDIRECT_URI="):
                    lines.append(f"NOTION_REDIRECT_URI={callback}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"NOTION_REDIRECT_URI={callback}\n")
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)


def _api_health() -> dict | None:
    try:
        data = urllib.request.urlopen(
            "http://127.0.0.1:8000/api/v1/health", timeout=2
        ).read()
        return json.loads(data)
    except Exception:
        return None


def _open_url(url: str):
    subprocess.Popen(
        ["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


# ── Service Management ───────────────────────────────────────────────────────


def start_qdrant() -> bool:
    if _port_in_use(6333):
        return True
    if not os.path.exists(QDRANT_BIN):
        return False
    log = os.path.join(LOG_DIR, "qdrant.log")
    args = [QDRANT_BIN, "--disable-telemetry"]
    if os.path.exists(QDRANT_CFG):
        args += ["--config-path", QDRANT_CFG]
    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=open(log, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid("qdrant", proc.pid)
    for _ in range(30):
        if _port_in_use(6333):
            return True
        time.sleep(0.5)
    return False


def start_api() -> bool:
    if _port_in_use(8000):
        return True
    log = os.path.join(LOG_DIR, "api.log")
    proc = subprocess.Popen(
        [VENV_PYTHON, os.path.join(ROOT, "scripts", "launch_api.py"), "--port", "8000"],
        stdin=subprocess.DEVNULL,
        stdout=open(log, "a"),
        stderr=subprocess.STDOUT,
        cwd=ROOT,
        start_new_session=True,
    )
    _write_pid("api", proc.pid)
    for _ in range(20):
        if _port_in_use(8000):
            return True
        time.sleep(0.5)
    return False


def start_ngrok() -> str | None:
    existing = _ngrok_url()
    if existing:
        return existing
    log = os.path.join(LOG_DIR, "ngrok.log")
    proc = subprocess.Popen(
        [
            "ngrok",
            "http",
            "8000",
            "--domain=glairy-ona-irreplaceable.ngrok-free.dev",
            "--log",
            "stdout",
            "--log-level",
            "info",
        ],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_pid("ngrok", proc.pid)
    for _ in range(20):
        url = _ngrok_url()
        if url:
            _update_env_redirect(url)
            return url
        time.sleep(0.5)
    return None


def start_webui() -> bool:
    if _port_in_use(8050):
        return True
    log = os.path.join(LOG_DIR, "webui.log")
    proc = subprocess.Popen(
        [VENV_PYTHON, os.path.join(ROOT, "scripts", "launch_app.py")],
        stdin=subprocess.DEVNULL,
        stdout=open(log, "a"),
        stderr=subprocess.STDOUT,
        cwd=ROOT,
        start_new_session=True,
    )
    _write_pid("webui", proc.pid)
    for _ in range(40):
        if _port_in_use(8050):
            return True
        time.sleep(0.5)
    return False


def _stop_service(name: str, port: int | None = None):
    pid = _read_pid(name)
    if _is_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    _rm_pid(name)
    # Also kill anything on the port
    if port:
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"], text=True
            ).strip()
            for p in out.split("\n"):
                if p.strip():
                    os.kill(int(p.strip()), signal.SIGTERM)
        except (subprocess.CalledProcessError, ValueError, OSError):
            pass


def stop_api():
    _stop_service("api", 8000)


def stop_ngrok():
    _stop_service("ngrok", 4040)


def stop_qdrant():
    _stop_service("qdrant", 6333)


def stop_webui():
    _stop_service("webui", 8050)


# ── Menu Bar App ─────────────────────────────────────────────────────────────


class ChronosApp(rumps.App):
    def __init__(self):
        super().__init__(
            "Chronos",
            title="⏳",
            quit_button=None,
        )

        self._pipeline_running = False

        # ── Status header ────────────────────────────────────────────
        self.status_item = rumps.MenuItem("Checking status…", callback=None)
        self.status_item.set_callback(None)
        self.ngrok_url_item = rumps.MenuItem("", callback=None)
        self.ngrok_url_item.set_callback(None)
        self.db_stats_item = rumps.MenuItem("", callback=None)
        self.db_stats_item.set_callback(None)

        # ── Service toggles ──────────────────────────────────────────
        self.api_toggle = rumps.MenuItem("API Server", callback=self.toggle_api)
        self.ngrok_toggle = rumps.MenuItem("ngrok Tunnel", callback=self.toggle_ngrok)
        self.qdrant_toggle = rumps.MenuItem("Qdrant", callback=self.toggle_qdrant)
        self.webui_toggle = rumps.MenuItem("Web UI", callback=self.toggle_webui)

        # ── Bulk actions ─────────────────────────────────────────────
        self.start_all_item = rumps.MenuItem("▶  Start All", callback=self.start_all)
        self.stop_all_item = rumps.MenuItem("■  Stop All", callback=self.stop_all)

        # ── Open links ───────────────────────────────────────────────
        self.open_webui = rumps.MenuItem("🌐  Open Web UI", callback=self.do_open_webui)
        self.open_api_docs = rumps.MenuItem(
            "📘  Open API Docs", callback=self.do_open_api_docs
        )
        self.open_qdrant_dash = rumps.MenuItem(
            "🔷  Open Qdrant Dashboard", callback=self.do_open_qdrant
        )
        self.open_ngrok_dash = rumps.MenuItem(
            "🔗  ngrok Inspector", callback=self.do_open_ngrok
        )
        self.copy_ngrok_url = rumps.MenuItem(
            "📋  Copy Notion Redirect URI", callback=self.do_copy_ngrok
        )

        # ── Pipeline ─────────────────────────────────────────────────
        self.run_pipeline = rumps.MenuItem(
            "🚀  Run Full Pipeline", callback=self.do_run_pipeline
        )
        self.run_sync = rumps.MenuItem("🔄  Sync from Plaud", callback=self.do_run_sync)

        # ── Logs ─────────────────────────────────────────────────────
        self.view_api_log = rumps.MenuItem("API Log", callback=self.do_view_api_log)
        self.view_ngrok_log = rumps.MenuItem(
            "ngrok Log", callback=self.do_view_ngrok_log
        )
        self.view_qdrant_log = rumps.MenuItem(
            "Qdrant Log", callback=self.do_view_qdrant_log
        )
        self.view_webui_log = rumps.MenuItem(
            "Web UI Log", callback=self.do_view_webui_log
        )
        self.view_pipeline_log = rumps.MenuItem(
            "Pipeline Log", callback=self.do_view_pipeline_log
        )
        self.logs_submenu = rumps.MenuItem("📄  View Logs")
        self.logs_submenu.update(
            [
                self.view_api_log,
                self.view_ngrok_log,
                self.view_qdrant_log,
                self.view_webui_log,
                self.view_pipeline_log,
            ]
        )

        # ── Quit ─────────────────────────────────────────────────────
        self.quit_and_stop = rumps.MenuItem(
            "Quit & Stop All Services", callback=self.do_quit_and_stop
        )
        self.quit_keep = rumps.MenuItem(
            "Quit (keep services running)", callback=self.do_quit_keep
        )

        self.menu = [
            self.status_item,
            self.ngrok_url_item,
            self.db_stats_item,
            rumps.separator,
            self.api_toggle,
            self.ngrok_toggle,
            self.qdrant_toggle,
            self.webui_toggle,
            rumps.separator,
            self.start_all_item,
            self.stop_all_item,
            rumps.separator,
            self.open_webui,
            self.open_api_docs,
            self.open_qdrant_dash,
            self.open_ngrok_dash,
            self.copy_ngrok_url,
            rumps.separator,
            self.run_pipeline,
            self.run_sync,
            rumps.separator,
            self.logs_submenu,
            rumps.separator,
            self.quit_and_stop,
            self.quit_keep,
        ]

        # Start status poller
        self._poller = rumps.Timer(self._poll_status, 5)
        self._poller.start()
        # Auto-start all services on launch
        threading.Thread(target=self._auto_start_on_launch, daemon=True).start()

    # ── Auto-Start ───────────────────────────────────────────────────────

    def _auto_start_on_launch(self):
        """Auto-start all 4 services when the menu bar app launches."""
        time.sleep(1)  # Let the menu bar render first
        rumps.notification(
            "Chronos", "Starting…", "Launching all services", sound=False
        )
        self._start_all_bg()

    # ── Status Polling ───────────────────────────────────────────────────

    def _poll_status(self, _):
        api_up = _port_in_use(8000)
        ngrok_up = False
        ngrok_public_url = None
        qdrant_up = _port_in_use(6333)
        webui_up = _port_in_use(8050)

        # Check ngrok + get URL
        try:
            ngrok_public_url = _ngrok_url()
            ngrok_up = ngrok_public_url is not None
        except Exception:
            pass

        # Use real health check for API instead of just port
        api_version = None
        if api_up:
            health = _api_health()
            if health:
                api_version = health.get("version")
            else:
                api_up = False  # Port open but not responding properly

        services = [api_up, ngrok_up, qdrant_up, webui_up]
        running = sum(services)

        # Update icon
        if running == 4:
            self.title = "🟢"
        elif running > 0:
            self.title = "🟡"
        else:
            self.title = "⏳"

        # Status line
        parts = []
        if api_up:
            parts.append(f"API ✓{f' v{api_version}' if api_version else ''}")
        if ngrok_up:
            parts.append("ngrok ✓")
        if qdrant_up:
            parts.append("Qdrant ✓")
        if webui_up:
            parts.append("Web ✓")

        if parts:
            self.status_item.title = f"{running}/4 services: {', '.join(parts)}"
        else:
            self.status_item.title = "No services running"

        # ngrok URL line
        if ngrok_public_url:
            # Truncate for menu readability
            short = ngrok_public_url.replace("https://", "")
            self.ngrok_url_item.title = f"🌍  {short}"
        else:
            self.ngrok_url_item.title = ""

        # DB stats (only when Qdrant is up)
        if qdrant_up:
            try:
                data = json.loads(
                    urllib.request.urlopen(
                        "http://127.0.0.1:6333/collections", timeout=2
                    ).read()
                )
                colls = data.get("result", {}).get("collections", [])
                total_points = 0
                for c in colls:
                    try:
                        cd = json.loads(
                            urllib.request.urlopen(
                                f"http://127.0.0.1:6333/collections/{c['name']}",
                                timeout=2,
                            ).read()
                        )
                        total_points += cd.get("result", {}).get("points_count", 0)
                    except Exception:
                        pass
                self.db_stats_item.title = (
                    f"💾  {len(colls)} collections · {total_points:,} vectors"
                )
            except Exception:
                self.db_stats_item.title = ""
        else:
            self.db_stats_item.title = ""

        # Toggle labels
        self.api_toggle.title = f"{'✅' if api_up else '⬜'}  API Server (8000)"
        self.ngrok_toggle.title = f"{'✅' if ngrok_up else '⬜'}  ngrok Tunnel"
        self.qdrant_toggle.title = f"{'✅' if qdrant_up else '⬜'}  Qdrant (6333)"
        self.webui_toggle.title = f"{'✅' if webui_up else '⬜'}  Web UI (8050)"

        # Pipeline state indicator
        if self._pipeline_running:
            self.run_pipeline.title = "🚀  Pipeline Running…"
            self.run_sync.title = "🔄  Pipeline Running…"
        else:
            self.run_pipeline.title = "🚀  Run Full Pipeline"
            self.run_sync.title = "🔄  Sync from Plaud"

    # ── Service Toggles ──────────────────────────────────────────────────

    @rumps.clicked("API Server")
    def toggle_api(self, sender):
        if _port_in_use(8000):
            stop_api()
            rumps.notification("Chronos", "API Server", "Stopped", sound=False)
        else:
            threading.Thread(target=self._start_api_bg, daemon=True).start()

    def _start_api_bg(self):
        if start_api():
            rumps.notification(
                "Chronos", "API Server", "Running on port 8000", sound=False
            )
        else:
            rumps.notification(
                "Chronos", "API Server", "Failed to start — check logs", sound=True
            )
        self._poll_status(None)

    @rumps.clicked("ngrok Tunnel")
    def toggle_ngrok(self, sender):
        if _ngrok_url():
            stop_ngrok()
            rumps.notification("Chronos", "ngrok", "Stopped", sound=False)
        else:
            threading.Thread(target=self._start_ngrok_bg, daemon=True).start()

    def _start_ngrok_bg(self):
        url = start_ngrok()
        if url:
            rumps.notification("Chronos", "ngrok Tunnel", f"URL: {url}", sound=False)
        else:
            rumps.notification(
                "Chronos", "ngrok", "Failed to start — check logs", sound=True
            )
        self._poll_status(None)

    @rumps.clicked("Qdrant")
    def toggle_qdrant(self, sender):
        if _port_in_use(6333):
            stop_qdrant()
            rumps.notification("Chronos", "Qdrant", "Stopped", sound=False)
        else:
            threading.Thread(target=self._start_qdrant_bg, daemon=True).start()

    def _start_qdrant_bg(self):
        if start_qdrant():
            rumps.notification("Chronos", "Qdrant", "Running on port 6333", sound=False)
        else:
            rumps.notification("Chronos", "Qdrant", "Failed to start", sound=True)
        self._poll_status(None)

    @rumps.clicked("Web UI")
    def toggle_webui(self, sender):
        if _port_in_use(8050):
            stop_webui()
            rumps.notification("Chronos", "Web UI", "Stopped", sound=False)
        else:
            threading.Thread(target=self._start_webui_bg, daemon=True).start()

    def _start_webui_bg(self):
        if start_webui():
            rumps.notification("Chronos", "Web UI", "Running on port 8050", sound=False)
        else:
            rumps.notification(
                "Chronos", "Web UI", "Failed to start — check logs", sound=True
            )
        self._poll_status(None)

    # ── Bulk Actions ─────────────────────────────────────────────────────

    @rumps.clicked("▶  Start All")
    def start_all(self, _):
        threading.Thread(target=self._start_all_bg, daemon=True).start()

    def _start_all_bg(self):
        results = []
        if not _port_in_use(6333):
            results.append(("Qdrant", start_qdrant()))
        else:
            results.append(("Qdrant", True))

        # ngrok before API so .env gets updated first
        if not _ngrok_url():
            url = start_ngrok()
            results.append(("ngrok", url is not None))
        else:
            results.append(("ngrok", True))

        if not _port_in_use(8000):
            results.append(("API", start_api()))
        else:
            results.append(("API", True))

        if not _port_in_use(8050):
            results.append(("Web UI", start_webui()))
        else:
            results.append(("Web UI", True))

        ok = [r[0] for r in results if r[1]]
        fail = [r[0] for r in results if not r[1]]

        msg = f"Running: {', '.join(ok)}" if ok else "None started"
        if fail:
            msg += f"\nFailed: {', '.join(fail)}"

        rumps.notification("Chronos", "Start All", msg, sound=bool(fail))
        self._poll_status(None)

    @rumps.clicked("■  Stop All")
    def stop_all(self, _):
        stop_webui()
        stop_api()
        stop_ngrok()
        stop_qdrant()
        rumps.notification("Chronos", "Stop All", "All services stopped", sound=False)
        time.sleep(1)
        self._poll_status(None)

    # ── Quick Links ──────────────────────────────────────────────────────

    @rumps.clicked("🌐  Open Web UI")
    def do_open_webui(self, _):
        if _port_in_use(8050):
            _open_url("http://localhost:8050")
        else:
            rumps.notification(
                "Chronos", "Web UI", "Not running — start it first", sound=True
            )

    @rumps.clicked("�  Open API Docs")
    def do_open_api_docs(self, _):
        if _port_in_use(8000):
            _open_url("http://localhost:8000/docs")
        else:
            rumps.notification(
                "Chronos", "API", "Not running — start it first", sound=True
            )

    @rumps.clicked("🔷  Open Qdrant Dashboard")
    def do_open_qdrant(self, _):
        if _port_in_use(6333):
            _open_url("http://localhost:6333/dashboard")
        else:
            rumps.notification(
                "Chronos", "Qdrant", "Not running — start it first", sound=True
            )

    @rumps.clicked("🔗  ngrok Inspector")
    def do_open_ngrok(self, _):
        if _port_in_use(4040):
            _open_url("http://127.0.0.1:4040")
        else:
            rumps.notification("Chronos", "ngrok", "Not running", sound=True)

    @rumps.clicked("📋  Copy Notion Redirect URI")
    def do_copy_ngrok(self, _):
        url = _ngrok_url()
        if url:
            callback = f"{url}/api/v1/auth/notion/callback"
            subprocess.run(["pbcopy"], input=callback.encode(), check=False)
            rumps.notification("Chronos", "Copied to clipboard", callback, sound=False)
        else:
            rumps.notification(
                "Chronos", "ngrok", "Not running — no URL to copy", sound=True
            )

    # ── Pipeline ─────────────────────────────────────────────────────────

    @rumps.clicked("🚀  Run Full Pipeline")
    def do_run_pipeline(self, _):
        if self._pipeline_running:
            rumps.notification(
                "Chronos", "Pipeline", "Already running — please wait", sound=True
            )
            return
        threading.Thread(
            target=self._run_pipeline_bg, args=("--full",), daemon=True
        ).start()

    @rumps.clicked("🔄  Sync from Plaud")
    def do_run_sync(self, _):
        if self._pipeline_running:
            rumps.notification(
                "Chronos", "Pipeline", "Already running — please wait", sound=True
            )
            return
        threading.Thread(
            target=self._run_pipeline_bg, args=("--ingest",), daemon=True
        ).start()

    def _run_pipeline_bg(self, flag: str):
        self._pipeline_running = True
        self._poll_status(None)
        rumps.notification("Chronos", "Pipeline", f"Running {flag}…", sound=False)
        log = os.path.join(LOG_DIR, "pipeline.log")
        result = subprocess.run(
            [VENV_PYTHON, os.path.join(ROOT, "scripts", "chronos_pipeline.py"), flag],
            stdin=subprocess.DEVNULL,
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        self._pipeline_running = False
        if result.returncode == 0:
            rumps.notification(
                "Chronos", "Pipeline", f"{flag} completed successfully ✓", sound=False
            )
        else:
            rumps.notification(
                "Chronos",
                "Pipeline",
                f"{flag} failed (exit {result.returncode}) — check logs",
                sound=True,
            )
        self._poll_status(None)

    # ── Logs ─────────────────────────────────────────────────────────────

    def _open_log(self, name: str):
        log = os.path.join(LOG_DIR, f"{name}.log")
        if os.path.exists(log):
            subprocess.Popen(["open", "-a", "Console", log])
        else:
            rumps.notification(
                "Chronos", "Logs", f"No {name} log file yet", sound=False
            )

    @rumps.clicked("API Log")
    def do_view_api_log(self, _):
        self._open_log("api")

    @rumps.clicked("ngrok Log")
    def do_view_ngrok_log(self, _):
        self._open_log("ngrok")

    @rumps.clicked("Qdrant Log")
    def do_view_qdrant_log(self, _):
        self._open_log("qdrant")

    @rumps.clicked("Web UI Log")
    def do_view_webui_log(self, _):
        self._open_log("webui")

    @rumps.clicked("Pipeline Log")
    def do_view_pipeline_log(self, _):
        self._open_log("pipeline")

    # ── Quit ─────────────────────────────────────────────────────────────

    @rumps.clicked("Quit & Stop All Services")
    def do_quit_and_stop(self, _):
        stop_webui()
        stop_api()
        stop_ngrok()
        stop_qdrant()
        rumps.quit_application()

    @rumps.clicked("Quit (keep services running)")
    def do_quit_keep(self, _):
        rumps.quit_application()


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ChronosApp().run()
