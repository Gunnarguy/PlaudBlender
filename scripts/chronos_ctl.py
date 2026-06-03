#!/usr/bin/env python3
"""Chronos stack controller.

Manage local Chronos services from one command:
- start / stop / restart
- status (quick view)
- analyze (deeper diagnostics)

Services managed:
- Qdrant (docker compose service)
- UI (scripts/launch_app.py)
- Auto-sync/Webhook (scripts/auto_sync.py)
- ngrok tunnel (optional, for public webhook URL)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = BASE_DIR / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = BASE_DIR / "venv" / "bin" / "python"
RUN_DIR = BASE_DIR / ".run"
LOG_DIR = BASE_DIR / "logs"
ENV_FILE = BASE_DIR / ".env"

UI_PORT = 8050
WEBHOOK_PORT = 8090
NGROK_API_PORT = 4040
QDRANT_PORT = 6333


@dataclass
class ServiceDef:
    name: str
    command: list[str]
    pid_file: Path
    log_file: Path
    port: Optional[int] = None


UI_SERVICE = ServiceDef(
    name="ui",
    command=[str(PYTHON_BIN), "scripts/launch_app.py"],
    pid_file=RUN_DIR / "ui.pid",
    log_file=LOG_DIR / "ui.log",
    port=UI_PORT,
)

SYNC_SERVICE = ServiceDef(
    name="auto_sync",
    command=[str(PYTHON_BIN), "scripts/auto_sync.py"],
    pid_file=RUN_DIR / "auto_sync.pid",
    log_file=LOG_DIR / "auto_sync.log",
    port=WEBHOOK_PORT,
)

NGROK_SERVICE = ServiceDef(
    name="ngrok",
    command=["ngrok", "http", str(WEBHOOK_PORT)],
    pid_file=RUN_DIR / "ngrok.pid",
    log_file=LOG_DIR / "ngrok.log",
    port=NGROK_API_PORT,
)


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(BASE_DIR),
        text=True,
        capture_output=True,
        check=check,
    )


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def write_pid(pid_file: Path, pid: int) -> None:
    pid_file.write_text(str(pid))


def remove_pid(pid_file: Path) -> None:
    if pid_file.exists():
        pid_file.unlink()


def service_running(service: ServiceDef) -> bool:
    pid = read_pid(service.pid_file)
    if pid and pid_running(pid):
        return True
    if service.port is not None and is_port_open(service.port):
        return True
    return False


def pids_on_port(port: int) -> list[int]:
    try:
        if sys.platform == "darwin":
            result = run_command(["lsof", "-ti", f"tcp:{port}"])
        else:
            result = run_command(["ss", "-tlnp", f"sport = :{port}"])
        if result.returncode != 0:
            return []
        pids: list[int] = []
        if sys.platform == "darwin":
            for raw in result.stdout.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    pids.append(int(raw))
                except ValueError:
                    continue
        else:
            import re
            for m in re.finditer(r'pid=(\d+)', result.stdout):
                pids.append(int(m.group(1)))
        return sorted(set(pids))
    except Exception:
        return []


def start_service(service: ServiceDef) -> None:
    if service_running(service):
        print(f"[ok] {service.name} already running")
        return

    ensure_dirs()

    if service.command[0].endswith("python") and not Path(service.command[0]).exists():
        print(f"[err] python env missing: {service.command[0]}")
        return

    with open(service.log_file, "a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            service.command,
            cwd=str(BASE_DIR),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    write_pid(service.pid_file, process.pid)
    print(f"[ok] started {service.name} (pid {process.pid})")


def stop_service(service: ServiceDef) -> None:
    pid = read_pid(service.pid_file)
    if not pid:
        if service.port is not None:
            port_pids = pids_on_port(service.port)
            if port_pids:
                for port_pid in port_pids:
                    try:
                        os.kill(port_pid, signal.SIGTERM)
                    except OSError:
                        continue
                time.sleep(1)
                for port_pid in pids_on_port(service.port):
                    try:
                        os.kill(port_pid, signal.SIGKILL)
                    except OSError:
                        continue
                print(f"[ok] stopped {service.name} by port :{service.port}")
                remove_pid(service.pid_file)
                return

        print(f"[ok] {service.name} not running")
        remove_pid(service.pid_file)
        return

    if not pid_running(pid):
        print(f"[ok] {service.name} stale pid cleaned")
        remove_pid(service.pid_file)
        return

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not pid_running(pid):
            remove_pid(service.pid_file)
            print(f"[ok] stopped {service.name}")
            return
        time.sleep(0.2)

    os.kill(pid, signal.SIGKILL)
    remove_pid(service.pid_file)
    print(f"[ok] force-stopped {service.name}")


def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def http_status(url: str) -> Optional[int]:
    try:
        with urlopen(url, timeout=3) as response:
            return int(getattr(response, "status", 200))
    except Exception:
        return None


def load_env_map() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_FILE.exists():
        return values

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def docker_qdrant_running() -> bool:
    try:
        result = run_command(["docker", "compose", "ps", "qdrant"])
        text = f"{result.stdout}\n{result.stderr}".lower()
        return "qdrant" in text and "up" in text
    except Exception:
        return False


def start_qdrant() -> None:
    try:
        result = run_command(["docker", "compose", "up", "-d", "qdrant"])
        if result.returncode == 0:
            print("[ok] qdrant started")
        else:
            print("[err] qdrant start failed")
            if result.stderr.strip():
                print(result.stderr.strip())
    except Exception as exc:
        print(f"[err] qdrant start failed: {exc}")


def stop_qdrant() -> None:
    try:
        result = run_command(["docker", "compose", "stop", "qdrant"])
        if result.returncode == 0:
            print("[ok] qdrant stopped")
        else:
            print("[err] qdrant stop failed")
            if result.stderr.strip():
                print(result.stderr.strip())
    except Exception as exc:
        print(f"[err] qdrant stop failed: {exc}")


def get_ngrok_public_url() -> Optional[str]:
    try:
        with urlopen(
            f"http://127.0.0.1:{NGROK_API_PORT}/api/tunnels", timeout=2
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        for tunnel in payload.get("tunnels", []):
            public_url = tunnel.get("public_url", "")
            if public_url.startswith("https://"):
                return public_url
        return None
    except Exception:
        return None


def cmd_start(args: argparse.Namespace) -> None:
    start_qdrant()
    start_service(SYNC_SERVICE)
    start_service(UI_SERVICE)
    if args.public:
        start_service(NGROK_SERVICE)

    print("\n[info] stack started")
    cmd_status(args)


def cmd_stop(args: argparse.Namespace) -> None:
    stop_service(UI_SERVICE)
    stop_service(SYNC_SERVICE)
    stop_service(NGROK_SERVICE)
    if not args.keep_qdrant:
        stop_qdrant()


def cmd_restart(args: argparse.Namespace) -> None:
    stop_args = argparse.Namespace(keep_qdrant=args.keep_qdrant)
    cmd_stop(stop_args)
    time.sleep(1)
    start_args = argparse.Namespace(public=args.public)
    cmd_start(start_args)


def cmd_status(_: argparse.Namespace) -> None:
    ui_ok = service_running(UI_SERVICE)
    sync_ok = service_running(SYNC_SERVICE)
    ngrok_ok = service_running(NGROK_SERVICE)
    qdrant_ok = docker_qdrant_running() and is_port_open(QDRANT_PORT)

    print("\nChronos stack status")
    print("-" * 42)
    print(f"qdrant      : {'UP' if qdrant_ok else 'DOWN'}")
    print(f"auto_sync   : {'UP' if sync_ok else 'DOWN'}")
    print(f"ui          : {'UP' if ui_ok else 'DOWN'}")
    print(f"ngrok       : {'UP' if ngrok_ok else 'DOWN'}")

    ui_http = http_status(f"http://127.0.0.1:{UI_PORT}/")
    webhook_http = http_status(f"http://127.0.0.1:{WEBHOOK_PORT}/health")
    print(f"ui http     : {ui_http if ui_http is not None else 'n/a'}")
    print(f"webhook http: {webhook_http if webhook_http is not None else 'n/a'}")

    ngrok_url = get_ngrok_public_url()
    if ngrok_url:
        print(f"public url  : {ngrok_url}")


def cmd_analyze(_: argparse.Namespace) -> None:
    env_values = load_env_map()
    webhook_url = env_values.get("PLAUD_WEBHOOK_URL", "")
    webhook_secret = env_values.get("PLAUD_WEBHOOK_SECRET", "")
    admin_token = env_values.get("WEBHOOK_ADMIN_TOKEN", "")

    print("\nChronos stack analysis")
    print("-" * 42)

    print(
        f"python env      : {'ok' if PYTHON_BIN.exists() else 'missing'} ({PYTHON_BIN})"
    )
    print(f".env            : {'present' if ENV_FILE.exists() else 'missing'}")
    print(f"qdrant running  : {'yes' if docker_qdrant_running() else 'no'}")
    print(f"ui port open    : {'yes' if is_port_open(UI_PORT) else 'no'}")
    print(f"webhook port    : {'yes' if is_port_open(WEBHOOK_PORT) else 'no'}")
    print(f"ngrok api port  : {'yes' if is_port_open(NGROK_API_PORT) else 'no'}")

    print(f"webhook secret  : {'set' if bool(webhook_secret) else 'missing'}")
    print(f"admin token     : {'set' if bool(admin_token) else 'missing'}")
    print(f"webhook url     : {webhook_url or 'missing'}")

    if webhook_url:
        parsed = urlparse(webhook_url)
        is_https = parsed.scheme == "https"
        is_localhost = parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}
        print(f"webhook https   : {'yes' if is_https else 'no'}")
        print(f"webhook public  : {'yes' if (is_https and not is_localhost) else 'no'}")

    health_code = http_status(f"http://127.0.0.1:{WEBHOOK_PORT}/health")
    if health_code is not None:
        print(f"webhook health  : {health_code}")

    ngrok_url = get_ngrok_public_url()
    if ngrok_url:
        print(f"ngrok public    : {ngrok_url}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronos local stack controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument(
        "--public",
        action="store_true",
        help="Also start ngrok tunnel for public webhook URL",
    )
    start_parser.set_defaults(func=cmd_start)

    stop_parser = subparsers.add_parser("stop", help="Stop services")
    stop_parser.add_argument(
        "--keep-qdrant",
        action="store_true",
        help="Keep qdrant container running",
    )
    stop_parser.set_defaults(func=cmd_stop)

    restart_parser = subparsers.add_parser("restart", help="Restart services")
    restart_parser.add_argument(
        "--public",
        action="store_true",
        help="Also start ngrok tunnel for public webhook URL",
    )
    restart_parser.add_argument(
        "--keep-qdrant",
        action="store_true",
        help="Keep qdrant container running during restart",
    )
    restart_parser.set_defaults(func=cmd_restart)

    status_parser = subparsers.add_parser("status", help="Quick service status")
    status_parser.set_defaults(func=cmd_status)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Detailed diagnostics and configuration checks"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
