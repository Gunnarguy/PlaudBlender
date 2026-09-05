"""Live sync: hold Plaud's notify WebSocket and run the sync the moment a
recording changes, instead of polling every 20 minutes.

Plaud's web app opens wss://<api>/ws/notify with the bearer token as the
WebSocket *subprotocol* (not a header) and exchanges PING/PONG. Every file
lifecycle step -- upload, transcript ready, summary ready -- arrives as a
`file_notify` event. Several land per recording, so the sync is started
once, 90 s after the last event of a burst. The 20-minute timer remains
as a safety net; this only makes the common case immediate.

    venv/bin/python scripts/plaud_v4_notify.py

Runs as a long-lived systemd service (deploy/systemd/chronos-plaud-notify.service).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets  # noqa: E402

from src.plaud_v4 import PlaudV4Client  # noqa: E402

log = logging.getLogger("plaud_notify")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

WS_URL = os.getenv("PLAUD_V4_NOTIFY_URL", "wss://api-test.plaud.ai/ws/notify")
SYNC_UNIT = os.getenv("PLAUD_V4_SYNC_UNIT", "chronos-plaud-v4-sync.service")
DEBOUNCE_S = float(os.getenv("PLAUD_V4_NOTIFY_DEBOUNCE", "90"))
PING_EVERY_S = 25.0
SILENCE_LIMIT_S = 75.0
HEARTBEAT = Path(__file__).resolve().parent.parent / "data" / "notify_heartbeat.json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


def _heartbeat(**fields) -> None:
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        current = json.loads(HEARTBEAT.read_text()) if HEARTBEAT.exists() else {}
        current.update(fields, updated_at=time.time())
        HEARTBEAT.write_text(json.dumps(current, indent=1))
    except OSError:
        pass


def _start_sync(reason: str) -> None:
    log.info("starting %s (%s)", SYNC_UNIT, reason)
    try:
        subprocess.run(["sudo", "-n", "systemctl", "start", SYNC_UNIT], check=False, timeout=30)
        _heartbeat(last_sync_trigger_at=time.time(), last_sync_reason=reason)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not start %s: %s", SYNC_UNIT, exc)


class Debouncer:
    """One sync per burst of events: fires DEBOUNCE_S after the last event."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.pending: set[str] = set()

    def touch(self, file_id: str) -> None:
        self.pending.add(file_id)
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._fire())

    async def _fire(self) -> None:
        try:
            await asyncio.sleep(DEBOUNCE_S)
        except asyncio.CancelledError:
            return
        ids = sorted(self.pending)
        self.pending.clear()
        _start_sync(f"{len(ids)} file event(s): {', '.join(i[:14] for i in ids[:4])}{'…' if len(ids) > 4 else ''}")


async def run_once(client: PlaudV4Client, debouncer: Debouncer) -> str:
    """One connection lifetime. Returns why it ended."""
    token = client._tokens.get("access")
    if not token:
        return "no-token"
    headers = {"Origin": "https://web.plaud.ai", "User-Agent": UA}
    async with websockets.connect(
        WS_URL,
        subprotocols=[token.replace("Bearer ", "").replace("bearer ", "")],
        additional_headers=headers,
        open_timeout=20,
        ssl=ssl.create_default_context(),
        max_size=4 * 1024 * 1024,
    ) as ws:
        log.info("connected")
        _heartbeat(connected_at=time.time(), state="connected")
        last_rx = time.monotonic()
        last_ping = 0.0
        while True:
            now = time.monotonic()
            if now - last_ping >= PING_EVERY_S:
                await ws.send("PING")
                last_ping = now
            if now - last_rx > SILENCE_LIMIT_S:
                return "silent"
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            last_rx = time.monotonic()
            if raw == "PONG":
                continue
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            kind = msg.get("type")
            if kind == "error":
                if msg.get("status") == -401:
                    return "unauthorized"
                log.warning("server error: %s", str(msg)[:200])
                continue
            if kind == "connected":
                continue
            if kind == "notify" and msg.get("sub_type") == "file_notify":
                data = msg.get("data") or {}
                event = data.get("event")
                nodes = data.get("nodes") or []
                for node in nodes:
                    fid = str(node.get("file_id") or node.get("id") or "")
                    if fid:
                        debouncer.touch(fid)
                log.info("file_notify %s x%d", event, len(nodes))
                _heartbeat(last_event_at=time.time(), last_event=event)
            else:
                log.info("event %s/%s", kind, msg.get("sub_type"))


async def main() -> int:
    client = PlaudV4Client()
    debouncer = Debouncer()
    backoff = 2.0
    while True:
        try:
            why = await run_once(client, debouncer)
        except Exception as exc:  # noqa: BLE001 -- a dropped socket is routine here
            why = f"{type(exc).__name__}: {str(exc)[:120]}"
        log.info("disconnected: %s", why)
        _heartbeat(state="reconnecting", last_disconnect=why)
        if why in ("unauthorized", "no-token"):
            ok = False
            try:
                ok = client.refresh()
            except Exception as exc:  # noqa: BLE001
                log.warning("refresh failed: %s", type(exc).__name__)
            if not ok:
                log.error("session cannot be refreshed; run scripts/plaud_v4_login.py. Retrying in 5 min.")
                await asyncio.sleep(300)
                client = PlaudV4Client()
                continue
            backoff = 2.0
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 120.0)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
