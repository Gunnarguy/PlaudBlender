"""Client for Plaud 4.0's account API.

The 4.0 web app (beta.plaud.ai) talks to `api-test.plaud.ai` under
`/file-app/v4/`. Unlike the third-party OAuth surface on platform.plaud.ai,
which stopped advancing past a fixed window of twenty recordings when the
account moved to 4.0, this API lists the whole library with a cursor and
serves every artifact the app itself shows: transcript with speakers,
polished transcript, summary, outline, audio.

Authentication is a login session, not an app credential. The password-free
route is an emailed one-time code:

    POST /auth/otp-send-code  {"username": <email>}   -> {"token": <challenge>}
    POST /auth/otp-login      {"code": <emailed>, "token": <challenge>}

The session comes back as cookies and/or token fields; both are kept. It is
renewed with `POST /auth/refresh-user-token`, and Plaud's own client treats a
refresh lifetime of -1 as never expiring, so one login is intended to last.

Nothing here logs or prints a token. The session file is chmod 600 and
gitignored, alongside the existing `.plaud_tokens.json`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

API_BASE = os.getenv("PLAUD_V4_API_BASE", "https://api-test.plaud.ai").rstrip("/")
WEB_ORIGIN = "https://beta.plaud.ai"
SESSION_FILE = Path(__file__).resolve().parent.parent / ".plaud_v4_session.json"

# Device codes as Plaud's own web bundle names them (`scene_source` in file
# detail). Codes it does not name are returned as the bare number.
DEVICE_MODELS = {
    880: "plaud_notepin",
    881: "plaud_note_pro",
    882: "plaud_notepin_s",
    888: "plaud_note",
}

_TOKEN_KEYS = ("access_token", "user_token", "workspace_token", "token")
_REFRESH_KEYS = ("refresh_token", "user_refresh_token")


class PlaudV4Error(RuntimeError):
    pass


class NotLoggedIn(PlaudV4Error):
    pass


def classic_id(file_id: str) -> str:
    """Map a v4 file id onto the id every other system already uses.

    Recordings migrated from 3.0 carry `of_` + the original 32-hex id, so the
    broker, the iOS app and Notion all match after the prefix is removed.
    4.0-native recordings use `f_` / `f_s_` and are new to everyone.
    """
    for prefix in ("of_", "f_s_", "f_"):
        if file_id.startswith(prefix):
            return file_id[len(prefix):]
    return file_id


def device_code(scene_source: Any) -> Optional[str]:
    """The device as a bare code string: "888", "860".

    Stored this way on purpose. Older syncs stored the device's full serial,
    which begins with the same three digits (888317281808436884), and the iOS
    app already identifies hardware by that three-digit prefix -- so a bare
    code and a full serial read as the same device without a migration.
    """
    if scene_source in (None, "", 0):
        return None
    try:
        return str(int(scene_source))
    except (TypeError, ValueError):
        return str(scene_source)


def device_model(scene_source: Any) -> Optional[str]:
    """Human name for a device code, where Plaud's bundle provides one."""
    code = device_code(scene_source)
    if code is None:
        return None
    try:
        return DEVICE_MODELS.get(int(code), code)
    except ValueError:
        return code


class PlaudV4Client:
    def __init__(self, session_file: Path = SESSION_FILE, timeout: float = 60.0):
        self.session_file = session_file
        self.timeout = timeout
        self.http = requests.Session()
        self.http.headers.update({
            "Accept": "application/json",
            "Origin": WEB_ORIGIN,
            "Referer": WEB_ORIGIN + "/",
            # Plaud's Cloudflare edge returns a 403 challenge page to any
            # non-browser User-Agent on the auth endpoints, verified from two
            # networks with identical requests. A browser string is the one
            # header that decides whether the request reaches the API at all.
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._tokens: dict[str, str] = {}
        self._load()

    # -- session persistence -------------------------------------------------

    def _load(self) -> None:
        if not self.session_file.exists():
            return
        try:
            data = json.loads(self.session_file.read_text())
        except json.JSONDecodeError:
            return
        for cookie in data.get("cookies", []):
            self.http.cookies.set(cookie["name"], cookie["value"], domain=cookie.get("domain"), path=cookie.get("path", "/"))
        self._tokens = {k: v for k, v in data.get("tokens", {}).items() if isinstance(v, str) and v}

    def _save(self) -> None:
        payload = {
            "saved_at": int(time.time()),
            "api_base": API_BASE,
            "cookies": [
                {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
                for c in self.http.cookies
            ],
            "tokens": self._tokens,
        }
        tmp = self.session_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.chmod(tmp, 0o600)
        tmp.replace(self.session_file)

    def _absorb(self, body: Any) -> None:
        """Keep any token-shaped fields a response carries."""
        data = body.get("data") if isinstance(body, dict) and isinstance(body.get("data"), dict) else body
        if not isinstance(data, dict):
            return
        for key in _TOKEN_KEYS:
            if isinstance(data.get(key), str) and data[key]:
                self._tokens["access"] = data[key]
                break
        for key in _REFRESH_KEYS:
            if isinstance(data.get(key), str) and data[key]:
                self._tokens["refresh"] = data[key]
                break

    @property
    def has_session(self) -> bool:
        return bool(self._tokens.get("access")) or any(self.http.cookies)

    def logout_local(self) -> None:
        self.http.cookies.clear()
        self._tokens = {}
        if self.session_file.exists():
            self.session_file.unlink()

    # -- transport ------------------------------------------------------------

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = dict(extra or {})
        if self._tokens.get("access"):
            h["Authorization"] = "Bearer " + self._tokens["access"]
        return h

    @staticmethod
    def _envelope(resp: requests.Response, want_text: bool = False):
        if want_text:
            return resp.text
        try:
            body = resp.json()
        except ValueError:
            raise PlaudV4Error(f"{resp.request.method} {resp.url} -> non-JSON {resp.status_code}: {resp.text[:200]}")
        # Plaud wraps as {"status": 0, "data": ..., "msg": ...}; status 0 is success.
        if isinstance(body, dict) and "status" in body and body["status"] not in (0, "0", None):
            raise PlaudV4Error(f"{resp.request.method} {resp.url} -> api status {body['status']}: {body.get('msg')}")
        return body

    def _request(self, method: str, path: str, *, retry_auth: bool = True, want_text: bool = False, **kw):
        url = API_BASE + path
        kw.setdefault("timeout", self.timeout)
        kw["headers"] = self._headers(kw.get("headers"))
        resp = self.http.request(method, url, **kw)
        if resp.status_code == 401 and retry_auth and self.has_session:
            if self.refresh():
                kw["headers"] = self._headers(kw.get("headers"))
                resp = self.http.request(method, url, **kw)
        if resp.status_code == 401:
            raise NotLoggedIn(f"{method} {path} -> 401. Run scripts/plaud_v4_login.py.")
        if resp.status_code == 422:
            raise PlaudV4Error(f"{method} {path} -> 422 {resp.text[:400]}")
        if resp.status_code >= 400:
            raise PlaudV4Error(f"{method} {path} -> {resp.status_code} {resp.text[:300]}")
        return self._envelope(resp, want_text=want_text)

    # -- auth -----------------------------------------------------------------

    def send_login_code(self, username: str) -> dict:
        """Ask Plaud to email a one-time code. Returns the challenge (has `token`)."""
        body = self._request("POST", "/auth/otp-send-code", json={"username": username}, retry_auth=False)
        data = body.get("data") if isinstance(body, dict) else None
        return data if isinstance(data, dict) else (body if isinstance(body, dict) else {})

    def login_with_code(self, challenge_token: str, code: str) -> dict:
        body = self._request("POST", "/auth/otp-login", json={"code": code.strip(), "token": challenge_token}, retry_auth=False)
        self._absorb(body)
        self._save()
        return body

    def refresh(self) -> bool:
        payload = {"refresh_token": self._tokens["refresh"]} if self._tokens.get("refresh") else None
        resp = self.http.post(API_BASE + "/auth/refresh-user-token", json=payload, headers=self._headers(), timeout=self.timeout)
        if resp.status_code != 200:
            return False
        try:
            body = resp.json()
        except ValueError:
            body = {}
        self._absorb(body)
        self._save()
        return True

    def me(self) -> dict:
        body = self._request("GET", "/user/me")
        return body.get("data", body) if isinstance(body, dict) else {}

    # -- library --------------------------------------------------------------

    def iter_recordings(self, page_size: int = 100, sort_by: str = "start_at", sort_order: str = "desc") -> Iterator[dict]:
        cursor: Optional[str] = None
        pages = 0
        while True:
            params = {"page_size": page_size, "sort_by": sort_by, "sort_order": sort_order}
            if cursor:
                params["cursor"] = cursor
            body = self._request("GET", "/file-app/v4/recordings/all", params=params)
            data = body.get("data") or {}
            for item in data.get("items") or []:
                yield item
            cursor = data.get("next_cursor") or None
            pages += 1
            if not cursor or pages >= 200:
                return

    def file_detail(self, file_id: str) -> dict:
        body = self._request("GET", f"/file-app/v4/files/detail/{file_id}")
        return body.get("data") or {}

    def content(self, content_id: str) -> str:
        return self._request("GET", f"/file-app/v4/files/contents/{content_id}", want_text=True)

    def content_json(self, content_id: str):
        text = self.content(content_id)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def objects_by_type(detail: dict) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for obj in detail.get("objects") or []:
            kind = str(obj.get("object_type") or "")
            if kind and kind not in out:
                out[kind] = obj
        return out
