"""
Plaud API Token Auth Client

Handles API token authentication for https://api.plaud.ai endpoints.
"""

import os
import base64
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path

TOKEN_FILE = Path(__file__).parent.parent / ".plaud_api_token.json"


class PlaudAPITokenClient:
    def __init__(self, client_id=None, client_secret=None):
        load_dotenv()
        self.client_id = client_id or os.getenv("PLAUD_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PLAUD_CLIENT_SECRET")
        self._api_token = None
        self._token_expiry = None
        self._load_token()
        if not self.client_id or not self.client_secret:
            raise ValueError("PLAUD_CLIENT_ID and PLAUD_CLIENT_SECRET must be set.")

    def _load_token(self):
        if TOKEN_FILE.exists():
            import json

            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                self._api_token = data.get("api_token")
                expiry = data.get("expiry")
                if expiry:
                    self._token_expiry = datetime.fromisoformat(expiry)

    def _save_token(self):
        import json

        data = {
            "api_token": self._api_token,
            "expiry": self._token_expiry.isoformat() if self._token_expiry else None,
            "saved_at": datetime.now().isoformat(),
        }
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f, indent=2)
        TOKEN_FILE.chmod(0o600)

    def get_api_token(self):
        if (
            self._api_token
            and self._token_expiry
            and datetime.now() < self._token_expiry - timedelta(minutes=5)
        ):
            return self._api_token
        self._fetch_token()
        return self._api_token

    def _fetch_token(self):
        url = "https://api.plaud.ai/apis/oauth/api-token"
        credentials = f"{self.client_id}:{self.client_secret}"
        b64 = base64.b64encode(credentials.encode()).decode()
        headers = {
            "Authorization": f"Bearer {b64}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        self._api_token = data["api_token"]
        expires_in = data.get("expires_in", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
        self._save_token()

    @property
    def is_authenticated(self):
        return bool(self.get_api_token())
