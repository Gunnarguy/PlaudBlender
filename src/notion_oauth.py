"""Notion OAuth Client — handles OAuth 2.0 authorization code flow.

Mirrors the Plaud OAuth client pattern:
- Authorization URL generation with CSRF state
- Code → token exchange
- Token storage in local JSON file
- Token validation via /v1/users/me
- No refresh tokens (Notion tokens don't expire — they're revoked by user)
"""

import json
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Notion OAuth endpoints
NOTION_AUTH_URL = "https://api.notion.com/v1/oauth/authorize"
NOTION_TOKEN_URL = "https://api.notion.com/v1/oauth/token"
NOTION_USERS_ME = "https://api.notion.com/v1/users/me"
NOTION_API_VERSION = "2022-06-28"

# Local storage
TOKEN_FILE = Path(__file__).parent.parent / ".notion_tokens.json"
DEFAULT_REDIRECT_URI = "http://localhost:8000/api/v1/auth/notion/callback"


class NotionOAuthClient:
    """OAuth 2.0 client for Notion API.

    Unlike Plaud, Notion access tokens do NOT expire — they remain valid
    until the user revokes the integration from their Notion settings.
    So there's no refresh flow; just store and validate.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("NOTION_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("NOTION_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv(
            "NOTION_REDIRECT_URI", DEFAULT_REDIRECT_URI
        )

        self._access_token: Optional[str] = None
        self._workspace_name: Optional[str] = None
        self._workspace_id: Optional[str] = None
        self._bot_id: Optional[str] = None
        self._owner: Optional[dict] = None
        self._authenticated_at: Optional[datetime] = None

        # Load existing tokens
        self._load_tokens()

    @property
    def has_credentials(self) -> bool:
        """Check if OAuth client credentials are configured."""
        return bool(self.client_id and self.client_secret)

    @property
    def is_authenticated(self) -> bool:
        """Check if we have a stored access token."""
        return bool(self._access_token)

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    @property
    def token_status(self) -> dict:
        """Return diagnostic token status (for /auth/notion/status endpoint)."""
        return {
            "is_authenticated": self.is_authenticated,
            "has_credentials": self.has_credentials,
            "workspace_name": self._workspace_name,
            "workspace_id": self._workspace_id,
            "bot_id": self._bot_id,
            "authenticated_at": (
                self._authenticated_at.isoformat() if self._authenticated_at else None
            ),
        }

    def get_authorization_url(self, state: Optional[str] = None) -> tuple:
        """Generate the Notion OAuth authorization URL.

        Returns: (auth_url, state)
        """
        if not self.client_id:
            raise ValueError(
                "NOTION_CLIENT_ID must be set in .env. "
                "Create an integration at https://www.notion.so/my-integrations"
            )

        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "owner": "user",
            "state": state,
        }

        from urllib.parse import urlencode
        auth_url = f"{NOTION_AUTH_URL}?{urlencode(params)}"
        return auth_url, state

    def exchange_code_for_token(self, code: str) -> dict:
        """Exchange the authorization code for an access token.

        Notion uses Basic Auth: base64(client_id:client_secret) in the
        Authorization header, with the code in the JSON body.
        """
        import base64

        credentials = f"{self.client_id}:{self.client_secret}"
        basic_auth = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_API_VERSION,
        }

        body = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
        }

        logger.info(
            "Notion token exchange → %s  code=%s…",
            NOTION_TOKEN_URL,
            code[:12] if len(code) > 12 else code,
        )

        response = requests.post(NOTION_TOKEN_URL, headers=headers, json=body)

        if not response.ok:
            logger.error(
                "Notion token exchange failed: %s %s — %s",
                response.status_code,
                response.reason,
                response.text[:500],
            )
            self._clear_tokens()
            response.raise_for_status()

        token_data = response.json()

        # Store token and workspace info
        self._access_token = token_data.get("access_token")
        self._workspace_name = token_data.get("workspace_name")
        self._workspace_id = token_data.get("workspace_id")
        self._bot_id = token_data.get("bot_id")
        self._owner = token_data.get("owner")
        self._authenticated_at = datetime.now()

        self._save_tokens()
        logger.info(
            "Notion OAuth success — workspace: %s", self._workspace_name
        )
        return token_data

    def validate_token(self) -> bool:
        """Validate the current token by calling /v1/users/me.

        Returns True if token is valid, False otherwise.
        """
        if not self._access_token:
            return False

        try:
            resp = requests.get(
                NOTION_USERS_ME,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Notion-Version": NOTION_API_VERSION,
                },
                timeout=5,
            )
            if resp.ok:
                return True
            else:
                logger.warning(
                    "Notion token validation failed: %s %s",
                    resp.status_code, resp.text[:200],
                )
                if resp.status_code == 401:
                    self._clear_tokens()
                return False
        except Exception as e:
            logger.warning(f"Notion token validation error: {e}")
            return False

    def _load_tokens(self):
        """Load tokens from local JSON file."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, "r") as f:
                    data = json.load(f)
                self._access_token = data.get("access_token")
                self._workspace_name = data.get("workspace_name")
                self._workspace_id = data.get("workspace_id")
                self._bot_id = data.get("bot_id")
                auth_at = data.get("authenticated_at")
                if auth_at:
                    self._authenticated_at = datetime.fromisoformat(auth_at)
                logger.info("Loaded existing Notion tokens (workspace: %s)", self._workspace_name)
            except Exception as e:
                logger.warning(f"Could not load Notion tokens: {e}")

    def _save_tokens(self):
        """Save tokens to local JSON file (chmod 600)."""
        data = {
            "access_token": self._access_token,
            "workspace_name": self._workspace_name,
            "workspace_id": self._workspace_id,
            "bot_id": self._bot_id,
            "authenticated_at": (
                self._authenticated_at.isoformat() if self._authenticated_at else None
            ),
            "saved_at": datetime.now().isoformat(),
        }
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f, indent=2)
        TOKEN_FILE.chmod(0o600)
        logger.info("Saved Notion token (workspace: %s)", self._workspace_name)

    def _clear_tokens(self):
        """Remove cached tokens (forces re-auth)."""
        self._access_token = None
        self._workspace_name = None
        self._workspace_id = None
        self._bot_id = None
        self._authenticated_at = None
        if TOKEN_FILE.exists():
            try:
                TOKEN_FILE.unlink()
            except Exception as e:
                logger.warning(f"Could not delete Notion token file: {e}")
