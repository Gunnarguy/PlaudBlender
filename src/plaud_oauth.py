"""
Plaud OAuth Client - Handles OAuth 2.0 authentication flow with Plaud API

Provides bulletproof authentication that:
- Proactively refreshes tokens 30 mins before expiry
- Validates tokens before use
- Automatically re-authenticates when needed
- Never throws unexpected exceptions
"""

import os
import json
import webbrowser
import secrets
import ssl
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class AuthenticationRequired(Exception):
    """Raised when authentication/re-authentication is required."""

    pass


class PlaudAuthError(Exception):
    """Base exception for Plaud authentication errors."""

    pass


# Plaud OAuth Configuration
# Auth is on app.plaud.ai, API is on platform.plaud.ai
PLAUD_AUTH_URL = "https://app.plaud.ai/platform/oauth"
PLAUD_TOKEN_URL = (
    "https://platform.plaud.ai/developer/api/oauth/third-party/access-token"
)
PLAUD_REFRESH_URL = PLAUD_TOKEN_URL + "/refresh"
PLAUD_API_BASE = "https://platform.plaud.ai/developer/api/open/third-party"

# Local callback configuration
#
# Plaud's developer portal typically allows (and examples use) a plain HTTP
# localhost callback during development.
#
# IMPORTANT: The redirect URI *must* exactly match what you registered in the
# Plaud developer portal (scheme, host, port, and path).
DEFAULT_REDIRECT_URI = "http://localhost:8050/auth/plaud/callback"
TOKEN_FILE = Path(__file__).parent.parent / ".plaud_tokens.json"
CERT_DIR = Path(__file__).parent.parent / ".certs"


class PlaudOAuthClient:
    """
    OAuth 2.0 client for Plaud API authentication.

    Handles the full OAuth flow including:
    - Authorization URL generation
    - Token exchange
    - Token refresh
    - Token storage
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ):
        """
        Initialize the Plaud OAuth client.

        Args:
            client_id: Plaud OAuth app client ID (or set PLAUD_CLIENT_ID env var)
            client_secret: Plaud OAuth app client secret (or set PLAUD_CLIENT_SECRET env var)
            redirect_uri: OAuth callback URL (default: http://localhost:8080/callback)
        """
        self.client_id = client_id or os.getenv("PLAUD_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PLAUD_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv(
            "PLAUD_REDIRECT_URI", DEFAULT_REDIRECT_URI
        )

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "PLAUD_CLIENT_ID and PLAUD_CLIENT_SECRET must be set. "
                "Get these from https://platform.plaud.ai/developer/portal"
            )

        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None

        # Try to load existing tokens
        self._load_tokens()

    def _clear_tokens(self):
        """Remove cached tokens to force re-auth when refresh fails."""
        self._access_token = None
        self._refresh_token = None
        self._token_expiry = None
        if TOKEN_FILE.exists():
            try:
                TOKEN_FILE.unlink()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not delete token file: {exc}")

    def _load_tokens(self):
        """Load tokens from local storage if available."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, "r") as f:
                    data = json.load(f)
                    self._access_token = data.get("access_token")
                    self._refresh_token = data.get("refresh_token")
                    expiry = data.get("expiry")
                    if expiry:
                        self._token_expiry = datetime.fromisoformat(expiry)
                    logger.info("✅ Loaded existing Plaud tokens")
            except Exception as e:
                logger.warning(f"Could not load tokens: {e}")

    def _save_tokens(self):
        """Save tokens to local storage."""
        data = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "expiry": self._token_expiry.isoformat() if self._token_expiry else None,
            "saved_at": datetime.now().isoformat(),
        }
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f, indent=2)
        # Secure the file
        TOKEN_FILE.chmod(0o600)
        logger.info("💾 Saved Plaud tokens")

    def get_authorization_url(
        self, scopes: Optional[list[str]] = None, state: str | None = None
    ) -> tuple[str, str]:
        """
        Generate the OAuth authorization URL.

        Args:
            scopes: List of OAuth scopes to request (not used by Plaud currently)
            state: CSRF protection state value (auto-generated if not provided)

        Returns:
            Tuple of (authorization_url, state)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        # Plaud uses simple OAuth params (no scopes). We still include `state`
        # for CSRF protection; the callback handler verifies it.
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
        }

        auth_url = f"{PLAUD_AUTH_URL}?{urlencode(params)}"
        return auth_url, state

    def exchange_code_for_token(self, code: str) -> dict:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth callback

        Returns:
            Token response dictionary
        """
        import base64

        # Plaud uses Basic auth header: base64(client_id:client_secret)
        credentials = f"{self.client_id}:{self.client_secret}"
        basic_auth = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = f"code={code}&redirect_uri={self.redirect_uri}"

        try:
            response = requests.post(PLAUD_TOKEN_URL, headers=headers, data=data)
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Plaud token refresh failed (%s) — clearing tokens, please re-authenticate",
                exc,
            )
            self._clear_tokens()
            raise

        token_data = response.json()

        # Store tokens
        self._access_token = token_data.get("access_token")
        self._refresh_token = token_data.get("refresh_token")

        # Calculate expiry
        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

        self._save_tokens()

        logger.info("🔐 Successfully obtained Plaud access token")
        return token_data

    def refresh_access_token(self) -> dict:
        """
        Refresh the access token using the refresh token.

        Returns:
            New token response dictionary
        """
        import base64

        if not self._refresh_token:
            raise ValueError("No refresh token available. Please re-authenticate.")

        # Try legacy refresh with Basic auth first
        credentials = f"{self.client_id}:{self.client_secret}"
        basic_auth = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = f"refresh_token={self._refresh_token}&grant_type=refresh_token"

        try:
            response = requests.post(PLAUD_TOKEN_URL, headers=headers, data=data)
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.warning("Legacy refresh failed (%s); trying /refresh endpoint", exc)
            # Plaud docs: /access-token/refresh takes just refresh_token, no auth header
            fallback_headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            fallback_data = f"refresh_token={self._refresh_token}"
            response = requests.post(
                PLAUD_REFRESH_URL, headers=fallback_headers, data=fallback_data
            )
            response.raise_for_status()

        token_data = response.json()

        # Update tokens
        self._access_token = token_data.get("access_token")
        if "refresh_token" in token_data:
            self._refresh_token = token_data["refresh_token"]

        expires_in = token_data.get("expires_in", 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

        self._save_tokens()

        logger.info("🔄 Refreshed Plaud access token")
        return token_data

    def _validate_token(self) -> bool:
        """
        Validate the current token by making a lightweight API call.

        Returns:
            True if token is valid, False otherwise
        """
        if not self._access_token:
            return False

        try:
            response = requests.get(
                f"{PLAUD_API_BASE}/users/current",
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Accept": "application/json",
                },
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Token validation failed: {e}")
            return False

    def ensure_valid_token(self) -> str:
        """
        Ensure we have a valid token, refreshing or re-authenticating as needed.

        This is the bulletproof method - it will ALWAYS return a valid token
        or raise an exception with clear instructions.

        Returns:
            Valid access token string
        """
        # Step 1: Check if we have a token that isn't expired
        if self._access_token and self._token_expiry:
            # Proactive refresh: refresh 30 mins before expiry
            if datetime.now() < self._token_expiry - timedelta(minutes=30):
                # Token should be valid, but verify it
                if self._validate_token():
                    return self._access_token
                else:
                    logger.warning("Token failed validation, attempting refresh...")

        # Step 2: Try to refresh if we have a refresh token
        if self._refresh_token:
            try:
                logger.info("Refreshing access token...")
                self.refresh_access_token()
                if self._validate_token():
                    return self._access_token  # type: ignore[return-value]
                else:
                    logger.warning("Refreshed token failed validation")
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}")

        # Step 3: Clear invalid tokens and require re-authentication
        self._clear_tokens()
        raise AuthenticationRequired(
            "Authentication required. Run: python plaud_setup.py\n"
            "Or call: client.oauth.authenticate_interactive()"
        )

    def get_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token string
        """
        # Check if we need to refresh (30 mins before expiry for safety)
        if self._token_expiry and datetime.now() >= self._token_expiry - timedelta(
            minutes=30
        ):
            logger.info("Token expired or expiring soon, refreshing...")
            try:
                self.refresh_access_token()
            except Exception as exc:
                logger.error("Automatic refresh failed: %s", exc)
                raise

        if not self._access_token:
            raise ValueError("No access token available. Please authenticate first.")

        return self._access_token

    @property
    def is_authenticated(self) -> bool:
        """
        Check if we have valid authentication.

        This does a lightweight check without network calls.
        For full validation, use ensure_valid_token().
        """
        if not self._access_token:
            return False
        if self._token_expiry and datetime.now() >= self._token_expiry:
            # Token expired - try refresh
            try:
                self.refresh_access_token()
                return True
            except Exception:
                return False
        return True

    @property
    def token_status(self) -> dict:
        """
        Get detailed token status for diagnostics.

        Returns:
            Dict with authentication status details
        """
        now = datetime.now()
        status = {
            "has_access_token": bool(self._access_token),
            "has_refresh_token": bool(self._refresh_token),
            "is_authenticated": self.is_authenticated,
            "token_valid": False,
            "expires_at": None,
            "expires_in_minutes": None,
            "needs_refresh": False,
        }

        if self._token_expiry:
            status["expires_at"] = self._token_expiry.isoformat()
            status["expires_in_minutes"] = (
                self._token_expiry - now
            ).total_seconds() / 60
            status["needs_refresh"] = now >= self._token_expiry - timedelta(minutes=30)
            status["token_valid"] = now < self._token_expiry

        return status

    def _open_browser_chrome_first(self, url: str):
        """Try Chrome first (better localhost handling), then fall back to default."""
        import subprocess
        import sys

        # Try Chrome on macOS
        if sys.platform == "darwin":
            chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
            ]
            for chrome in chrome_paths:
                if os.path.exists(chrome):
                    try:
                        subprocess.Popen([chrome, url])
                        return
                    except Exception:
                        continue

        # Fall back to default browser
        webbrowser.open(url)

    def authenticate_interactive(self):
        """
        Run interactive OAuth flow - opens browser and handles callback.
        """
        auth_url, state = self.get_authorization_url()

        print("\n" + "=" * 60)
        print("🔐 PLAUD AUTHENTICATION")
        print("=" * 60)
        print("\nOpening browser for Plaud authentication...")
        print(f"\nIf browser doesn't open, visit:\n{auth_url}\n")

        # Open browser (prefer Chrome for better localhost handling)
        self._open_browser_chrome_first(auth_url)

        # Start local server to catch callback
        received_code: list[str | None] = [
            None
        ]  # Use list to modify in nested function
        received_state: list[str | None] = [None]

        class CallbackHandler(BaseHTTPRequestHandler):
            def _send_cors_headers(self):
                """Send CORS headers to allow cross-origin requests from Plaud."""
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")

            def do_OPTIONS(self):
                """Handle CORS preflight requests."""
                self.send_response(200)
                self._send_cors_headers()
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/callback":
                    params = parse_qs(parsed.query)
                    received_code[0] = params.get("code", [None])[0]
                    received_state[0] = params.get("state", [None])[0]

                    self.send_response(200)
                    self._send_cors_headers()
                    self.send_header("Content-type", "text/html")
                    self.end_headers()

                    success_html = """
                    <html>
                    <head><title>PlaudBlender - Authenticated!</title></head>
                    <body style="font-family: -apple-system, sans-serif; text-align: center; padding: 50px;">
                        <h1>✅ Authentication Successful!</h1>
                        <p>You can close this window and return to the app.</p>
                        <script>window.close();</script>
                    </body>
                    </html>
                    """
                    self.wfile.write(success_html.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        # Parse port from redirect URI
        parsed = urlparse(self.redirect_uri)
        port = parsed.port or 8080
        use_https = parsed.scheme == "https"

        # Bind to 127.0.0.1 explicitly (avoid IPv6 issues)
        server = HTTPServer(("127.0.0.1", port), CallbackHandler)

        # Wrap with SSL if using HTTPS
        if use_https:
            cert_file = CERT_DIR / "localhost.crt"
            key_file = CERT_DIR / "localhost.key"
            if cert_file.exists() and key_file.exists():
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(str(cert_file), str(key_file))
                server.socket = context.wrap_socket(server.socket, server_side=True)
                logger.info("Using HTTPS for OAuth callback (self-signed cert)")
            else:
                # We can't silently fall back to HTTP here because the browser will
                # still attempt to redirect to the *registered* redirect URI.
                raise RuntimeError(
                    "PLAUD_REDIRECT_URI is set to https:// but no TLS cert/key were found. "
                    "Either set PLAUD_REDIRECT_URI to http://localhost:8050/auth/plaud/callback (and register it in Plaud), "
                    f"or provide TLS files at {CERT_DIR}/localhost.crt and {CERT_DIR}/localhost.key."
                )

        server.timeout = 300  # 5 minute timeout

        print(f"Waiting for authentication callback on port {port}...")

        # Wait for callback
        while received_code[0] is None:
            server.handle_request()

        server.server_close()

        # Verify state
        if received_state[0] != state:
            raise ValueError("State mismatch! Possible CSRF attack.")

        # Exchange code for token
        self.exchange_code_for_token(received_code[0])

        print("\n✅ Successfully authenticated with Plaud!")
        print("=" * 60 + "\n")


def authenticate():
    """Convenience function to run OAuth flow."""
    client = PlaudOAuthClient()
    client.authenticate_interactive()
    return client


if __name__ == "__main__":
    # Run interactive authentication
    authenticate()
