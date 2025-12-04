"""
Plaud OAuth Client - Handles OAuth 2.0 authentication flow with Plaud API
"""
import os
import json
import webbrowser
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Plaud OAuth Configuration
# Auth is on app.plaud.ai, API is on platform.plaud.ai
PLAUD_AUTH_URL = "https://app.plaud.ai/platform/oauth"
PLAUD_TOKEN_URL = "https://platform.plaud.ai/developer/api/oauth/third-party/access-token"
PLAUD_API_BASE = "https://platform.plaud.ai/developer/api/open/third-party"

# Local callback configuration
DEFAULT_REDIRECT_URI = "http://localhost:8080/callback"
TOKEN_FILE = Path(__file__).parent.parent / ".plaud_tokens.json"


class PlaudOAuthClient:
    """
    OAuth 2.0 client for Plaud API authentication.
    
    Handles the full OAuth flow including:
    - Authorization URL generation
    - Token exchange
    - Token refresh
    - Token storage
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, redirect_uri: str = None):
        """
        Initialize the Plaud OAuth client.
        
        Args:
            client_id: Plaud OAuth app client ID (or set PLAUD_CLIENT_ID env var)
            client_secret: Plaud OAuth app client secret (or set PLAUD_CLIENT_SECRET env var)
            redirect_uri: OAuth callback URL (default: http://localhost:8080/callback)
        """
        self.client_id = client_id or os.getenv("PLAUD_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PLAUD_CLIENT_SECRET")
        self.redirect_uri = redirect_uri or os.getenv("PLAUD_REDIRECT_URI", DEFAULT_REDIRECT_URI)
        
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
    
    def _load_tokens(self):
        """Load tokens from local storage if available."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, 'r') as f:
                    data = json.load(f)
                    self._access_token = data.get('access_token')
                    self._refresh_token = data.get('refresh_token')
                    expiry = data.get('expiry')
                    if expiry:
                        self._token_expiry = datetime.fromisoformat(expiry)
                    logger.info("✅ Loaded existing Plaud tokens")
            except Exception as e:
                logger.warning(f"Could not load tokens: {e}")
    
    def _save_tokens(self):
        """Save tokens to local storage."""
        data = {
            'access_token': self._access_token,
            'refresh_token': self._refresh_token,
            'expiry': self._token_expiry.isoformat() if self._token_expiry else None,
            'saved_at': datetime.now().isoformat()
        }
        with open(TOKEN_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        # Secure the file
        TOKEN_FILE.chmod(0o600)
        logger.info("💾 Saved Plaud tokens")
    
    def get_authorization_url(self, scopes: list = None, state: str = None) -> tuple:
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
        
        # Plaud uses simple OAuth params (no scopes)
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
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
            "Accept": "application/json"
        }
        
        data = f"code={code}&redirect_uri={self.redirect_uri}"
        
        response = requests.post(PLAUD_TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        
        # Store tokens
        self._access_token = token_data.get('access_token')
        self._refresh_token = token_data.get('refresh_token')
        
        # Calculate expiry
        expires_in = token_data.get('expires_in', 3600)
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
        
        # Plaud uses Basic auth header
        credentials = f"{self.client_id}:{self.client_secret}"
        basic_auth = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        
        data = f"refresh_token={self._refresh_token}&grant_type=refresh_token"
        
        response = requests.post(PLAUD_TOKEN_URL, headers=headers, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        
        # Update tokens
        self._access_token = token_data.get('access_token')
        if 'refresh_token' in token_data:
            self._refresh_token = token_data['refresh_token']
        
        expires_in = token_data.get('expires_in', 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
        
        self._save_tokens()
        
        logger.info("🔄 Refreshed Plaud access token")
        return token_data
    
    def get_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token string
        """
        # Check if we need to refresh
        if self._token_expiry and datetime.now() >= self._token_expiry - timedelta(minutes=5):
            logger.info("Token expired or expiring soon, refreshing...")
            self.refresh_access_token()
        
        if not self._access_token:
            raise ValueError("No access token available. Please authenticate first.")
        
        return self._access_token
    
    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication."""
        if not self._access_token:
            return False
        if self._token_expiry and datetime.now() >= self._token_expiry:
            # Try to refresh
            try:
                self.refresh_access_token()
                return True
            except:
                return False
        return True
    
    def authenticate_interactive(self):
        """
        Run interactive OAuth flow - opens browser and handles callback.
        """
        auth_url, state = self.get_authorization_url()
        
        print("\n" + "="*60)
        print("🔐 PLAUD AUTHENTICATION")
        print("="*60)
        print("\nOpening browser for Plaud authentication...")
        print(f"\nIf browser doesn't open, visit:\n{auth_url}\n")
        
        # Open browser
        webbrowser.open(auth_url)
        
        # Start local server to catch callback
        received_code = [None]  # Use list to modify in nested function
        received_state = [None]
        
        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == '/callback':
                    params = parse_qs(parsed.query)
                    received_code[0] = params.get('code', [None])[0]
                    received_state[0] = params.get('state', [None])[0]
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    success_html = """
                    <html>
                    <head><title>PlaudBlender - Authenticated!</title></head>
                    <body style="font-family: -apple-system, sans-serif; text-align: center; padding: 50px;">
                        <h1>✅ Authentication Successful!</h1>
                        <p>You can close this window and return to the terminal.</p>
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
        
        server = HTTPServer(('localhost', port), CallbackHandler)
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
        print("="*60 + "\n")


def authenticate():
    """Convenience function to run OAuth flow."""
    client = PlaudOAuthClient()
    client.authenticate_interactive()
    return client


if __name__ == "__main__":
    # Run interactive authentication
    authenticate()
