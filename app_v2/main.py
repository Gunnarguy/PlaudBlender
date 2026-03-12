"""Chronos App v2 - Recording-Centric UI

Run with: python -m app_v2.main
"""

import logging
import platform
import secrets
import subprocess
import threading
import time

from dash import Dash
from flask import redirect, request, jsonify, make_response
from flask_compress import Compress
from markupsafe import escape

from app_v2.layout import create_layout
from app_v2.callbacks import register_all_callbacks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# CSRF state storage for in-app OAuth flow
_oauth_pending_states: dict[str, bool] = {}

# Redirect URI for in-app OAuth (Dash server at port 8050)
INAPP_REDIRECT_URI = "http://localhost:8050/auth/plaud/callback"


def _register_auth_routes(server):
    """Register Flask routes for Plaud OAuth flow on the Dash server."""

    @server.route("/auth/plaud")
    def auth_plaud_start():
        """Start Plaud OAuth — opens Chrome and shows a waiting page.

        Safari blocks mixed-content XHR from Plaud's HTTPS auth page to our
        HTTP localhost callback.  Chrome has a localhost exemption, so we open
        the auth URL in Chrome specifically and return a waiting page that
        polls /auth/plaud/status until tokens arrive.
        """
        try:
            from src.plaud_oauth import PlaudOAuthClient

            client = PlaudOAuthClient(redirect_uri=INAPP_REDIRECT_URI)
            auth_url, state = client.get_authorization_url()
            _oauth_pending_states[state] = True

            # Open Chrome (has localhost mixed-content exemption)
            _open_in_chrome(auth_url)

            return _auth_waiting_page()
        except Exception as e:
            safe_msg = escape(str(e))
            return (
                _auth_error_page(
                    "Configuration Error",
                    f"{safe_msg}<br><br>"
                    "Make sure <code>PLAUD_CLIENT_ID</code> and "
                    "<code>PLAUD_CLIENT_SECRET</code> are set in your "
                    "<code>.env</code> file.",
                ),
                500,
            )

    def _cors_response(body, status=200, content_type="text/html"):
        """Wrap a response with CORS headers for Plaud OAuth XHR callbacks."""
        resp = make_response(body, status)
        # Plaud's OAuth page may send Origin: null (sandboxed redirect) or
        # its own domain.  We must reflect the request Origin so the browser
        # accepts the XHR response.
        origin = request.headers.get("Origin", "")
        allowed = {"https://app.plaud.ai", "https://resource.plaud.ai", "null"}
        if origin in allowed:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            # Fallback — allow any origin for this one endpoint since
            # only a valid state+code can trigger token exchange.
            resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "*"
        if content_type:
            resp.headers["Content-Type"] = content_type
        return resp

    @server.route("/auth/plaud/callback", methods=["GET", "OPTIONS"])
    def auth_plaud_callback():
        """Handle OAuth callback from Plaud, exchange code for tokens."""
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return _cors_response("", 204)

        error = request.args.get("error")
        if error:
            return _cors_response(
                _auth_error_page("Plaud Denied Access", escape(str(error))), 400
            )

        code = request.args.get("code")
        state = request.args.get("state")
        if not code:
            return _cors_response(
                _auth_error_page("Missing Code", "No authorization code received."),
                400,
            )

        if state not in _oauth_pending_states:
            return _cors_response(
                _auth_error_page("Invalid State", "CSRF state mismatch — try again."),
                403,
            )
        del _oauth_pending_states[state]

        try:
            from src.plaud_oauth import PlaudOAuthClient

            client = PlaudOAuthClient(redirect_uri=INAPP_REDIRECT_URI)
            client.exchange_code_for_token(code)
            return _cors_response(_auth_success_page())
        except Exception as e:
            safe_msg = escape(str(e))
            return _cors_response(
                _auth_error_page("Token Exchange Failed", safe_msg), 500
            )

    @server.route("/auth/plaud/status")
    def auth_plaud_status():
        """Return JSON auth status for AJAX polling."""
        try:
            from src.plaud_oauth import PlaudOAuthClient

            client = PlaudOAuthClient()
            return jsonify(client.token_status)
        except Exception as e:
            return jsonify({"is_authenticated": False, "error": str(e)})


def _auth_success_page() -> str:
    return """<!DOCTYPE html>
<html><head><title>Chronos — Connected!</title></head>
<body style="font-family:-apple-system,sans-serif;text-align:center;padding:60px;
background:#0f172a;color:#e2e8f0;">
<h1 style="color:#10b981;">✅ Plaud Connected!</h1>
<p>Authentication successful. This tab will close automatically.</p>
<p style="color:#64748b;font-size:0.85rem;">You can also close it manually and
return to Chronos.</p>
<script>
if(window.opener){window.opener.postMessage('plaud-auth-success','*');}
setTimeout(function(){window.close();},2500);
</script>
</body></html>"""


def _auth_error_page(title: str, detail: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><title>Chronos — Auth Error</title></head>
<body style="font-family:-apple-system,sans-serif;text-align:center;padding:60px;
background:#0f172a;color:#e2e8f0;">
<h1 style="color:#ef4444;">❌ {title}</h1>
<p>{detail}</p>
<a href="/" style="color:#60a5fa;text-decoration:underline;">Return to Chronos</a>
</body></html>"""


def _open_in_chrome(url: str):
    """Open a URL in Chrome (it has a localhost mixed-content exemption)."""
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(
                ["open", "-a", "Google Chrome", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif platform.system() == "Linux":
            subprocess.Popen(
                ["google-chrome", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            import webbrowser

            webbrowser.open(url)
    except Exception:
        # Fallback to default browser
        import webbrowser

        webbrowser.open(url)


def _auth_waiting_page() -> str:
    """Page shown while user completes OAuth in Chrome."""
    return """<!DOCTYPE html>
<html><head><title>Chronos — Authenticating…</title></head>
<body style="font-family:-apple-system,sans-serif;text-align:center;padding:60px;
background:#0f172a;color:#e2e8f0;">
<h1>🔐 Authenticating with Plaud…</h1>
<p>A Chrome window has opened. Complete the login there.</p>
<p id="status" style="color:#64748b;">Waiting for authorization…</p>
<div style="margin:30px auto;width:40px;height:40px;border:4px solid #334155;
border-top:4px solid #60a5fa;border-radius:50%;animation:spin 1s linear infinite;"></div>
<style>@keyframes spin{to{transform:rotate(360deg);}}</style>
<script>
(function poll(){
  fetch('/auth/plaud/status')
    .then(r=>r.json())
    .then(d=>{
      if(d.is_authenticated){
        document.getElementById('status').textContent='Connected!';
        document.querySelector('h1').textContent='✅ Plaud Connected!';
        document.querySelector('div').style.display='none';
        if(window.opener){window.opener.postMessage('plaud-auth-success','*');}
        setTimeout(function(){window.close();},2000);
      } else {
        setTimeout(poll,2000);
      }
    })
    .catch(()=>setTimeout(poll,3000));
})();
</script>
</body></html>"""


def _start_token_keepalive():
    """Daemon thread that refreshes Plaud tokens every 20 minutes."""

    def _loop():
        while True:
            time.sleep(20 * 60)
            try:
                from src.plaud_oauth import PlaudOAuthClient

                client = PlaudOAuthClient()
                if client.is_authenticated:
                    client.ensure_valid_token()
                    logger.info("Token keepalive: refresh OK")
            except Exception as e:
                logger.debug(f"Token keepalive: {e}")

    t = threading.Thread(target=_loop, daemon=True, name="plaud-token-keepalive")
    t.start()
    logger.info("Plaud token keepalive thread started")


def create_app() -> Dash:
    """Create and configure the Dash app."""
    app = Dash(
        __name__,
        title="Chronos",
        assets_folder="assets",
        suppress_callback_exceptions=True,
        eager_loading=False,
    )

    # Enable gzip/brotli compression on all responses
    app.server.config["COMPRESS_ALGORITHM"] = ["br", "gzip"]
    app.server.config["COMPRESS_MIN_SIZE"] = 500
    Compress(app.server)

    # Register Flask routes for in-app Plaud OAuth
    _register_auth_routes(app.server)

    # Set layout
    app.layout = create_layout()

    # Register callbacks
    register_all_callbacks(app)

    # Start auto-sync service in background
    try:
        from src.plaud_auto_sync import get_auto_sync

        auto_sync = get_auto_sync()
        auto_sync.start()
        logger.info("Auto-sync service started in background")
    except Exception as e:
        logger.warning(f"Could not start auto-sync: {e}")

    # Start background token keepalive
    _start_token_keepalive()

    logger.info("Chronos app v2 initialized")
    return app


def main():
    """Run the app."""
    app = create_app()

    logger.info("Starting Chronos v2 at http://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
