"""Chronos App v2 - Recording-Centric UI

Run with: python -m app_v2.main
"""

import logging
import os
import platform
import secrets
import shutil
import subprocess
import threading
import time
from urllib.parse import urlencode, urlsplit, urlunsplit

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
# HTTPS avoids Safari mixed-content block (HTTPS Plaud -> HTTP localhost)
INAPP_REDIRECT_URI = "https://localhost:8050/auth/plaud/callback"
# Notion OAuth now flows through the FastAPI backend (single redirect URI
# configured on notion.so). Read from .env so it stays in sync.
NOTION_REDIRECT_URI = os.environ.get(
    "NOTION_REDIRECT_URI", "http://localhost:8000/api/v1/auth/notion/callback"
)


def _notion_api_authorize_url(return_to: str = "") -> str:
    """Build the FastAPI Notion authorize URL from the configured callback host."""
    parsed = urlsplit(NOTION_REDIRECT_URI)
    if parsed.scheme and parsed.netloc:
        base = urlunsplit(
            (parsed.scheme, parsed.netloc, "/api/v1/auth/notion/web-authorize", "", "")
        )
    else:
        base = "http://localhost:8000/api/v1/auth/notion/web-authorize"

    if return_to:
        return f"{base}?{urlencode({'return_to': return_to})}"
    return base


def _should_start_embedded_auto_sync() -> tuple[bool, str]:
    """Decide whether the Dash app should spawn its in-process auto-sync worker."""
    override = str(os.environ.get("CHRONOS_EMBEDDED_AUTO_SYNC", "")).strip().lower()
    if override in {"0", "false", "off", "no"}:
        return False, "disabled by CHRONOS_EMBEDDED_AUTO_SYNC"
    if override in {"1", "true", "on", "yes"}:
        return True, "forced by CHRONOS_EMBEDDED_AUTO_SYNC"

    if platform.system() != "Linux":
        return True, "non-Linux runtime"

    if not shutil.which("systemctl"):
        return True, "systemctl unavailable"

    try:
        enabled = subprocess.run(
            ["systemctl", "is-enabled", "chronos-auto-sync.service"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        enabled_state = (enabled.stdout or enabled.stderr or "").strip() or "unknown"

        active = subprocess.run(
            ["systemctl", "is-active", "chronos-auto-sync.service"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        active_state = (active.stdout or active.stderr or "").strip() or "unknown"

        managed_states = {"enabled", "enabled-runtime", "linked", "alias", "static", "indirect"}
        active_states = {"active", "activating", "reloading"}
        if enabled_state in managed_states or active_state in active_states:
            return (
                False,
                f"systemd manages chronos-auto-sync.service ({active_state}/{enabled_state})",
            )
    except Exception as exc:
        logger.debug("Could not inspect systemd auto-sync service: %s", exc)

    return True, "no systemd-managed auto-sync detected"


def _register_auth_routes(server):
    """Register Flask routes for Plaud OAuth flow on the Dash server."""

    @server.route("/auth/plaud")
    def auth_plaud_start():
        """Start Plaud OAuth — redirect to Plaud's authorization page."""
        try:
            from src.plaud_oauth import PlaudOAuthClient

            client = PlaudOAuthClient(redirect_uri=INAPP_REDIRECT_URI)
            auth_url, state = client.get_authorization_url()
            _oauth_pending_states[state] = True

            return redirect(auth_url)
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
        logger.info(
            "CALLBACK HIT: method=%s url=%s args=%s origin=%s",
            request.method,
            request.url,
            dict(request.args),
            request.headers.get("Origin", "(none)"),
        )

        # Handle CORS preflight
        if request.method == "OPTIONS":
            return _cors_response("", 204)

        # Detect XHR vs browser redirect — XHR needs CORS-wrapped JSON,
        # browser redirect can use redirect() or HTML pages.
        is_xhr = (
            bool(request.headers.get("Origin"))
            or request.headers.get("X-Requested-With", "").lower() == "xmlhttprequest"
        )

        error = request.args.get("error")
        if error:
            msg = escape(str(error))
            if is_xhr:
                return _cors_response(f'{{"error": "{msg}"}}', 400, "application/json")
            return _cors_response(
                _auth_error_page("Plaud Denied Access", str(msg)), 400
            )

        code = request.args.get("code")
        state = request.args.get("state")
        if not code:
            if is_xhr:
                return _cors_response(
                    '{"error": "missing code"}', 400, "application/json"
                )
            return _cors_response(
                _auth_error_page("Missing Code", "No authorization code received."),
                400,
            )

        # Clean up state if it exists (may be lost on debug reloader restart)
        if state:
            _oauth_pending_states.pop(state, None)

        from src.plaud_oauth import PlaudOAuthClient

        # Idempotent: if tokens already valid, skip exchange
        try:
            client = PlaudOAuthClient(redirect_uri=INAPP_REDIRECT_URI)
            if client.is_authenticated:
                logger.info("CALLBACK: already authenticated — skipping exchange")
                if is_xhr:
                    return _cors_response(
                        '{"status": "ok", "already": true}', 200, "application/json"
                    )
                return redirect("/")
        except Exception:
            pass

        try:
            client = PlaudOAuthClient(redirect_uri=INAPP_REDIRECT_URI)
            token_data = client.exchange_code_for_token(code, state=state)
            logger.info(
                "CALLBACK: token exchange SUCCESS — keys=%s",
                list(token_data.keys()),
            )
            if is_xhr:
                return _cors_response('{"status": "ok"}', 200, "application/json")
            return redirect("/")
        except Exception as e:
            logger.error("CALLBACK: token exchange FAILED — %s", e)
            # Race-condition guard: Plaud fires XHR + browser redirect with
            # the same code.  If the XHR already consumed the code, the
            # browser redirect fails — but tokens are saved.  Re-check.
            import time as _time

            _time.sleep(0.5)
            try:
                check = PlaudOAuthClient(redirect_uri=INAPP_REDIRECT_URI)
                if check.is_authenticated:
                    logger.info("CALLBACK: parallel request already exchanged — OK")
                    if is_xhr:
                        return _cors_response(
                            '{"status": "ok", "already": true}',
                            200,
                            "application/json",
                        )
                    return redirect("/")
            except Exception:
                pass
            safe_msg = escape(str(e))
            detail = (
                "<br><br><b>Check the server terminal for detailed per-strategy "
                "logs showing exactly what was sent and what Plaud returned.</b>"
            )
            if is_xhr:
                return _cors_response(
                    f'{{"error": "{safe_msg}"}}', 200, "application/json"
                )
            return _cors_response(
                _auth_error_page("Token Exchange Failed", str(safe_msg) + detail),
                500,
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

    # ── Notion OAuth routes ───────────────────────────────────────────
    # All Notion OAuth now flows through FastAPI (single redirect URI).
    # The Dash app just redirects to the FastAPI web-authorize endpoint.

    @server.route("/auth/notion")
    def auth_notion_start():
        """Start Notion OAuth — redirect through FastAPI backend."""
        return_to = request.host_url.rstrip("/") + "/"
        return redirect(_notion_api_authorize_url(return_to=return_to))

    @server.route("/auth/notion/callback", methods=["GET", "OPTIONS"])
    def auth_notion_callback():
        """Legacy callback — FastAPI handles this now.

        If Dash somehow receives a callback, forward to success page.
        Also handles the ``?notion_connected=1`` redirect from FastAPI.
        """
        _invalidate_notion_service()
        return redirect("/")

    @server.route("/auth/notion/status")
    def auth_notion_status():
        """Return JSON auth status for AJAX polling."""
        try:
            from src.notion_oauth import NotionOAuthClient

            client = NotionOAuthClient()
            return jsonify(client.token_status)
        except Exception as e:
            return jsonify({"is_authenticated": False, "error": str(e)})


def _invalidate_notion_service():
    """Reset the NotionService singleton so it picks up fresh OAuth tokens."""
    try:
        from src.notion_service import get_notion_service
        svc = get_notion_service()
        svc.invalidate_client()
    except Exception:
        pass


def _auth_error_page(title: str, detail: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><title>Chronos — Auth Error</title></head>
<body style="font-family:-apple-system,sans-serif;text-align:center;padding:60px;
background:#0f172a;color:#e2e8f0;">
<h1 style="color:#ef4444;">&#10060; {title}</h1>
<p>{detail}</p>
<a href="/" style="color:#60a5fa;text-decoration:underline;">Return to Chronos</a>
</body></html>"""


def _register_xray_routes(server):
    """Register Flask API routes for the X-ray Activity Monitor PiP panel."""

    @server.after_request
    def _xray_cors(response):
        """Allow same-origin fetch to X-ray endpoints under self-signed HTTPS."""
        if request.path.startswith("/xray/"):
            response.headers["Access-Control-Allow-Origin"] = request.host_url.rstrip(
                "/"
            )
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @server.route("/xray/api/events")
    def xray_api_events():
        from app_v2.services.xray import get_recent_events
        since = request.args.get("since", 0, type=int)
        events = get_recent_events(200, since_seq=since)
        return jsonify(events)

    @server.route("/xray/api/clear", methods=["POST"])
    def xray_api_clear():
        from app_v2.services.xray import clear_events
        clear_events()
        return jsonify({"ok": True})

    @server.route("/xray/api/costs")
    def xray_api_costs():
        """Return session + historical cost data for the cost ticker."""
        from src.chronos.cost_tracker import get_session_cost, get_cost_summary

        days = request.args.get("days", 30, type=int)
        return jsonify(
            {
                "session": get_session_cost(),
                "historical": get_cost_summary(days=days),
            }
        )

    @server.route("/xray/api/throughput")
    def xray_api_throughput():
        """Return rolling event-rate buckets for the sparkline."""
        from app_v2.services.xray import get_throughput

        buckets = request.args.get("buckets", 30, type=int)
        return jsonify({"buckets": get_throughput(buckets)})


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

    # Register X-ray Activity Monitor routes (standalone window)
    _register_xray_routes(app.server)

    # Set layout
    app.layout = create_layout()

    # Register callbacks
    register_all_callbacks(app)

    # Start auto-sync service in background
    try:
        from src.plaud_oauth import PlaudOAuthClient
        from src.plaud_auto_sync import get_auto_sync

        oauth_client = PlaudOAuthClient()
        should_start_auto_sync, auto_sync_reason = _should_start_embedded_auto_sync()
        if not should_start_auto_sync:
            logger.info("Skipping embedded auto-sync startup: %s", auto_sync_reason)
        elif oauth_client.is_authenticated:
            auto_sync = get_auto_sync()
            auto_sync.start()
            logger.info("Auto-sync service started in background")
        else:
            logger.info(
                "Plaud is not authenticated locally; skipping auto-sync startup"
            )
    except Exception as e:
        logger.warning(f"Could not start auto-sync: {e}")

    # Start background token keepalive
    _start_token_keepalive()

    logger.info("Chronos app v2 initialized")
    return app


def main():
    """Run the app."""
    app = create_app()

    logger.info("Starting Chronos v2 at https://localhost:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
