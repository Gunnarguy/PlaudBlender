"""OAuth authentication flow endpoints (Plaud + Notion)."""

import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response

from api.schemas.responses import AuthURLResponse, TokenExchangeRequest, TokenStatusOut

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# In-memory map: OAuth state → metadata about the flow source/return path.
_plaud_oauth_pending: dict[str, dict[str, str]] = {}
_plaud_oauth_completed: dict[str, dict[str, str]] = {}
_notion_oauth_pending: dict[str, dict[str, str]] = {}


# ── Plaud OAuth ─────────────────────────────────────────────


@router.get("/plaud/status", response_model=TokenStatusOut)
async def plaud_status():
    """Check Plaud authentication status."""
    from src.plaud_oauth import PlaudOAuthClient

    client = PlaudOAuthClient()
    ts = client.token_status
    return TokenStatusOut(
        is_authenticated=ts.get("is_authenticated", False),
        has_access_token=ts.get("has_access_token", False),
        expires_at=ts.get("expires_at"),
        extra={
            k: v
            for k, v in ts.items()
            if k not in ("is_authenticated", "has_access_token", "expires_at")
        },
    )


@router.get("/plaud/authorize", response_model=AuthURLResponse)
async def plaud_authorize(request: Request):
    """Get Plaud OAuth authorization URL.

    Pass ``?mobile=true`` from iOS so the callback redirects back into the app
    after the server exchanges the code. Web callers can omit it and optionally
    provide ``return_to`` to land back on the browser UI.
    """
    from src.plaud_oauth import PlaudOAuthClient

    mobile = request.query_params.get("mobile", "").lower() in ("true", "1")
    return_to = _clean_return_to(request.query_params.get("return_to", ""))
    redirect_uri = _plaud_redirect_uri(request)
    client = PlaudOAuthClient(redirect_uri=redirect_uri)
    url, state = client.get_authorization_url()
    _plaud_oauth_pending[state] = {
        "source": "mobile" if mobile else "web",
        "return_to": return_to,
    }
    return AuthURLResponse(auth_url=url, state=state)


@router.options("/plaud/callback")
async def plaud_callback_options():
    """Plaud may preflight the callback URL before sending the real GET."""
    return Response(status_code=204)


@router.get("/plaud/callback")
async def plaud_callback(
    request: Request, code: str = "", state: str = "", error: str = ""
):
    """Handle Plaud OAuth redirect and bounce back to mobile or web.

    Plaud may hit this route more than once for the same state.  We keep a small
    in-memory completion map so duplicate GETs redirect consistently instead of
    failing after the first successful exchange.
    """
    pending = _plaud_oauth_pending.get(state, {})
    completed = _plaud_oauth_completed.get(state)
    source = pending.get("source") or (completed or {}).get("source", "mobile")
    return_to = pending.get("return_to") or (completed or {}).get("return_to", "")

    if completed:
        if completed.get("status") == "success":
            return _plaud_redirect(source, return_to=return_to, success=True)
        return _plaud_redirect(
            source,
            return_to=return_to,
            error=completed.get("error", "authorization_failed"),
        )

    if error:
        _plaud_oauth_pending.pop(state, None)
        _plaud_oauth_completed[state] = {
            "source": source,
            "return_to": return_to,
            "status": "error",
            "error": error,
        }
        return _plaud_redirect(source, return_to=return_to, error=error)

    if not code:
        callback_error = "no_code"
        _plaud_oauth_pending.pop(state, None)
        _plaud_oauth_completed[state] = {
            "source": source,
            "return_to": return_to,
            "status": "error",
            "error": callback_error,
        }
        return _plaud_redirect(source, return_to=return_to, error=callback_error)

    from src.plaud_oauth import PlaudOAuthClient

    redirect_uri = _plaud_redirect_uri(request)
    client = PlaudOAuthClient(redirect_uri=redirect_uri)
    try:
        client.exchange_code_for_token(code=code, state=state)
    except Exception as exc:
        callback_error = str(exc)
        _plaud_oauth_pending.pop(state, None)
        _plaud_oauth_completed[state] = {
            "source": source,
            "return_to": return_to,
            "status": "error",
            "error": callback_error,
        }
        return _plaud_redirect(source, return_to=return_to, error=callback_error)

    _plaud_oauth_pending.pop(state, None)
    _plaud_oauth_completed[state] = {
        "source": source,
        "return_to": return_to,
        "status": "success",
    }
    return _plaud_redirect(source, return_to=return_to, success=True)


@router.post("/plaud/token", response_model=TokenStatusOut)
async def plaud_token_exchange(body: TokenExchangeRequest):
    """Exchange auth code for Plaud access token."""
    from src.plaud_oauth import PlaudOAuthClient

    client = PlaudOAuthClient()
    try:
        client.exchange_code_for_token(code=body.code, state=body.state)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ts = client.token_status
    return TokenStatusOut(
        is_authenticated=ts.get("is_authenticated", False),
        has_access_token=ts.get("has_access_token", False),
        expires_at=ts.get("expires_at"),
    )


@router.post("/plaud/refresh", response_model=TokenStatusOut)
async def plaud_refresh():
    """Refresh Plaud access token."""
    from src.plaud_oauth import PlaudOAuthClient

    client = PlaudOAuthClient()
    try:
        client.refresh_access_token()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ts = client.token_status
    return TokenStatusOut(
        is_authenticated=ts.get("is_authenticated", False),
        has_access_token=ts.get("has_access_token", False),
        expires_at=ts.get("expires_at"),
    )


# ── Notion OAuth ────────────────────────────────────────────


@router.get("/notion/status", response_model=TokenStatusOut)
async def notion_status():
    """Check Notion authentication status."""
    from src.notion_oauth import NotionOAuthClient
    from src.config import get_settings
    from notion_client import Client

    client = NotionOAuthClient()
    ts = client.token_status
    settings = get_settings()
    integration_token = (getattr(settings, "notion_token", None) or "").strip()
    integration_token_valid = False

    if integration_token:
        try:
            integration_client = Client(auth=integration_token, timeout_ms=5_000)
            integration_client.users.me()
            integration_token_valid = True
        except Exception:
            integration_token_valid = False

    is_authenticated = ts.get("is_authenticated", False) or integration_token_valid
    has_access_token = bool(client.access_token) or integration_token_valid
    auth_mode = (
        "oauth"
        if bool(client.access_token)
        else ("integration_token" if integration_token_valid else "none")
    )

    return TokenStatusOut(
        is_authenticated=is_authenticated,
        has_access_token=has_access_token,
        extra={
            "workspace_name": ts.get("workspace_name"),
            "workspace_id": ts.get("workspace_id"),
            "auth_mode": auth_mode,
        },
    )


@router.get("/notion/authorize", response_model=AuthURLResponse)
async def notion_authorize(request: Request):
    """Get Notion OAuth authorization URL.

    Pass ``?mobile=true`` from iOS or omit for web.  The source is stored
    so the callback knows where to redirect after the token exchange.
    """
    from src.notion_oauth import NotionOAuthClient

    mobile = request.query_params.get("mobile", "").lower() in ("true", "1")
    return_to = _clean_return_to(request.query_params.get("return_to", ""))
    redirect_uri = _notion_redirect_uri(request)
    client = NotionOAuthClient(redirect_uri=redirect_uri)

    url, state = client.get_authorization_url()
    _notion_oauth_pending[state] = {
        "source": "mobile" if mobile else "web",
        "return_to": return_to,
    }
    return AuthURLResponse(auth_url=url, state=state)


@router.get("/notion/web-authorize")
async def notion_web_authorize(request: Request):
    """Browser-redirect entry point for the Dash web UI.

    Instead of returning JSON, this 302-redirects straight to Notion so
    the Dash app can simply link here.
    """
    from src.notion_oauth import NotionOAuthClient

    redirect_uri = _notion_redirect_uri(request)
    client = NotionOAuthClient(redirect_uri=redirect_uri)
    url, state = client.get_authorization_url()
    _notion_oauth_pending[state] = {
        "source": "web",
        "return_to": _clean_return_to(request.query_params.get("return_to", "")),
    }
    return RedirectResponse(url=url)


@router.get("/notion/callback")
async def notion_callback(
    request: Request, code: str = "", state: str = "", error: str = ""
):
    """Handle Notion OAuth redirect.

    After exchanging the code, redirect to:
    - ``plaudblender://`` for iOS (ASWebAuthenticationSession catches it)
    - ``http://localhost:8050/`` for the Dash web UI
    """
    pending = _notion_oauth_pending.pop(state, {})
    source = pending.get("source", "mobile")
    return_to = pending.get("return_to", "")

    if error:
        return _notion_redirect(source, return_to=return_to, error=error)

    if not code:
        return _notion_redirect(source, return_to=return_to, error="no_code")

    from src.notion_oauth import NotionOAuthClient

    redirect_uri = _notion_redirect_uri(request)
    client = NotionOAuthClient(redirect_uri=redirect_uri)
    try:
        client.exchange_code_for_token(code=code)
    except Exception as exc:
        return _notion_redirect(source, return_to=return_to, error=str(exc))

    return _notion_redirect(source, return_to=return_to, success=True)


# ── helpers ──────────────────────────────────────────────────


def _notion_redirect_uri(request: Request) -> str:
    """Return the single redirect URI (from env, or derived from request)."""
    env_uri = os.getenv("NOTION_REDIRECT_URI")
    if env_uri:
        return env_uri
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/v1/auth/notion/callback"


def _plaud_redirect_uri(request: Request) -> str:
    """Return the public Plaud callback URI used by the API OAuth flow."""
    env_uri = os.getenv("PLAUD_API_REDIRECT_URI")
    if env_uri:
        return env_uri
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/v1/auth/plaud/callback"


def _plaud_redirect(
    source: str, *, return_to: str = "", success: bool = False, error: str = ""
) -> RedirectResponse:
    """Build the post-auth redirect for Plaud mobile/web callers."""
    if source == "web":
        base = return_to or "http://localhost:8050/"
        if error:
            return RedirectResponse(url=_append_query(base, {"plaud_error": error}))
        return RedirectResponse(url=_append_query(base, {"plaud_connected": "1"}))
    else:
        if error:
            return RedirectResponse(url=f"plaudblender://plaud-callback?error={error}")
        return RedirectResponse(url="plaudblender://plaud-callback?success=true")


def _notion_redirect(
    source: str, *, return_to: str = "", success: bool = False, error: str = ""
) -> RedirectResponse:
    """Build the post-auth redirect for the given source."""
    if source == "web":
        base = return_to or "http://localhost:8050/"
        if error:
            return RedirectResponse(url=_append_query(base, {"notion_error": error}))
        return RedirectResponse(url=_append_query(base, {"notion_connected": "1"}))
    else:
        if error:
            return RedirectResponse(url=f"plaudblender://notion-callback?error={error}")
        return RedirectResponse(url="plaudblender://notion-callback?success=true")


def _clean_return_to(value: str) -> str:
    """Allow only absolute http/https return URLs for web OAuth redirects."""
    if not value:
        return ""

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit(parsed)


def _append_query(url: str, params: dict[str, str]) -> str:
    """Append query params to a URL without losing existing parameters."""
    parsed = urlsplit(url)
    existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
    existing.update(params)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(existing),
            parsed.fragment,
        )
    )


@router.post("/notion/token", response_model=TokenStatusOut)
async def notion_token_exchange(body: TokenExchangeRequest):
    """Exchange auth code for Notion access token."""
    from src.notion_oauth import NotionOAuthClient

    client = NotionOAuthClient()
    try:
        client.exchange_code_for_token(code=body.code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ts = client.token_status
    return TokenStatusOut(
        is_authenticated=ts.get("is_authenticated", False),
        has_access_token=bool(client.access_token),
        extra={
            "workspace_name": ts.get("workspace_name"),
            "workspace_id": ts.get("workspace_id"),
        },
    )
