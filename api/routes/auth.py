"""OAuth authentication flow endpoints (Plaud + Notion)."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from api.schemas.responses import AuthURLResponse, TokenExchangeRequest, TokenStatusOut

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# In-memory map: OAuth state → source ("mobile" | "web")
_notion_oauth_pending: dict[str, str] = {}


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
async def plaud_authorize():
    """Get Plaud OAuth authorization URL."""
    from src.plaud_oauth import PlaudOAuthClient

    client = PlaudOAuthClient()
    url, state = client.get_authorization_url()
    return AuthURLResponse(auth_url=url, state=state)


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

    client = NotionOAuthClient()
    ts = client.token_status
    settings = get_settings()
    integration_token_present = bool(getattr(settings, "notion_token", None))

    is_authenticated = ts.get("is_authenticated", False) or integration_token_present
    has_access_token = bool(client.access_token) or integration_token_present
    auth_mode = (
        "oauth"
        if bool(client.access_token)
        else ("integration_token" if integration_token_present else "none")
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
    redirect_uri = _notion_redirect_uri(request)
    client = NotionOAuthClient(redirect_uri=redirect_uri)

    url, state = client.get_authorization_url()
    _notion_oauth_pending[state] = "mobile" if mobile else "web"
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
    _notion_oauth_pending[state] = "web"
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
    source = _notion_oauth_pending.pop(state, "mobile")

    if error:
        return _notion_redirect(source, error=error)

    if not code:
        return _notion_redirect(source, error="no_code")

    from src.notion_oauth import NotionOAuthClient

    redirect_uri = _notion_redirect_uri(request)
    client = NotionOAuthClient(redirect_uri=redirect_uri)
    try:
        client.exchange_code_for_token(code=code)
    except Exception as exc:
        return _notion_redirect(source, error=str(exc))

    return _notion_redirect(source, success=True)


# ── helpers ──────────────────────────────────────────────────


def _notion_redirect_uri(request: Request) -> str:
    """Return the single redirect URI (from env, or derived from request)."""
    env_uri = os.getenv("NOTION_REDIRECT_URI")
    if env_uri:
        return env_uri
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/v1/auth/notion/callback"


def _notion_redirect(
    source: str, *, success: bool = False, error: str = ""
) -> RedirectResponse:
    """Build the post-auth redirect for the given source."""
    if source == "web":
        if error:
            return RedirectResponse(url=f"http://localhost:8050/?notion_error={error}")
        return RedirectResponse(url="http://localhost:8050/?notion_connected=1")
    else:
        if error:
            return RedirectResponse(url=f"plaudblender://notion-callback?error={error}")
        return RedirectResponse(url="plaudblender://notion-callback?success=true")


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
