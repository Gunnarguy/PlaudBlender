"""
JWT bearer token authentication.

For v1, we use a simple shared API key (from env) to keep things moving.
A full JWT flow with user accounts can be added later.
"""

import os
import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _get_api_key() -> str:
    """Read the API key from environment."""
    key = os.getenv("CHRONOS_API_KEY", "")
    if not key:
        # In development, allow unauthenticated access
        return ""
    return key


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Validate the Bearer token. Returns the token string.

    If CHRONOS_API_KEY is unset (dev mode), all requests are allowed
    unless CHRONOS_REQUIRE_AUTH=1 is configured.
    """
    api_key = _get_api_key()
    require_auth_env = os.getenv("CHRONOS_REQUIRE_AUTH", "") == "1"

    # Dev mode: no key configured → skip auth (unless forced by CHRONOS_REQUIRE_AUTH=1)
    if not api_key:
        if require_auth_env:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required, but CHRONOS_API_KEY is not configured",
            )
        return "dev"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    if not hmac.compare_digest(credentials.credentials, api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return credentials.credentials
