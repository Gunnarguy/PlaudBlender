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

    Requests require a valid Bearer token matching CHRONOS_API_KEY.
    """
    api_key = _get_api_key()

    # Fail closed: if no key is configured, reject all requests.
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Server is improperly configured (missing CHRONOS_API_KEY)",
        )

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
