"""
JWT bearer token authentication with deployment-mode safety.
"""

import os
import hmac
import ipaddress

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _get_api_key() -> str:
    """Read the API key from environment."""
    return os.getenv("CHRONOS_API_KEY", "").strip()


def _get_client_ip(request: Request) -> str:
    # Check X-Forwarded-For from proxies
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def _is_loopback(ip_str: str) -> bool:
    if ip_str == "testclient":
        return True
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_loopback
    except ValueError:
        return False


def _is_private_or_tailscale(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        tailscale_net = ipaddress.ip_network("100.64.0.0/10")
        return ip.is_private or ip in tailscale_net
    except ValueError:
        return False


def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Validate the Bearer token based on deployment mode and client IP."""
    mode = os.getenv("CHRONOS_DEPLOYMENT_MODE", "trusted_lan").strip().lower()
    client_ip = _get_client_ip(request)

    # Determine if auth is bypassed based on mode
    bypass_auth = False
    if mode == "loopback":
        if _is_loopback(client_ip):
            bypass_auth = True
    elif mode == "trusted_lan":
        if _is_loopback(client_ip) or _is_private_or_tailscale(client_ip):
            bypass_auth = True

    if bypass_auth:
        return "bypassed"

    api_key = _get_api_key()
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
