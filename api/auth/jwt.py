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


_UNTRUSTED = "0.0.0.0"  # not loopback, not private: forces the API key check


def resolve_client_ip(headers, peer_host: str | None) -> str:
    """Return the address to make trust decisions on.

    X-Forwarded-For is attacker-controlled. It is only honored when the direct
    peer is loopback, i.e. the local ngrok agent, and then only its LAST entry,
    which the proxy appends and the caller cannot forge. The first entry is
    whatever the caller sent. A direct LAN/Tailscale peer's own address wins
    over any header it sends.
    """
    peer = peer_host or "127.0.0.1"
    if not _is_loopback(peer):
        return peer
    xff = headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[-1].strip() or _UNTRUSTED
    # Proxied without an XFF entry: don't let it inherit loopback trust.
    if headers.get("X-Forwarded-Proto") or headers.get("X-Forwarded-Host"):
        return _UNTRUSTED
    return peer


def _get_client_ip(request: Request) -> str:
    return resolve_client_ip(
        request.headers, request.client.host if request.client else None
    )


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
        # Keyless access is only valid when the deployment mode explicitly
        # allowed this client address above. A public proxy/tunnel request must
        # never inherit trusted-LAN behavior merely because no key is set.
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
