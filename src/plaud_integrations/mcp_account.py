"""Official PLAUD MCP adapter with runtime tool discovery and normalization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

from .call_ledger import PlaudCallLedger, default_ledger
from .errors import (
    PlaudAuthenticationError,
    PlaudConfigurationError,
    PlaudIntegrationError,
    PlaudUnknownToolError,
)
from .legacy_account import _first
from .models import (
    PlaudCallEvent,
    PlaudFile,
    PlaudFileListRequest,
    PlaudFilePage,
    PlaudIntegrationCapability,
    PlaudNote,
    PlaudSpeaker,
    PlaudTranscript,
    PlaudTranscriptSegment,
    PlaudUser,
    payload_hash,
    utc_now,
)
from .redaction import redact

EXPECTED_PUBLIC_TOOLS = {
    "login": "auth",
    "logout": "auth",
    "get_current_user": "read-only",
    "list_files": "read-only",
    "get_file": "read-only",
    "get_note": "read-only",
    "get_transcript": "read-only",
}
REVIEWED_TOOLS = frozenset(EXPECTED_PUBLIC_TOOLS)
MCP_ACCOUNT_OAUTH_SOURCE = "plaudblender_account_oauth"
MCP_TOKEN_MINIMUM_VALIDITY = timedelta(minutes=2)
MCP_AUTH_STATUS_CACHE_SECONDS = 30


@dataclass(frozen=True)
class MCPAuthenticationStatus:
    available: bool
    authenticated: bool
    state: str
    message: str
    credential_source: str | None = None
    expires_at: str | None = None
    verified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "authenticated": self.authenticated,
            "state": self.state,
            "message": self.message,
            "credential_source": self.credential_source,
            "expires_at": self.expires_at,
            "verified_at": self.verified_at,
        }


@dataclass
class MCPToolResult:
    tool_name: str
    input_payload: dict[str, Any]
    raw_result: Any
    structured_content: Any
    text_content: str
    duration_ms: int
    is_error: bool
    schema_hash: str | None


class PlaudMCPAccountAdapter:
    def __init__(
        self,
        *,
        command: str | None = None,
        args: Sequence[str] | None = None,
        ledger: PlaudCallLedger = default_ledger,
        token_path: Path | None = None,
        account_token_provider: Callable[[], tuple[str, datetime | None]] | None = None,
    ):
        self.command = command or os.getenv("PLAUD_MCP_COMMAND") or shutil.which("npx") or "npx"
        self.args = list(args or (os.getenv("PLAUD_MCP_ARGS") or "-y @plaud-ai/mcp@latest --no-login").split())
        self.ledger = ledger
        self._token_path = token_path or Path.home() / ".plaud" / "tokens-mcp.json"
        self._account_token_provider = account_token_provider
        self._bridge_lock = threading.Lock()
        self._auth_status_lock = threading.Lock()
        self._cached_auth_status: MCPAuthenticationStatus | None = None
        self._auth_status_cached_at = 0.0
        self._capabilities: dict[str, PlaudIntegrationCapability] = {}
        # Persistent stdio session: spawning npx per call cost 4-5 s of
        # node startup + MCP handshake on every request. One long-lived
        # process (guarded by the lock; restarted on failure) makes each
        # call a single JSON-RPC round trip.
        self._stdio_lock = threading.Lock()
        self._stdio_proc: subprocess.Popen | None = None
        self._stdio_discovered: dict[str, dict[str, Any]] = {}
        self._stdio_server_version: str | None = None
        self._stdio_next_id = 10

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    @classmethod
    def _mcp_token_expiry(cls, token: dict[str, Any]) -> datetime | None:
        value = token.get("expires_at")
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
        return cls._parse_datetime(value)

    def _read_mcp_token(self) -> dict[str, Any]:
        try:
            token = json.loads(self._token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return token if isinstance(token, dict) else {}

    def _mcp_token_is_usable(self, token: dict[str, Any]) -> bool:
        if not token.get("access_token"):
            return False
        expires_at = self._mcp_token_expiry(token)
        return expires_at is None or expires_at > datetime.now(timezone.utc) + MCP_TOKEN_MINIMUM_VALIDITY

    def _mcp_command_available(self) -> bool:
        return bool(shutil.which(self.command) or Path(self.command).is_file())

    def _account_oauth_token(self, *, validate: bool) -> tuple[str, datetime | None]:
        try:
            if self._account_token_provider is not None:
                access_token, expires_at = self._account_token_provider()
            else:
                from src.plaud_oauth import PlaudOAuthClient

                client = PlaudOAuthClient()
                access_token = client.ensure_valid_token() if validate else client.get_access_token()
                expires_at = self._parse_datetime(client.token_status.get("expires_at"))
            if not access_token:
                raise ValueError("No Plaud account access token is available")
            return access_token, self._parse_datetime(expires_at)
        except PlaudAuthenticationError:
            raise
        except Exception as exc:
            raise PlaudAuthenticationError(
                "Plaud account authentication is required before MCP can reconnect. Reconnect Plaud and try again."
            ) from exc

    def _write_account_oauth_token(self, access_token: str, expires_at: datetime | None) -> None:
        expiry = expires_at or (datetime.now(timezone.utc) + timedelta(minutes=10))
        payload = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_at": int(expiry.timestamp() * 1000),
            "source": MCP_ACCOUNT_OAUTH_SOURCE,
        }
        temp_name: str | None = None
        try:
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self._token_path.parent, 0o700)
            except OSError:
                pass
            descriptor, temp_name = tempfile.mkstemp(prefix=".tokens-mcp-", dir=str(self._token_path.parent))
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self._token_path)
            temp_name = None
        except OSError as exc:
            raise PlaudIntegrationError(
                "Could not synchronize the Plaud MCP session", code="mcp_token_sync_failed", status_code=500
            ) from exc
        finally:
            if temp_name:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass

    def _clear_auth_status_cache(self) -> None:
        with self._auth_status_lock:
            self._cached_auth_status = None
            self._auth_status_cached_at = 0.0

    def _reset_after_token_change(self) -> None:
        with self._stdio_lock:
            self._reset_stdio()

    def synchronize_account_session(self, *, force: bool = False, validate: bool = False) -> bool:
        """Give the official MCP a current account OAuth access token.

        The MCP receives only the short-lived access token. PlaudBlender keeps
        the refresh token, avoiding competing refresh-token rotation between
        the two clients.
        """
        with self._bridge_lock:
            current = self._read_mcp_token()
            if not force and self._mcp_token_is_usable(current):
                return False
            access_token, expires_at = self._account_oauth_token(validate=validate)
            self._write_account_oauth_token(access_token, expires_at)
            self._reset_after_token_change()
            self._clear_auth_status_cache()
            return True

    @staticmethod
    def _is_authentication_error(error: Any) -> bool:
        message = str(error).lower()
        return any(marker in message for marker in (
            "not authenticated",
            "authentication required",
            "unauthorized",
            "invalid or expired token",
            "invalid token",
            " 401",
            "status 401",
        ))

    def _invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        return (
            self._run(self._call_async(tool_name, arguments))
            if os.getenv("PLAUD_MCP_USE_PYTHON_SDK") == "1"
            else self._call_stdio(tool_name, arguments)
        )

    def _raise_mcp_tool_error(self, result: MCPToolResult) -> None:
        response = result.structured_content if result.structured_content is not None else result.text_content
        if self._is_authentication_error(response):
            raise PlaudAuthenticationError("PLAUD MCP session is not authenticated")
        raise PlaudIntegrationError(str(response), code="mcp_tool_error", status_code=502)

    def _credential_source(self) -> str | None:
        token = self._read_mcp_token()
        if token.get("source") == MCP_ACCOUNT_OAUTH_SOURCE:
            return "Plaud account OAuth"
        if token.get("access_token"):
            return "Official MCP OAuth"
        return None

    def _cache_auth_status(self, status: MCPAuthenticationStatus) -> MCPAuthenticationStatus:
        with self._auth_status_lock:
            self._cached_auth_status = status
            self._auth_status_cached_at = time.monotonic()
        return status

    def authentication_status(self, *, force_refresh: bool = False) -> MCPAuthenticationStatus:
        if not self._mcp_command_available():
            return self._cache_auth_status(MCPAuthenticationStatus(
                available=False,
                authenticated=False,
                state="unavailable",
                message="Official Plaud MCP is unavailable because npx is not installed.",
            ))
        with self._auth_status_lock:
            cached = self._cached_auth_status
            is_fresh = time.monotonic() - self._auth_status_cached_at < MCP_AUTH_STATUS_CACHE_SECONDS
        if cached is not None and is_fresh and not force_refresh:
            return cached

        try:
            self.call_tool("get_current_user")
        except PlaudAuthenticationError:
            return self._cache_auth_status(MCPAuthenticationStatus(
                available=True,
                authenticated=False,
                state="reauthorization_required",
                message="Reconnect Plaud to restore the Official MCP session.",
                credential_source=self._credential_source(),
            ))
        except PlaudIntegrationError as exc:
            return self._cache_auth_status(MCPAuthenticationStatus(
                available=True,
                authenticated=False,
                state="unavailable",
                message=f"Official Plaud MCP is unavailable: {exc}",
                credential_source=self._credential_source(),
            ))
        except Exception:
            return self._cache_auth_status(MCPAuthenticationStatus(
                available=True,
                authenticated=False,
                state="unavailable",
                message="Official Plaud MCP could not be checked. Try reconnecting Plaud.",
                credential_source=self._credential_source(),
            ))

        token = self._read_mcp_token()
        expires_at = self._mcp_token_expiry(token)
        return self._cache_auth_status(MCPAuthenticationStatus(
            available=True,
            authenticated=True,
            state="connected",
            message="Official Plaud MCP is connected and follows the Plaud account session.",
            credential_source=self._credential_source(),
            expires_at=expires_at.isoformat() if expires_at else None,
            verified_at=datetime.now(timezone.utc).isoformat(),
        ))

    def reconnect_from_account_oauth(self) -> MCPAuthenticationStatus:
        self.synchronize_account_session(force=True, validate=True)
        status = self.authentication_status(force_refresh=True)
        if not status.authenticated:
            raise PlaudAuthenticationError(status.message)
        return status

    def _stdio_process(self) -> subprocess.Popen:
        env = os.environ.copy()
        env.setdefault(
            "NPM_CONFIG_CACHE",
            str(os.path.join(tempfile.gettempdir(), "plaud-mcp-npm-cache")),
        )
        return subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    @staticmethod
    def _stdio_send(process: subprocess.Popen, payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise PlaudIntegrationError("PLAUD MCP stdin is unavailable", code="mcp_transport_error")
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()

    @staticmethod
    def _stdio_receive(process: subprocess.Popen, request_id: int, timeout: float = 45) -> dict[str, Any]:
        if process.stdout is None:
            raise PlaudIntegrationError("PLAUD MCP stdout is unavailable", code="mcp_transport_error")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            line = process.stdout.readline()
            if not line:
                break
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if payload.get("id") == request_id:
                if "error" in payload:
                    raise PlaudIntegrationError(
                        f"PLAUD MCP protocol error: {payload['error']}", code="mcp_protocol_error"
                    )
                return payload.get("result", {})
        detail = ""
        if process.poll() is not None and process.stderr is not None:
            detail = process.stderr.read()[-500:]
        raise PlaudIntegrationError(
            f"PLAUD MCP timed out waiting for response{': ' + detail if detail else ''}",
            code="mcp_timeout",
            status_code=504,
            retryable=True,
        )

    def _stdio_initialize(self, process: subprocess.Popen) -> dict[str, Any]:
        self._stdio_send(process, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "PlaudBlender", "version": "1.0"},
            },
        })
        initialized = self._stdio_receive(process, 1)
        self._stdio_send(process, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        return initialized

    def _stdio_tools(self, process: subprocess.Popen) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        initialized = self._stdio_initialize(process)
        self._stdio_send(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        return initialized, self._stdio_receive(process, 2).get("tools", [])

    @staticmethod
    def _stop_stdio(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def _capabilities_from_stdio(
        self, tools: list[dict[str, Any]], server_version: str | None
    ) -> list[PlaudIntegrationCapability]:
        capabilities = []
        for tool in tools:
            name = str(tool.get("name", ""))
            input_schema = tool.get("inputSchema")
            output_schema = tool.get("outputSchema")
            schema_hash = payload_hash({"input": input_schema, "output": output_schema})
            capability = PlaudIntegrationCapability(
                operation_id=f"mcp.{name}", transport="plaud_mcp", authentication_model="MCP OAuth",
                safety=EXPECTED_PUBLIC_TOOLS.get(name, "unknown"),
                implementation_status="implemented" if name in REVIEWED_TOOLS else "discovered-unreviewed",
                test_status="runtime-discovered", source_file="src/plaud_integrations/mcp_account.py",
                tool_name=name, description=tool.get("description"), input_schema=input_schema,
                output_schema=output_schema, schema_hash=schema_hash, discovered_at_runtime=True,
                source_version=server_version,
            )
            self._capabilities[name] = capability
            capabilities.append(capability)
        return capabilities

    def _ensure_stdio(self) -> subprocess.Popen:
        """Return the live persistent process, starting and handshaking it
        if needed. Caller must hold ``_stdio_lock``."""
        if self._stdio_proc is not None and self._stdio_proc.poll() is None:
            return self._stdio_proc
        self._reset_stdio()
        process = self._stdio_process()
        try:
            initialized, tools = self._stdio_tools(process)
        except BaseException:
            self._stop_stdio(process)
            raise
        self._stdio_proc = process
        self._stdio_discovered = {str(tool.get("name")): tool for tool in tools}
        self._stdio_server_version = (initialized.get("serverInfo") or {}).get("version")
        self._stdio_next_id = 10
        return process

    def _reset_stdio(self) -> None:
        """Tear down the persistent process. Caller must hold ``_stdio_lock``."""
        if self._stdio_proc is not None:
            self._stop_stdio(self._stdio_proc)
        self._stdio_proc = None
        self._stdio_discovered = {}

    def _discover_stdio(self) -> list[PlaudIntegrationCapability]:
        with self._stdio_lock:
            self._ensure_stdio()
            return self._capabilities_from_stdio(
                list(self._stdio_discovered.values()), self._stdio_server_version
            )

    @staticmethod
    def _mcp_imports():
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise PlaudConfigurationError("The Python 'mcp' package is required for official PLAUD MCP access") from exc
        return ClientSession, StdioServerParameters, stdio_client

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        result: list[Any] = []
        failure: list[BaseException] = []

        def runner() -> None:
            try:
                result.append(asyncio.run(coro))
            except BaseException as exc:  # pragma: no cover - defensive bridge
                failure.append(exc)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()
        if failure:
            raise failure[0]
        return result[0]

    @staticmethod
    def _schema(tool: Any, camel: str, snake: str) -> dict[str, Any] | None:
        value = getattr(tool, camel, None)
        if value is None:
            value = getattr(tool, snake, None)
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        return value if isinstance(value, dict) else None

    async def _discover_async(self) -> list[PlaudIntegrationCapability]:
        ClientSession, StdioServerParameters, stdio_client = self._mcp_imports()
        params = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                capabilities = []
                for tool in response.tools:
                    input_schema = self._schema(tool, "inputSchema", "input_schema")
                    output_schema = self._schema(tool, "outputSchema", "output_schema")
                    schema_hash = payload_hash({"input": input_schema, "output": output_schema})
                    capability = PlaudIntegrationCapability(
                        operation_id=f"mcp.{tool.name}",
                        transport="plaud_mcp",
                        authentication_model="MCP OAuth",
                        safety=EXPECTED_PUBLIC_TOOLS.get(tool.name, "unknown"),
                        implementation_status="implemented" if tool.name in REVIEWED_TOOLS else "discovered-unreviewed",
                        test_status="runtime-discovered",
                        source_file="src/plaud_integrations/mcp_account.py",
                        tool_name=tool.name,
                        description=getattr(tool, "description", None),
                        input_schema=input_schema,
                        output_schema=output_schema,
                        schema_hash=schema_hash,
                        discovered_at_runtime=True,
                        source_version=None,
                    )
                    self._capabilities[tool.name] = capability
                    capabilities.append(capability)
                return capabilities

    def discover_tools(self) -> list[PlaudIntegrationCapability]:
        if os.getenv("PLAUD_MCP_USE_PYTHON_SDK") == "1":
            return self._run(self._discover_async())
        return self._discover_stdio()

    def _call_stdio(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        started = time.perf_counter()
        with self._stdio_lock:
            # One transparent restart-and-retry: the persistent process (or
            # its upstream MCP session) can die while idle between calls.
            for attempt in (1, 2):
                process = self._ensure_stdio()
                if tool_name not in REVIEWED_TOOLS or tool_name not in self._stdio_discovered:
                    raise PlaudUnknownToolError(tool_name)
                tool = self._stdio_discovered[tool_name]
                schema_hash = payload_hash({"input": tool.get("inputSchema"), "output": tool.get("outputSchema")})
                request_id = self._stdio_next_id
                self._stdio_next_id += 1
                try:
                    self._stdio_send(process, {
                        "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    })
                    # Browser authorization is human-paced. The default 45-second
                    # transport timeout is appropriate for normal tools but can report
                    # a false failure after OAuth has already persisted successfully.
                    response_timeout = 300 if tool_name == "login" else 45
                    raw = self._stdio_receive(process, request_id, timeout=response_timeout)
                except (PlaudIntegrationError, OSError, ValueError):
                    # OSError/ValueError: writing to a process that died
                    # while idle (broken pipe / closed stream).
                    self._reset_stdio()
                    if attempt == 2:
                        raise
                    continue
                content = raw.get("content", []) or []
                text_content = "\n".join(
                    str(item.get("text", item)) for item in content if isinstance(item, dict)
                ).strip()
                structured = raw.get("structuredContent")
                if structured is None and text_content:
                    try:
                        structured = json.loads(text_content)
                    except (TypeError, ValueError):
                        pass
                return MCPToolResult(
                    tool_name=tool_name, input_payload=arguments, raw_result=raw,
                    structured_content=structured, text_content=text_content,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    is_error=bool(raw.get("isError", False)), schema_hash=schema_hash,
                )
        raise PlaudIntegrationError(  # pragma: no cover - loop always returns or raises
            "PLAUD MCP call loop exited unexpectedly", code="mcp_transport_error"
        )

    async def _call_async(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        ClientSession, StdioServerParameters, stdio_client = self._mcp_imports()
        params = StdioServerParameters(command=self.command, args=self.args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                discovered = {tool.name: tool for tool in tools.tools}
                if tool_name not in REVIEWED_TOOLS or tool_name not in discovered:
                    raise PlaudUnknownToolError(tool_name)
                tool = discovered[tool_name]
                input_schema = self._schema(tool, "inputSchema", "input_schema")
                output_schema = self._schema(tool, "outputSchema", "output_schema")
                schema_hash = payload_hash({"input": input_schema, "output": output_schema})
                started = time.perf_counter()
                result = await session.call_tool(tool_name, arguments)
                duration_ms = int((time.perf_counter() - started) * 1000)
                content = getattr(result, "content", []) or []
                text_content = "\n".join(
                    str(getattr(item, "text", item)) for item in content if getattr(item, "text", item)
                ).strip()
                structured = getattr(result, "structuredContent", None)
                if structured is None:
                    structured = getattr(result, "structured_content", None)
                if structured is None and text_content:
                    try:
                        structured = json.loads(text_content)
                    except (TypeError, ValueError):
                        pass
                raw = result.model_dump(mode="json") if hasattr(result, "model_dump") else repr(result)
                return MCPToolResult(
                    tool_name=tool_name,
                    input_payload=arguments,
                    raw_result=raw,
                    structured_content=structured,
                    text_content=text_content,
                    duration_ms=duration_ms,
                    is_error=bool(getattr(result, "isError", getattr(result, "is_error", False))),
                    schema_hash=schema_hash,
                )

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        if tool_name not in REVIEWED_TOOLS:
            raise PlaudUnknownToolError(tool_name)
        correlation_id = str(uuid4())
        arguments = arguments or {}
        started = time.perf_counter()
        error_name: str | None = None
        result: MCPToolResult | None = None
        status = "error"
        response: Any = {"error": "MCP call was not started"}
        try:
            if tool_name not in {"login", "logout"}:
                self.synchronize_account_session()
            try:
                result = self._invoke_tool(tool_name, arguments)
                if result.is_error:
                    self._raise_mcp_tool_error(result)
            except PlaudIntegrationError as exc:
                if tool_name in {"login", "logout"} or not self._is_authentication_error(exc):
                    raise
                # The MCP token may have been revoked independently of its
                # local expiry. Force the account owner to validate/refresh,
                # rebuild the token file, then retry this one safe request.
                self.synchronize_account_session(force=True, validate=True)
                try:
                    result = self._invoke_tool(tool_name, arguments)
                    if result.is_error:
                        self._raise_mcp_tool_error(result)
                except PlaudIntegrationError as retry_error:
                    if self._is_authentication_error(retry_error):
                        raise PlaudAuthenticationError(
                            "PLAUD MCP could not restore its session. Reconnect Plaud and try again."
                        ) from retry_error
                    raise
            status = "success"
            response = result.structured_content if result.structured_content is not None else result.text_content
            return result
        except Exception as exc:
            status = "error"
            response = {"error": str(exc)}
            error_name = type(exc).__name__
            raise
        finally:
            self.ledger.record(
                PlaudCallEvent(
                    timestamp=utc_now().isoformat(),
                    correlation_id=correlation_id,
                    transport="plaud_mcp",
                    operation=tool_name,
                    safety=EXPECTED_PUBLIC_TOOLS.get(tool_name, "unknown"),
                    request_summary=f"MCP tool {tool_name}",
                    redacted_request=redact(arguments),
                    response_status=status,
                    redacted_response=redact(response),
                    duration_ms=(result.duration_ms if result else int((time.perf_counter() - started) * 1000)),
                    schema_hash=result.schema_hash if result else None,
                    source_version="@plaud-ai/mcp@latest",
                    error_classification=error_name,
                )
            )

    @staticmethod
    def _payload(result: MCPToolResult) -> Any:
        return result.structured_content if result.structured_content is not None else {}

    @staticmethod
    def _provenance(result: MCPToolResult) -> dict[str, Any]:
        raw = result.raw_result
        return {
            "source_transport": "plaud_mcp",
            "source_operation": result.tool_name,
            "source_version": "@plaud-ai/mcp@latest",
            "source_payload_hash": payload_hash(raw),
            "raw_payload_available": True,
            "raw_payload": raw,
        }

    def get_current_user(self) -> PlaudUser:
        result = self.call_tool("get_current_user")
        data = self._payload(result)
        data = data.get("data", data) if isinstance(data, dict) else {}
        return PlaudUser(
            id=_first(data, "id", "user_id"), email=data.get("email"), name=_first(data, "name", "nickname"),
            **self._provenance(result),
        )

    def list_files(self, request: PlaudFileListRequest) -> PlaudFilePage:
        args = {key: value for key, value in vars(request).items() if value is not None}
        result = self.call_tool("list_files", args)
        payload = self._payload(result)
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        rows = data if isinstance(data, list) else _first(data, "files", "items", default=[])
        files = [self._normalize_file(row, result) for row in rows if isinstance(row, dict)]
        meta = data if isinstance(data, dict) else {}
        return PlaudFilePage(
            files=files, page=meta.get("page"), page_size=meta.get("page_size"),
            total=_first(meta, "total", "total_count"), next_page=meta.get("next_page"),
            **self._provenance(result),
        )

    def _normalize_file(self, data: dict[str, Any], result: MCPToolResult) -> PlaudFile:
        return PlaudFile(
            id=str(_first(data, "id", "file_id", default="")), name=data.get("name"),
            created_at=data.get("created_at"), start_at=data.get("start_at"),
            duration_ms=data.get("duration"), serial_number=data.get("serial_number"),
            presigned_url=data.get("presigned_url"), **self._provenance(result),
        )

    def get_file(self, file_id: str) -> PlaudFile:
        result = self.call_tool("get_file", {"file_id": file_id})
        data = self._payload(result)
        data = data.get("data", data) if isinstance(data, dict) else {}
        return self._normalize_file(data, result)

    @staticmethod
    def _note_markdown(blocks: Sequence[Any]) -> str | None:
        """Join the markdown out of MCP note blocks (auto_sum_note and friends)."""
        parts = [
            str(block["data_content"])
            for block in blocks
            if isinstance(block, dict) and block.get("data_content")
        ]
        return "\n\n".join(parts) or None

    def get_note(self, file_id: str) -> PlaudNote:
        result = self.call_tool("get_note", {"file_id": file_id})
        data = self._payload(result)
        # The server returns a list of note blocks whose markdown lives under
        # data_content. Dict-shaped responses are kept for older servers.
        markdown: str | None = None
        action_items: list[Any] = []
        topics: list[Any] = []
        if isinstance(data, list):
            markdown = self._note_markdown(data)
        elif isinstance(data, dict):
            inner = data.get("data", data)
            if isinstance(inner, list):
                markdown = self._note_markdown(inner)
            elif isinstance(inner, dict):
                markdown = _first(inner, "markdown", "text", "content", "summary")
                action_items = inner.get("action_items", []) or []
                topics = inner.get("topics", []) or []
        return PlaudNote(
            file_id=file_id, markdown=markdown,
            action_items=action_items, topics=topics,
            **self._provenance(result),
        )

    def get_transcript(self, file_id: str) -> PlaudTranscript:
        result = self.call_tool("get_transcript", {"file_id": file_id})
        data = self._payload(result)
        data = data.get("data", data) if isinstance(data, dict) else {}
        rows = _first(data, "segments", "source_list", "results", default=[]) or []
        segments = []
        for row in rows:
            speaker_id = _first(row, "speaker_id", "speaker") if isinstance(row, dict) else None
            if isinstance(row, dict):
                segments.append(PlaudTranscriptSegment(
                    start_seconds=row.get("start"), end_seconds=row.get("end"), text=row.get("text"),
                    speaker=PlaudSpeaker(id=str(speaker_id), label=str(speaker_id)) if speaker_id else None,
                    language=row.get("language"), language_probability=_first(row, "language_probability", "language_probabilitiy"),
                ))
        return PlaudTranscript(
            file_id=file_id, text=_first(data, "text", "transcript"), language=data.get("language"),
            duration_seconds=data.get("duration"), segments=segments, **self._provenance(result),
        )
