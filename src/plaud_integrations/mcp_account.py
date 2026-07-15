"""Official PLAUD MCP adapter with runtime tool discovery and normalization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Sequence
from uuid import uuid4

from .call_ledger import PlaudCallLedger, default_ledger
from .errors import PlaudConfigurationError, PlaudIntegrationError, PlaudUnknownToolError
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
    ):
        self.command = command or os.getenv("PLAUD_MCP_COMMAND") or shutil.which("npx") or "npx"
        self.args = list(args or (os.getenv("PLAUD_MCP_ARGS") or "-y @plaud-ai/mcp@latest --no-login").split())
        self.ledger = ledger
        self._capabilities: dict[str, PlaudIntegrationCapability] = {}

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

    def _discover_stdio(self) -> list[PlaudIntegrationCapability]:
        process = self._stdio_process()
        try:
            initialized, tools = self._stdio_tools(process)
            server_version = (initialized.get("serverInfo") or {}).get("version")
            return self._capabilities_from_stdio(tools, server_version)
        finally:
            self._stop_stdio(process)

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
        process = self._stdio_process()
        started = time.perf_counter()
        try:
            initialized, tools = self._stdio_tools(process)
            discovered = {str(tool.get("name")): tool for tool in tools}
            if tool_name not in REVIEWED_TOOLS or tool_name not in discovered:
                raise PlaudUnknownToolError(tool_name)
            tool = discovered[tool_name]
            schema_hash = payload_hash({"input": tool.get("inputSchema"), "output": tool.get("outputSchema")})
            self._stdio_send(process, {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })
            raw = self._stdio_receive(process, 3)
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
        finally:
            self._stop_stdio(process)

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
        try:
            result = (
                self._run(self._call_async(tool_name, arguments))
                if os.getenv("PLAUD_MCP_USE_PYTHON_SDK") == "1"
                else self._call_stdio(tool_name, arguments)
            )
            status = "error" if result.is_error else "success"
            response = result.structured_content if result.structured_content is not None else result.text_content
            if result.is_error:
                raise PlaudIntegrationError(str(response), code="mcp_tool_error", status_code=502)
            return result
        except Exception as exc:
            result = None
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

    def get_note(self, file_id: str) -> PlaudNote:
        result = self.call_tool("get_note", {"file_id": file_id})
        data = self._payload(result)
        data = data.get("data", data) if isinstance(data, dict) else {}
        return PlaudNote(
            file_id=file_id, markdown=_first(data, "markdown", "text", "content", "summary"),
            action_items=data.get("action_items", []) or [], topics=data.get("topics", []) or [],
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
