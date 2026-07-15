#!/usr/bin/env python3
"""Inspect and drive the official Plaud MCP from this repository.

This script is intentionally separate from PlaudBlender's direct Plaud API
OAuth flow. Use it when you want to verify the official Plaud MCP install,
check auth state, or trigger the MCP login flow for supported AI clients.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plaud_integrations.mcp_account import PlaudMCPAccountAdapter


DEFAULT_MCP_COMMAND = os.getenv("PLAUD_MCP_COMMAND") or shutil.which("npx") or "npx"
DEFAULT_MCP_ARGS = tuple(
    part
    for part in (
        os.getenv("PLAUD_MCP_ARGS") or "-y @plaud-ai/mcp@latest --no-login"
    ).split()
    if part
)


@dataclass
class ToolCallSummary:
    """Small normalized view of an MCP tool result."""

    name: str
    ok: bool
    text: str
    payload: Any


class PlaudMCPDoctor:
    """Thin sync wrapper around the official Plaud MCP stdio server."""

    def __init__(self, command: str = DEFAULT_MCP_COMMAND, args: Sequence[str] = DEFAULT_MCP_ARGS):
        self.command = command
        self.args = list(args)
        self.adapter = PlaudMCPAccountAdapter(command=command, args=args)

    def _server_params(self) -> StdioServerParameters:
        return StdioServerParameters(command=self.command, args=self.args)

    @staticmethod
    def _normalize_text(content: Sequence[Any]) -> str:
        parts: list[str] = []
        for item in content or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _maybe_parse_json(text: str) -> Any:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _format_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, indent=2, sort_keys=True)
        return str(payload)

    def _run(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import queue
        import threading

        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)

        def runner() -> None:
            try:
                result_queue.put((True, asyncio.run(coro)))
            except Exception as exc:  # pragma: no cover - defensive
                result_queue.put((False, exc))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        ok, payload = result_queue.get()
        thread.join()
        if ok:
            return payload
        raise payload

    async def _gather_status_async(self) -> dict[str, Any]:
        async with stdio_client(self._server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_names = [tool.name for tool in tools.tools]

                user_summary = await self._call_tool_async(session, "get_current_user", {})
                return {
                    "command": self.command,
                    "args": self.args,
                    "tools": tool_names,
                    "authenticated": user_summary.ok,
                    "user": user_summary.payload,
                    "user_text": user_summary.text,
                }

    async def _call_tool_async(
        self, session: ClientSession, tool_name: str, args: dict[str, Any] | None = None
    ) -> ToolCallSummary:
        result = await session.call_tool(tool_name, args or {})
        text = self._normalize_text(getattr(result, "content", []))
        payload = getattr(result, "structuredContent", None)
        if payload is None:
            payload = self._maybe_parse_json(text)

        return ToolCallSummary(
            name=tool_name,
            ok=not bool(getattr(result, "isError", False)),
            text=text,
            payload=payload,
        )

    async def _tool_once_async(self, tool_name: str, args: dict[str, Any] | None = None) -> ToolCallSummary:
        async with stdio_client(self._server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await self._call_tool_async(session, tool_name, args)

    def status(self) -> dict[str, Any]:
        capabilities = self.adapter.discover_tools()
        tool_names = [item.tool_name for item in capabilities if item.tool_name]
        try:
            user_summary = self.call_tool("get_current_user")
        except Exception as exc:
            user_summary = ToolCallSummary(
                name="get_current_user", ok=False, text=str(exc), payload=None
            )
        return {
            "command": self.command,
            "args": self.args,
            "tools": tool_names,
            "tool_schemas": {
                item.tool_name: {
                    "description": item.description,
                    "input_schema": item.input_schema,
                    "output_schema": item.output_schema,
                    "schema_hash": item.schema_hash,
                }
                for item in capabilities
                if item.tool_name
            },
            "authenticated": user_summary.ok,
            "user": user_summary.payload,
            "user_text": user_summary.text,
        }

    def call_tool(self, tool_name: str, args: dict[str, Any] | None = None) -> ToolCallSummary:
        result = self.adapter.call_tool(tool_name, args)
        return ToolCallSummary(
            name=result.tool_name,
            ok=not result.is_error,
            text=result.text_content,
            payload=result.structured_content,
        )


def _read_version(command: str) -> str:
    try:
        result = subprocess.run(
            [command, "-v"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        return f"unavailable ({exc})"

    text = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        return f"unavailable ({text or f'exit {result.returncode}'})"
    return text or "unknown"


def _print_status(doctor: PlaudMCPDoctor, as_json: bool) -> int:
    summary = doctor.status()
    versions = {
        "node": _read_version("node"),
        "npx": _read_version("npx"),
    }
    output = {**summary, "versions": versions}

    if as_json:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    print("Plaud MCP status")
    print(f"  command: {summary['command']} {' '.join(summary['args'])}".rstrip())
    print(f"  node:    {versions['node']}")
    print(f"  npx:     {versions['npx']}")
    print(f"  tools:   {', '.join(summary['tools'])}")

    if summary["authenticated"]:
        print("  auth:    authenticated")
        if summary["user"] is not None:
            print("  user:")
            print(_indent_block(PlaudMCPDoctor._format_payload(summary["user"]), "    "))
        elif summary["user_text"]:
            print(f"  user:    {summary['user_text']}")
    else:
        print("  auth:    not authenticated")
        if summary["user_text"]:
            print(f"  detail:  {summary['user_text']}")
        print("  next:    ./venv/bin/python scripts/plaud_mcp_doctor.py --login")

    return 0


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _print_tool_result(result: ToolCallSummary, as_json: bool) -> int:
    output = {
        "tool": result.name,
        "ok": result.ok,
        "text": result.text,
        "payload": result.payload,
    }
    if as_json:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if result.ok else 1

    print(f"Plaud MCP tool: {result.name}")
    print(f"  status: {'ok' if result.ok else 'error'}")
    if result.payload is not None:
        print(_indent_block(PlaudMCPDoctor._format_payload(result.payload), "  "))
    elif result.text:
        print(_indent_block(result.text, "  "))
    return 0 if result.ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="Show install and auth status")
    action.add_argument("--login", action="store_true", help="Trigger the Plaud MCP OAuth login flow")
    action.add_argument("--logout", action="store_true", help="Log out the current Plaud MCP session")
    action.add_argument("--current-user", action="store_true", help="Fetch the authenticated Plaud MCP user")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not shutil.which("node") or not shutil.which("npx"):
        parser.error("Node.js and npx are required for the official Plaud MCP.")

    doctor = PlaudMCPDoctor()

    if args.login:
        result = doctor.call_tool("login")
        code = _print_tool_result(result, args.json)
        if code == 0 and not args.json:
            print("  next: finish the browser authorization, then rerun --status")
        return code

    if args.logout:
        return _print_tool_result(doctor.call_tool("logout"), args.json)

    if args.current_user:
        return _print_tool_result(doctor.call_tool("get_current_user"), args.json)

    return _print_status(doctor, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
