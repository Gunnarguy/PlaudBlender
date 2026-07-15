"""Generate the machine-readable PLAUD public capability manifest."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable

from .call_ledger import PlaudCallLedger, default_ledger
from .mcp_account import EXPECTED_PUBLIC_TOOLS
from .models import PlaudIntegrationCapability

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = ROOT / "plaud-capability-manifest.json"
DOCUMENTATION_SOURCES = [
    "https://docs.plaud.ai/llms.txt",
    "https://docs.plaud.ai/plaud-mcp-cli/mcp",
    "https://docs.plaud.ai/openapi/auth.json",
    "https://docs.plaud.ai/openapi/file.json",
    "https://docs.plaud.ai/openapi/transcription.json",
]


def _rest_capabilities() -> list[PlaudIntegrationCapability]:
    source = "src/plaud_integrations"
    return [
        PlaudIntegrationCapability("createPartnerAccessToken", "plaud_embedded_rest", "HTTP Basic client_id:secret_key", "auth", "implemented", "unit-tested", f"{source}/embedded_auth.py", "POST", "/oauth/partner/access-token"),
        PlaudIntegrationCapability("refreshPartnerAccessToken", "plaud_embedded_rest", "HTTP Basic plus refresh_token form body", "auth", "implemented", "unit-tested", f"{source}/embedded_auth.py", "POST", "/oauth/partner/access-token/refresh"),
        PlaudIntegrationCapability("createUserAccessToken", "plaud_embedded_rest", "Bearer partner access token", "auth", "implemented", "unit-tested", f"{source}/embedded_auth.py", "POST", "/open/partner/users/access-token"),
        PlaudIntegrationCapability("generatePresignedUrls", "plaud_embedded_rest", "Bearer user access token", "mutating", "implemented", "unit-tested", f"{source}/embedded_upload.py", "POST", "/open/partner/files/upload/generate-presigned-urls"),
        PlaudIntegrationCapability("uploadPresignedPart", "plaud_upload", "Presigned URL", "mutating", "implemented", "unit-tested", f"{source}/embedded_upload.py", "PUT", "<runtime presigned URL>"),
        PlaudIntegrationCapability("completeUpload", "plaud_embedded_rest", "Bearer user access token", "mutating", "implemented", "unit-tested", f"{source}/embedded_upload.py", "POST", "/open/partner/files/upload/complete-upload"),
        PlaudIntegrationCapability("createTranscription", "plaud_embedded_rest", "X-Client-Id + X-Client-Api-Key", "mutating", "implemented", "unit-tested", f"{source}/transcription.py", "POST", "/open/partner/ai/transcriptions/"),
        PlaudIntegrationCapability("getTranscription", "plaud_embedded_rest", "X-Client-Id + X-Client-Api-Key", "read-only", "implemented", "unit-tested", f"{source}/transcription.py", "GET", "/open/partner/ai/transcriptions/{transcription_id}"),
        PlaudIntegrationCapability("legacyAccountREST", "plaud_account_rest", "Third-party OAuth bearer token", "read-only", "compatibility-awaiting-public-verification", "regression-tested", f"{source}/legacy_account.py"),
    ]


def _documented_mcp_capabilities() -> list[PlaudIntegrationCapability]:
    return [
        PlaudIntegrationCapability(
            operation_id=f"mcp.{name}", transport="plaud_mcp", authentication_model="MCP OAuth",
            safety=safety, implementation_status="implemented-awaiting-runtime-discovery",
            test_status="mock-contract-tested", source_file="src/plaud_integrations/mcp_account.py",
            tool_name=name, discovered_at_runtime=False,
        )
        for name, safety in EXPECTED_PUBLIC_TOOLS.items()
    ]


def _apply_ledger(capabilities: list[PlaudIntegrationCapability], ledger: PlaudCallLedger) -> None:
    by_operation = {item.operation_id: item for item in capabilities}
    by_operation.update({item.tool_name: item for item in capabilities if item.tool_name})
    for event in ledger.recent(1000):
        capability = by_operation.get(event.get("operation"))
        if not capability:
            continue
        if event.get("error_classification"):
            capability.last_failure = f"{event.get('timestamp')}: {event.get('error_classification')}"
        elif str(event.get("response_status", "")).lower() in {"success", "200", "201", "204"}:
            capability.last_successful_call_time = event.get("timestamp")
        capability.last_latency_ms = event.get("duration_ms")


def generate_manifest(
    discovered_mcp_tools: Iterable[PlaudIntegrationCapability] | None = None,
    *,
    ledger: PlaudCallLedger = default_ledger,
) -> dict:
    documented = _documented_mcp_capabilities()
    if discovered_mcp_tools is not None:
        discovered = list(discovered_mcp_tools)
        discovered_names = {item.tool_name for item in discovered}
        documented = discovered + [item for item in documented if item.tool_name not in discovered_names]
    capabilities = _rest_capabilities() + documented
    _apply_ledger(capabilities, ledger)
    runtime_versions = sorted({
        item.source_version for item in capabilities
        if item.transport == "plaud_mcp" and item.source_version
    })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documentation_sources": DOCUMENTATION_SOURCES,
        "mcp": {
            "package": "@plaud-ai/mcp@latest",
            "remote_server": "https://mcp.plaud.ai/mcp",
            "discovery_method": "ClientSession.list_tools()",
            "runtime_discovery_succeeded": discovered_mcp_tools is not None,
            "discovered_server_versions": runtime_versions,
        },
        "regional_base_urls": {
            "us": "https://platform-us.plaud.ai/developer/api",
            "jp": "https://platform-jp.plaud.ai/developer/api",
        },
        "legacy_adapter_status": "Compatibility; current public documentation status awaiting verification",
        "capabilities": [item.to_dict() for item in capabilities],
    }


def write_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    discovered_mcp_tools: Iterable[PlaudIntegrationCapability] | None = None,
) -> dict:
    manifest = generate_manifest(discovered_mcp_tools)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return generate_manifest()
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return generate_manifest()


if __name__ == "__main__":
    discovered = None
    if os.getenv("PLAUD_MANIFEST_DISCOVER_MCP") == "1":
        from .mcp_account import PlaudMCPAccountAdapter

        discovered = PlaudMCPAccountAdapter().discover_tools()
    write_manifest(discovered_mcp_tools=discovered)
