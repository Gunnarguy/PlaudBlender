"""Contract-focused tests for separated public PLAUD integrations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from api.routes import auth as auth_routes
from src.plaud_integrations.call_ledger import PlaudCallLedger
from src.plaud_integrations.capability_manifest import generate_manifest
from src.plaud_integrations.embedded_auth import PlaudEmbeddedAuthClient
from src.plaud_integrations.embedded_upload import PlaudEmbeddedUploadClient
from src.plaud_integrations.errors import PlaudAuthenticationError, PlaudIntegrationError, PlaudUnknownToolError
from src.plaud_integrations.legacy_account import PlaudLegacyAccountAdapter
from src.plaud_integrations.mcp_account import (
    MCP_ACCOUNT_OAUTH_SOURCE,
    MCPToolResult,
    PlaudMCPAccountAdapter,
)
from src.plaud_integrations.models import PlaudFileListRequest
from src.plaud_integrations.redaction import REDACTED, redact
from src.plaud_integrations.transcription import PlaudTranscriptionClient


class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None):
        self._payload = payload or {}
        self.status_code = status
        self.headers = headers or {}
        self.content = json.dumps(self._payload).encode() if self._payload else b""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, *args, **kwargs):
        return self._next("POST", *args, **kwargs)

    def put(self, *args, **kwargs):
        return self._next("PUT", *args, **kwargs)

    def request(self, method, *args, **kwargs):
        return self._next(method, *args, **kwargs)


class PlaudIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ledger = PlaudCallLedger(Path(self.tempdir.name) / "ledger.jsonl")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_capability_manifest_contains_all_documented_operations(self):
        manifest = generate_manifest(ledger=self.ledger)
        operations = {item["operation_id"] for item in manifest["capabilities"]}
        self.assertIn("createPartnerAccessToken", operations)
        self.assertIn("completeUpload", operations)
        self.assertIn("createTranscription", operations)
        self.assertIn("mcp.get_transcript", operations)
        self.assertEqual(manifest["mcp"]["discovery_method"], "ClientSession.list_tools()")

    def test_secret_redaction_is_recursive_and_removes_url_query_secrets(self):
        value = redact({
            "Authorization": "Bearer eyJabcdefghijklmnopqrstuv",
            "nested": {"refresh_token": "secret"},
            "url": "https://example.test/callback?code=abc&safe=yes",
        })
        self.assertEqual(value["Authorization"], REDACTED)
        self.assertEqual(value["nested"]["refresh_token"], REDACTED)
        self.assertIn("code=%5BREDACTED%5D", value["url"])
        self.assertNotIn("abc", value["url"])

    def test_secret_redaction_removes_signed_storage_credentials_from_urls_and_text(self):
        signed_url = (
            "https://storage.example/avatar.jpg?AWSAccessKeyId=temp-id"
            "&Signature=signed-value&x-amz-security-token=session-value"
        )
        value = redact({"url": signed_url, "text": f'{{"avatar":"{signed_url}"}}'})

        for item in value.values():
            self.assertNotIn("temp-id", item)
            self.assertNotIn("signed-value", item)
            self.assertNotIn("session-value", item)
            self.assertTrue(REDACTED in item or "%5BREDACTED%5D" in item)

    def test_oauth_state_must_match_a_locally_issued_value(self):
        auth_routes._plaud_oauth_pending.clear()
        auth_routes._plaud_oauth_pending["expected-state"] = {"source": "mobile", "return_to": ""}
        self.assertEqual(auth_routes._matching_pending_state("expected-state"), "expected-state")
        self.assertIsNone(auth_routes._matching_pending_state("attacker-state"))
        self.assertIsNone(auth_routes._matching_pending_state(""))

    def test_partner_refresh_and_user_token_use_distinct_auth_models(self):
        session = FakeSession([
            FakeResponse({"access_token": "partner", "refresh_token": "new-refresh", "token_type": "bearer", "expires_in": 3600}),
            FakeResponse({"access_token": "user-token", "token_type": "bearer", "expires_in": 86400}),
        ])
        client = PlaudEmbeddedAuthClient(
            "client", "secret", session=session, ledger=self.ledger
        )
        refreshed = client.refresh_partner_token("old-refresh")
        user = client.issue_user_token(refreshed.access_token, "stable-user")
        self.assertEqual(refreshed.refresh_token, "new-refresh")
        self.assertEqual(user.access_token, "user-token")
        refresh_kwargs = session.calls[0][2]
        user_kwargs = session.calls[1][2]
        self.assertEqual(refresh_kwargs["auth"], ("client", "secret"))
        self.assertEqual(user_kwargs["headers"]["Authorization"], "Bearer partner")
        self.assertNotIn("old-refresh", Path(self.ledger.path).read_text())

    def test_multipart_upload_retries_part_and_completes(self):
        session = FakeSession([
            FakeResponse({
                "FileId": "file-1", "UploadId": "upload-1", "ChunkSize": 3,
                "Parts": [{"PartNumber": 1, "PresignedUrl": "https://s3.test/part"}],
            }),
            requests.Timeout("retry"),
            FakeResponse({}, headers={"ETag": '"etag-1"'}),
            FakeResponse({"FileId": "file-1", "FileType": "mp3", "DownloadUrl": "https://s3.test/download", "FileMd5": "abc"}),
        ])
        client = PlaudEmbeddedUploadClient("user", session=session, ledger=self.ledger, max_retries=2)
        upload = client.generate_presigned_urls(3, "mp3")
        part = client.upload_part(upload.parts[0]["PresignedUrl"], b"abc", part_number=1)
        completed = client.complete_upload(upload, [part], "mp3", "abc")
        self.assertEqual(part["ETag"], '"etag-1"')
        self.assertEqual(completed.download_url, "https://s3.test/download")
        self.assertEqual([call[0] for call in session.calls], ["POST", "PUT", "PUT", "POST"])

    def test_transcription_submission_and_success_normalization(self):
        session = FakeSession([
            FakeResponse({"transcription_id": "task-1", "status": "PENDING", "data": {}}),
            FakeResponse({
                "transcription_id": "task-1", "status": "SUCCESS",
                "data": {"text": "Hello", "language": "en", "duration": 2, "results": [
                    {"start": 0, "end": 2, "text": "Hello", "speaker_id": "Speaker 1", "language_probability": "0.94"}
                ]},
            }),
        ])
        client = PlaudTranscriptionClient("client", "api-key", session=session, ledger=self.ledger)
        created = client.submit("https://s3.test/download", diarization=True)
        finished = client.get(created.transcription_id)
        self.assertEqual(created.status, "PENDING")
        self.assertEqual(finished.transcript.text, "Hello")
        self.assertEqual(finished.transcript.segments[0].speaker.label, "Speaker 1")
        self.assertNotIn("api-key", Path(self.ledger.path).read_text())

    def test_transcription_poll_timeout(self):
        class PendingClient(PlaudTranscriptionClient):
            def get(self, transcription_id):
                return self._job({"transcription_id": transcription_id, "status": "PENDING", "data": {}})

        client = PendingClient("client", "api-key", session=FakeSession([]), ledger=self.ledger)
        with self.assertRaises(PlaudIntegrationError) as caught:
            client.poll("task-1", timeout_seconds=0, interval_seconds=0)
        self.assertEqual(caught.exception.code, "transcription_timeout")

    def test_unknown_mcp_tool_is_rejected_before_server_invocation(self):
        with self.assertRaises(PlaudUnknownToolError):
            PlaudMCPAccountAdapter(ledger=self.ledger).call_tool("delete_everything")

    def test_mcp_account_session_bridge_writes_access_token_only(self):
        token_path = Path(self.tempdir.name) / "tokens-mcp.json"
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        adapter = PlaudMCPAccountAdapter(
            ledger=self.ledger,
            token_path=token_path,
            account_token_provider=lambda: ("account-access-token", expiry),
        )

        self.assertTrue(adapter.synchronize_account_session(force=True))

        stored = json.loads(token_path.read_text())
        self.assertEqual(stored["access_token"], "account-access-token")
        self.assertEqual(stored["source"], MCP_ACCOUNT_OAUTH_SOURCE)
        self.assertNotIn("refresh_token", stored)
        self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)

    def test_mcp_retries_unauthenticated_tool_with_account_session(self):
        token_path = Path(self.tempdir.name) / "tokens-mcp.json"
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        adapter = PlaudMCPAccountAdapter(
            ledger=self.ledger,
            token_path=token_path,
            account_token_provider=lambda: ("account-access-token", expiry),
        )
        unauthenticated = MCPToolResult(
            tool_name="list_files", input_payload={}, raw_result={},
            structured_content=None, text_content="Not authenticated. Please login first.",
            duration_ms=1, is_error=True, schema_hash="schema",
        )
        recovered = MCPToolResult(
            tool_name="list_files", input_payload={}, raw_result={},
            structured_content={"data": []}, text_content="", duration_ms=1,
            is_error=False, schema_hash="schema",
        )

        with patch.object(adapter, "_invoke_tool", side_effect=[unauthenticated, recovered]) as invoke:
            result = adapter.call_tool("list_files")

        self.assertEqual(result.structured_content, {"data": []})
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(json.loads(token_path.read_text())["source"], MCP_ACCOUNT_OAUTH_SOURCE)

    def test_mcp_returns_401_when_session_cannot_be_repaired(self):
        token_path = Path(self.tempdir.name) / "tokens-mcp.json"
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        adapter = PlaudMCPAccountAdapter(
            ledger=self.ledger,
            token_path=token_path,
            account_token_provider=lambda: ("account-access-token", expiry),
        )
        unauthenticated = MCPToolResult(
            tool_name="list_files", input_payload={}, raw_result={},
            structured_content=None, text_content="Not authenticated. Please login first.",
            duration_ms=1, is_error=True, schema_hash="schema",
        )

        with patch.object(adapter, "_invoke_tool", return_value=unauthenticated):
            with self.assertRaises(PlaudAuthenticationError) as caught:
                adapter.call_tool("list_files")

        self.assertEqual(caught.exception.status_code, 401)

    def test_mcp_tool_discovery_captures_runtime_schema_and_version(self):
        adapter = PlaudMCPAccountAdapter(ledger=self.ledger)
        process = object()
        discovered_payload = [{
            "name": "get_current_user",
            "description": "Return the authenticated user",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {"type": "object", "properties": {"id": {"type": "string"}}},
        }]
        with patch.object(adapter, "_stdio_process", return_value=process), patch.object(
            adapter, "_stdio_tools",
            return_value=({"serverInfo": {"version": "0.3.5"}}, discovered_payload),
        ), patch.object(adapter, "_stop_stdio"):
            tools = adapter.discover_tools()

        self.assertEqual([tool.tool_name for tool in tools], ["get_current_user"])
        self.assertEqual(tools[0].source_version, "0.3.5")
        self.assertTrue(tools[0].schema_hash)
        self.assertEqual(tools[0].input_schema["type"], "object")

    def test_mcp_response_normalization_preserves_provenance(self):
        result = MCPToolResult(
            tool_name="get_transcript", input_payload={"file_id": "file-1"}, raw_result={"raw": True},
            structured_content={"data": {"text": "Hi", "segments": [{"start": 0, "end": 1, "text": "Hi"}]}},
            text_content="", duration_ms=4, is_error=False, schema_hash="hash",
        )

        class FakeMCP(PlaudMCPAccountAdapter):
            def call_tool(self, tool_name, arguments=None):
                return result

        transcript = FakeMCP(ledger=self.ledger).get_transcript("file-1")
        self.assertEqual(transcript.text, "Hi")
        self.assertEqual(transcript.source_transport, "plaud_mcp")
        self.assertTrue(transcript.raw_payload_available)

    def test_legacy_adapter_keeps_compatibility_behavior(self):
        class LegacyClient:
            def get_user(self):
                return {"data": {"id": "user-1", "email": "u@example.test"}}

            def list_recordings(self, page, page_size):
                return {"data": {"recordings": [{"id": "file-1", "name": "Standup"}], "total": 1}}

        adapter = PlaudLegacyAccountAdapter(LegacyClient(), ledger=self.ledger)
        self.assertEqual(adapter.get_current_user().id, "user-1")
        page = adapter.list_files(PlaudFileListRequest())
        self.assertEqual(page.files[0].name, "Standup")
        self.assertEqual(page.source_version, "compatibility-third-party-rest")
        ledger_text = Path(self.ledger.path).read_text()
        self.assertIn('"transport": "plaud_account_rest"', ledger_text)


if __name__ == "__main__":
    unittest.main()
