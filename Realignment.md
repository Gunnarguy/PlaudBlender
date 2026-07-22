# PLAUDBLENDER PUBLIC PLAUD INTEGRATION AUDIT AND MODERNIZATION

## Expert Role

Act as a senior integration engineer specializing in:

- PLAUD public developer APIs
- Model Context Protocol
- OAuth 2.0
- Python and FastAPI
- Swift and SwiftUI
- iOS authentication and Keychain security
- OpenAPI client generation
- Backward-compatible API migrations
- Contract testing and integration diagnostics

You are working inside the existing repository:

`Gunnarguy/PlaudBlender`

The repository contains:

- The Python Chronos backend
- FastAPI routes
- Existing PLAUD OAuth and REST integrations
- An official PLAUD MCP diagnostic script
- A native `PlaudBlenderiOS` SwiftUI application
- Existing timeline, sync, diagnostics, X-Ray, settings, search, and recording interfaces

## Primary Objective

Audit and modernize every **publicly documented** PLAUD REST and MCP integration used by PlaudBlender as of **July 15, 2026**.

Do not redesign or replace the entire application.

Do not remove working legacy behavior until the replacement has demonstrated functional parity.

Do not add beta-builder, NDA-only, private, unpublished, reverse-engineered, or guessed functionality.

The repository must remain safe to keep public.

## Non-Negotiable Rules

1. Audit the current repository before modifying code.
2. Preserve all existing working Chronos behavior.
3. Treat existing PLAUD account REST integration as a compatibility adapter.
4. Add current public PLAUD MCP and Embedded REST integrations beside the compatibility adapter.
5. Do not collapse different PLAUD authentication systems into one client.
6. Do not hardcode credentials, secrets, tokens, domains, device identifiers, or personal infrastructure addresses.
7. Never place a PLAUD client secret, partner secret, API key, refresh token, or webhook secret inside the iOS application.
8. Do not use undocumented endpoints merely because they appear in an old file.
9. Do not invent SDK symbols, endpoint paths, request fields, response fields, or authentication headers.
10. Verify every implemented operation against the current public PLAUD documentation or OpenAPI specification.
11. Dynamically discover MCP tools using `list_tools()`.
12. Build and test after each phase.
13. Do not commit NDA documentation, beta frameworks, private SDK artifacts, or unpublished capabilities.
14. Do not directly modify `main` without first creating a dedicated working branch.
15. Do not delete compatibility code during this task.

## Phase 0: Repository Audit

Before changing anything:

1. Identify the current branch and latest commit.
2. Read the following relevant files and nearby dependencies:
   - `src/config.py`
   - `src/plaud_oauth.py`
   - `src/plaud_client.py`
   - `src/plaud_api_token.py`
   - `scripts/plaud_mcp_doctor.py`
   - `scripts/mcp_server.py`
   - `api/main.py`
   - `api/routes/auth.py`
   - `api/routes/sync.py`
   - `api/routes/settings.py`
   - `PlaudBlenderiOS/PlaudBlenderiOS/Services/APIClient.swift`
   - `PlaudBlenderiOS/PlaudBlenderiOS/Services/AuthManager.swift`
   - `PlaudBlenderiOS/PlaudBlenderiOS/Services/KeychainService.swift`
   - `PlaudBlenderiOS/PlaudBlenderiOS/ViewModels/SettingsViewModel.swift`
   - `PlaudBlenderiOS/PlaudBlenderiOS/Views/Settings/SettingsView.swift`
3. Search the entire repository for:
   - PLAUD endpoint URLs
   - `PLAUD_CLIENT_ID`
   - `PLAUD_CLIENT_SECRET`
   - `PLAUD_API_KEY`
   - OAuth and refresh-token logic
   - `.plaud_tokens.json`
   - `.plaud_api_token.json`
   - `@plaud-ai/mcp`
   - MCP `list_tools`
   - PLAUD request and response schemas
   - duplicated or conflicting PLAUD clients
4. Produce an initial audit table:
   - Current implementation
   - Authentication model
   - Endpoint or MCP tool
   - Publicly documented
   - Working
   - Tested
   - Legacy
   - Missing
   - Security concern
   - Proposed action
5. Run the existing Python tests and build the iOS target before making changes.
6. Record all existing failures separately. Do not attribute pre-existing failures to this work.

## Current Public Capability Baseline

Verify this baseline against PLAUD’s live public documentation before coding.

### Public PLAUD MCP tools

Expected currently documented tools:

- `login`
- `logout`
- `get_current_user`
- `list_files`
- `get_file`
- `get_note`
- `get_transcript`

Do not hardcode the tool count as authoritative.

At runtime:

1. Connect to the official PLAUD MCP implementation.
2. Call `list_tools()`.
3. Store the discovered tool name, description, input schema, output schema when available, and schema hash.
4. Compare the live tool set with the expected baseline.
5. Report additions, removals, or schema changes.
6. Do not automatically invoke newly discovered mutating tools.

### Public Embedded REST operations

Verify the exact regional base URLs, authentication headers, paths, schemas, and availability.

Expected public operations include:

#### Partner authentication

- `POST /oauth/partner/access-token`
- `POST /oauth/partner/access-token/refresh`

#### User token

- `POST /open/partner/users/access-token`

#### File upload

- `POST /open/partner/files/upload/generate-presigned-urls`
- `PUT` file chunks to returned presigned URLs
- `POST /open/partner/files/upload/complete-upload`

#### Transcription

- `POST /open/partner/ai/transcriptions/`
- `GET /open/partner/ai/transcriptions/{transcription_id}`

Confirm whether transcription currently uses:

- `X-Client-Id`
- `X-Client-Api-Key`

Do not assume that the transcription API uses the same bearer token as the file-upload API.

## Target Backend Structure

Introduce a dedicated integration package without breaking the existing import paths:

```text
src/plaud_integrations/
├── __init__.py
├── capability_manifest.py
├── models.py
├── errors.py
├── account_protocol.py
├── legacy_account.py
├── mcp_account.py
├── embedded_auth.py
├── embedded_upload.py
├── transcription.py
├── call_ledger.py
└── redaction.py
```

### Required adapter separation

#### `PlaudLegacyAccountAdapter`

Wrap the existing account REST behavior.

It may continue using the current third-party account REST implementation.

Do not label it deprecated unless PLAUD explicitly labels it deprecated.

Classify it as one of:

- Current public
- Compatibility
- Legacy
- Undocumented in current public documentation
- Awaiting verification

#### `PlaudMCPAccountAdapter`

Use the official PLAUD MCP implementation.

Provide typed application-level methods for:

```python
get_current_user()
list_files(...)
get_file(file_id)
get_note(file_id)
get_transcript(file_id)
```

Normalize MCP responses into stable internal models.

Preserve:

- Raw MCP result
- Structured content
- Text content
- Tool name
- Input payload
- Duration
- Error state
- MCP schema hash

#### `PlaudEmbeddedAuthClient`

Handle:

- Partner-token acquisition
- Partner-token refresh
- User-token issuance
- Regional PLAUD host selection
- Expiration metadata

Partner and client secrets remain backend-only.

#### `PlaudEmbeddedUploadClient`

Handle:

- Presigned multipart URL generation
- Direct part uploads
- Completion call
- Retry and timeout behavior
- ETag or equivalent returned upload metadata
- Progress events
- Correlation IDs

#### `PlaudTranscriptionClient`

Handle:

- Submission
- Polling
- Timeout
- Success
- Failure
- Timestamped segments
- Speaker information when available
- VAD and diarization options
- Language configuration
- Raw response preservation

## Shared Account Protocol

Create an internal interface allowing the legacy REST and MCP account sources to be evaluated side by side:

```python
class PlaudAccountSource(Protocol):
    def get_current_user(self) -> PlaudUser: ...
    def list_files(self, request: PlaudFileListRequest) -> PlaudFilePage: ...
    def get_file(self, file_id: str) -> PlaudFile: ...
    def get_note(self, file_id: str) -> PlaudNote: ...
    def get_transcript(self, file_id: str) -> PlaudTranscript: ...
```

Do not force an adapter to fabricate unsupported fields.

Use optional values and provenance metadata instead.

## Canonical Models

Create typed internal models for at least:

- `PlaudUser`
- `PlaudFile`
- `PlaudFilePage`
- `PlaudNote`
- `PlaudTranscript`
- `PlaudTranscriptSegment`
- `PlaudSpeaker`
- `PlaudUploadSession`
- `PlaudTranscriptionJob`
- `PlaudIntegrationCapability`
- `PlaudIntegrationStatus`
- `PlaudCallEvent`

Every normalized object must preserve provenance:

```python
source_transport
source_operation
source_version
source_payload_hash
retrieved_at
raw_payload_available
```

## Capability Manifest

Generate:

`plaud-capability-manifest.json`

The manifest must contain:

- Generated timestamp
- Documentation source references
- MCP package or remote server information
- Discovered MCP tools
- MCP input schemas
- MCP schema hashes
- REST operation identifiers
- HTTP methods
- Paths
- Authentication model
- Implementation status
- Test status
- Read-only, mutating, destructive, auth, or unknown safety classification
- Legacy adapter status
- Last successful call time
- Last failure
- Source file implementing the operation

The generated manifest, not a manually maintained spreadsheet, becomes the machine-readable source of truth.

## Backend API Routes

Add a new authenticated FastAPI router, preferably:

`api/routes/plaud_integrations.py`

Suggested public-app routes:

```text
GET  /api/v1/plaud/integrations/status
GET  /api/v1/plaud/integrations/capabilities

GET  /api/v1/plaud/mcp/tools
POST /api/v1/plaud/mcp/tools/{tool_name}

POST /api/v1/plaud/embedded/user-token
POST /api/v1/plaud/embedded/uploads/presign
POST /api/v1/plaud/embedded/uploads/complete

POST /api/v1/plaud/embedded/transcriptions
GET  /api/v1/plaud/embedded/transcriptions/{transcription_id}
```

Requirements:

- Protect routes using the existing backend authentication mechanism.
- Never return backend secrets.
- Never return complete refresh tokens.
- Redact sensitive headers and payload fields.
- Restrict arbitrary MCP invocation to reviewed tools.
- Require explicit confirmation for mutating calls.
- Return typed error objects.
- Emit correlation IDs.
- Write every call to the integration call ledger.

## Call Ledger

Create one transport-neutral call model for:

- Existing account REST
- Official MCP
- Embedded REST
- Presigned file upload
- Chronos backend operations

Record:

- Timestamp
- Correlation ID
- Transport
- Operation
- Safe or mutating classification
- Request summary
- Redacted request
- Response status
- Redacted response
- Duration
- Retry count
- Schema hash
- Source version
- Error classification

Do not store raw credentials.

## iOS Application Changes

Do not redesign the full application.

Add a focused PLAUD Platform section to the existing Settings or System interface.

Display:

```text
Account REST       Connected / Error / Unverified
Official MCP       Connected / Error / Unavailable
MCP tools          7 discovered, or actual runtime count
Embedded Auth      Configured / Missing
File Upload        Ready / Missing prerequisites
Transcription      Ready / Missing prerequisites
Region             Current configured region
Last Verified      Timestamp
```

Add navigation to a diagnostic detail screen showing:

- Capabilities
- MCP tools
- Operation status
- Last call
- Latency
- Last error
- Authentication model
- Implementation source

Do not expose:

- Client secret
- Partner secret
- API key
- Refresh token
- Full bearer token

Use the existing:

- `APIClient`
- `ClientNetworkEvent`
- `AuthManager`
- `KeychainService`
- Settings architecture
- X-Ray concepts

Extend the network event model only as needed:

```swift
enum IntegrationTransport: String, Codable, Sendable {
    case chronosREST
    case plaudAccountREST
    case plaudMCP
    case plaudEmbeddedREST
    case plaudUpload
}
```

## Security Corrections

Audit and correct these issues when confirmed:

1. Ensure `.plaud_api_token.json` is ignored by Git.
2. Do not log OAuth authorization codes, callback query strings, access tokens, refresh tokens, secrets, or API keys.
3. Validate OAuth state locally before accepting a callback.
4. Redact sensitive request and response fields before saving call history.
5. Reconcile the Keychain implementation with the security documentation.
6. Prefer `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` for persistent app credentials unless a documented requirement prevents it.
7. Remove hardcoded personal LAN, Tailscale, recovery, or ngrok addresses from public Swift source.
8. Move local endpoint overrides to ignored configuration or local build settings.
9. Verify that public deployment mode cannot silently bypass authentication.
10. Never include private PLAUD builder artifacts in tests or fixtures.

## Tests

Add tests covering:

### Backend unit tests

- Capability manifest generation
- MCP tool discovery
- MCP response normalization
- Unknown MCP tool rejection
- Secret redaction
- OAuth state validation
- Partner token refresh
- User-token issuance
- Multipart upload orchestration
- Transcription polling
- Timeout handling
- Retry handling
- Compatibility adapter behavior

### Contract tests

Use mocked or recorded sanitized fixtures.

Do not commit real personal recordings, real access tokens, private API responses, or NDA data.

### iOS tests

- Integration status decoding
- Capability decoding
- Tool-list decoding
- Redaction display
- Missing feature fallback
- Existing Settings and Sync behavior remains intact
- Existing app navigation does not regress

### Builds

Run:

- Relevant Python tests
- Entire backend test suite when practical
- Swift unit tests
- iOS simulator build
- Physical-device build only when available and required

## Migration Strategy

Use this order:

1. Audit
2. Shared models
3. Capability manifest
4. MCP adapter
5. Embedded auth adapter
6. Embedded upload adapter
7. Transcription adapter
8. FastAPI routes
9. Call ledger
10. iOS status interface
11. Contract tests
12. Side-by-side parity testing
13. Documentation
14. Migration recommendation

Do not switch production ingestion from legacy REST to MCP automatically.

At the end, provide a measured recommendation:

- Keep legacy account REST
- Prefer MCP
- Hybrid selection
- Await missing parity

Support the recommendation with test evidence.

## Required Final Report

At completion, provide:

1. Branch name
2. Final commit SHA
3. Files added
4. Files modified
5. Public PLAUD operations implemented
6. MCP tools discovered at runtime
7. Legacy paths preserved
8. Security issues fixed
9. Tests run
10. Test results
11. Build results
12. Remaining unknowns
13. Documentation assumptions
14. Items intentionally excluded because they were beta, private, unpublished, or unverifiable
15. Recommended next phase

Do not claim completion without build and test evidence.
