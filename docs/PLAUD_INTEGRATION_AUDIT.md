# PLAUD Public Integration Audit

Audit date: July 15, 2026

Working branch: `codex/plaud-public-integration-realignment`
Baseline commit: `66479d0`

## Public documentation verified

- PLAUD MCP: <https://docs.plaud.ai/plaud-mcp-cli/mcp>
- Documentation index: <https://docs.plaud.ai/llms.txt>
- Embedded authentication OpenAPI: <https://docs.plaud.ai/openapi/auth.json>
- File upload OpenAPI: <https://docs.plaud.ai/openapi/file.json>
- Transcription OpenAPI: <https://docs.plaud.ai/openapi/transcription.json>
- Public MCP support article: <https://support.plaud.ai/hc/en-us/articles/57751078986265-Plaud-MCP>

The live MCP package reported server version `0.3.5` and seven tools on July 15, 2026: `login`, `list_files`, `get_file`, `get_note`, `get_transcript`, `get_current_user`, and `logout`. Tool count is not hardcoded as authoritative; runtime `tools/list` discovery supplies names, descriptions, schemas, and schema hashes.

## Initial audit

| Current implementation | Authentication model | Operation | Publicly documented now | Working before change | Tested before change | Classification | Concern | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `src/plaud_oauth.py` + `src/plaud_client.py` | Third-party OAuth and bearer token | Account recordings, notes, transcripts | Not present in the current Embedded or MCP public references | Existing production path | Existing regression coverage | Compatibility; awaiting verification | Mixed with newer PLAUD paths | Preserved behind `PlaudLegacyAccountAdapter` |
| `src/plaud_api_token.py` | Client ID and secret to older API-token endpoint | API token acquisition | Not in current public OpenAPI | Unknown | Limited | Undocumented in current public docs | Token file was not ignored | Preserved; token file now ignored; not used by new clients |
| `src/plaud_workflow.py` | Bearer API token | Older workflow transcription | Not in current public OpenAPI | Existing optional path | Existing workflow tests | Compatibility / awaiting verification | Different host and auth model | Preserved; not promoted as current public API |
| `scripts/plaud_mcp_doctor.py` | Official MCP OAuth | Tool discovery and current user | Yes | Blocked by heavyweight Python MCP import in this environment | No schema capture | Current public | Tool list only; no schema hashes | Uses protocol `tools/list`, captures schemas and hashes |
| No Embedded partner client | HTTP Basic `client_id:secret_key` | Partner token and refresh | Yes | Missing | Missing | Current public | Missing capability | Added backend-only client |
| No Embedded user-token client | Bearer partner access token | User token | Yes | Missing | Missing | Current public | Missing capability | Added backend-only client |
| Existing upload methods in compatibility client | Compatibility bearer token | Upload | Does not match current Embedded upload contract | Existing compatibility path | Partial | Compatibility | Contract/auth ambiguity | Added separate multipart client; preserved old upload path |
| Existing workflow transcription | Older bearer/API-token path | Submit and poll | Current API instead requires `X-Client-Id` + `X-Client-Api-Key` | Existing compatibility path | Partial | Compatibility | Auth systems were easy to conflate | Added separate transcription client |
| iOS Settings Plaud connection | Chronos API + server-side account OAuth | Account connection status | App-specific | Working | Existing tests | Current app behavior | No platform-level diagnostics | Added focused PLAUD Platform section and detail list |

## Implemented public operations

| Operation | Method/tool | Authentication | Safety | Implementation |
| --- | --- | --- | --- | --- |
| Partner token | `POST /oauth/partner/access-token` | HTTP Basic | Auth | `embedded_auth.py` |
| Partner refresh | `POST /oauth/partner/access-token/refresh` | HTTP Basic + form refresh token | Auth | `embedded_auth.py` |
| User token | `POST /open/partner/users/access-token` | Bearer partner token | Auth | `embedded_auth.py` |
| Presign upload | `POST /open/partner/files/upload/generate-presigned-urls` | Bearer user token | Mutating | `embedded_upload.py` |
| Upload part | `PUT <presigned URL>` | Presigned URL | Mutating | `embedded_upload.py` |
| Complete upload | `POST /open/partner/files/upload/complete-upload` | Bearer user token | Mutating | `embedded_upload.py` |
| Submit transcription | `POST /open/partner/ai/transcriptions/` | `X-Client-Id` + `X-Client-Api-Key` | Mutating | `transcription.py` |
| Get transcription | `GET /open/partner/ai/transcriptions/{id}` | `X-Client-Id` + `X-Client-Api-Key` | Read-only | `transcription.py` |
| Official account MCP | Runtime-discovered tools | MCP OAuth | Per discovered/reviewed tool | `mcp_account.py` |

US and Japan are the only regions present in the verified OpenAPI server lists. The regional base URLs are `https://platform-us.plaud.ai/developer/api` and `https://platform-jp.plaud.ai/developer/api`.

## Security corrections

- Added `.plaud_api_token.json` to `.gitignore`.
- OAuth callbacks now require a locally issued, constant-time-compared state before errors or codes are accepted.
- Removed authorization code, state, client ID fragments, and token response bodies from OAuth logs.
- Added recursive ledger and iOS network-preview redaction.
- New routes use the existing `require_auth` dependency. Public proxy/tunnel requests fail closed when `CHRONOS_API_KEY` is absent, including when trusted-LAN mode remains enabled for private clients.
- Embedded partner secret and transcription API key are environment-only backend settings.
- Keychain persistence uses `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`.
- Removed personal LAN, Tailscale IP, and ngrok endpoints from Swift source and executable startup defaults. Local values belong in ignored environment or `LocalOverrides.plist` configuration.

## Verification evidence and baseline limitations

- Baseline backend suite: could not reach collection; the existing virtualenv stalled while importing pytest/Pygments metadata and was stopped after more than two minutes.
- New backend contract suite: 11 tests pass.
- Full backend regression suite: 415 passed and one pre-existing failure remained in `tests/test_config.py::test_resolve_openai_api_key`; the baseline implementation returns a configured key even when `CHRONOS_OPENAI_ENABLED=0`, while the pre-existing test expects `None`. This code was unchanged by the PLAUD work. The run also collected the pre-existing untracked duplicate test files in the working tree; those passed and are not part of this commit.
- Live MCP discovery: succeeds; package server version `0.3.5`, seven tools, not authenticated. Account data calls were intentionally not attempted.
- iOS simulator app build: succeeds.
- Generic iOS device build with signing disabled: succeeds.
- iOS unit-test bundle: compiles after correcting two pre-existing `PipelineStatus` initializer drifts in the existing test file.
- Unit-test execution and UI launch were not performed because no simulator was booted. The available iPhone 17 Pro simulator was left shutdown rather than booted implicitly.
- A pre-existing Swift warning remains at `SettingsViewModel.swift:113` concerning `TokenStatus` actor-isolated decoding in a `@Sendable` closure.

## Migration recommendation

Use a hybrid selection for now:

- Keep compatibility account REST as the production ingestion source.
- Prefer official MCP for new read-only account access once authenticated parity calls confirm files, notes, transcripts, pagination, and error semantics against sanitized fixtures.
- Use Embedded REST only for the documented partner/user-token, multipart upload, and transcription workflows.
- Do not switch ingestion automatically. The legacy account REST documentation status and authenticated MCP parity are still unresolved.

## Intentionally excluded

- PLAUD Embedded binary iOS/Android SDK artifacts and any private or separately licensed framework content.
- Beta-builder, NDA, unpublished, reverse-engineered, or guessed endpoints and fields.
- Audio recording control, transcript editing, and other operations the public MCP documentation says are unavailable.
- Any newly discovered future MCP tool not on the reviewed allowlist. Such tools appear as `discovered-unreviewed` and cannot be invoked by the backend route.

The generated [`plaud-capability-manifest.json`](../plaud-capability-manifest.json) is the machine-readable source of truth. Regenerate it with live MCP discovery using:

```bash
PLAUD_MANIFEST_DISCOVER_MCP=1 python -m src.plaud_integrations.capability_manifest
```
