# Security Policy

PlaudBlender processes sensitive personal voice recordings, transcripts, metadata, and AI-generated summaries.

## Sensitive Data

Treat the following as private:

- Plaud OAuth credentials
- Plaud access/refresh tokens
- `.env`
- `.plaud_tokens.json`
- `.notion_tokens.json`
- SQLite databases under `data/`
- audio files
- transcripts
- logs
- exported JSON/HTML
- API keys for Gemini, OpenAI, Notion, Qdrant, ngrok, or other services

## Local-First Default

PlaudBlender is intended to run locally by default.

Use caution when enabling:

- ngrok
- public tunnels
- LAN exposure
- remote FastAPI access
- remote Qdrant access
- shared iOS endpoints

## Authentication

For any deployment reachable outside your own machine, configure:

```bash
CHRONOS_API_KEY=<strong-random-token>
CHRONOS_REQUIRE_AUTH=1
```

Do not expose admin/debug routes publicly without authentication.

## Qdrant Exposure

Qdrant should normally bind to localhost.

Prefer:

```yaml
ports:
  - "127.0.0.1:6333:6333"
```

Only bind to `0.0.0.0` if you intentionally want LAN/public access and have appropriate network controls.

## Reporting Issues

If you find a security issue, do not post sensitive exploit details in a public issue.

Open a minimal issue saying a security concern exists, or contact the maintainer privately if contact information is available.

## Secret Rotation

If a credential was ever committed, pasted into logs, shared in screenshots, or exposed through a tunnel:

1. Revoke it.
2. Rotate it.
3. Remove it from local files.
4. Confirm it is not present in Git history.
5. Re-run secret scanning if available.
