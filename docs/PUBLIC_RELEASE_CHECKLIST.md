# Public Release Checklist

Use this checklist before posting PlaudBlender publicly.

## 1. Secrets and Identity
- [ ] Confirm `.env`, token files, local DBs, and logs are not tracked.
- [ ] Keep all API keys and OAuth secrets local-only.
- [ ] Verify docs and source headers do not include personal identifiers.
- [ ] Ensure examples use placeholders, never live endpoints.

## 2. Security Defaults
- [ ] Set `CHRONOS_REQUIRE_AUTH=1` in deployed environments.
- [ ] Use a strong `CHRONOS_API_KEY` for any shared endpoint.
- [ ] Do not expose admin/debug routes publicly without auth.
- [ ] Limit remote access to trusted tunnels or private networks.

## 3. Reproducibility
- [ ] Follow `README.md` quickstart from a clean clone.
- [ ] Run tests: `./venv/bin/python -m pytest tests/ -q`.
- [ ] Confirm iOS local overrides use user-owned endpoints only.
- [ ] Verify no machine-specific paths are required in docs.

## 4. Release Post Content
- [ ] Share only the primary monorepo.
- [ ] State clearly: bring your own Plaud/OpenAI/Gemini/Notion credentials.
- [ ] Mention known performance limits (upstream API/model latency).
- [ ] Include setup link and contribution/security docs.

## 5. Optional Hardening
- [ ] Rotate any credentials that were ever copied to test files or terminals.
- [ ] Enable secret scanning in GitHub repository settings.
- [ ] Add branch protection for `main` with required CI checks.

## 6. Narrative Consistency
- [ ] Root README says Qdrant is the current vector store.
- [ ] Pinecone appears only in clearly marked historical/migration contexts.
- [ ] Notion is described as an optional bridge/import/sync layer, not stale architecture.
- [ ] Discord quickstart exists and frames the repo as developer/power-user project.
- [ ] GitHub repo description does not mention Pinecone.

## 7. Discord Post Readiness
- [ ] Include repo link.
- [ ] State BYO credentials.
- [ ] State this is not a polished consumer app.
- [ ] Mention optional Notion bridge.
- [ ] Mention Qdrant as current vector store.
- [ ] Avoid implying nontechnical plug-and-play setup.

