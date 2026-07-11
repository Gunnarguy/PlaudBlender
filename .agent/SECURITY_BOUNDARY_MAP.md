# SECURITY BOUNDARY MAP

## Authentication Boundaries
* **REST API & Web UI endpoints**: Configured with `CHRONOS_REQUIRE_AUTH` (calls fail with 401 Unauthorized if request headers lack valid `Authorization: Bearer <key>` containing `CHRONOS_API_KEY`).
* **OAuth Callback**: Plaud OAuth redirects trust browser code exchange. Bound to `8090` callback URLs.
* **Webhook endpoints**: Validated via Plaud signature verification.
* **Public Tunnels**: ngrok public access must be guarded by token comparison.

## CORS Origins Map
* Currently allowed origins reflect incoming requests. CORS configurations in API routes must validate exact domain list to prevent cross-origin reflection attacks.
