# DEPLOYMENT TOPOLOGY

## Network Exposures & Listeners
* **FastAPI Backend (8000)**: Bound to `0.0.0.0:8000`. Exposes REST API to LAN, Tailscale, and public ngrok tunnels (iOS client uses this).
* **Dash UI (8050)**: Bound to `0.0.0.0:8050`. Exposes web layout to LAN, Tailscale, and ngrok.
* **Plaud Webhook (8090)**: Bound to `0.0.0.0:8090`. Exposes webhook receiver to Plaud OAuth callbacks and events.
* **Qdrant DB (6333)**: Bound to `127.0.0.1:6333`. Local access only; not accessible from LAN or Tailscale directly.
* **ngrok API (4040)**: Bound to `127.0.0.1:4040`. Local control plane only.
* **SSH (22)**: Bound to `0.0.0.0:22` (LAN) and Tailscale SSH.
* **VNC (5900/5901)**: Bound to `0.0.0.0`. Remote desktop access.

## Tunnels & Proxies
* **ngrok Tunnels**: Public URLs are mapped to ports 8000, 8050, and 8090.
* **Tailscale**: Node `gunzino` provides private authenticated network access via Tailscale IP `100.76.130.109`.
