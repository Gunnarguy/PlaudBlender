# HOST SERVICE INVENTORY

## Running systemd units (Chronos)
* **`chronos-api.service`**: FastAPI backend server (Port 8000). RSS: ~328.5 MB.
* **`chronos-auto-sync.service`**: Auto-sync daemon (Port 8090 / webhooks / directory polling). RSS: ~317.3 MB.
* **`chronos-ui.service`**: Dash frontend web app (Port 8050). RSS: ~22.8 MB.
* **`chronos-qdrant.service`**: Qdrant Vector database wrapper.
* **`chronos-ngrok.service`**: Local ngrok tunnel orchestrator (Port 4040 API). RSS: ~16.5 MB.

## Running docker containers
1. **`qdrant`** (Image: `qdrant/qdrant:v1.17.1`)
   * *Status*: Up 6 days
   * *Ports*: `127.0.0.1:6333->6333/tcp`
   * *RSS*: ~31.9 MB
2. **`jobscoutos`** (Image: `jobscoutos-jobscoutos`)
   * *Status*: Up 12 hours
   * *Ports*: `127.0.0.1:8787->8787/tcp`
   * *RSS*: ~477.4 MB (Docker proxy uses 5.1 MB)

## Other servers & daemons
* **Ollama Service** (`ollama.service`): Local LLM server (Port 11434). RSS: ~10.1 MB (idle).
* **Racklink Commander** (pid 772): Custom server (Port 5000). RSS: ~7.1 MB.
* **OLED Controller** (pid 275): Display loop. RSS: ~12.3 MB.
* **Fan Controller** (pid 770): Fan hardware loops. RSS: ~5.3 MB.
* **Tailscaled** (`tailscaled.service`): VPN Node agent. RSS: ~39.6 MB.
* **Xvnc-core / VNC** (pid 862): Virtual desktop host (Ports 5900, 5901, 6001). RSS: ~7.9 MB.
