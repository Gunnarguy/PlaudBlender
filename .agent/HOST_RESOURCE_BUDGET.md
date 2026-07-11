# HOST RESOURCE BUDGET

## Allocation Overview
* **Physical RAM**: 3,788 MB (3.7 GB)
* **Kernel & OS Baseline**: ~800 MB
* **Other servers (JobscoutOS + VNC + Tailscale + Controllers)**: ~650 MB
* **Ollama (Idle)**: ~10 MB (up to 1,500 MB when active)
* **Emergency Reserve**: 500 MB

## Chronos Allocation Envelopes
* **Qdrant**: 1 GB limit in Docker Compose (Currently using 32 MB idle, spikes up to 400 MB on heavy indexing).
* **Chronos API**: MemoryHigh: 512 MB, MemoryMax: 768 MB. CPUQuota: 100%. (Currently using 328 MB RSS).
* **Chronos UI**: MemoryHigh: 384 MB, MemoryMax: 640 MB. CPUQuota: 80%. (Currently using 23 MB RSS).
* **Chronos auto-sync**: MemoryHigh: 768 MB, MemoryMax: 1100 MB. CPUQuota: 150%. (Currently using 317 MB RSS).
* **Chronos pipeline**: MemoryHigh: 900 MB, MemoryMax: 1300 MB. CPUQuota: 125%.
* **Chronos MCP**: MemoryHigh: 256 MB, MemoryMax: 512 MB. CPUQuota: 50%. (Not running/active).

> [!IMPORTANT]
> The total memory ceilings (max caps) sum to ~5.4 GB, which exceeds the physical 4 GB RAM. We rely on systemd `MemoryHigh` throttling and swap pressure buffer.
