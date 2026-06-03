# Portfolio Case Study: Chronos iOS Client

An engineering retrospective on constructing **PlaudBlenderiOS** (Chronos Mobile)—a native SwiftUI interface for managing voice recording knowledge ingestion, similarity vector graphs, and WebSocket diagnostics.

---

## 1. Problem Space

The desktop version of Chronos operates as a monolithic Python/Dash dashboard running on a local workstation. While powerful, it requires the developer to be at their desk to trigger ingestion, search semantic records, or review connection status. 

To make Chronos genuinely useful for ambient knowledge gathering, it needed a **mobile companion application** that could:
1. Sync and upload voice files captured on-the-go by Plaud hardware.
2. Search semantic databases and generate RAG-enriched summaries with low latency.
3. Render complex entity-relationship graphs on mobile screens.
4. Provide raw, developer-grade observability of backend service states.

---

## 2. Technical Constraints

- **Dynamic Connectivity Bounds**: The user’s mobile device transitions between home Wi-Fi networks (with local Pi IP routers), cellular data (routing over public ngrok tunnels), and the secure corporate tailnet (relying on Tailscale IP subnets).
- **Layout Performance**: Native Swift graph libraries lack layout-optimization algorithms (like cose or concentric) capable of displaying hundreds of entity relationships without choking the main thread.
- **Resource Constraints**: High-frequency telemetry polling strains iOS battery lifespans and cellular bandwidth allowances.
- **State Integrity**: Allocating ViewModels dynamically inside SwiftUI view structures risks recursion loops and runtime crashes.

---

## 3. Architecture Overview

To resolve these constraints, PlaudBlenderiOS adopts an **isolated Model-View-ViewModel-Service (MVVM-S)** framework:

```
                  ┌────────────────────────────────┐
                  │          SwiftUI Views         │
                  └───────────────┬────────────────┘
                                  │ Reactive bindings
                                  ▼
                  ┌────────────────────────────────┐
                  │    Observable ViewModels       │
                  └───────────────┬────────────────┘
                                  │ Environment context
                                  ▼
                  ┌────────────────────────────────┐
                  │   Network & Telemetry Services │
                  └───────────────┬────────────────┘
                                  │ HTTPS / WebSocket
                                  ▼
                  ┌────────────────────────────────┐
                  │       FastAPI Backend API      │
                  └────────────────────────────────┘
```

---

## 4. Key Technical Challenges & Solutions

### Challenge 1: Multi-Network Environment Resolution
Mobile devices change networks frequently. Bindiing the app to a single hardcoded server URL means the user loses access the moment they step out of range of their home Wi-Fi or turn off their Tailscale VPN.

* **Solution**: `AuthManager` resolves a prioritized list of host URL candidates. On startup or connection drop, `APIClient` executes a sequential health probe (`/api/v1/health` with a 5-second timeout) to dynamically lock onto the fastest available route:
  1. Stored User Override
  2. Info.plist default config
  3. Tailscale MagicDNS (`your-device.your-tailnet.ts.net`)
  4. Tailscale stable IP (`100.x.y.z`)
  5. Public ngrok tunnel (`your-ngrok-domain.ngrok-free.dev`)
  6. Home LAN IP (`10.x.y.z`)

---

### Challenge 2: Mobile Knowledge Graph Rendering
Drawing entity maps containing hundreds of nodes and connections in native SwiftUI code degrades rendering frames, causing choppy scrolling and lags.

* **Solution**: Integrated a `WKWebView` wrapper ([GraphContainerView.swift](PlaudBlenderiOS/Views/Graph/GraphContainerView.swift)) that bundles Cytoscape.js. The iOS client fetches JSON nodes from the `/api/v1/graph` endpoint and sends them across the WebKit Javascript bridge. This isolates graph layout math (cose, concentric, grid) to WebKit’s native rendering threads, keeping the parent SwiftUI app responsive.

---

### Challenge 3: Eliminating "Modifying State During View Update" Warnings
SwiftUI logs runtime errors and crashes if views modify observable states during layout calculations.

* **Solution**: Isolated ViewModel instantiation inside a plain, non-observable class (`ViewModelCache`) declared as a state variable in the root ContentView container. Because the cache class itself is not observed, caching ViewModels during the layout evaluation phase does not trigger SwiftUI state re-evaluation loops.

---

### Challenge 4: High-Frequency Telemetry Ingestion
Monitoring live sync pipeline metrics via typical REST HTTP polling calls saturates network bandwidth and triggers request queue blocks.

* **Solution**: Swapped HTTP polling for a persistent WebSocket service ([WebSocketManager.swift](PlaudBlenderiOS/Services/WebSocketManager.swift)). The client establishes a single WSS connection to `/api/v1/xray/ws`, consuming structured JSON events and feeding them to an in-memory ring buffer.

---

## 5. Architectural Tradeoffs

- **WebKit Overhead vs. Swift Charts Complexity**: Choosing a WKWebView bridge over a native SwiftUI Canvas layout saves weeks of development time and reuses desktop style scripts. However, it requires spinning up web thread contexts and increases the overall memory footprint of the application.
- **Dynamic Probing Timeout vs. Initial Lags**: Sequential health checks prevent connection loss across networks. However, if multiple routes are down, the user may experience an initial 1.5s lag while the app probes fallback routes.
- **No Local Caching vs. Direct Live Sync**: Storing all timeline items in transient memory avoids complex SQLite synchronizations on-device, but prevents offline usage if the server is unreachable.

---

## 6. Outcome and Metrics

The finalized PlaudBlenderiOS client demonstrates:
- **7 Integrated API Routers**: Seamless communication across Timeline, Search, Stats, Graph, Sync, System, and Notion backend targets.
- **Multi-network Resiliency**: Auto-recovery times under 2 seconds during network handovers.
- **Redacted Logging**: Complete sanitization of keys and Authorization tokens.
- **Modular Design**: View and service boundaries are clean, maintaining 100% build compatibility with the latest Xcode tools.

---

## 7. Future Roadmap

1. **SwiftData Caching Integration**: Refactor the database client to cache timeline entries locally, supporting offline review.
2. **Local Graph Library Migration**: Transition from WKWebView to a native SwiftUI graph renderer once iOS layouts mature, eliminating Javascript bridging overhead.
