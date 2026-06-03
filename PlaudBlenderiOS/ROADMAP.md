# Project Roadmap: PlaudBlenderiOS

This document outlines the development phases, current implementation status, outstanding technical debt, and next milestones for the Chronos iOS native client.

---

## 1. Project Status

- **Current Version**: `0.8.0-Beta`
- **Build Status**: Compiling and passing test suites in simulator and target devices.
- **Target Audience**: Gunnar Hostetler (Self-hosted/Developer execution).

---

## 2. Completed Milestones (`v0.1` to `v0.8`)

- [x] **Phase 1: Bootstrapping Foundation**
  - Keychain-safe credentials storage configuration.
  - Multi-endpoint ping probes (WAN, LAN, Tailscale DNS, Tailscale stable IP).
  - Versioned API endpoint paths rewriting middleware in `APIClient`.
- [x] **Phase 2: Core Views Implementation**
  - Timeline listing with calendar heatmaps and expanding cards.
  - Stats view integrated with Swift Charts rendering category breakdowns, hourly summaries, and token costs.
  - Notion sync workspace linking matched recordings to Notion page hooks.
- [x] **Phase 3: Real-time Telemetry & Visualization**
  - cytoscape.js bridge rendered inside WKWebView containers.
  - WebSocket-based telemetry stream mapping JSON payloads directly to an X-Ray logger.
  - OSLog integrations covering connection, security, and rendering operations.

---

## 3. Active Work (`v0.9`)

- [ ] **Data Upload Polish**
  - Validating upload candidate scanning for local recording files.
  - Testing `multipart/form-data` uploads to the `/api/v1/sync/upload/process` routes.
- [ ] **Graph Interaction Upgrades**
  - Enhancing search options within the web graph view to filter nodes on-the-fly.
  - Exposing similarity metrics in the node context sheets.

---

## 4. Planned Improvements (`v1.0+`)

- [ ] **Offline Caching Storage**
  - Integrating SwiftData to cache `DaySummary` and `ChronosRecording` arrays local-first.
  - Adding automatic cache invalidation policies when network connections restore.
- [ ] **Native Push Notifications**
  - Integrating Apple Push Notification service (APNs) payloads.
  - Sending foreground notices when pipeline processing completions or sync alerts trigger.
- [ ] **Shared Extension Supports**
  - Exposing an iOS Share Extension to upload raw voice files directly from the Files or Voice Memos applications.

---

## 5. Technical Debt

- **WKWebView JS Assets Lifecycle**: Cytoscape Javascript libraries are currently loaded from remote CDN references inside the static graph HTML template. This should be replaced with local bundle assets to enable full offline graph rendering.
- **Network Call Retry Backoff**: Retries in `APIClient` are limited to a fixed sleep delay on GET timeout exceptions. This should be refactored to an exponential backoff scheduler.
- **Mock Canvas Previews**: SwiftUI previews for view containers like `GraphContainerView` rely on live internet connections. They should be mocked using static HTML resources.

---

## 6. Release Readiness Checklist

- [ ] Clear all debugger warnings and OSLog compiler notes.
- [ ] Implement local Cytoscape.js file bundling to eliminate CDN dependencies.
- [ ] Validate Keychain access flags to ensure data remains accessible while the device is locked.
- [ ] Verify Alembic SQL migrations on the backend to match all client Pydantic structs.
- [ ] Validate TestFlight configuration parameters.
