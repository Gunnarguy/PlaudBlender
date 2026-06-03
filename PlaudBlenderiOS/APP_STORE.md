# App Store & TestFlight Guide: PlaudBlenderiOS

This document provides setup checklists, TestFlight distribution details, and App Store review configurations for PlaudBlenderiOS.

---

## 1. App Store Metadata Reference

| Field | Value | Max Length |
|---|---|---:|
| **App Name** | Chronos Timeline Client | 30 chars |
| **Subtitle** | AI Voice Recording Timeline | 30 chars |
| **Primary Category** | Productivity | — |
| **Secondary Category**| Developer Tools | — |
| **Description** | A native SwiftUI client for Chronos, translating voice recordings into semantic timelines, analytics charts, and connection graphs. *Note: Requires a self-hosted Chronos backend server.* | 4000 chars |

---

## 2. TestFlight Configuration Checklist

When distributing beta builds via TestFlight:
- [ ] **Xcode Target Bundles**: Ensure the bundle version matches the main repository tag (e.g., `0.8.0`).
- [ ] **Provisioning Profiles**: Set up development profiles matching your Apple Developer ID team.
- [ ] **TestFlight Testing Group**: Add developer testers to your App Store Connect tester lists.
- [ ] **Preflight Connection Flag**: Enable the loopback test argument (`-PlaudBlenderAllowLoopbackServer`) in local configurations if debugging with a local simulator.

---

## 3. App Reviewer Guidance

Because PlaudBlenderiOS connects to a user-configured server rather than a central public database, Apple's reviewers require specific guidelines and testing credentials to bypass initial blocks.

### Step-by-Step Reviewer Configuration
1. **Provide a Demo Server Endpoint**: Deploy a public demo server (using the FastAPI server on Render/Railway or an ngrok tunnel) populated with mock recording data.
2. **Review Notes Instructions**: Add the following note in the App Store Connect submission details:
   > "This application is a native client designed to interact with a self-hosted voice recording sync engine. To facilitate testing, we have prepared a public demo server. Please input the credentials below on the Settings screen."
3. **Reviewer Credentials**:
   - **Server URL**: `https://demo-chronos.ngrok-free.dev`
   - **Chronos API Key**: `reviewer_token_12345`

---

## 4. App Store Review Checklist

Ensure the app passes the standard validation rules before submitting:
- [ ] **IPv6 Reachability**: Ensure the URLSession queries in `APIClient.swift` work over standard IPv6-only networks.
- [ ] **Keychain Fallbacks**: Verify the app does not crash if the Keychain is empty.
- [ ] **WKWebView Handling**: Ensure the web-view elements gracefully render an offline message if Cytoscape resources fail to load.
