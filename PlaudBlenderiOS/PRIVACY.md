# Privacy Policy: PlaudBlenderiOS

**Last Updated: May 29, 2026**

This document describes how PlaudBlenderiOS collects, processes, and protects user data, voice recordings, credentials, and diagnostic logs.

---

## 1. Privacy Philosophy

PlaudBlenderiOS is built around a **user-owned data philosophy**. The app does not transmit your personal data, audio files, or API credentials to any third-party telemetry, tracking, or advertising networks. You have complete visibility and control over where your data is stored and which external services it interacts with.

---

## 2. Data Boundary Map

```
┌──────────────────────────────────────────────────────────────┐
│                    On-Device Sandbox (Private)               │
│                                                              │
│  - Keychain: Chronos API Keys, server target URL overrides   │
│  - Memory: Transient view states, diagnostic events, caches  │
│  - Cache: Temp audio file streams (deleted after playback)  │
└──────────────────────────────┬───────────────────────────────┘
                               │ User-controlled transit
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                  Self-Hosted API Backend (User-Owned)        │
│                                                              │
│  - PostgreSQL: Day Summaries, transcriptions, telemetry logs  │
│  - Qdrant: Vector embeddings and similarity index            │
│  - Storage: Ingested WAV audio archive files                 │
└──────────────────────────────┬───────────────────────────────┘
                               │ Optional external uplink
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Third-Party Integrations                 │
│                                                              │
│  - Plaud Cloud API: Fetch recording chunks (Auth required)   │
│  - Notion API: Sync pages and summaries (Auth required)      │
│  - Gemini / OpenAI APIs: Generate vector models & RAG context │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. What Data Stays on the Device

- **API Credentials**: Your Chronos access tokens and custom target URLs are stored exclusively inside the device's secure Keychain.
- **Diagnostics log**: Real-time network and telemetry events displayed in the X-Ray monitor are stored in a transient in-memory ring buffer (up to 300 HTTP events and 500 WebSocket events) and are destroyed when the application processes are terminated.
- **Audio Stream Cache**: Audio segments downloaded during playback are stored temporarily within the application's sandbox directory and purged automatically when playback completes.

---

## 4. What Data is Sent to Remote Gateways

- **Transcriptions and Vector Metadata**: Transcripts and event text segments are transmitted to your configured Chronos backend for vector classification and PostgreSQL storage.
- **Audio Uploads**: File candidates selected in the **Data** tab (`DataView`) are sent directly to your self-hosted `/api/v1/sync/upload/process` endpoint.
- **RAG Prompts**: Custom search strings entered in the **Search** interface are sent to the FastAPI backend, which packages them with retrieved context and forwards them to Gemini or OpenAI for summary generation.

---

## 5. Third-Party Integrations

PlaudBlenderiOS links to the following external APIs:
- **Plaud.ai**: To retrieve audio assets from your Plaud voice recorder.
- **Notion.so**: To import notes and push summaries.
- **OpenAI / Google Gemini**: To run transcription, semantic search, and summary prompts.

These integrations are direct API calls initiated by your self-hosted FastAPI server. The iOS app does not talk directly to Plaud, Notion, or LLM servers, routing all communication through your central gateway.

---

## 6. How to Delete or Reset Data

- **API Token Revocation**: Logging out in the Settings tab (`SettingsView`) completely deletes all stored keys and custom URL endpoints from the iOS Keychain.
- **Server Data Purge**: You can wipe backend caches, reset stuck recordings, and delete database records by running the Admin tools located in the **Data** tab (`DataView`).

---

## 7. App Store Privacy Declarations

If distributing this app through TestFlight or private App Store accounts, declare the following data usages:

| Data Type | Declared Usage | Link to User Identity |
|---|---|---|
| **Audio/Voice Recordings** | App Functionality (Sync) | No |
| **Diagnostics / Logs** | Developer Observability | No |
| **Identifiers (API Tokens)**| Secure Authorization | No |
