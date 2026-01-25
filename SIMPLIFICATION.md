# Chronos Simplification Guide

**Goal:** Make Chronos usable, granular, and less confusing — while keeping the power-user depth you love.

---

## Before → After

### Navigation (was 3 radio groups, now 1 clean list)

**Before:**

```
📊 Core        ⚙️ Operations      🔧 System
○ Dashboard    ○ (none)           ○ (none)
○ Search       ○ Controls         ○ Settings
○ Timeline     ○ Plaud Hub        ○ Logs
○ Recordings   ○ Devices
               ○ Workflows
               ○ Webhooks
```

**After:**

```
🏠 Home           ← Quick status + one-click actions
🔍 Search         ← Semantic + temporal queries
📚 Library        ← Browse/manage all recordings
⚡ Pipeline       ← Simple 3-step ingest→process→index
📱 Plaud          ← All Plaud features in one place
⚙️ Settings       ← Config + diagnostics
```

---

### Pipeline (was 8 controls, now 3 clear steps)

**Before:** 4 limit inputs + full limit + recording ID + force + preflight checkbox

**After:**

```
Step 1: FETCH     [Pull from Plaud]     ← Gets your recordings
Step 2: PROCESS   [Run Gemini]          ← Cleans transcripts
Step 3: INDEX     [Push to Qdrant]      ← Makes searchable

Advanced ▼  (collapsed by default)
├─ Custom limits
├─ Single recording override
├─ Force reprocess
└─ Preflight diagnostics
```

---

### Plaud Features (was 4 pages, now 1 tabbed panel)

**Before:** Plaud Hub / Devices / Workflows / Webhooks as separate pages

**After:** Single "Plaud" page with tabs:

```
📱 Plaud Integration
├─ Overview   (connection status, quick stats)
├─ Devices    (NotePin, Note, NotePro management)
├─ Workflows  (AI transcription pipelines)
└─ Webhooks   (async event handling)
```

---

## Key Principles

1. **Progressive disclosure** — Simple first, power-user options hidden in "Advanced"
2. **One place for each thing** — No duplicate paths to the same action
3. **Show state clearly** — Is it connected? How many recordings? What's pending?
4. **Command visibility** — Keep the "command preview" (you like seeing what runs)

---

## What Gets Removed

| Item                           | Reason                                           |
| ------------------------------ | ------------------------------------------------ |
| `archive/` references          | Already deprecated, shouldn't affect active code |
| Legacy `Recording` table usage | Chronos uses `ChronosRecording` exclusively      |
| Pinecone mentions in docs      | We're 100% Qdrant now                            |
| Triple radio navigation        | Replaced with clean single-level nav             |
| Duplicate "process" buttons    | One clear workflow instead                       |

---

## What Gets Kept (The Good Stuff)

- ✅ Command previews before running
- ✅ Latency display in status bar
- ✅ Debug mode for raw payloads
- ✅ Saved searches
- ✅ Granular limits (just collapsed)
- ✅ Force reprocess option
- ✅ All temporal filters (day-of-week, hour, date range)
- ✅ Full Plaud API access (devices, workflows, webhooks)
