# PlaudBlender AI Instructions

## Project Overview

PlaudBlender transforms Plaud voice recordings into a searchable, visual knowledge graph. It integrates Plaud's API, Google's Gemini models (via LlamaIndex), Notion, and Pinecone to process, store, and visualize transcript data.

## Architecture & Data Flow

1.  **Ingestion:** `src/plaud_client.py` fetches recordings/transcripts from Plaud API using OAuth (`src/plaud_oauth.py`).
2.  **Processing:** `src/dual_store_processor.py` uses Gemini to extract themes and summaries.
3.  **Storage (Dual Store):**
    - **Pinecone:** Stores vectors in two namespaces: `full_text` (chunked transcripts) and `summaries` (AI syntheses).
    - **Notion:** `src/notion_client.py` syncs metadata and content to Notion pages.
4.  **Visualization:** `gui.py` provides a Tkinter interface for management and visualization (using `vis.js` in webviews).

## Critical Workflows

- **Setup:** Run `python plaud_setup.py` to handle OAuth authentication and `.env` generation.
- **GUI:** The main entry point is `python gui.py`. It uses `threading` to keep the UI responsive during API calls.
- **Sync:** `scripts/sync_to_pinecone.py` handles the batch synchronization pipeline.
- **Testing:** Use `python test_components.py` for integration tests. This script uses `rich` for formatted console output.

## Coding Conventions

- **Environment Variables:** All secrets (API keys, Client IDs) must be loaded from `.env` using `python-dotenv`. Never hardcode credentials.
- **Logging:** Use the standard `logging` library. Initialize with `logging.basicConfig(level=logging.INFO)` in scripts.
- **UI Development:** `gui.py` uses `tkinter`. Heavy operations must run in separate threads to avoid freezing the main loop.
- **LLM Integration:** Use `llama_index` abstractions for Gemini interactions. Respect token limits when chunking (`MAX_CHUNK_TOKENS = 8000`).

## Key Files

- `gui.py`: Main application logic and UI event handling.
- `src/dual_store_processor.py`: Core logic for AI processing and Pinecone interaction.
- `src/plaud_client.py`: Wrapper for Plaud API endpoints.
- `test_components.py`: Integration tests for individual components (Notion, Gemini, Plaud).

## Process & Checklist

- **Live audit checklist:** `docs/audit-checklist.md` is the single source of truth for UX/feature/infra tasks. Update it as items are completed so contributors and Copilot have current status and next actions.
