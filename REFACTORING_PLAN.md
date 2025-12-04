# PlaudBlender GUI Refactoring Plan

## 1. Current State Analysis
The current `gui.py` is a monolithic "God Class" (`PlaudBlenderGUI`) containing approximately 6000 lines of code. It handles:
- Application initialization and window management.
- UI layout for all tabs (Dashboard, Transcripts, Pinecone, Search, Settings, Logs).
- State management (transcripts, search results, API clients).
- Business logic (syncing, fetching, processing).
- Event handling and threading.

### Issues
- **Maintainability**: Hard to navigate and modify.
- **Coupling**: UI code is tightly coupled with logic and state.
- **Scalability**: Adding new features increases the mess.
- **Readability**: Mixed levels of abstraction.

## 2. Refactoring Goals
- **Modularization**: Break down the monolith into smaller, focused modules.
- **Separation of Concerns**: Separate UI (View) from Logic (Controller/Model).
- **Intuitive Flow**: Redesign the UI to be more user-friendly and intuitive.
- **Retain Functionality**: Ensure all existing features work as expected.

## 3. Proposed Architecture

We will move `gui.py` to a `gui/` package with the following structure:

```
gui/
├── __init__.py
├── main_window.py       # Main application window and layout
├── state.py             # Centralized state management (AppModel)
├── theme.py             # UI styles, colors, and fonts
├── assets/              # Icons and images (if any)
├── components/          # Reusable UI components
│   ├── __init__.py
│   ├── stat_card.py     # Dashboard stat cards
│   ├── status_bar.py    # Bottom status bar
│   └── tooltips.py      # Tooltip helpers
├── tabs/                # Individual tab implementations
│   ├── __init__.py
│   ├── base_tab.py      # Abstract base class for tabs
│   ├── dashboard.py     # Dashboard tab
│   ├── transcripts.py   # Transcripts browser
│   ├── pinecone_tab.py  # Pinecone management
│   ├── search.py        # Search interface
│   ├── settings.py      # Settings and configuration
│   └── logs.py          # Application logs
└── utils/               # GUI helpers
    ├── __init__.py
    ├── async_handler.py # Threading and async task management
    └── dialogs.py       # Custom dialogs
```

## 4. Step-by-Step Implementation Plan

### Phase 1: Setup and Infrastructure
1.  Create the `gui/` directory structure.
2.  Implement `gui/theme.py` to define the look and feel (colors, styles).
3.  Implement `gui/state.py` to hold the application state (clients, data).
4.  Implement `gui/utils/async_handler.py` to handle background tasks.

### Phase 2: Core UI Components
1.  Implement `gui/components/stat_card.py`.
2.  Implement `gui/components/status_bar.py`.
3.  Implement `gui/tabs/base_tab.py` to define the interface for tabs.

### Phase 3: Tab Migration (Iterative)
1.  **Settings Tab**: Move settings logic to `gui/tabs/settings.py`.
2.  **Logs Tab**: Move logging logic to `gui/tabs/logs.py`.
3.  **Dashboard Tab**: Reimplement the dashboard in `gui/tabs/dashboard.py` using the new components.
4.  **Transcripts Tab**: Move transcript browsing and filtering to `gui/tabs/transcripts.py`.
5.  **Pinecone Tab**: Move Pinecone management to `gui/tabs/pinecone_tab.py`.
6.  **Search Tab**: Move search functionality to `gui/tabs/search.py`.

### Phase 4: Main Window and Integration
1.  Implement `gui/main_window.py` to assemble the tabs and manage the main loop.
2.  Update `gui.py` (or create a new entry point) to launch the new `MainWindow`.

### Phase 5: Cleanup and Polish
1.  Verify all functionality.
2.  Remove the old `gui.py` code (or replace it with a redirect).
3.  Ensure intuitive flow (e.g., better navigation between tabs, clear feedback).

## 5. Key Design Changes for "Intuitive Flow"
- **Unified Status**: A global status bar that clearly shows background activity (syncing, loading).
- **Contextual Actions**: Actions (like "Sync to Pinecone") should be available where they make sense (e.g., on the Transcripts tab selection) rather than just in a global menu.
- **Visual Feedback**: Better loading indicators and success/error messages.
- **Simplified Dashboard**: Focus on key metrics and most recent activity.

## 6. Reference: Existing Functionality to Retain
- **Plaud Integration**: OAuth, fetching recordings.
- **Pinecone Integration**: Syncing, deleting, stats, namespace management.
- **Notion Integration**: (Via backend, triggered by GUI).
- **Search**: Semantic search with filters.
- **Visualization**: Mind map generation (calls `visualizer.py`).
- **Settings**: API keys, configuration.
