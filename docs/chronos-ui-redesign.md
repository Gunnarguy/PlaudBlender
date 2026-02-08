# Chronos UI Redesign - Architecture Document

## Problem Statement

Current UI shows 571 extracted entities in a graph visualization. This is academic, not practical.

**User's Reality:**

- Continuous life capture via Plaud recordings
- 5-hour recording limit splits longer sessions
- Mix of work, personal, random conversations
- Hours of data every day
- Need to find specific moments, trace topics, understand patterns

## Design Philosophy

1. **Recording-Centric**: Start from the source (recordings), not extracted entities
2. **Time as Primary Axis**: Days → Recordings → Events (not random entity soup)
3. **Surface What Matters**: Categories, keywords, important moments
4. **Contextual Everything**: Always show when/where something happened
5. **Progressive Disclosure**: Overview first, drill down on demand

## Data Model (Current State)

```
Qdrant Events (163 total):
├── recording_id (13 unique recordings)
├── start_ts, end_ts, timestamp
├── day_of_week, hour_of_day
├── clean_text (narrative)
├── category (work, personal, reflection, idea, meeting, deep_work, break)
├── sentiment (-1 to 1)
├── keywords (list)
├── speaker (conversation, monologue, etc.)
└── duration_seconds

SQLite: Empty (need to aggregate from events)
```

## UI Architecture

### Primary Views

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CHRONOS                    🔍 Search...              [Settings] [Sync] │
├─────────┬───────────────────────────────────────────────────────────────┤
│         │                                                               │
│  NAV    │  MAIN CONTENT AREA                                            │
│         │                                                               │
│ 📅 Days │  (Switches based on nav selection)                            │
│ 🎙️ Recs │                                                               │
│ 💡 Topics│                                                               │
│ 📊 Stats │                                                               │
│         │                                                               │
└─────────┴───────────────────────────────────────────────────────────────┘
```

### 1. Day View (Primary)

Shows recordings grouped by day with aggregate stats.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ October 2025                                     ◀ ────────────── ▶     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ 📅 Wednesday, Oct 29                              6.2 hours total   │ │
│ │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │ │
│ │                                                                     │ │
│ │ ┌─────────────────────────────────┐ ┌─────────────────────────────┐ │ │
│ │ │ 🎙️ Recording 1 (5:00)          │ │ 🎙️ Recording 2 (1:12)       │ │ │
│ │ │ 8:05 AM - 1:05 PM              │ │ 1:30 PM - 2:42 PM           │ │ │
│ │ │ ████████ work ███ meeting      │ │ █████ work ██ personal      │ │ │
│ │ │ 11 events                       │ │ 7 events                    │ │ │
│ │ │ Topics: Q4 planning, budget... │ │ Topics: 1:1 with manager... │ │ │
│ │ └─────────────────────────────────┘ └─────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ 📅 Tuesday, Oct 28                               4.1 hours total    │ │
│ │ ...                                                                 │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2. Recording Detail View

When you click a recording, expand to show all events.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ← Back to Oct 29                                                        │
│                                                                         │
│ 🎙️ Recording: Oct 29, 8:05 AM - 1:05 PM (5:00:00)                       │
│                                                                         │
│ Category Breakdown:                                                     │
│ [████████████ work 68%] [████ meeting 20%] [██ break 12%]              │
│                                                                         │
│ Keywords: Q4 planning, budget review, team sync, project deadline       │
│                                                                         │
│ ─────────────────────────────────────────────────────────────────────── │
│                                                                         │
│ TIMELINE                                                                │
│ 8:05 ──●────────●────────────●──────●──────●────────────●──── 1:05     │
│        │        │            │      │      │            │               │
│                                                                         │
│ EVENTS (11)                                                             │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ 8:05 AM - 8:12 AM │ meeting │ sentiment: -0.5                     │   │
│ │ Discussed workplace dynamics and staff changes. We talked about   │   │
│ │ Davina and how she only started to...                             │   │
│ │ Keywords: staffing, workplace culture, process frustration        │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ 8:15 AM - 8:30 AM │ work │ sentiment: 0.2                         │   │
│ │ ...                                                                │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3. Topic Timeline View

Track a specific topic/keyword across all recordings.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 💡 Topic: "remote job"                                                  │
│                                                                         │
│ Mentioned in 4 recordings across 3 days                                 │
│                                                                         │
│ TIMELINE                                                                │
│ Oct 24 ─────●─────────────────●────── Oct 29                           │
│             │                 │                                         │
│             Oct 27 ───●───●───                                          │
│                                                                         │
│ OCCURRENCES                                                             │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ Oct 29, 10:45 AM │ Recording 1                                    │   │
│ │ "...thinking about the remote job search again. Need to update    │   │
│ │ my resume and start applying more seriously..."                   │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ Oct 27, 3:20 PM │ Recording 2                                     │   │
│ │ "...remote job market seems tough right now but I need to..."     │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4. Search Results View

Search shows results with full context.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🔍 "salary negotiation"                                    3 results    │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ 📅 Oct 27 │ 🎙️ Recording 3 │ 2:45 PM │ 92% match                  │   │
│ │ "...preparing for the salary negotiation conversation. I need     │   │
│ │ to research market rates and document my accomplishments..."      │   │
│ │ [View in Context] [Play Audio]                                    │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │ 📅 Oct 24 │ 🎙️ Recording 1 │ 11:30 AM │ 78% match                 │   │
│ │ "...the salary negotiation went better than expected. They        │   │
│ │ offered a 15% increase which was..."                              │   │
│ │ [View in Context] [Play Audio]                                    │   │
│ └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Structure

```
app_v2/
├── main.py                 # Entry point
├── layout.py               # Main app layout with sidebar nav
├── callbacks/
│   ├── __init__.py
│   ├── navigation.py       # View switching
│   ├── day_view.py         # Day expansion, recording selection
│   ├── recording_view.py   # Recording detail interactions
│   ├── search.py           # Search and filtering
│   └── sync.py             # Plaud sync operations
├── components/
│   ├── __init__.py
│   ├── sidebar.py          # Navigation sidebar
│   ├── day_card.py         # Collapsible day summary
│   ├── recording_card.py   # Recording summary in day view
│   ├── recording_detail.py # Full recording with events
│   ├── event_card.py       # Individual event display
│   ├── topic_timeline.py   # Topic tracking view
│   ├── search_results.py   # Search results display
│   └── stats_panel.py      # Statistics overview
├── services/
│   ├── __init__.py
│   ├── data_service.py     # Main data access layer
│   ├── recording_service.py # Recording aggregation
│   └── search_service.py   # Search functionality
└── assets/
    └── style.css           # Custom styles
```

## Data Service API

```python
class ChronosDataService:
    # Aggregation
    def get_days(self, start_date=None, end_date=None) -> List[DaySummary]
    def get_day_detail(self, date: str) -> DayDetail
    def get_recording_detail(self, recording_id: str) -> RecordingDetail

    # Events
    def get_events_for_recording(self, recording_id: str) -> List[Event]
    def get_event_detail(self, event_id: str) -> Event

    # Topics
    def get_all_topics(self) -> List[TopicSummary]  # Keywords aggregated
    def get_topic_timeline(self, topic: str) -> TopicTimeline

    # Search
    def search(self, query: str) -> List[SearchResult]

    # Stats
    def get_stats() -> Stats

@dataclass
class DaySummary:
    date: str
    total_duration_seconds: float
    recording_count: int
    event_count: int
    top_categories: Dict[str, int]
    top_keywords: List[str]

@dataclass
class RecordingSummary:
    recording_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    event_count: int
    categories: Dict[str, int]
    keywords: List[str]

@dataclass
class Event:
    id: str
    recording_id: str
    start_ts: datetime
    end_ts: datetime
    clean_text: str
    category: str
    sentiment: float
    keywords: List[str]
    speaker: str
```

## Implementation Order

1. **Data Service** - Aggregation logic (recording grouping, day stats)
2. **Day View** - Primary navigation
3. **Recording Detail** - Drill-down capability
4. **Search** - Find specific content
5. **Topic Timeline** - Track themes (bonus)
6. **Polish** - Transitions, loading states, responsive

## Key Interactions

1. **Day Click** → Expand to show recordings
2. **Recording Click** → Navigate to recording detail
3. **Event Click** → Show full text, possibly highlight in context
4. **Keyword Click** → Open topic timeline
5. **Search Submit** → Show results with context
6. **Back Navigation** → Return to previous view

## Success Metrics

- Can find a specific conversation within 3 clicks
- Can see "what happened on Oct 27" in one view
- Can trace a topic across multiple days
- Search results show useful context
- UI loads in <1s for 200 events
