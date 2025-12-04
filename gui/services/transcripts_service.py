from __future__ import annotations

from datetime import datetime
from typing import List, Dict

from gui.services.clients import get_plaud_client
from gui.state import state
from gui.utils.logging import log


def fetch_transcripts() -> List[Dict]:
    """Fetch recordings from Plaud API and normalize for display."""
    client = get_plaud_client()
    # Use default limit to match legacy behavior; Plaud API may reject large limits
    recordings = client.list_recordings()
    enhanced = [_normalize_recording(rec) for rec in recordings]
    state.transcripts = enhanced
    state.filtered_transcripts = enhanced
    log('INFO', f"Fetched {len(enhanced)} Plaud recordings")
    return enhanced


def filter_transcripts(query: str) -> List[Dict]:
    query_lower = query.lower()
    state.filtered_transcripts = [rec for rec in state.transcripts if query_lower in rec['display_name'].lower()]
    return state.filtered_transcripts


def _normalize_recording(rec: Dict) -> Dict:
    name = rec.get('name') or rec.get('title') or rec.get('file_name') or 'Untitled'
    start_at = rec.get('start_at')
    if start_at:
        try:
            dt = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M')
        except ValueError:
            date_str = start_at[:10]
            time_str = start_at[11:16]
    else:
        date_str = time_str = '—'

    duration_ms = rec.get('duration', 0) or rec.get('duration_ms', 0)
    minutes = duration_ms // 60000
    seconds = (duration_ms % 60000) // 1000
    duration_str = f"{minutes}:{seconds:02d}" if duration_ms else '—'

    rec = rec.copy()
    rec.update({
        'display_name': name,
        'display_date': date_str,
        'display_time': time_str,
        'display_duration': duration_str,
        'short_id': f"{rec.get('id','')[:10]}…" if rec.get('id') else '—',
    })
    return rec
