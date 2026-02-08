"""T1.1: Fix broken recordings — diagnose and repair stuck/failed recordings."""

import sqlite3
import sys

conn = sqlite3.connect("data/brain.db")
cur = conn.cursor()

# 1. Show current state
print("=" * 70)
print("RECORDING STATUS BREAKDOWN")
print("=" * 70)
cur.execute(
    """
    SELECT processing_status, COUNT(*) as cnt
    FROM chronos_recordings
    GROUP BY processing_status
    ORDER BY cnt DESC
"""
)
for status, cnt in cur.fetchall():
    print(f"  {status:15s}  {cnt}")

# 2. Show stuck "processing" recordings
print(f"\n{'=' * 70}")
print("STUCK 'processing' RECORDINGS")
print("=" * 70)
cur.execute(
    """
    SELECT recording_id, title, created_at, duration_seconds, source,
           LENGTH(transcript) as transcript_len, error_message
    FROM chronos_recordings
    WHERE processing_status = 'processing'
    ORDER BY created_at
"""
)
for row in cur.fetchall():
    rec_id, title, created, dur, source, t_len, err = row
    print(
        f"  {rec_id[:20]}  created={created}  dur={dur}s  src={source}  transcript={t_len or 0} chars  err={err}"
    )

# 3. Show failed recordings
print(f"\n{'=' * 70}")
print("FAILED RECORDINGS")
print("=" * 70)
cur.execute(
    """
    SELECT recording_id, title, created_at, duration_seconds, source,
           LENGTH(transcript) as transcript_len, error_message
    FROM chronos_recordings
    WHERE processing_status = 'failed'
    ORDER BY created_at
"""
)
for row in cur.fetchall():
    rec_id, title, created, dur, source, t_len, err = row
    print(
        f"  {rec_id[:20]}  created={created}  dur={dur}s  transcript={t_len or 0} chars"
    )
    print(f"    ERROR: {err}")

# 4. Check if any have events already (partial processing)
print(f"\n{'=' * 70}")
print("EVENTS FOR BROKEN RECORDINGS")
print("=" * 70)
cur.execute(
    """
    SELECT r.recording_id, r.processing_status, COUNT(e.event_id) as event_count
    FROM chronos_recordings r
    LEFT JOIN chronos_events e ON r.recording_id = e.recording_id
    WHERE r.processing_status IN ('processing', 'failed')
    GROUP BY r.recording_id
    HAVING event_count > 0
"""
)
rows = cur.fetchall()
if rows:
    for rec_id, status, cnt in rows:
        print(f"  {rec_id[:20]}  status={status}  events={cnt}")
else:
    print("  (none have events)")

# 5. Fix: Reset stuck recordings to 'pending'
if "--fix" in sys.argv:
    print(f"\n{'=' * 70}")
    print("FIXING: Resetting 'processing' → 'pending'")
    print("=" * 70)
    cur.execute(
        """
        UPDATE chronos_recordings
        SET processing_status = 'pending', error_message = NULL
        WHERE processing_status = 'processing'
    """
    )
    print(f"  Reset {cur.rowcount} recordings to 'pending'")

    # Also reset failed ones that had transcripts (worth retrying)
    cur.execute(
        """
        UPDATE chronos_recordings
        SET processing_status = 'pending', error_message = NULL
        WHERE processing_status = 'failed'
        AND transcript IS NOT NULL
        AND LENGTH(transcript) > 100
    """
    )
    print(f"  Reset {cur.rowcount} failed recordings with transcripts to 'pending'")

    # For failed ones without transcripts, keep as failed but clear for re-ingest
    cur.execute(
        """
        UPDATE chronos_recordings
        SET processing_status = 'pending', error_message = NULL
        WHERE processing_status = 'failed'
        AND (transcript IS NULL OR LENGTH(transcript) < 100)
    """
    )
    print(f"  Reset {cur.rowcount} failed recordings without transcripts to 'pending'")

    conn.commit()

    # Show new state
    print(f"\nNEW STATUS:")
    cur.execute(
        """
        SELECT processing_status, COUNT(*) as cnt
        FROM chronos_recordings
        GROUP BY processing_status
        ORDER BY cnt DESC
    """
    )
    for status, cnt in cur.fetchall():
        print(f"  {status:15s}  {cnt}")

conn.close()
