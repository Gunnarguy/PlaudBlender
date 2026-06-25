import sqlite3
import time
import sys
from src.chronos.transcript_processor import TranscriptProcessor
from src.chronos.qdrant_client import ChronosQdrantClient

def backfill():
    conn = sqlite3.connect('/home/gunnarhostetler/PlaudBlender/data/brain.db', timeout=30.0)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT event_id, clean_text, qdrant_point_id
        FROM chronos_events
        WHERE sentiment IN (0.0, 0.05)
        """
    )
    events = cursor.fetchall()
    
    if not events:
        print("No events to backfill.")
        return

    print(f"Backfilling {len(events)} events (throttled to save CPU)...")
    processor = TranscriptProcessor(db_session=None)
    qdrant = ChronosQdrantClient()
    
    updated = 0
    for event_id, text, qdrant_point_id in events:
        # Get sentiment using processor (uses local LLM with 30s timeout now)
        sentiment = processor._local_sentiment_for_text(text)
        
        # Avoid exact 0.0 and 0.05 so they don't get selected for backfill again
        if sentiment == 0.0:
            sentiment = 0.01
        elif sentiment == 0.05:
            sentiment = 0.051
            
        # Update SQLite
        success = False
        while not success:
            try:
                cursor.execute("UPDATE chronos_events SET sentiment = ? WHERE event_id = ?", (sentiment, event_id))
                success = True
            except sqlite3.OperationalError as e:
                if 'locked' in str(e):
                    time.sleep(1.0)
                else:
                    raise e
                    
        # Update Qdrant only after this event has actually been indexed.
        # Otherwise Qdrant returns 404 for every unindexed point and the
        # backfill spends hours producing noise while competing with indexing.
        if qdrant_point_id:
            try:
                qdrant.client.set_payload(
                    collection_name=qdrant.collection_name,
                    payload={'sentiment': sentiment},
                    points=[qdrant_point_id]
                )
            except Exception as e:
                print(f"⚠️ Failed to update Qdrant payload for event {event_id}: {e}")
            
        updated += 1
        
        if updated % 10 == 0:
            print(f"✅ Graded {updated} / {len(events)} chunks (Last score: {sentiment})")
            while True:
                try:
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if 'locked' in str(e):
                        time.sleep(1.0)
                    else:
                        raise e
            
        time.sleep(0.5)

    conn.commit()
    conn.close()
    print(f"Finished updating {updated} events.")

if __name__ == "__main__":
    backfill()
