#!/usr/bin/env python3
"""Test the data service."""

from app_v2.services.data_service import get_data_service


def main():
    svc = get_data_service()

    # Test get_days
    days = svc.get_days()
    print(f"Days: {len(days)}")
    for d in days[:3]:
        print(
            f"  {d.date_display}: {d.recording_count} recordings, {d.event_count} events, {d.duration_formatted}"
        )
        for r in d.recordings[:2]:
            print(
                f"    - {r.time_range_formatted} ({r.duration_formatted}): {r.top_category}"
            )

    # Test stats
    stats = svc.get_stats()
    print(f"\nStats:")
    print(f"  Recordings: {stats.total_recordings}")
    print(f"  Events: {stats.total_events}")
    print(f"  Days: {stats.total_days}")
    print(f"  Duration: {stats.total_duration_hours:.1f} hours")
    print(f"  Categories: {stats.categories}")
    print(f"  Top keywords: {stats.top_keywords[:5]}")

    # Test topic timeline
    topic = svc.get_topic_timeline("job")
    print(f"\nTopic 'job':")
    print(f"  Occurrences: {topic.total_occurrences}")
    print(f"  Recordings: {topic.recording_count}")
    for o in topic.occurrences[:2]:
        print(f"  - {o.timestamp}: {o.text_snippet[:80]}...")


if __name__ == "__main__":
    main()
