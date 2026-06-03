from app_v2.services import xray


def setup_function():
    xray.clear_events()


def test_get_recent_events_returns_incremental_results_newest_first():
    start_seq = xray._seq_counter

    xray.xray_log("nav", "switch", "Moved to the timeline")
    xray.xray_log("search", "query", "Searched for meetings")
    xray.xray_log("graph", "build", "Rebuilt the graph")

    latest = xray.get_recent_events(limit=2)
    assert [event["seq"] for event in latest] == [start_seq + 3, start_seq + 2]

    incremental = xray.get_recent_events(limit=10, since_seq=start_seq + 1)
    assert [event["seq"] for event in incremental] == [start_seq + 3, start_seq + 2]


def test_get_recent_events_skips_work_when_client_is_already_current():
    xray.xray_log("nav", "switch", "Moved to the stats page")
    latest_seq = xray.get_recent_events(limit=1)[0]["seq"]

    assert xray.get_recent_events(limit=50, since_seq=latest_seq) == []
