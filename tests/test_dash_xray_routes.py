from flask import Flask

import app_v2.main as app_main
from app_v2.components import stats as stats_component
from src.chronos import cost_tracker
from app_v2.services import xray as xray_service


def _build_xray_test_client():
    app = Flask(__name__)
    app_main._register_xray_routes(app)
    return app.test_client()


def test_xray_costs_clamps_negative_days(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cost_tracker,
        "get_session_cost",
        lambda: {
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": {},
            "recent": [],
        },
    )

    def fake_get_cost_summary(days=30):
        captured["days"] = days
        return {
            "days": days,
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": {},
            "by_day": [],
        }

    monkeypatch.setattr(cost_tracker, "get_cost_summary", fake_get_cost_summary)

    client = _build_xray_test_client()
    response = client.get("/xray/api/costs?days=-5")

    assert response.status_code == 200
    assert captured["days"] == 1
    assert response.get_json()["historical"]["days"] == 1


def test_xray_throughput_clamps_bucket_count(monkeypatch):
    captured = {}

    def fake_get_throughput(buckets=30):
        captured["buckets"] = buckets
        return [0] * buckets

    monkeypatch.setattr(xray_service, "get_throughput", fake_get_throughput)

    client = _build_xray_test_client()
    response = client.get("/xray/api/throughput?buckets=999")

    assert response.status_code == 200
    assert captured["buckets"] == 60
    assert len(response.get_json()["buckets"]) == 60


def test_create_cost_section_uses_15_second_refresh(monkeypatch):
    monkeypatch.setattr(
        cost_tracker,
        "get_session_cost",
        lambda: {
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": {},
            "recent": [],
        },
    )
    monkeypatch.setattr(
        cost_tracker,
        "get_cost_summary",
        lambda days=30: {
            "days": days,
            "total_cost_usd": 0.0,
            "total_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": {},
            "by_day": [],
        },
    )
    monkeypatch.setattr(cost_tracker, "get_model_pricing_table", lambda: [])

    section = stats_component.create_cost_section()
    interval = next(
        child
        for child in section.children
        if getattr(child, "id", None) == "cost-refresh-interval"
    )

    assert interval.interval == 15000