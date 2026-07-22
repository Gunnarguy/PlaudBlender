from dash import html, dcc
import pytest
from app_v2.layout import create_layout


def _get_element_by_id(element, target_id):
    """Recursively search for an element with the given id."""

    if hasattr(element, "id"):
        element_id = getattr(element, "id")
        if isinstance(element_id, dict) and element_id.get("id") == target_id:
            return element
        if element_id == target_id:
            return element

    if getattr(element, "id", None) == target_id:
        return element

    children = getattr(element, "children", None)

    if children is None:
        return None

    if isinstance(children, list) or isinstance(children, tuple):
        for child in children:
            if child is not None:
                found = _get_element_by_id(child, target_id)
                if found is not None:
                    return found
    elif children is not None:
        return _get_element_by_id(children, target_id)

    return None


def test_get_element_by_id_helper():
    """Test our helper works"""
    layout = html.Div(
        children=[html.Div(id="target-div"), dcc.Store(id="target-store")]
    )
    assert _get_element_by_id(layout, "target-div") is not None
    assert _get_element_by_id(layout, "target-store") is not None
    assert _get_element_by_id(layout, "nonexistent") is None


class TestLayout:
    def test_create_layout_returns_div(self):
        """Verify create_layout returns a valid dash html.Div."""
        layout = create_layout()
        assert isinstance(layout, html.Div)
        assert layout.className == "chronos-app"

    def test_layout_contains_required_stores(self):
        """Verify all required state management stores are present."""
        layout = create_layout()
        stores_to_check = [
            "current-view",
            "selected-recording",
            "selected-topic",
            "search-query",
            "days-data",
            "heatmap-scroll-target",
            "app-preferences",
            "active-workflows-count",
        ]
        for store_id in stores_to_check:
            store = _get_element_by_id(layout, store_id)
            assert store is not None, f"Store '{store_id}' not found in layout"
            assert isinstance(store, dcc.Store) or type(store).__name__ == "Store", (
                f"Element '{store_id}' is not a dcc.Store, it is {type(store)}"
            )

    def test_layout_contains_required_intervals(self):
        """Verify required polling intervals are present."""
        layout = create_layout()
        intervals_to_check = ["auto-refresh", "pipeline-progress-poll", "workflow-poll"]
        for interval_id in intervals_to_check:
            interval = _get_element_by_id(layout, interval_id)
            assert interval is not None, f"Interval '{interval_id}' not found in layout"
            assert (
                isinstance(interval, dcc.Interval)
                or type(interval).__name__ == "Interval"
            ), f"Element '{interval_id}' is not a dcc.Interval"

    def test_layout_contains_main_content_areas(self):
        """Verify layout includes main sections like the content container and detail panel."""
        layout = create_layout()

        # Test content container
        content = _get_element_by_id(layout, "content-container")
        assert content is not None
        assert isinstance(content, html.Div) or type(content).__name__ == "Div"

        # Test detail panel
        detail_panel = _get_element_by_id(layout, "detail-panel")
        assert detail_panel is not None
        assert (
            isinstance(detail_panel, html.Aside)
            or type(detail_panel).__name__ == "Aside"
        )

        # Test loading overlay
        loading = _get_element_by_id(layout, "loading-overlay")
        assert loading is not None
        assert isinstance(loading, dcc.Loading) or type(loading).__name__ == "Loading"
