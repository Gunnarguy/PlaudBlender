from app_v2.callbacks.navigation import merge_preferences


def test_merge_preferences_none():
    """Test merge_preferences with None returns defaults."""
    merged = merge_preferences(None)
    # Should include default keys
    assert "auto_refresh_enabled" in merged
    assert "auto_refresh_seconds" in merged
    assert "default_view" in merged
    assert merged["auto_refresh_seconds"] == 60
    assert merged["default_view"] == "timeline"


def test_merge_preferences_override():
    """Test merge_preferences with partial overrides."""
    prefs = {"auto_refresh_enabled": False, "default_view": "topics"}
    merged = merge_preferences(prefs)
    assert merged["auto_refresh_enabled"] is False
    assert merged["default_view"] == "topics"
    assert merged["auto_refresh_seconds"] == 60  # Default is kept


def test_merge_preferences_coercion():
    """Test merge_preferences clamps and coerces auto_refresh_seconds."""
    # Test valid coercion
    assert (
        merge_preferences({"auto_refresh_seconds": "120"})["auto_refresh_seconds"]
        == 120
    )

    # Test lower bound
    assert merge_preferences({"auto_refresh_seconds": 10})["auto_refresh_seconds"] == 15

    # Test upper bound
    assert (
        merge_preferences({"auto_refresh_seconds": 1000})["auto_refresh_seconds"] == 300
    )

    # Test invalid coercion falls back to 60
    assert (
        merge_preferences({"auto_refresh_seconds": "invalid"})["auto_refresh_seconds"]
        == 60
    )


def test_merge_preferences_view_validation():
    """Test merge_preferences validates default_view."""
    # Valid view
    assert merge_preferences({"default_view": "graph"})["default_view"] == "graph"

    # Invalid view falls back to timeline
    assert (
        merge_preferences({"default_view": "invalid_view"})["default_view"]
        == "timeline"
    )
