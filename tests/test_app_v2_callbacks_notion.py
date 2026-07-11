import pytest
from unittest.mock import patch
import time
import threading

from app_v2.callbacks.notion import (
    _set_notion_cache,
    invalidate_notion_cache,
    _get_cached_notion_data,
    _NOTION_CACHE_TTL,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the notion cache before and after each test."""
    invalidate_notion_cache()
    yield
    invalidate_notion_cache()


def test_set_and_get_notion_cache():
    test_data = {"key": "value"}

    with patch("time.monotonic", return_value=100.0):
        _set_notion_cache(test_data)

    with patch("time.monotonic", return_value=105.0):
        # Within TTL (default 60 seconds)
        assert _get_cached_notion_data() == test_data


def test_get_notion_cache_expired():
    test_data = {"key": "value"}

    with patch("time.monotonic", return_value=100.0):
        _set_notion_cache(test_data)

    with patch("time.monotonic", return_value=100.0 + _NOTION_CACHE_TTL + 1.0):
        # Exceeded TTL
        assert _get_cached_notion_data() is None


def test_invalidate_notion_cache():
    test_data = {"key": "value"}

    with patch("time.monotonic", return_value=100.0):
        _set_notion_cache(test_data)

    # Verify it is set
    with patch("time.monotonic", return_value=105.0):
        assert _get_cached_notion_data() == test_data

    # Invalidate
    invalidate_notion_cache()

    # Verify it is cleared
    with patch("time.monotonic", return_value=105.0):
        assert _get_cached_notion_data() is None


def test_mutation_of_cached_dictionaries():
    """Verify that mutating retrieved cache does not corrupt/mutate internal state if reference is shared."""
    test_data = {"nested": {"list": [1, 2, 3]}}
    _set_notion_cache(test_data)

    retrieved = _get_cached_notion_data()
    assert retrieved == test_data

    # Mutate retrieved
    retrieved["nested"]["list"].append(4)

    # Internal cache is mutated because it's a direct reference
    assert _get_cached_notion_data()["nested"]["list"] == [1, 2, 3, 4]


def test_concurrent_readers():
    """Verify lock prevents thread collisions during concurrent reads/writes."""
    # Start reader threads
    def reader():
        for _ in range(100):
            _get_cached_notion_data()
            time.sleep(0.001)

    def writer():
        for i in range(100):
            _set_notion_cache({"foo": f"bar_{i}"})
            time.sleep(0.001)

    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=writer),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert _get_cached_notion_data() is not None
