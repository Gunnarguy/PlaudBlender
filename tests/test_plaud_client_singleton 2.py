from src.plaud_client import get_client, reset_client
import pytest
import threading


def test_get_client_singleton(monkeypatch):
    # Mock environment variables needed for PlaudOAuthClient initialization
    monkeypatch.setenv("PLAUD_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("PLAUD_CLIENT_SECRET", "test_client_secret")

    # Ensure starting clean
    reset_client()

    client1 = get_client()
    client2 = get_client()

    assert client1 is not None
    assert client1 is client2


def test_reset_client(monkeypatch):
    monkeypatch.setenv("PLAUD_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("PLAUD_CLIENT_SECRET", "test_client_secret")

    reset_client()
    client1 = get_client()

    reset_client()
    client2 = get_client()

    assert client1 is not client2


def test_get_client_thread_safety(monkeypatch):
    monkeypatch.setenv("PLAUD_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("PLAUD_CLIENT_SECRET", "test_client_secret")

    reset_client()
    instances = []

    def target():
        instances.append(get_client())

    threads = [threading.Thread(target=target) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should have retrieved the exact same instance
    assert len(instances) == 10
    first = instances[0]
    for inst in instances:
        assert inst is first
