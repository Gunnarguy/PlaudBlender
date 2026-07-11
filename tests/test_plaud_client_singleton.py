from src.plaud_client import get_client


def test_get_client_singleton(monkeypatch):
    # Mock environment variables needed for PlaudOAuthClient initialization
    monkeypatch.setenv("PLAUD_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("PLAUD_CLIENT_SECRET", "test_client_secret")

    # Ensure starting clean
    monkeypatch.setattr("src.plaud_client._client_instance", None)

    client1 = get_client()
    client2 = get_client()

    assert client1 is not None
    assert client1 is client2
