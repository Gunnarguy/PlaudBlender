import pytest

from src.notion_service import (
    NotionConnectionError,
    NotionService,
    NotionTimeoutError,
)


def test_fetch_recordings_raise_on_error_classifies_timeout(monkeypatch):
    service = NotionService()
    service._settings.notion_database_id = "db-123"
    monkeypatch.setattr(service, "_resolve_token", lambda: "token")

    class DummyDataSources:
        def retrieve(self, **kwargs):
            raise RuntimeError("request timed out while paging Notion")

    class DummyClient:
        data_sources = DummyDataSources()

    monkeypatch.setattr(service, "_get_client", lambda: DummyClient())

    with pytest.raises(NotionTimeoutError) as excinfo:
        service.fetch_recordings(limit=10, raise_on_error=True)

    assert "Fetching Notion recordings timed out" in str(excinfo.value)


def test_fetch_page_content_raise_on_error_classifies_connection_error(monkeypatch):
    service = NotionService()

    class DummyChildren:
        def list(self, **kwargs):
            raise RuntimeError("Connection refused")

    class DummyBlocks:
        children = DummyChildren()

    class DummyClient:
        blocks = DummyBlocks()

    monkeypatch.setattr(service, "_get_client", lambda: DummyClient())

    with pytest.raises(NotionConnectionError) as excinfo:
        service.fetch_page_content("page-1", raise_on_error=True)

    assert "Fetching Notion page content for page-1 failed" in str(excinfo.value)


def test_fetch_recordings_still_returns_empty_list_without_raise(monkeypatch):
    service = NotionService()
    service._settings.notion_database_id = "db-123"
    monkeypatch.setattr(service, "_resolve_token", lambda: "token")
    monkeypatch.setattr(service, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("Connection refused")))

    assert service.fetch_recordings(limit=10) == []
