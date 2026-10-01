from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.chronos_repository import get_tombstoned_recording_ids
from src.plaud_v4 import PlaudV4Client, PlaudV4Error, PlaudV4NotFound


class _Resp:
    def __init__(self, body):
        self._body = body
        self.request = SimpleNamespace(method="GET")
        self.url = "https://api-test.plaud.ai/file-app/v4/files/detail/f_s_x"
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._body


def test_file_not_found_is_its_own_error_but_still_a_plaud_error():
    with pytest.raises(PlaudV4NotFound) as caught:
        PlaudV4Client._envelope(_Resp({"status": -1800311, "msg": "file not found"}))
    assert isinstance(caught.value, PlaudV4Error)  # existing `except PlaudV4Error` still catches it


def test_other_api_errors_stay_generic():
    with pytest.raises(PlaudV4Error) as caught:
        PlaudV4Client._envelope(_Resp({"status": -1, "msg": "boom"}))
    assert not isinstance(caught.value, PlaudV4NotFound)
    assert PlaudV4Client._envelope(_Resp({"status": 0, "data": 1})) == {"status": 0, "data": 1}


def _session():
    engine = create_engine("sqlite://")
    return sessionmaker(bind=engine)()


def test_tombstones_read_and_missing_table_is_empty():
    session = _session()
    assert get_tombstoned_recording_ids(session) == set()  # fresh DB: no table yet
    session.execute(text("CREATE TABLE janitor_tombstones (recording_id TEXT PRIMARY KEY)"))
    session.execute(text("INSERT INTO janitor_tombstones VALUES ('notion:abc'), ('f_s_1')"))
    assert get_tombstoned_recording_ids(session) == {"notion:abc", "f_s_1"}
