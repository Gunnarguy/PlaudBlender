"""
scripts/plaud_v4_audio.py: a wrapped .opus must not be re-downloaded every run.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.plaud_v4_audio import existing_audio, record_audio, sha256_file  # noqa: E402


def test_row_path_present(tmp_path):
    f = tmp_path / "r1.ogg"
    f.write_bytes(b"x")
    rec = SimpleNamespace(local_audio_path=str(f))
    assert existing_audio(rec, tmp_path, "r1") == f


def test_row_points_at_deleted_opus_finds_ogg_twin(tmp_path):
    ogg = tmp_path / "r1.ogg"
    ogg.write_bytes(b"x")
    rec = SimpleNamespace(local_audio_path=str(tmp_path / "r1.opus"))
    assert existing_audio(rec, tmp_path, "r1") == ogg


def test_no_row_but_file_in_raw_dir(tmp_path):
    mp3 = tmp_path / "r1.mp3"
    mp3.write_bytes(b"x")
    assert existing_audio(None, tmp_path, "r1") == mp3


def test_nothing_on_disk(tmp_path):
    rec = SimpleNamespace(local_audio_path=str(tmp_path / "r1.opus"))
    assert existing_audio(rec, tmp_path, "r1") is None
    assert existing_audio(None, tmp_path, "r1") is None


def test_partial_download_is_not_audio(tmp_path):
    (tmp_path / "r1.opus.part").write_bytes(b"x")
    assert existing_audio(None, tmp_path, "r1") is None


def test_record_audio_sets_only_audio_fields(tmp_path):
    f = tmp_path / "r1.ogg"
    f.write_bytes(b"abc")
    rec = SimpleNamespace(local_audio_path="old.opus", checksum="old", title="keep", time_is_estimated=True)
    session = MagicMock()
    record_audio(session, rec, f, sha256_file(f))
    assert rec.local_audio_path == str(f)
    assert rec.checksum == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert rec.title == "keep" and rec.time_is_estimated is True
    session.commit.assert_called_once()
