"""
src/plaud_v4.py: dropped connections are retried, but only for idempotent requests.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.plaud_v4 import _RETRY, PlaudV4Client  # noqa: E402


def test_session_retries_https_requests():
    client = PlaudV4Client()
    assert client.http.get_adapter("https://api.plaud.ai/x").max_retries.total == 3


def test_reads_retry_but_writes_do_not():
    assert _RETRY.is_retry("GET", 503)
    assert not _RETRY.is_retry("POST", 503)
    assert not _RETRY.is_retry("GET", 404)
