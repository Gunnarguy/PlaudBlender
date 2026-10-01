"""requirements.lock must pin every direct requirement, and the pins must satisfy requirements.txt.

The deploy scripts install with `-c requirements.lock`; a requirement missing from the lock
would float to the newest release again (how mcp 2.x broke the MCP server on 2026-10-01).
"""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _requirements():
    reqs = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            reqs.append(Requirement(line))
    return reqs


def _lock():
    pins = {}
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        assert match, f"lock lines must be exact pins, got: {line!r}"
        pins[canonicalize_name(match.group(1))] = Version(match.group(2))
    return pins


def test_every_requirement_is_pinned_and_satisfies_its_bounds():
    pins = _lock()
    for req in _requirements():
        name = canonicalize_name(req.name)
        assert name in pins, f"{req.name} is in requirements.txt but not pinned in requirements.lock"
        assert req.specifier.contains(pins[name], prereleases=True), (
            f"requirements.lock pins {req.name}=={pins[name]}, outside requirements.txt's {req.specifier}"
        )


def test_mcp_stays_on_1x():
    # scripts/mcp_server.py imports FastMCP from mcp.server, which 2.x moved.
    assert _lock()[canonicalize_name("mcp")] < Version("2")


def test_lock_has_no_local_or_editable_installs():
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        if line.strip().startswith("#"):
            continue  # the header documents the grep that strips these
        assert "file://" not in line and not line.startswith("-e "), line
