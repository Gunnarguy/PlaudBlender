"""Thought signature tracking (stub).

Gemini 3 can return "thoughtSignature" parts when using tool/function calling.
The full implementation is out of scope for unit tests; the tests require that
these symbols exist and can be instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional


@dataclass(frozen=True)
class ThoughtSignature:
    name: str
    value: str


class ThoughtSignatureManager:
    def __init__(self):
        self._signatures: Dict[str, ThoughtSignature] = {}

    def put(self, sig: ThoughtSignature) -> None:
        self._signatures[sig.name] = sig

    def get(self, name: str) -> Optional[ThoughtSignature]:
        return self._signatures.get(name)


@lru_cache(maxsize=1)
def get_thought_manager() -> ThoughtSignatureManager:
    return ThoughtSignatureManager()
