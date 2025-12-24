"""ColPali ingestion (stub).

Some early experiments referenced ColPali-style document ingestion.
The current project focus is Chronos + transcripts; tests expect this module.
"""

from __future__ import annotations


class GeminiVisionAnalyzer:
    def __init__(self):
        pass


class ColPaliProcessor:
    def __init__(self, analyzer: GeminiVisionAnalyzer | None = None):
        self.analyzer = analyzer or GeminiVisionAnalyzer()
