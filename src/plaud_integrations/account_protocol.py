"""Application-level account protocol shared by REST and MCP sources."""

from typing import Protocol

from .models import (
    PlaudFile,
    PlaudFileListRequest,
    PlaudFilePage,
    PlaudNote,
    PlaudTranscript,
    PlaudUser,
)


class PlaudAccountSource(Protocol):
    def get_current_user(self) -> PlaudUser: ...
    def list_files(self, request: PlaudFileListRequest) -> PlaudFilePage: ...
    def get_file(self, file_id: str) -> PlaudFile: ...
    def get_note(self, file_id: str) -> PlaudNote: ...
    def get_transcript(self, file_id: str) -> PlaudTranscript: ...
