import os
from functools import lru_cache
from typing import Tuple

from src.plaud_oauth import PlaudOAuthClient
from src.plaud_client import PlaudClient
from src.pinecone_client import PineconeClient
from gui.state import state


def get_oauth_client() -> PlaudOAuthClient:
    if not state.plaud_oauth_client:
        state.plaud_oauth_client = PlaudOAuthClient()
    return state.plaud_oauth_client


def get_plaud_client() -> PlaudClient:
    if not state.plaud_client:
        state.plaud_client = PlaudClient(get_oauth_client())
    return state.plaud_client


def get_pinecone_client() -> PineconeClient:
    if not state.pinecone_client:
        state.pinecone_client = PineconeClient()
    return state.pinecone_client


def current_index_name() -> str:
    return os.getenv('PINECONE_INDEX_NAME', 'transcripts')
