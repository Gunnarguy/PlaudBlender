from enum import Enum


class Provider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    PINECONE = "pinecone"  # Pinecone hosted inference embeddings


DEFAULT_PROVIDER = Provider.GOOGLE
