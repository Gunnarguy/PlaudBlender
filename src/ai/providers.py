from enum import Enum


class Provider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"


DEFAULT_PROVIDER = Provider.GOOGLE
