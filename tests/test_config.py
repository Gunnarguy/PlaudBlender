import os
import pytest
from unittest.mock import patch
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

from src.config import (
    normalize_openai_model_name,
    _env_flag,
    _resolve_chronos_gemini_api_key,
    _openai_key_configured,
    _resolve_openai_api_key,
    Settings,
    get_settings,
    get_local_timezone,
)

# 2. Write tests for normalize_openai_model_name
def test_normalize_openai_model_name():
    # In Python, `"" or "gpt-5.5"` evaluates to `"gpt-5.5"`, because `""` is falsy.
    # Therefore, normalize_openai_model_name("") will return "gpt-5.5".
    assert normalize_openai_model_name(None) == "gpt-5.5"
    assert normalize_openai_model_name("") == "gpt-5.5"

    # However, `"   "` is NOT falsy. `"   " or "gpt-5.5"` evaluates to `"   "`.
    # Then `("   ").strip()` evaluates to `""`.
    # Then `_OPENAI_MODEL_ALIASES.get("", "")` evaluates to `""`.
    assert normalize_openai_model_name("   ") == ""

    # Test alias mapping
    assert normalize_openai_model_name("gpt-5-mini") == "gpt-5.4-mini"
    assert normalize_openai_model_name("gpt-5-nano") == "gpt-5.4-nano"

    # Test exact string returned if no alias
    assert normalize_openai_model_name("gpt-4") == "gpt-4"
    assert normalize_openai_model_name("claude-3-opus") == "claude-3-opus"

# 3. Write tests for _env_flag
@patch.dict(os.environ, {"TEST_FLAG_1": "1", "TEST_FLAG_0": "0", "TEST_FLAG_OTHER": "yes"}, clear=True)
def test_env_flag():
    # Test returns True when env var is "1"
    assert _env_flag("TEST_FLAG_1") is True

    # Test returns False when env var is "0", missing, or other values
    assert _env_flag("TEST_FLAG_0") is False
    assert _env_flag("TEST_FLAG_MISSING") is False
    assert _env_flag("TEST_FLAG_MISSING", default="1") is True
    assert _env_flag("TEST_FLAG_OTHER") is False

# 4. Write tests for _resolve_chronos_gemini_api_key
def test_resolve_chronos_gemini_api_key():
    # Test with CHRONOS_GEMINI_API_KEY set
    with patch.dict(os.environ, {"CHRONOS_GEMINI_API_KEY": "chronos_key", "GEMINI_API_KEY": "shared_key"}, clear=True):
        assert _resolve_chronos_gemini_api_key() == "chronos_key"

    # Test with CHRONOS_ALLOW_SHARED_GEMINI_KEY=1 and GEMINI_API_KEY set
    with patch.dict(os.environ, {"CHRONOS_ALLOW_SHARED_GEMINI_KEY": "1", "GEMINI_API_KEY": "shared_key"}, clear=True):
        assert _resolve_chronos_gemini_api_key() == "shared_key"

    # Test with CHRONOS_ALLOW_SHARED_GEMINI_KEY=0 and GEMINI_API_KEY set
    with patch.dict(os.environ, {"CHRONOS_ALLOW_SHARED_GEMINI_KEY": "0", "GEMINI_API_KEY": "shared_key"}, clear=True):
        assert _resolve_chronos_gemini_api_key() is None

    # Test without both
    with patch.dict(os.environ, {}, clear=True):
        assert _resolve_chronos_gemini_api_key() is None

# 5. Write tests for _openai_key_configured
def test_openai_key_configured():
    # Test returns True when OPENAI_API_KEY is set and non-empty
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-1234"}, clear=True):
        assert _openai_key_configured() is True

    # Test returns False otherwise
    with patch.dict(os.environ, {"OPENAI_API_KEY": "   "}, clear=True):
        assert _openai_key_configured() is False

    with patch.dict(os.environ, {}, clear=True):
        assert _openai_key_configured() is False

# 6. Write tests for _resolve_openai_api_key
def test_resolve_openai_api_key():
    # Test with CHRONOS_OPENAI_ENABLED=1 and OPENAI_API_KEY set
    with patch.dict(os.environ, {"CHRONOS_OPENAI_ENABLED": "1", "OPENAI_API_KEY": "sk-1234"}, clear=True):
        assert _resolve_openai_api_key() == "sk-1234"

    # Test with CHRONOS_OPENAI_ENABLED=1 but OPENAI_API_KEY empty
    with patch.dict(os.environ, {"CHRONOS_OPENAI_ENABLED": "1", "OPENAI_API_KEY": "   "}, clear=True):
        assert _resolve_openai_api_key() is None

    # Test with CHRONOS_OPENAI_ENABLED=0
    with patch.dict(os.environ, {"CHRONOS_OPENAI_ENABLED": "0", "OPENAI_API_KEY": "sk-1234"}, clear=True):
        assert _resolve_openai_api_key() is None

# 7. Write tests for Settings class
def test_settings_initialization():
    settings_default = Settings()
    # Check that some fields are set to something (from import time environ)
    assert hasattr(settings_default, 'gemini_api_version')
    assert hasattr(settings_default, 'plaud_default_language')

    # Check that kwargs work as expected for dataclasses
    settings_custom = Settings(
        gemini_api_version="v1",
        plaud_default_language="fr",
        chronos_cleaning_model="gemini-1.5-pro"
    )
    assert settings_custom.gemini_api_version == "v1"
    assert settings_custom.plaud_default_language == "fr"
    assert settings_custom.chronos_cleaning_model == "gemini-1.5-pro"

def test_get_settings():
    # Simple test for the factory function
    settings = get_settings()
    assert isinstance(settings, Settings)

# 8. Write tests for get_local_timezone
def test_get_local_timezone_from_tz_env():
    # Clear the cache before testing
    get_local_timezone.cache_clear()

    with patch.dict(os.environ, {"TZ": "Europe/London"}, clear=True):
        tz = get_local_timezone()
        assert isinstance(tz, ZoneInfo)
        assert tz.key == "Europe/London"

def test_get_local_timezone_invalid_tz_env():
    # Clear the cache before testing
    get_local_timezone.cache_clear()

    # Setup mock to simulate missing TZ and let it fallback to system time
    with patch.dict(os.environ, {"TZ": "Invalid/Zone"}, clear=True):
        with patch("src.config.os.path.realpath") as mock_realpath:
            mock_realpath.return_value = "/etc/localtime"

            # This should fall through to datetime.now().astimezone().tzinfo
            with patch("src.config.datetime") as mock_dt:
                mock_tz = timezone.utc
                mock_dt.now.return_value.astimezone.return_value.tzinfo = mock_tz
                tz = get_local_timezone()
                assert tz == mock_tz

def test_get_local_timezone_from_localtime():
    # Clear the cache before testing
    get_local_timezone.cache_clear()

    with patch.dict(os.environ, {}, clear=True):
        with patch("src.config.os.path.realpath") as mock_realpath:
            mock_realpath.return_value = "/var/db/timezone/zoneinfo/America/New_York"
            tz = get_local_timezone()
            assert isinstance(tz, ZoneInfo)
            assert tz.key == "America/New_York"
