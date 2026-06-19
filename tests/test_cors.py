import re

# The new regex implemented in api/main.py
CORS_REGEX = r"^https?://(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?$"

def test_cors_regex_valid_origins():
    valid_origins = [
        "http://10.0.0.1",
        "https://10.0.0.1",
        "http://10.255.255.255:3000",
        "http://192.168.1.100",
        "https://192.168.1.100:8000",
        "http://172.16.0.1",
        "http://172.31.255.255",
        "https://172.20.10.5:8080",
    ]
    for origin in valid_origins:
        assert re.match(CORS_REGEX, origin) is not None, f"Expected {origin} to be valid"

def test_cors_regex_invalid_origins():
    invalid_origins = [
        "http://10.attacker.com",
        "http://192.168.com",
        "https://192.168.attacker.com",
        "http://172.32.0.1",
        "http://172.15.255.255",
        "http://10.0.0.1.attacker.com",
        "http://192.168.1.100.com",
        "http://10.0.0.1:3000/path",  # Origin header doesn't contain path
        "ftp://10.0.0.1",
        "http://evil10.0.0.1",
    ]
    for origin in invalid_origins:
        assert re.match(CORS_REGEX, origin) is None, f"Expected {origin} to be invalid"
