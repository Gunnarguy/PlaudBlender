#!/usr/bin/env python3
"""
Plaud OAuth helpers:
- Print the consent URL for manual authorization
- Validate the stored token (auto-refresh if possible)

Usage:
  python scripts/plaud_auth_utils.py --print-consent-url
  python scripts/plaud_auth_utils.py --check-token [--no-refresh]
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure env vars are loaded from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.plaud_oauth import PlaudOAuthClient  # noqa: E402
from src.plaud_client import PlaudClient  # noqa: E402


def print_consent_url():
    client = PlaudOAuthClient()
    url, state = client.get_authorization_url()
    print("\n🔗 Plaud consent URL:\n" + url)
    print(f"\nRedirect URI : {client.redirect_uri}")
    print(f"State        : {state} (for your reference; Plaud may ignore)")
    print("\nOpen that URL in a browser, approve access, and capture the ?code=... from the redirect.")
    print("You can paste that code into plaud_setup.py when prompted, or rerun the GUI auth flow.\n")


def check_token(auto_refresh: bool = True) -> int:
    try:
        oauth = PlaudOAuthClient()
    except Exception as exc:  # config missing
        print(f"❌ Could not initialize Plaud OAuth client: {exc}")
        return 1

    def _validate() -> int:
        api = PlaudClient(oauth)
        user = api.get_user()
        who = user.get("email") or user.get("username") or user
        print(f"✅ Access token valid. User: {who}")
        return 0

    # Try current token
    try:
        oauth.get_access_token()
    except Exception as exc:
        print(f"❌ No valid access token: {exc}\nRun: python plaud_setup.py")
        return 1

    try:
        return _validate()
    except Exception as exc:
        print(f"⚠️  Token check failed: {exc}")
        if not auto_refresh:
            return 1
        try:
            print("🔄 Attempting refresh...")
            oauth.refresh_access_token()
            return _validate()
        except Exception as exc2:
            print(f"❌ Refresh failed: {exc2}\nRun: python plaud_setup.py")
            return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plaud OAuth utilities")
    parser.add_argument("--print-consent-url", action="store_true", help="Print the Plaud consent URL and exit")
    parser.add_argument("--check-token", action="store_true", help="Validate stored token against Plaud API")
    parser.add_argument("--no-refresh", action="store_true", help="Do not attempt auto-refresh on failure")
    args = parser.parse_args(argv)

    if args.print_consent_url:
        print_consent_url()
        return 0

    if args.check_token:
        return check_token(auto_refresh=not args.no_refresh)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
