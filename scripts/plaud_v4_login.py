#!/usr/bin/env python3
"""One-time Plaud 4.0 login for PlaudBlender. Password-free.

    venv/bin/python scripts/plaud_v4_login.py --email you@example.com

Plaud emails a one-time code; paste it here. The resulting session is saved
to .plaud_v4_session.json (chmod 600, gitignored) and PlaudBlender renews it
on its own from then on. Run again only if a sync reports NotLoggedIn.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plaud_v4 import PlaudV4Client, PlaudV4Error, classic_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", help="the email Plaud has for your account")
    parser.add_argument("--check", action="store_true", help="only verify the saved session, do not log in")
    args = parser.parse_args()

    client = PlaudV4Client()

    if args.check or (client.has_session and not args.email):
        return verify(client)

    email = args.email or input("Plaud account email: ").strip()
    if not email:
        print("an email is required", file=sys.stderr)
        return 2

    print(f"Requesting a one-time code for {email} ...")
    try:
        challenge = client.send_login_code(email)
    except PlaudV4Error as error:
        # A 422 here names the exact fields the endpoint wants; surface it whole.
        print(f"could not request a code:\n  {error}", file=sys.stderr)
        return 1

    token = challenge.get("token") or challenge.get("challenge_token") or challenge.get("otp_token")
    if not token:
        print("Plaud accepted the request but returned no challenge token. Response keys:", sorted(challenge.keys()), file=sys.stderr)
        return 1
    print("Code sent. Check your email (and spam).")

    code = getpass.getpass("Enter the code: ").strip()
    if not code:
        print("no code entered", file=sys.stderr)
        return 2

    try:
        client.login_with_code(token, code)
    except PlaudV4Error as error:
        print(f"login failed:\n  {error}", file=sys.stderr)
        return 1

    print(f"Session saved to {client.session_file.name} (chmod 600, gitignored).")
    return verify(client)


def verify(client: PlaudV4Client) -> int:
    try:
        me = client.me()
        nickname = me.get("nickname") or me.get("name") or me.get("email") or "?"
        first = next(client.iter_recordings(page_size=5), None)
        if not first:
            print(f"Logged in as {nickname}, but the library listed no recordings.")
            return 1
        title = (first.get("name") or "")[:60]
        print(f"Logged in as {nickname}.")
        print(f"Newest recording: {title}  (id {classic_id(first['file_id'])[:12]}…)")
        print("PlaudBlender can now sync. Next: venv/bin/python scripts/plaud_v4_sync.py")
        return 0
    except PlaudV4Error as error:
        print(f"session check failed:\n  {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
