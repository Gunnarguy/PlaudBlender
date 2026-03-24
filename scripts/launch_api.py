"""
Launch the Chronos REST API.

Usage:
    python scripts/launch_api.py
    python scripts/launch_api.py --port 8000
    python scripts/launch_api.py --reload
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="Launch Chronos API server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    import uvicorn

    print(f"\n  Chronos API  →  http://localhost:{args.port}/docs\n")
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
