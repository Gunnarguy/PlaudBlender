"""
Launch the Chronos REST API.

Usage:
    python scripts/launch_api.py
    python scripts/launch_api.py --port 8000
    python scripts/launch_api.py --reload
"""

import argparse
import os
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
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("CHRONOS_API_WORKERS", "2")),
        help="Number of Uvicorn worker processes",
    )
    args = parser.parse_args()

    import uvicorn

    print(f"\n  Chronos API  →  http://localhost:{args.port}/docs\n")
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else max(1, args.workers),
    )


if __name__ == "__main__":
    main()
