#!/usr/bin/env python3
"""
scripts/run_server.py
=====================
CLI runner for the SAR Change Intelligence FastAPI Server.

Usage:
    python scripts/run_server.py
    python scripts/run_server.py --port 8000 --reload
"""

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows so box/check characters render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure repo root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Run SAR Change Intelligence FastAPI Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  🛰️  Starting SAR Change Intelligence API Server")
    print(f"  📡  Address: http://localhost:{args.port}")
    print(f"  📖  Docs:    http://localhost:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
