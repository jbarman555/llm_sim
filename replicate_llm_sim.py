#!/usr/bin/env python3
"""Convenience wrapper around the package CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from llm_sim_replication.cli import main


if __name__ == "__main__":
    main()
