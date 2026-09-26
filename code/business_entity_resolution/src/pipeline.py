"""Placeholder entry point for the final submission package.

The live pipeline is developed at the repo root (src/run_pipeline.py). When packaging,
copy the finished root src/ here and make this file call run_pipeline.main().
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_SRC = Path(__file__).resolve().parents[3] / "src"


def main(argv=None):
    sys.path.insert(0, str(ROOT_SRC))
    from run_pipeline import main as run  # noqa: WPS433
    run(argv)


if __name__ == "__main__":
    main()
