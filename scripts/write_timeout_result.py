#!/usr/bin/env python3
"""Emit a fallback benchmark result when execution fails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Output JSON path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    payload = {
        "results": [],
        "metadata": {
            "error": "timeout",
            "model": args.model,
            "dataset": args.dataset,
        },
    }

    Path(args.path).write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
