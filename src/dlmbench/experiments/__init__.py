"""Paper analyses from released measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import (
    PAPER_RESOURCES,
    load_resource,
    metric_agreement,
    verify_all,
    verify_rebenchmark,
)
from .replay import EXPERIMENTS, reproduce_run


def _run(args) -> int:
    if args.list:
        for name, description in EXPERIMENTS.items():
            print(f"{name}\t{description}")
        return 0
    if args.out is None:
        raise ValueError("provide --out for reproduction, or use --list to inspect experiments")
    result = reproduce_run(args.out, experiment=args.experiment)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def attach_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("reproduce", help="reconstruct paper analyses from released measurements")
    parser.add_argument("experiment", nargs="?", default="all", choices=["all", *EXPERIMENTS])
    parser.add_argument("--list", action="store_true", help="list available analyses")
    parser.add_argument("--out", type=Path, help="new output directory")
    parser.set_defaults(func=_run)


__all__ = [
    "PAPER_RESOURCES", "attach_parser", "load_resource", "metric_agreement",
    "reproduce_run", "verify_all", "verify_rebenchmark",
]
