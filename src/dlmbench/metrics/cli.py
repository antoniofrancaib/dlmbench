"""Commands for metric evaluation and protocol inspection."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from ..banks import load_bank
from .api import evaluate, list_metrics


def attach_parser(subparsers) -> None:
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate automatic metrics on sample banks"
    )
    evaluate_parser.add_argument("generated", nargs="?", help="sample bank ID or JSONL path")
    evaluate_parser.add_argument("--reference")
    evaluate_parser.add_argument(
        "--metrics", nargs="*", default=None, help="metrics to evaluate; defaults to all"
    )
    evaluate_parser.add_argument("--profile", default="paper")
    evaluate_parser.add_argument("--device", default=None)
    evaluate_parser.add_argument("--out", default="-")
    evaluate_parser.set_defaults(func=_evaluate)

    evaluate_parser.add_argument("--list", action="store_true", help="list metrics without evaluation")
    evaluate_parser.add_argument("--json", action="store_true", help="with --list, export metric protocols")


def _evaluate(args: argparse.Namespace) -> int:
    if args.list:
        if args.generated is not None:
            raise ValueError("--list does not accept a sample bank")
        return _metrics(args)
    if args.json:
        raise ValueError("--json is only used with --list; evaluation already emits JSON")
    if args.generated is None:
        raise ValueError("provide a sample bank, or use --list to inspect metrics")
    generated = load_bank(args.generated)
    reference = load_bank(args.reference) if args.reference else None
    result = evaluate(
        generated,
        reference,
        metrics=args.metrics,
        profile=args.profile,
        device=args.device,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out == "-":
        print(payload, end="")
    else:
        _atomic_write(Path(args.out), payload)
    failures = [value for value in result["metrics"].values() if value.get("status") != "ok"]
    return 1 if failures else 0


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _metrics(args: argparse.Namespace) -> int:
    if args.json:
        from .protocols import PAPER_PROFILE, paper_profile_source
        print(json.dumps({"profile": "paper-v1", "source": paper_profile_source(),
                          "metrics": PAPER_PROFILE}, indent=2, allow_nan=False))
        return 0
    for metric in list_metrics():
        print(metric)
    return 0
