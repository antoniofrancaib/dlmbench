"""Portable command line entry point; optional computation stays lazy."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from . import __version__
from .artifacts import fetch_banks, select_banks, verify_bank
from .banks import load_bank


def _json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _samples_list(args) -> int:
    rows = select_banks(suite=args.suite, dataset=args.dataset)
    if args.json:
        _json(rows)
    else:
        print("BANK\tDATASET\tSAMPLES\tKIND\tDOWNLOAD")
        for row in rows:
            availability = ("unavailable" if not row.get("sha256") or not row.get("bytes") else
                            "GitHub" if row.get("download_urls") else "local")
            print(f"{row['id']}\t{row['dataset']}\t{row.get('n_samples', '?')}\t"
                  f"{row.get('kind', '?')}\t{availability}")
    return 0


def _samples_show(args) -> int:
    bank = load_bank(args.bank, root=args.root)
    if args.index < 0 or args.count < 1 or args.index + args.count > len(bank):
        raise ValueError(f"sample slice must be inside [0, {len(bank)}), with positive count")
    for index in range(args.index, args.index + args.count):
        if args.json:
            print(json.dumps({"id": bank.ids[index], "text": bank.texts[index]},
                             ensure_ascii=False))
        else:
            print(f"[{bank.ids[index]}]\n{bank.texts[index]}")
    return 0


def _samples_action(args) -> int:
    if not args.banks and not args.suite:
        raise ValueError("specify at least one bank ID or --suite")
    entries = select_banks(args.banks, suite=args.suite)
    if not entries:
        raise ValueError("no banks match this selection")
    if args.action == "fetch":
        outcomes = fetch_banks([entry["id"] for entry in entries], root=args.root,
                               source=args.source, base_url=args.base_url)
        _json(outcomes)
        return int(any(row["status"] != "verified" for row in outcomes))
    outcomes = []
    for entry in entries:
        try:
            outcome = verify_bank(entry["id"], root=args.root)
        except (OSError, ValueError) as error:
            outcome = {"id": entry["id"], "status": "error", "error": str(error)}
        outcomes.append(outcome)
    _json(outcomes)
    return int(any(row["status"] != "verified" for row in outcomes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlmbench", description="Samples, metrics, and paper reproduction for text generation.")
    parser.add_argument("--version", action="version", version=f"dlmbench {__version__}")
    parser.add_argument("--debug", action="store_true", help="show Python tracebacks on failure")
    commands = parser.add_subparsers(dest="command", required=True)
    samples = commands.add_parser("samples", help="list, fetch, inspect, and verify sample banks")
    actions = samples.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", help="list the release catalog without loading samples")
    listing.add_argument("--suite")
    listing.add_argument("--dataset")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_samples_list)
    showing = actions.add_parser("show", help="inspect samples by zero-based position")
    showing.add_argument("bank", help="catalog ID, JSONL file, or bank directory")
    showing.add_argument("--root", help="directory containing samples/ (or DLMBENCH_DATA_DIR; default: .)")
    showing.add_argument("--index", type=int, default=0)
    showing.add_argument("--count", type=int, default=1)
    showing.add_argument("--json", action="store_true", help="emit lossless JSONL")
    showing.set_defaults(func=_samples_show)
    for action in ("fetch", "verify"):
        command = actions.add_parser(action, help=f"{action} immutable release samples")
        command.add_argument("banks", nargs="*")
        command.add_argument("--suite")
        command.add_argument("--root", help="directory containing samples/ (or DLMBENCH_DATA_DIR; default: .)")
        if action == "fetch":
            source = command.add_mutually_exclusive_group()
            source.add_argument("--from", dest="source", help="local artifact directory or tar bundle")
            source.add_argument("--base-url", help="explicit HTTP(S) mirror containing catalog paths")
        command.set_defaults(func=_samples_action)
    for module in ("metrics", "generators", "experiments"):
        extension = importlib.import_module(f"dlmbench.{module}")
        extension.attach_parser(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        if args.debug:
            raise
        print(f"dlmbench: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("dlmbench: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
