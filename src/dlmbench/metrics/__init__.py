"""Automatic metric evaluation for ordered text banks."""

from .api import evaluate, list_metrics

__all__ = ["attach_parser", "evaluate", "list_metrics"]


def attach_parser(subparsers):
    """Lazily attach the CLI without importing metric implementations."""

    from .cli import attach_parser as _attach_parser

    return _attach_parser(subparsers)
