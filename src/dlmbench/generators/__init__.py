"""Portable generators and their explicit provenance contracts."""


class UnsupportedGenerationError(RuntimeError):
    """Raised when a requested generation path is unavailable or ambiguous."""


def _lazy_exports():
    from .artifacts import (
        CountArtifact,
        PhraseArtifact,
        load_count_artifact,
        load_phrase_artifact,
    )
    from .naive import (
        historical_mirror,
        historical_periodic,
        historical_phrase_sample,
        historical_top_token_ids,
        historical_topk_iid,
    )
    return {
        "CountArtifact": CountArtifact,
        "PhraseArtifact": PhraseArtifact,
        "load_count_artifact": load_count_artifact,
        "load_phrase_artifact": load_phrase_artifact,
        "historical_mirror": historical_mirror,
        "historical_periodic": historical_periodic,
        "historical_phrase_sample": historical_phrase_sample,
        "historical_top_token_ids": historical_top_token_ids,
        "historical_topk_iid": historical_topk_iid,
    }


def __getattr__(name: str):
    exports = _lazy_exports()
    if name not in exports:
        raise AttributeError(name)
    return exports[name]


def attach_parser(subparsers):
    """Attach the generator CLI without importing computation dependencies."""
    from .cli import attach_parser as _attach_parser
    return _attach_parser(subparsers)


__all__ = [
    "CountArtifact",
    "PhraseArtifact",
    "UnsupportedGenerationError",
    "attach_parser",
    "historical_mirror",
    "historical_periodic",
    "historical_phrase_sample",
    "historical_top_token_ids",
    "historical_topk_iid",
    "load_count_artifact",
    "load_phrase_artifact",
]
