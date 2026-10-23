"""Validated compact source artifacts for the paper's naive samplers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..provenance import sha256_file
from . import UnsupportedGenerationError

COUNT_SCHEMA_VERSION = 1
PHRASE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CountArtifact:
    counts: np.ndarray
    sha256: str
    dataset: str
    tokenizer: dict[str, Any]
    source: dict[str, Any]
    vocabulary_size: int
    n_tokens: int


@dataclass(frozen=True)
class PhraseArtifact:
    phrases: tuple[tuple[int, ...], ...]
    sha256: str
    dataset: str
    tokenizer: dict[str, Any]
    source: dict[str, Any]
    ngram: int


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise UnsupportedGenerationError(f"source artifact is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedGenerationError(f"invalid JSON source artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise UnsupportedGenerationError(f"source artifact must contain a JSON object: {path}")
    return payload, sha256_file(path)


def _validate_common(
    payload: dict[str, Any], *, schema: int, kind: str, path: Path
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if payload.get("schema_version") != schema or payload.get("kind") != kind:
        raise UnsupportedGenerationError(
            f"unsupported source artifact at {path}; expected kind={kind!r}, schema={schema}"
        )
    dataset = payload.get("dataset")
    tokenizer = payload.get("tokenizer")
    source = payload.get("source")
    if not isinstance(dataset, str) or not dataset:
        raise UnsupportedGenerationError(f"artifact {path} has no dataset identifier")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("id"), str):
        raise UnsupportedGenerationError(f"artifact {path} has no declared tokenizer id")
    if not isinstance(source, dict):
        raise UnsupportedGenerationError(f"artifact {path} has no source provenance object")
    return dataset, tokenizer, source, payload


def load_count_artifact(path: str | Path) -> CountArtifact:
    """Load an exact compact token-count artifact without synthesizing values."""

    artifact_path = Path(path).expanduser()
    payload, digest = _read_json(artifact_path)
    dataset, tokenizer, source, payload = _validate_common(
        payload, schema=COUNT_SCHEMA_VERSION, kind="token_counts", path=artifact_path
    )
    counts = np.asarray(payload.get("counts"), dtype=np.float64)
    if counts.ndim != 1 or counts.size < 1:
        raise UnsupportedGenerationError(f"artifact {artifact_path} has no one-dimensional count vector")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise UnsupportedGenerationError(f"artifact {artifact_path} has invalid count values")
    if np.count_nonzero(counts) == 0:
        raise UnsupportedGenerationError(f"artifact {artifact_path} has zero total token mass")
    vocabulary_size = payload.get("vocabulary_size")
    n_tokens = payload.get("n_tokens")
    if type(vocabulary_size) is not int or vocabulary_size != counts.size:
        raise UnsupportedGenerationError(
            f"artifact {artifact_path} vocabulary_size must equal its count-vector length"
        )
    if type(n_tokens) is not int or n_tokens < 1:
        raise UnsupportedGenerationError(f"artifact {artifact_path} has no positive n_tokens")
    if np.all(counts == np.floor(counts)) and int(counts.sum()) != n_tokens:
        raise UnsupportedGenerationError(
            f"artifact {artifact_path} n_tokens disagrees with its integer count vector"
        )
    return CountArtifact(
        counts=counts,
        sha256=digest,
        dataset=dataset,
        tokenizer=dict(tokenizer),
        source=dict(source),
        vocabulary_size=vocabulary_size,
        n_tokens=n_tokens,
    )


def load_phrase_artifact(path: str | Path, *, ngram: int = 5) -> PhraseArtifact:
    """Load the exact uniform corpus phrase bank used by the paper sampler."""

    artifact_path = Path(path).expanduser()
    payload, digest = _read_json(artifact_path)
    dataset, tokenizer, source, payload = _validate_common(
        payload, schema=PHRASE_SCHEMA_VERSION, kind="phrase_bank", path=artifact_path
    )
    if payload.get("ngram") != ngram:
        raise UnsupportedGenerationError(
            f"artifact {artifact_path} ngram={payload.get('ngram')!r}; expected {ngram}"
        )
    rows = payload.get("phrases")
    if not isinstance(rows, list) or not rows:
        raise UnsupportedGenerationError(f"artifact {artifact_path} has no phrase rows")
    phrases: list[tuple[int, ...]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != ngram or any(type(token) is not int or token < 0 for token in row):
            raise UnsupportedGenerationError(
                f"artifact {artifact_path} contains a malformed {ngram}-token phrase"
            )
        phrases.append(tuple(row))
    return PhraseArtifact(
        phrases=tuple(phrases),
        sha256=digest,
        dataset=dataset,
        tokenizer=dict(tokenizer),
        source=dict(source),
        ngram=ngram,
    )
