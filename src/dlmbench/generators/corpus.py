"""Build compact custom-corpus artifacts for the native naive samplers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..provenance import sha256_file
from . import UnsupportedGenerationError
from .tokenization import TokenizerSnapshot


def _tokenizer_identity(snapshot: TokenizerSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.identifier,
        "revision": snapshot.revision,
        "snapshot_sha256": snapshot.digest,
        "acquisition": (
            "requested identity and revision are user-supplied assertions; "
            "snapshot_sha256 is the verified content digest"
        ),
    }


def _corpus_path(corpus_path: Path) -> tuple[Path, str]:
    path = Path(corpus_path).expanduser()
    digest = sha256_file(path)
    if path.is_dir():
        path = path / "samples.jsonl"
    if not path.is_file():
        raise UnsupportedGenerationError(
            f"corpus is not a file or Bank directory: {corpus_path}"
        )
    return path, digest


def _decode_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedGenerationError(f"invalid JSON source provenance {path}: {error}") from error
    if not isinstance(payload, dict):
        raise UnsupportedGenerationError(f"source provenance must be a JSON object: {path}")
    return payload


def build_count_artifact(
    *,
    corpus_path: Path,
    tokenizer: TokenizerSnapshot,
    dataset: str,
    source: dict[str, Any],
    output_path: Path,
) -> Path:
    """Count a JSONL/Bank text corpus into the exact loader schema."""

    if not str(dataset):
        raise UnsupportedGenerationError("a custom corpus artifact requires a dataset identifier")
    if not isinstance(source, dict) or not source:
        raise UnsupportedGenerationError("a custom corpus artifact requires source provenance")
    if tokenizer.tokenizer is None:
        raise UnsupportedGenerationError("tokenizer snapshot was not loaded")
    vocabulary_size = int(getattr(tokenizer.tokenizer, "vocab_size", 0))
    if vocabulary_size < 1:
        raise UnsupportedGenerationError("loaded tokenizer does not report a positive vocabulary size")
    path, corpus_sha256 = _corpus_path(corpus_path)
    counts = [0] * vocabulary_size
    n_tokens = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise UnsupportedGenerationError(f"invalid corpus JSON at {path}:{line_number}") from error
            text = row.get("text") if isinstance(row, dict) else row
            if not isinstance(text, str):
                raise UnsupportedGenerationError(f"corpus row {line_number} has no string text")
            for token in tokenizer.tokenizer.encode(text, add_special_tokens=False):
                token_id = int(token)
                if token_id < 0 or token_id >= vocabulary_size:
                    raise UnsupportedGenerationError(f"tokenizer emitted token ID {token_id} outside its vocabulary")
                counts[token_id] += 1
                n_tokens += 1
    if n_tokens < 1:
        raise UnsupportedGenerationError("custom corpus contains no tokens")
    payload = {
        "schema_version": 1,
        "kind": "token_counts",
        "dataset": dataset,
        "tokenizer": _tokenizer_identity(tokenizer),
        "source": {
            **dict(source),
            "corpus_path": str(Path(corpus_path).expanduser()),
            "corpus_sha256": corpus_sha256,
        },
        "vocabulary_size": vocabulary_size,
        "n_tokens": n_tokens,
        "counts": counts,
    }
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise UnsupportedGenerationError(f"refusing to overwrite existing corpus artifact {destination}")
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return destination


def build_phrase_artifact(
    *,
    corpus_path: Path,
    tokenizer: TokenizerSnapshot,
    dataset: str,
    source: dict[str, Any],
    output_path: Path,
    max_phrases: int,
) -> Path:
    """Build a non-empty five-token phrase bank with the documented prefix policy."""

    if max_phrases < 1:
        raise UnsupportedGenerationError("max_phrases must be positive")
    if not str(dataset):
        raise UnsupportedGenerationError("a custom corpus artifact requires a dataset identifier")
    if not isinstance(source, dict) or not source:
        raise UnsupportedGenerationError("a custom corpus artifact requires source provenance")
    if tokenizer.tokenizer is None:
        raise UnsupportedGenerationError("tokenizer snapshot was not loaded")
    path, corpus_sha256 = _corpus_path(corpus_path)
    phrases: list[list[int]] = []
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise UnsupportedGenerationError(f"invalid corpus JSON at {path}:{line_number}") from error
        text = row.get("text") if isinstance(row, dict) else row
        if not isinstance(text, str):
            raise UnsupportedGenerationError(f"corpus row {line_number} has no string text")
        token_ids = [
            int(token)
            for token in tokenizer.tokenizer.encode(text, add_special_tokens=False)
        ]
        for start in range(len(token_ids) - 4):
            phrases.append(token_ids[start:start + 5])
            if len(phrases) == max_phrases:
                break
        if len(phrases) == max_phrases:
            break
    if not phrases:
        raise UnsupportedGenerationError("custom corpus contains no complete five-token phrase")
    payload = {
        "schema_version": 1,
        "kind": "phrase_bank",
        "dataset": dataset,
        "tokenizer": _tokenizer_identity(tokenizer),
        "construction_policy": (
            "first complete five-token windows in corpus order, bounded by max_phrases"
        ),
        "source": {
            **dict(source),
            "corpus_path": str(Path(corpus_path).expanduser()),
            "corpus_sha256": corpus_sha256,
        },
        "ngram": 5,
        "phrases": phrases,
    }
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise UnsupportedGenerationError(f"refusing to overwrite existing corpus artifact {destination}")
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return destination


__all__ = ["build_count_artifact", "build_phrase_artifact"]
