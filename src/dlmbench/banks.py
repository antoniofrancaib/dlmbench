"""Lossless ordered text banks, shared by generation, metrics, and replay."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from .provenance import digest_json, sha256_file


@dataclass(frozen=True)
class Bank:
    """An ordered collection; repeated texts are separate observations.

    ``sha256`` identifies the source file bytes. ``text_sha256`` identifies
    the ordered UTF-8 text using length prefixes, independently of serialization.
    Token IDs retain their original tokenizer space; metrics must verify that
    tokenizer identity before using them as scorer inputs.
    """

    ids: tuple[str | int, ...]
    texts: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    token_ids: tuple[tuple[int, ...] | None, ...] | None = None
    path: Path | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("a sample bank must contain at least one sample")
        if len(self.ids) != len(self.texts):
            raise ValueError("sample IDs and texts have different lengths")
        if any(type(value) not in (str, int) for value in self.ids):
            raise ValueError("sample IDs must be strings or integers")
        if len({(type(value), value) for value in self.ids}) != len(self.ids):
            raise ValueError("sample IDs must be unique (duplicate text is allowed)")
        for text in self.texts:
            if not isinstance(text, str):
                raise ValueError("sample text must be a string")
            text.encode("utf-8", errors="strict")
        if self.token_ids is not None and len(self.token_ids) != len(self.texts):
            raise ValueError("token rows and texts have different lengths")
        if self.token_ids is not None:
            for row in self.token_ids:
                if row is not None and (not isinstance(row, (list, tuple)) or
                                       any(type(token) is not int or token < 0 for token in row)):
                    raise ValueError("token IDs must be nonnegative integers")

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def id(self) -> str:
        return str(self.metadata.get("id") or self.metadata.get("generator_id") or
                   self.metadata.get("bank_id") or "custom")

    @property
    def dataset(self) -> str | None:
        return self.metadata.get("dataset") or self.metadata.get("generation_config", {}).get("dataset")

    @property
    def text_sha256(self) -> str:
        digest = hashlib.sha256()
        for text in self.texts:
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    def identity(self) -> dict[str, Any]:
        return {"id": self.id, "dataset": self.dataset, "n_samples": len(self),
                "sha256": self.sha256, "text_sha256": self.text_sha256,
                "ids_sha256": digest_json(self.ids),
                "token_ids_sha256": digest_json(self.token_ids) if self.token_ids is not None else None}


def data_root(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get("DLMBENCH_DATA_DIR", ".")).expanduser()


def load_catalog() -> dict[str, Any]:
    resource = files("dlmbench.resources").joinpath("banks.json")
    if not resource.is_file():
        return {"schema_version": 1, "banks": [], "suites": {}}
    return json.loads(resource.read_text(encoding="utf-8"))


def catalog_entry(bank_id: str) -> dict[str, Any]:
    matches = [row for row in load_catalog()["banks"]
               if bank_id == row["id"] or bank_id in row.get("aliases", [])]
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous bank {bank_id!r}; use 'dlmbench samples list'")
    return matches[0]


def bank_relative_path(entry: dict[str, Any]) -> Path:
    path = Path(entry.get("path") or f"samples/{entry['dataset']}/{entry['id']}/samples.jsonl")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("catalog sample paths must be safe relative paths")
    return path


def load_bank(source: str | Path, *, root: str | Path | None = None,
              expected_sha256: str | None = None) -> Bank:
    """Load a JSONL file, bank directory, or downloaded catalog ID.

    No normalization, deduplication, tokenization, random selection, or silent
    malformed-row filtering occurs. Empty text is retained for metric validation.
    """
    path = Path(source).expanduser()
    metadata: dict[str, Any] = {}
    if not path.exists():
        entry = catalog_entry(str(source))
        if entry.get("provenance", {}).get("status") == "unavailable-reference":
            raise FileNotFoundError(
                f"the exact frozen reference bank {source!r} is unavailable in this release; "
                "supply your own reference JSONL for a custom evaluation"
            )
        metadata.update(entry)
        expected_sha256 = expected_sha256 or entry.get("sha256")
        path = data_root(root) / bank_relative_path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"bank {source!r} is not downloaded; use 'dlmbench samples fetch {source}'")
    if path.is_dir():
        path = path / "samples.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"bank manifest must be a JSON object: {manifest_path}")
        metadata = {**manifest, **metadata}
        expected_sha256 = expected_sha256 or manifest.get("output_sha256") or manifest.get("samples_sha256")
    # Curated catalog provenance carries the original scorer-relevant fields.
    # Expose them to metric consumers while preserving the provenance record.
    scientific = metadata.get("provenance", {}).get("scientific_metadata", {})
    for key, value in scientific.items():
        if value is not None:
            if metadata.get(key) is not None and metadata[key] != value:
                raise ValueError(f"conflicting recorded bank metadata: {key}")
            metadata[key] = value
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"sample checksum mismatch: {path.name}")
    texts, identifiers, tokens = [], [], []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path.name}:{line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"expected an object with string text at {path.name}:{line_number}")
            identifiers.append(row.get("id", len(texts)))
            texts.append(row["text"])
            token_row = row.get("token_ids")
            if token_row is not None:
                if not isinstance(token_row, list) or any(type(token) is not int or token < 0 for token in token_row):
                    raise ValueError(f"invalid token IDs at {path.name}:{line_number}")
                token_row = tuple(token_row)
            tokens.append(token_row)
    expected_count = metadata.get("n_samples", metadata.get("sample_count"))
    if expected_count is not None and int(expected_count) != len(texts):
        raise ValueError(f"sample count mismatch: expected {expected_count}, got {len(texts)}")
    return Bank(ids=tuple(identifiers), texts=tuple(texts), metadata=metadata,
                token_ids=tuple(tokens) if any(row is not None for row in tokens) else None,
                path=path, sha256=actual_sha256)
