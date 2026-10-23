"""Explicit tokenizer loading and decoding for generated token IDs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import UnsupportedGenerationError


@dataclass(frozen=True)
class TokenizerSnapshot:
    path: Path
    identifier: str
    revision: str | None
    digest: str
    tokenizer: Any

TOKENIZER_ACQUISITION_NOTE = (
    "requested identity and revision are user-supplied assertions; "
    "snapshot_sha256 is the verified content digest"
)


def directory_digest(path: str | Path) -> str:
    """Hash a tokenizer snapshot's relative names and bytes, not its local root."""

    root = Path(path).expanduser()
    if not root.is_dir():
        raise UnsupportedGenerationError(f"tokenizer snapshot is not a directory: {root}")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = file_path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def load_tokenizer_snapshot(
    path: str | Path,
    *,
    expected_id: str,
    expected_revision: str | None = None,
) -> TokenizerSnapshot:
    """Load a local tokenizer snapshot without network access."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise UnsupportedGenerationError(
            "text generation requires transformers; install the isolated generation environment"
        ) from error
    root = Path(path).expanduser()
    if not root.is_dir():
        raise UnsupportedGenerationError(f"tokenizer snapshot is not a directory: {root}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
    except Exception as error:
        raise UnsupportedGenerationError(f"could not load tokenizer snapshot {root}: {error}") from error
    identifier = str(expected_id or "")
    if not identifier:
        identifier = str(getattr(tokenizer, "name_or_path", "") or "")
    if not identifier:
        raise UnsupportedGenerationError(f"tokenizer snapshot {root} does not report its identity")
    if expected_id and expected_id not in identifier and root.name != expected_id:
        raise UnsupportedGenerationError(
            f"tokenizer snapshot {root} identifies as {identifier!r}, expected {expected_id!r}"
        )
    return TokenizerSnapshot(
        path=root,
        identifier=str(expected_id or identifier),
        revision=expected_revision,
        digest=directory_digest(root),
        tokenizer=tokenizer,
    )


def decode_token_rows(snapshot: TokenizerSnapshot, token_rows: list[list[int]]) -> list[str]:
    """Decode exact token sequences without normalization or special-token removal."""

    if snapshot.tokenizer is None:
        raise UnsupportedGenerationError("tokenizer snapshot is not loaded")
    return [
        snapshot.tokenizer.decode(row, clean_up_tokenization_spaces=False)
        for row in token_rows
    ]


def provenance(snapshot: TokenizerSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.identifier,
        "revision": snapshot.revision,
        "snapshot_sha256": snapshot.digest,
        "acquisition": TOKENIZER_ACQUISITION_NOTE,
    }
