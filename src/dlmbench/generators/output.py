"""Atomic generator output directories with strict compatibility checks."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..provenance import runtime_record, sha256_file
from . import UnsupportedGenerationError

MANIFEST_SCHEMA_VERSION = "dlmbench-generation-v1"


def prepare_output(
    out_dir: str | Path, output_sha256: str | None = None, *, require_empty: bool = False
) -> Path:
    """Create a new empty output directory before generation starts."""

    target = Path(out_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    existing = target / "manifest.json"
    if existing.is_file():
        try:
            prior = json.loads(existing.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise UnsupportedGenerationError(
                f"refusing to write over unreadable output manifest {existing}: {error}"
            ) from error
        prior_output_sha256 = prior.get("output_sha256")
        if not isinstance(prior_output_sha256, str) or (
            output_sha256 is not None and prior_output_sha256 != output_sha256
        ):
            raise UnsupportedGenerationError(
                f"output directory {target} already contains a different manifest"
            )
        if existing_output_matches(target, prior):
            if require_empty:
                raise UnsupportedGenerationError(
                    f"output directory {target} already contains a completed bank"
                )
            return target
    if require_empty and any(target.iterdir()):
        raise UnsupportedGenerationError(f"output directory {target} is not empty")
    if existing.is_file():
        raise UnsupportedGenerationError(
            f"output directory {target} has a matching manifest but missing or changed output files"
        )
    if any(target.iterdir()):
        raise UnsupportedGenerationError(f"output directory {target} is not empty")
    return target


def existing_output_matches(target: Path, manifest: dict[str, Any]) -> bool:
    """Validate all output files listed by an existing manifest."""

    output_sha256 = manifest.get("output_sha256")
    if not isinstance(output_sha256, str) or len(output_sha256) != 64:
        return False
    sample_path = target / "samples.jsonl"
    if not sample_path.is_file() or sha256_file(sample_path) != output_sha256:
        return False
    expected_checksums = {}
    checksum_path = target / "checksums.sha256"
    if not checksum_path.is_file():
        return False
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected_checksums[filename.strip()] = digest
    return expected_checksums == {
        "samples.jsonl": output_sha256,
        "manifest.json": sha256_file(target / "manifest.json"),
    }


def write_output(
    rows: list[dict[str, Any]], out_dir: str | Path, manifest: dict[str, Any]
) -> Path:
    """Atomically write samples, manifest, and checksums for one generation."""

    target = Path(out_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise UnsupportedGenerationError(f"output directory {target} is not empty")
    samples = target / "samples.jsonl"
    temporary = samples.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(samples)
    output_sha256 = sha256_file(samples)
    complete_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "runtime": runtime_record(),
        **manifest,
        "output_sha256": output_sha256,
        "n_samples": len(rows),
    }
    if int(complete_manifest["n_samples"]) != int(manifest["n_samples"]):
        raise UnsupportedGenerationError("generated row count disagrees with the requested count")
    manifest_sha256 = _write_manifest(target / "manifest.json", complete_manifest)
    _write_checksums(target, output_sha256, manifest_sha256)
    return target


def _write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256_file(path)


def _write_checksums(target: Path, output_sha256: str, manifest_sha256: str) -> None:
    path = target / "checksums.sha256"
    payload = (
        f"{output_sha256}  samples.jsonl\n"
        f"{manifest_sha256}  manifest.json\n"
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=target)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
