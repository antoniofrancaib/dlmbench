"""Shared provenance helpers for executable upstream adapters."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from ...provenance import sha256_file
from .. import UnsupportedGenerationError

UPSTREAM_SOURCES = {
    "flm-hydra": ("https://github.com/david3684/flm", "a1918d5164e5038e37d0b7a4fb2010ce75b863b3"),
    "langflow": ("https://github.com/nealchen2003/LangFlow", "a712b08570c56c787d6ef24f8e906a2dcdf768f5"),
    "rdlm": ("https://github.com/harryjo97/RDLM", "67443aa6a2d0fa981eb7c6105f9cbc563e59c5c1"),
    "sdtt-official": ("https://github.com/jdeschena/sdtt", "1150985e90b8f2d5749e4469d5154eff9ec922c4"),
    "elf": ("https://github.com/lillian039/ELF", "b29d8833609e9ab7f67cd9da39435ac5cea04837"),
    "cobit": ("https://github.com/GBATZOLIS/BitstreamDiffusion", "7c93d1da98194226ca9eff50d5f0cb7f376db4a1"),
}


def expected_tokenizer(config: dict[str, Any]) -> str:
    dataset = str(config.get("dataset", "")).lower()
    if dataset == "lm1b":
        return "bert-base-uncased"
    if dataset == "owt":
        return "gpt2"
    raise UnsupportedGenerationError(f"no pinned tokenizer is recorded for dataset {dataset!r}")


def effective_config(
    config: dict[str, Any], *, sample_count: int, seed: int
) -> dict[str, Any]:
    effective = deepcopy(config)
    effective["n_samples"] = int(sample_count)
    effective["seed"] = int(seed)
    return effective


def source_identity(family: str, source_root: Path, *, entrypoint: str) -> dict[str, str]:
    try:
        repository, expected_revision = UPSTREAM_SOURCES[family]
    except KeyError as error:
        raise UnsupportedGenerationError(f"no upstream source provenance is recorded for family {family!r}") from error
    actual_revision = _git_revision(Path(source_root))
    if actual_revision != expected_revision:
        raise UnsupportedGenerationError(
            f"source revision mismatch for {family}: expected {expected_revision}, got {actual_revision}"
        )
    return {
        "repository": repository,
        "expected_revision": expected_revision,
        "revision": actual_revision,
        "root": str(Path(source_root).expanduser()),
        "entrypoint": entrypoint,
    }


def _git_revision(root: Path) -> str:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            raise UnsupportedGenerationError(
                f"could not verify upstream source revision at {root}: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    revision = run("rev-parse", "HEAD")
    if not revision:
        raise UnsupportedGenerationError(f"upstream source has no Git revision at {root}")
    status = run("status", "--porcelain")
    if status:
        raise UnsupportedGenerationError(f"upstream source checkout is modified at {root}: {status}")
    return revision


def checkpoint_identity(checkpoint: Path) -> dict[str, str]:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        digest = sha256_file(path)
    elif path.is_dir():
        digest = _tree_digest(path)
    else:
        raise UnsupportedGenerationError(f"checkpoint path does not exist: {path}")
    return {"path": str(path), "sha256": digest}


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise UnsupportedGenerationError(f"checkpoint directory is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def generation_manifest(
    config: dict[str, Any],
    *,
    source_root: Path,
    checkpoint: Path,
    tokenizer: dict[str, Any],
    sampler: dict[str, Any],
    entrypoint: str,
    elapsed_seconds: float,
    execution_mode: str,
    command: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    decoding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "generator_id": config.get("id"),
        "label": config.get("label"),
        "dataset": config.get("dataset"),
        "sequence_length": config.get("sequence_length"),
        "family": config.get("family"),
        "source": source_identity(str(config.get("family")), Path(source_root), entrypoint=entrypoint),
        "checkpoint": checkpoint_identity(Path(checkpoint)),
        "tokenizer": tokenizer,
        "sampler": sampler,
        "execution_mode": execution_mode,
        "runtime_seconds": round(float(elapsed_seconds), 6),
        "output_mode": "text_and_token_ids",
        "n_samples": int(config.get("n_samples", 0)),
        "support": config.get("support"),
    }
    if command is not None:
        manifest["executed_command"] = command
    if extra:
        manifest.update(extra)
    if decoding is not None:
        manifest["decoding"] = decoding
    return manifest


def elapsed(start: float) -> float:
    return time.monotonic() - start


def require_paths(root: Path, *relative_paths: str) -> None:
    for relative in relative_paths:
        path = Path(root).expanduser() / relative
        if not path.is_file():
            raise UnsupportedGenerationError(f"pinned upstream file is missing: {path}")


def make_scratch(out_dir: Path) -> Path:
    """Create an exclusive sibling scratch directory for upstream intermediates."""

    output_root = Path(out_dir).expanduser()
    if not output_root.parent.exists():
        raise UnsupportedGenerationError(
            f"output parent does not exist: {output_root.parent}"
        )
    try:
        return Path(tempfile.mkdtemp(prefix=f".{output_root.name}.scratch-", dir=output_root.parent))
    except OSError as error:
        raise UnsupportedGenerationError(f"could not create adapter scratch directory: {error}") from error


def remove_scratch(scratch: Path) -> None:
    """Remove one adapter-owned scratch directory without following symlinks."""

    root = Path(scratch).expanduser()
    if root.is_symlink() or not root.is_dir():
        return
    try:
        shutil.rmtree(root)
    except OSError as error:
        raise UnsupportedGenerationError(f"could not remove adapter scratch directory {root}: {error}") from error


def retain_scratch(scratch: Path, out_dir: Path) -> None:
    """Keep a failed adapter scratch directory for explicit operator inspection."""

    root = Path(scratch).expanduser()
    if not root.is_dir() or root.is_symlink():
        return
    marker = Path(out_dir).expanduser().with_name(f"{Path(scratch).name}.retained")
    if marker.exists():
        return
    marker.write_text(
        f"adapter scratch retained after a failed run: {scratch}\n"
        "inspect it, then remove both the scratch directory and this marker\n",
        encoding="utf-8",
    )


def release_scratch(scratch: Path) -> None:
    """Remove one adapter-owned scratch directory and its failure marker."""

    remove_scratch(scratch)
    marker = scratch.with_name(f"{scratch.name}.retained")
    marker.unlink(missing_ok=True)


@contextmanager
def adapter_scratch(out_dir: Path):
    """Give an adapter exclusive scratch, retaining it only after a failure."""

    scratch = make_scratch(out_dir)
    try:
        yield scratch
    except BaseException:
        retain_scratch(scratch, out_dir)
        raise
    try:
        release_scratch(scratch)
    except BaseException:
        retain_scratch(scratch, out_dir)
        raise
