"""Executable inference adapter for the CoBit official evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .. import UnsupportedGenerationError
from .provenance import (
    effective_config,
    elapsed,
    generation_manifest,
    require_paths,
    source_identity,
)


def _select_cache(paths: list[Path], nfe: int, s_churn: float) -> Path:
    targets = (f"tnfe-{nfe}", f"anfe-{nfe}", f"steps={nfe}", f"steps-{nfe}")
    candidates: list[Path] = []
    for path in paths:
        key_path = path.parent / "_cache_key.json"
        cache_key = ""
        if key_path.is_file():
            try:
                cache_key = json.loads(key_path.read_text(encoding="utf-8")).get("cache_key", "")
            except (OSError, json.JSONDecodeError):
                cache_key = ""
        churn = re.search(r"(?:^|\|)s_churn=([^|]+)", cache_key)
        exact_churn = bool(churn and abs(float(churn.group(1)) - s_churn) < 1e-9)
        if any(target in cache_key or target in path.name for target in targets) and exact_churn:
            candidates.append(path)
    if not candidates:
        raise UnsupportedGenerationError(
            f"no CoBit cache matches target NFE={nfe} and s_churn={s_churn}"
        )
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}
    if len(digests) != 1:
        raise UnsupportedGenerationError(
            "multiple non-identical CoBit caches match the requested operating point"
        )
    return candidates[0]


def execute(
    config: dict[str, Any],
    *,
    source_root: Path,
    checkpoint: Path,
    tokenizer: Any,
    out_dir: Path,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the official external-ppl driver and harvest only generated texts."""

    root = Path(source_root).expanduser()
    checkpoint_path = Path(checkpoint).expanduser()
    del out_dir
    require_paths(
        root,
        "evaluation/run_eval.py",
        "configs/owt/eval_cobit_m_750K.py",
    )
    source_identity("cobit", root, entrypoint="evaluation.run_eval --metrics external_ppl")
    effective = effective_config(config, sample_count=sample_count, seed=seed)
    sampling = dict(effective.get("sampling") or {})
    if "s_churn" not in sampling:
        raise UnsupportedGenerationError(
            f"CoBit generator {effective['id']!r} does not declare exact sampling.s_churn"
        )
    dataset_cache = root / "datasets" / "openwebtext_gpt2_trainm100k"
    required = (
        "cache_train_gpt2_flat_eos1.uint16",
        "cache_train_gpt2_gpt2id_bpe16_v65536_base1024_len1024.uint16",
        "tokenizer_gpt2id_bpe16_65536_base1024.json",
        "tokenizer_gpt2id_bpe16_65536_base1024.meta.json",
    )
    missing = [name for name in required if not (dataset_cache / name).is_file()]
    if missing:
        raise UnsupportedGenerationError(
            f"CoBit OWT dataset cache is incomplete under {dataset_cache}; missing {missing}"
        )
    if not checkpoint_path.is_file():
        raise UnsupportedGenerationError(f"CoBit checkpoint file does not exist: {checkpoint_path}")
    experiment = (
        "paper/unconditional_text/owt/continuous_rate_raw_binary_bits_medium_24x1024"
    )
    staged_checkpoint = root / "runs" / experiment / "checkpoints" / "step=000750000.pt"
    _stage_checkpoint(checkpoint_path, staged_checkpoint)
    nfe = int(effective["nfe"])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        "-m",
        "evaluation.run_eval",
        "--config",
        str(root / "configs" / "owt" / "eval_cobit_m_750K.py"),
        "--metrics",
        "external_ppl",
        "--external_ppl_num_samples",
        str(sample_count),
        "--external_ppl_steps",
        str(nfe),
    ]
    env = dict(os.environ)
    env["EVAL_SEED"] = str(seed)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HOME", str(root / "hf_cache"))
    start = time.monotonic()
    completed = subprocess.run(command, env=env, cwd=str(root), check=False)
    if completed.returncode != 0:
        raise UnsupportedGenerationError(
            f"CoBit command exited {completed.returncode} for {effective['id']!r}"
        )
    generated_files = sorted((root / "runs" / experiment).rglob("shared_text_cache/**/*.jsonl"))
    selected = _select_cache(generated_files, nfe, float(sampling["s_churn"]))
    rows: list[dict[str, Any]] = []
    with selected.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if "_meta" in payload:
                continue
            text = str(payload.get("full", ""))
            rows.append(
                {
                    "id": len(rows),
                    "text": text,
                    "token_ids": tokenizer.tokenizer.encode(text, add_special_tokens=False),
                }
            )
            if len(rows) == sample_count:
                break
    if len(rows) != sample_count:
        raise UnsupportedGenerationError(
            f"CoBit harvested {len(rows)} samples, expected {sample_count}"
        )
    manifest = generation_manifest(
        effective,
        source_root=root,
        checkpoint=checkpoint_path,
        tokenizer={"id": tokenizer.identifier, "revision": tokenizer.revision},
        sampler={
            "name": "CoBit official external_ppl sampler",
            "parameters": sampling,
            "seed": seed,
            "batch_size": int(effective["batch_size"]),
            "precision": "float model; integer token IDs",
            "nfe": nfe,
            "actual_nfe": int(effective["actual_nfe"]),
        },
        entrypoint="evaluation.run_eval --metrics external_ppl",
        elapsed_seconds=elapsed(start),
        execution_mode="recorded official evaluator subprocess; metric output ignored",
        command=command,
    )
    return rows, manifest


def _stage_checkpoint(source: Path, destination: Path) -> None:
    """Link the supplied checkpoint into the path expected by CoBit's config."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve(strict=False) == source.resolve(strict=False):
            return
        destination.unlink()
    token = uuid.uuid4().hex
    temporary = destination.with_name(f".{destination.name}.{token}.tmp")
    os.symlink(source.resolve(strict=True), temporary)
    try:
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
