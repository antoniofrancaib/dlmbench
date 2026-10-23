"""Executable inference adapter for the ELF-M PyTorch evaluator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ...provenance import sha256_file
from .. import UnsupportedGenerationError
from .provenance import (
    adapter_scratch,
    effective_config,
    elapsed,
    generation_manifest,
    require_paths,
    source_identity,
)


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
    """Run the inference-only ELF evaluator and harvest its generated JSONL."""

    root = Path(source_root).expanduser()
    checkpoint_path = Path(checkpoint).expanduser()
    output_root = Path(out_dir).expanduser()
    require_paths(root, "src/eval.py", "src/configs/training_configs/train_owt_ELF-M.yml")
    source_identity("elf", root, entrypoint="src/eval.py")
    if not checkpoint_path.is_file():
        raise UnsupportedGenerationError(f"ELF checkpoint file does not exist: {checkpoint_path}")
    effective = effective_config(config, sample_count=sample_count, seed=seed)
    nfe = int(effective["nfe"])
    sampling = dict(effective.get("sampling") or {})
    start = time.monotonic()
    with adapter_scratch(output_root) as scratch:
        sampling_config = scratch / f"elf_sampling_{nfe}nfe.yml"
        sampling_config.write_text(
            "- sampling_method: sde\n"
            f"  num_sampling_steps: [{nfe}]\n"
            "  cfgs: [1]\n"
            f"  sde_gamma: {sampling.get('sde_gamma', 1.0)}\n"
            "  self_cond_cfg_scales: [3]\n"
            "  time_schedule: logit_normal\n",
            encoding="utf-8",
        )
        elf_out = scratch / "elf_out"
        command = [
            sys.executable,
            str(root / "src" / "eval.py"),
            "--config",
            str(root / "src" / "configs" / "training_configs" / "train_owt_ELF-M.yml"),
            "--checkpoint_path",
            str(checkpoint_path),
            "--seed",
            str(seed),
            "--config_override",
            f"num_samples={sample_count}",
            "--config_override",
            f"global_batch_size={int(effective['batch_size'])}",
            "--config_override",
            f"sampling_configs_path={sampling_config}",
            "--config_override",
            f"output_dir={elf_out}",
            "--config_override",
            "online_eval=false",
            "--config_override",
            "use_wandb=false",
            "--config_override",
            "eval_data_path=none",
            "--config_override",
            "hf_repo_id=none",
            "--config_override",
            "data_path=none",
        ]
        env = dict(os.environ)
        env["TOKENIZERS_PARALLELISM"] = "false"
        completed = subprocess.run(command, env=env, cwd=str(root), check=False)
        if completed.returncode != 0:
            raise UnsupportedGenerationError(
                f"ELF command exited {completed.returncode} for {effective['id']!r}"
            )
        generated_files = sorted(elf_out.rglob("all_generated_*.jsonl"))
        if not generated_files:
            raise UnsupportedGenerationError(f"ELF produced no generated JSONL under {elf_out}")
        generated = generated_files[-1]
        upstream_sha256 = sha256_file(generated)
        rows: list[dict[str, Any]] = []
        with generated.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                text = str(payload.get("generated", ""))
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
                f"ELF harvested {len(rows)} samples, expected {sample_count}"
            )
    manifest = generation_manifest(
        effective,
        source_root=root,
        checkpoint=checkpoint_path,
        tokenizer={"id": tokenizer.identifier, "revision": tokenizer.revision},
        sampler={
            "name": "ELF SDE evaluator",
            "parameters": sampling,
            "seed": seed,
            "batch_size": int(effective["batch_size"]),
            "precision": "float model; integer token IDs",
            "nfe": nfe,
            "actual_nfe": int(effective["actual_nfe"]),
        },
        entrypoint="src/eval.py",
        elapsed_seconds=elapsed(start),
        execution_mode="recorded upstream evaluator subprocess with online evaluation disabled",
        command=command,
        extra={"upstream_output_sha256": upstream_sha256},
    )
    return rows, manifest
