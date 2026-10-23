"""Executable adapter for the official SDTT Hydra sampling entrypoint."""

from __future__ import annotations

import math
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
    """Run ``mode=sample`` and normalize the official NPZ token output."""

    root = Path(source_root).expanduser()
    model_path = Path(checkpoint).expanduser()
    output_root = Path(out_dir).expanduser()
    require_paths(root, "src/sdtt/main.py")
    source_identity("sdtt-official", root, entrypoint="src/sdtt/main.py mode=sample")
    if model_path.suffix != ".ckpt" or not model_path.is_file():
        raise UnsupportedGenerationError(
            f"SDTT requires a staged Lightning .ckpt checkpoint: {model_path}"
        )
    effective = effective_config(config, sample_count=sample_count, seed=seed)
    sampling = dict(effective.get("sampling") or {})
    requested_batch = int(effective["batch_size"])
    batch_size = (
        requested_batch
        if sample_count % requested_batch == 0
        else math.gcd(sample_count, requested_batch)
    )
    batch_size = max(1, batch_size)
    start = time.monotonic()
    with adapter_scratch(output_root) as scratch:
        run_dir = scratch / "sdtt_hydra"
        compat_root = scratch / "runtime_compat"
        command = [
            sys.executable,
            str(root / "src" / "sdtt" / "main.py"),
            "mode=sample",
            "model=dit-orig-small",
            "parameterization=multi-round-sdtt",
            f"checkpointing.resume_ckpt_path={model_path}",
            "checkpointing.resume_from_ckpt=true",
            f"parameterization.checkpoint_path={model_path}",
            "parameterization.start_from_hf=false",
            "training.ema=0",
            "parameterization.sampling.uncond.run=true",
            "parameterization.sampling.cond_prefix.run=false",
            f"parameterization.sampling.uncond.num_samples={sample_count}",
            f"parameterization.sampling.uncond.batch_size={batch_size}",
            f"parameterization.sampling.uncond.num_steps={int(effective['steps'])}",
            f"parameterization.sampling.uncond.seq_len={int(effective['sequence_length'])}",
            f"parameterization.sampling.uncond.from_ema={str(bool(sampling.get('from_ema', False))).lower()}",
            f"parameterization.sampling.uncond.sampler={sampling.get('sampler', 'ancestral')}",
            "parameterization.sampling.uncond.add_bos=false",
            "parameterization.sampling.uncond.add_eos=false",
            f"+seed={int(sampling.get('effective_seed', seed))}",
            "eval.ppl_with_ar.run=false",
            "eval.mauve.run=false",
            "eval.lambada_openai.run=false",
            f"hydra.run.dir={run_dir}",
            "hydra.job.chdir=true",
        ]
        env = dict(os.environ)
        compat_root.mkdir(parents=True, exist_ok=True)
        (compat_root / "pkg_resources.py").write_text(
            "def declare_namespace(_package):\n    return None\n", encoding="utf-8"
        )
        torchdata_root = compat_root / "torchdata"
        torchdata_root.mkdir(parents=True, exist_ok=True)
        (torchdata_root / "__init__.py").write_text("", encoding="utf-8")
        (torchdata_root / "stateful_dataloader.py").write_text(
            "from torch.utils.data import DataLoader as StatefulDataLoader\n", encoding="utf-8"
        )
        env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(compat_root), str(root / "src"), env.get("PYTHONPATH", ""))
            if part
        )
        completed = subprocess.run(command, env=env, cwd=str(scratch), check=False)
        if completed.returncode != 0:
            raise UnsupportedGenerationError(
                f"SDTT command exited {completed.returncode} for {effective['id']!r}"
            )
        npz_files = sorted((run_dir / "samples" / "uncond").glob("*.npz"))
        if not npz_files:
            raise UnsupportedGenerationError(f"SDTT produced no unconditional NPZ under {run_dir}")
        import numpy as np

        npz_path = npz_files[-1]
        data = np.load(str(npz_path))
        token_ids = data["samples"] if "samples" in data.files else data[data.files[0]]
        if token_ids.ndim != 2 or token_ids.shape[0] != sample_count:
            raise UnsupportedGenerationError(
                f"unexpected SDTT token array shape {token_ids.shape}; expected {sample_count} rows"
            )
        upstream_sha256 = sha256_file(npz_path)
        token_rows = token_ids.astype(int).tolist()
        texts = [
            tokenizer.tokenizer.decode(row, clean_up_tokenization_spaces=False)
            for row in token_rows
        ]
        rows = [
            {"id": index, "text": text, "token_ids": token_rows[index]}
            for index, text in enumerate(texts)
        ]
    manifest = generation_manifest(
        effective,
        source_root=root,
        checkpoint=model_path,
        tokenizer={"id": tokenizer.identifier, "revision": tokenizer.revision},
        sampler={
            "name": "official SDTT unconditional ancestral sampler",
            "parameters": sampling,
            "seed": int(sampling.get("effective_seed", seed)),
            "requested_seed": seed,
            "batch_size": batch_size,
            "precision": "float model; integer token IDs",
            "nfe": int(effective["nfe"]),
            "actual_nfe": int(effective["actual_nfe"]),
        },
        entrypoint="src/sdtt/main.py mode=sample",
        elapsed_seconds=elapsed(start),
        execution_mode="recorded official Hydra subprocess",
        command=command,
        extra={"upstream_output_sha256": upstream_sha256},
    )
    return rows, manifest
