"""Executable inference adapter for the recorded FLM Hydra backend."""

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


def _algorithm(config: dict[str, Any]) -> str:
    names = {
        "ar": "ar", "duo_base": "duo_base", "duo": "duo", "flm": "flm",
        "fmlm": "fmlm", "mdlm": "mdlm", "sedd": "sedd",
    }
    try:
        return names[str(config["algorithm"])]
    except KeyError as error:
        raise UnsupportedGenerationError(f"unsupported FLM algorithm {config.get('algorithm')!r}") from error


def _overrides(config: dict[str, Any], checkpoint: Path, output: Path) -> list[str]:
    count = int(config["n_samples"])
    batch_size = int(config["batch_size"])
    if count < 1 or batch_size < 1:
        raise UnsupportedGenerationError("FLM sample count and batch size must be positive")
    data = "lm1b-wrap" if config["dataset"] == "lm1b" else "openwebtext-split"
    sampling = dict(config.get("sampling") or {})
    overrides = [
        "mode=sample_eval",
        f"seed={int(config['seed'])}",
        "model=small",
        f"model.length={int(config['sequence_length'])}",
        f"data={data}",
        f"algo={_algorithm(config)}",
        f"eval.checkpoint_path={checkpoint}",
        f"loader.eval_batch_size={batch_size}",
        f"sampling.num_sample_batches={(count + batch_size - 1) // batch_size}",
        f"sampling.steps={int(config['steps'])}",
        "+wandb.offline=true",
        f"eval.generated_samples_path={output}",
    ]
    for key in ("predictor", "noise_removal", "gamma"):
        if key in sampling:
            overrides.append(f"sampling.{key}={sampling[key]}")
    if config.get("double_temb") is not None:
        overrides.append(f"algo.double_temb={str(bool(config['double_temb'])).lower()}")
    if config.get("disable_ema") is not None:
        overrides.append(f"eval.disable_ema={str(bool(config['disable_ema'])).lower()}")
    return overrides


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
    """Run the upstream Hydra sample-eval path and recover token IDs."""

    root = Path(source_root).expanduser()
    model_path = Path(checkpoint).expanduser()
    output_root = Path(out_dir).expanduser()
    require_paths(root, "main.py")
    source_identity("flm-hydra", root, entrypoint="main.py mode=sample_eval")
    if not model_path.is_file():
        raise UnsupportedGenerationError(f"FLM checkpoint file does not exist: {model_path}")
    effective = effective_config(config, sample_count=sample_count, seed=seed)
    start = time.monotonic()
    with adapter_scratch(output_root) as scratch:
        output = scratch / "flm_generated.json"
        overrides = _overrides(effective, model_path, output)
        runner = (
            "from dlmbench.generators.adapters.flm_checkpoint_compat import install; "
            "install(); import runpy; runpy.run_path"
        )
        command = [sys.executable, "-u", "-c", runner, str(root / "main.py"), *overrides]
        env = dict(os.environ)
        env["HYDRA_FULL_ERROR"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(root), env.get("PYTHONPATH", "")) if value
        )
        completed = subprocess.run(command, env=env, cwd=str(root), check=False)
        if completed.returncode != 0:
            raise UnsupportedGenerationError(
                f"FLM command exited {completed.returncode} for {config['id']!r}"
            )
        if not output.is_file():
            raise UnsupportedGenerationError(f"FLM completed but {output} was not written")
        payload = json.loads(output.read_text(encoding="utf-8"))
        texts = payload.get("generated_seqs", [])
        if len(texts) != sample_count:
            raise UnsupportedGenerationError(
                f"FLM produced {len(texts)} samples, expected {sample_count}"
            )
        token_ids = [
            tokenizer.tokenizer.encode(text, add_special_tokens=False)
            for text in texts
        ]
        upstream_sha256 = sha256_file(output)
        rows = [
            {"id": index, "text": str(text), "token_ids": token_ids[index]}
            for index, text in enumerate(texts)
        ]
    manifest = generation_manifest(
        effective,
        source_root=root,
        checkpoint=model_path,
        tokenizer={"id": tokenizer.identifier, "revision": tokenizer.revision},
        sampler={
            "name": "flm-hydra sample_eval",
            "parameters": effective.get("sampling", {}),
            "seed": seed,
            "batch_size": int(effective["batch_size"]),
            "precision": "float model; integer token IDs",
            "nfe": int(effective["nfe"]),
            "actual_nfe": int(effective["actual_nfe"]),
        },
        entrypoint="main.py mode=sample_eval",
        elapsed_seconds=elapsed(start),
        execution_mode="recorded upstream subprocess",
        command=command,
    )
    manifest["upstream_output_sha256"] = upstream_sha256
    return rows, manifest
