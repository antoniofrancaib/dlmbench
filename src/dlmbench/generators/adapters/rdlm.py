"""Executable inference adapter for the recorded RDLM official sampler loop."""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
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
    """Load once and run the official RDLM batched sampler on one device."""

    del out_dir
    root = Path(source_root).expanduser()
    checkpoint_path = Path(checkpoint).expanduser()
    require_paths(root, "main.py", "run_sample.py", "sampling.py", "data.py")
    source_identity("rdlm", root, entrypoint="sampling.get_sampling_fn")
    if not checkpoint_path.is_file():
        raise UnsupportedGenerationError(f"RDLM checkpoint file does not exist: {checkpoint_path}")
    effective = effective_config(config, sample_count=sample_count, seed=seed)
    start = time.monotonic()

    import torch
    import torch.distributed as dist
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:0"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import data  # type: ignore
    import sampling  # type: ignore
    from hypersphere import Hypersphere  # type: ignore
    from model.ema import ExponentialMovingAverage  # type: ignore
    from utils import seq_utils, utils  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_state = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )
    if "config" not in loaded_state or "model" not in loaded_state:
        raise UnsupportedGenerationError(
            f"RDLM checkpoint is missing required config/model keys: {checkpoint_path}"
        )
    train_cfg = loaded_state["config"]
    sampling_cfg = OmegaConf.load(root / "configs" / "exp" / "sample_lm1b.yaml")
    OmegaConf.set_struct(sampling_cfg, False)
    for key in ("model", "training", "data", "tokens", "original_tokens"):
        if hasattr(train_cfg, key):
            OmegaConf.update(sampling_cfg, key, getattr(train_cfg, key))

    utils.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    nfe = int(effective["nfe"])
    sampling_cfg.sampling.steps = nfe
    batch_size = int(effective["batch_size"])
    sampling_cfg.sampling.batch_per_gpu = batch_size

    token_size = int(train_cfg.tokens)
    vocab_size = token_size + int(train_cfg.sde.prior_dist.add_mask_token)
    manifold = Hypersphere(vocab_size - 1)
    base_max_length = math.ceil(
        math.log(int(train_cfg.original_tokens)) / math.log(token_size)
    )
    sequence_length = int(train_cfg.model.length) * base_max_length
    scheduler = instantiate(train_cfg.scheduler, device=device)
    prior_dist = instantiate(
        train_cfg.sde.prior_dist,
        device=device,
        batch_dims=(batch_size, sequence_length, vocab_size),
    )
    sde = None
    sde_path = checkpoint_path.with_name("sde.pkl")
    if sde_path.is_file():
        try:
            with sde_path.open("rb") as handle:
                alphas, rhos = pickle.load(handle)
            sde = instantiate(
                train_cfg.sde,
                manifold=manifold,
                scheduler=scheduler,
                prior_dist=prior_dist,
                preprocessed=(alphas.to(device), rhos.to(device)),
            )
        except (OSError, EOFError, pickle.UnpicklingError, TypeError, ValueError) as error:
            print(f"[rdlm] preprocessed SDE load failed ({error}); using config")
    if sde is None:
        sde = instantiate(
            sampling_cfg.sde,
            manifold=manifold,
            scheduler=scheduler,
            prior_dist=prior_dist,
            device=device,
        )

    drift_model = instantiate(train_cfg.model).to(device)
    state_dict = loaded_state["model"]
    if "_orig_mod" in next(iter(state_dict)):
        state_dict = {
            key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()
        }
    drift_model.load_state_dict(state_dict)
    drift_model.eval()
    ema = ExponentialMovingAverage(
        drift_model.parameters(), decay=float(train_cfg.training.ema)
    )
    if "ema" in loaded_state:
        ema.load_state_dict(loaded_state["ema"])
        ema.store(drift_model.parameters())
        ema.copy_to(drift_model.parameters())

    upstream_tokenizer = data.get_tokenizer(sampling_cfg.data.train)
    sampling_fn = sampling.get_sampling_fn(
        sampling_cfg,
        sde,
        (batch_size, sequence_length, vocab_size),
        1e-5,
        device,
        proj_fn=lambda sample: sample,
    )
    shift_and_decode = seq_utils.find_bos_and_shift_fn(
        token_size, base_max_length, upstream_tokenizer
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        while len(rows) < sample_count:
            sample = sampling_fn(drift_model).detach().cpu()
            sentences, _ = shift_and_decode(sample)
            for text in sentences:
                if len(rows) >= sample_count:
                    break
                rows.append({"id": len(rows), "text": text})
    if len(rows) != sample_count:
        raise UnsupportedGenerationError(
            f"RDLM produced {len(rows)} samples, expected {sample_count}"
        )
    token_ids = [
        tokenizer.tokenizer.encode(row["text"], add_special_tokens=False) for row in rows
    ]
    for row, token_row in zip(rows, token_ids, strict=True):
        row["token_ids"] = token_row

    manifest = generation_manifest(
        effective,
        source_root=root,
        checkpoint=checkpoint_path,
        tokenizer={"id": tokenizer.identifier, "revision": tokenizer.revision},
        sampler={
            "name": "rdlm-grw official sampler",
            "parameters": effective.get("sampling", {}),
            "seed": seed,
            "batch_size": batch_size,
            "precision": "float model; integer token IDs",
            "nfe": nfe,
            "actual_nfe": int(effective["actual_nfe"]),
        },
        entrypoint="sampling.get_sampling_fn",
        elapsed_seconds=elapsed(start),
        execution_mode="recorded single-process official sampler loop",
        extra={
            "deviations": [
                "single-device process without a DistributedDataParallel wrapper",
                "torch.compile is not applied",
                "sample count is configurable",
            ],
            "rdlm_auxiliary_assets": {
                "sde_pickle": str(sde_path),
                "sde_pickle_present": sde_path.is_file(),
            },
        },
    )
    return rows, manifest
