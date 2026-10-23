"""Executable inference adapter for the pinned LangFlow release."""

from __future__ import annotations

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

LANGFLOW_DECODING = {
    "method": "transformers.batch_decode",
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": None,
    "note": "tokenizer default",
}


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
    """Run the recorded LangFlow inference path and return ordered Bank rows."""

    del out_dir
    root = Path(source_root).expanduser()
    model_path = Path(checkpoint).expanduser()
    require_paths(root, "inference.py", "langflow/model.py", "langflow/config.py", "langflow/config.json")
    if not model_path.is_file():
        raise UnsupportedGenerationError(f"LangFlow checkpoint file does not exist: {model_path}")

    source_identity("langflow", root, entrypoint="inference.py")

    effective = effective_config(config, sample_count=sample_count, seed=seed)
    start = time.monotonic()
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from langflow import LangFlow, LangFlowConfig  # type: ignore

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_config = LangFlowConfig.from_pretrained(str(root / "langflow"))
    state_dict = load_file(str(model_path), device="cpu")
    embedding_key = "backbone.vocab_embed.embedding"
    try:
        flow_config.vocab_size = int(state_dict[embedding_key].shape[0])
    except KeyError as error:
        raise UnsupportedGenerationError(
            f"LangFlow checkpoint is missing required tensor {embedding_key!r}"
        ) from error
    model = LangFlow(flow_config)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    tokenizer_path = Path(tokenizer.path)
    upstream_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    sequence_length = int(effective["sequence_length"])
    steps = int(effective["steps"])
    batch_size = int(effective["batch_size"])
    rows: list[dict[str, Any]] = []
    produced = 0
    with torch.inference_mode():
        while produced < sample_count:
            batch = min(batch_size, sample_count - produced)
            samples = model.generate_samples(
                num_samples=batch,
                seq_length=sequence_length,
                num_steps=steps,
                device=device,
            ).detach().cpu().tolist()
            texts = upstream_tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            if len(samples) != len(texts) or len(samples) != batch:
                raise UnsupportedGenerationError("LangFlow returned a malformed sample batch")
            rows.extend(
                {"id": produced + index, "text": text, "token_ids": token_ids}
                for index, (text, token_ids) in enumerate(zip(texts, samples, strict=True))
            )
            produced += batch

    if len(rows) != sample_count:
        raise UnsupportedGenerationError(
            f"LangFlow produced {len(rows)} samples, expected {sample_count}"
        )
    manifest = generation_manifest(
        effective,
        source_root=root,
        checkpoint=model_path,
        tokenizer={"id": tokenizer.identifier, "revision": tokenizer.revision},
        sampler={
            "name": "langflow-euler-edm",
            "parameters": effective.get("sampling", {}),
            "seed": seed,
            "batch_size": batch_size,
            "precision": "float32 model; integer token IDs",
            "nfe": int(effective["nfe"]),
            "actual_nfe": int(effective["actual_nfe"]),
        },
        entrypoint="langflow.generate_samples",
        elapsed_seconds=elapsed(start),
        execution_mode="recorded in-process upstream sampler",
        extra={"command": None},
        decoding=LANGFLOW_DECODING,
    )
    return rows, manifest
