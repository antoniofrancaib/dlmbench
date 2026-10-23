"""Lazy dispatch for executable upstream generation adapters."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .. import UnsupportedGenerationError

ADAPTERS = {
    "flm-hydra": "flm",
    "langflow": "langflow",
    "rdlm": "rdlm",
    "sdtt-official": "sdtt",
    "elf": "elf",
    "cobit": "cobit",
}


def execute(
    config: dict[str, Any],
    *,
    source_root: Path,
    checkpoint: Path,
    tokenizer: Path,
    out_dir: Path,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one executable upstream adapter and return Bank rows plus manifest data."""

    family = str(config.get("family"))
    try:
        module_name = ADAPTERS[family]
    except KeyError as error:
        raise UnsupportedGenerationError(f"no executable adapter is registered for family {family!r}") from error
    module = importlib.import_module(f".{module_name}", package=__package__)
    return module.execute(
        config,
        source_root=source_root,
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        out_dir=out_dir,
        sample_count=sample_count,
        seed=seed,
    )


__all__ = ["ADAPTERS", "execute"]
