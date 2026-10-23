"""Generator catalog loading and status resolution."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..provenance import sha256_file
from . import UnsupportedGenerationError

CATALOG_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "resources" / "generators" / "catalog.json"
    if not path.is_file():
        return {"schema_version": CATALOG_SCHEMA_VERSION, "generators": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise UnsupportedGenerationError(
            f"unsupported generator catalog schema {payload.get('schema_version')!r}"
        )
    return payload


def load_catalog() -> dict[str, Any]:
    """Return a defensive copy of the packaged generator catalog."""

    return json.loads(json.dumps(_catalog()))


def generators() -> list[dict[str, Any]]:
    return list(load_catalog()["generators"])


def generator(generator_id: str) -> dict[str, Any]:
    matches = [row for row in generators() if row.get("id") == generator_id]
    if len(matches) != 1:
        raise UnsupportedGenerationError(f"unknown generator {generator_id!r}")
    return matches[0]


def catalog_sha256() -> str:
    path = Path(__file__).resolve().parents[1] / "resources" / "generators" / "catalog.json"
    if not path.is_file():
        raise UnsupportedGenerationError("generator catalog resource is missing")
    return sha256_file(path)
