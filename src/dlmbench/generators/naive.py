"""Exact implementations of the paper's four naive sampler families."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from . import UnsupportedGenerationError


def historical_top_token_ids(values: Sequence[float], k: int) -> np.ndarray:
    """Return top-k IDs with the recorded NumPy argpartition behavior."""

    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 1:
        raise UnsupportedGenerationError("marginal must be a non-empty vector")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise UnsupportedGenerationError("marginal must contain finite non-negative values")
    if type(k) is not int or not 1 <= k <= probabilities.size:
        raise UnsupportedGenerationError(
            f"k must be between 1 and the vocabulary size ({probabilities.size})"
        )
    selected = np.argpartition(-probabilities, k - 1)[:k]
    return selected[np.argsort(-probabilities[selected])].astype(np.int64)


def historical_topk_iid(
    probabilities: Sequence[float], *, k: int, shape: tuple[int, int], seed: int
) -> np.ndarray:
    """Draw token IDs independently with the historical NumPy RNG."""

    values = np.asarray(probabilities, dtype=np.float64)
    ids = historical_top_token_ids(values, k)
    weights = values[ids]
    mass = weights.sum()
    if mass <= 0:
        raise UnsupportedGenerationError("top-k tokens have zero total probability mass")
    weights = weights / mass
    if any(type(value) is not int or value < 1 for value in shape):
        raise UnsupportedGenerationError("sample count and sequence length must be positive")
    return np.random.default_rng(seed).choice(ids, size=shape, replace=True, p=weights)


def historical_mirror(
    probabilities: Sequence[float], *, source_k: int, n_samples: int, seq_len: int, seed: int
) -> np.ndarray:
    """Draw a floor-half block, mirror it, and append its first token if odd."""

    half = seq_len // 2
    first = historical_topk_iid(
        probabilities,
        k=source_k,
        shape=(n_samples, half),
        seed=seed,
    )
    pieces = [first, first.copy()]
    if seq_len % 2:
        pieces.append(first[:, :1])
    return np.concatenate(pieces, axis=1)


def historical_periodic(probabilities: Sequence[float], *, k: int, seq_len: int) -> np.ndarray:
    """Repeat the descending-frequency top-k IDs to the target length."""

    if type(seq_len) is not int or seq_len < 1:
        raise UnsupportedGenerationError("sequence length must be positive")
    ids = historical_top_token_ids(probabilities, k).tolist()
    repeats = seq_len // len(ids) + 1
    return np.asarray((ids * repeats)[:seq_len], dtype=np.int64)


def historical_phrase_sample(
    phrases: Sequence[Sequence[int]], *, n_samples: int, seq_len: int, seed: int
) -> list[list[int]]:
    """Choose phrases with the historical Python random.choice loop."""

    try:
        bank = [list(map(int, phrase)) for phrase in phrases]
    except TypeError as error:
        raise UnsupportedGenerationError("phrase bank rows must be token-ID sequences") from error
    if not bank or any(not phrase for phrase in bank):
        raise UnsupportedGenerationError("phrase bank must be non-empty")
    if type(n_samples) is not int or n_samples < 1 or type(seq_len) is not int or seq_len < 1:
        raise UnsupportedGenerationError("sample count and sequence length must be positive")
    rng = random.Random(seed)
    rows: list[list[int]] = []
    for _ in range(n_samples):
        unit: list[int] = []
        while len(unit) < seq_len:
            unit.extend(rng.choice(bank))
        rows.append(unit[:seq_len])
    return rows
