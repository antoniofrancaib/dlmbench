from __future__ import annotations

import math
import random

import numpy as np
import pytest

from dlmbench.generators import (
    UnsupportedGenerationError,
    historical_mirror,
    historical_periodic,
    historical_phrase_sample,
    historical_top_token_ids,
    historical_topk_iid,
)


def test_top_token_ids_preserve_historical_argpartition_order() -> None:
    values = np.asarray([0.3, 0.1, 0.9, 0.8, 0.2, 0.7, 0.4, 0.6])
    np.testing.assert_array_equal(historical_top_token_ids(values, 4), [2, 3, 5, 7])


def test_topk_iid_is_restricted_renormalized_and_seed_stable() -> None:
    values = np.asarray([0.1, 0.7, 0.0, 0.2])
    first = historical_topk_iid(values, k=3, shape=(1000, 20), seed=7)
    second = historical_topk_iid(values, k=3, shape=(1000, 20), seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (1000, 20)
    assert set(np.unique(first)).issubset({1, 3, 0})
    frequencies = np.bincount(first.ravel(), minlength=4) / first.size
    expected = np.asarray([0.1, 0.7, 0.2]) / 1.0
    for token, expected_mass in zip((0, 1, 3), expected, strict=True):
        assert math.isclose(frequencies[token], expected_mass, abs_tol=0.03)
    assert frequencies[2] == 0


def test_mirror_uses_floor_half_and_repeats_first_token_for_odd_length() -> None:
    values = np.asarray([0.6, 0.0, 0.4])
    observed = historical_mirror(values, source_k=2, n_samples=3, seq_len=5, seed=4)
    block = historical_topk_iid(values, k=2, shape=(3, 2), seed=4)
    expected = np.concatenate([block, block.copy(), block[:, :1]], axis=1)
    np.testing.assert_array_equal(observed, expected)


def test_periodic_is_frequency_ordered_and_identical_for_every_sample() -> None:
    values = np.asarray([0.1, 0.5, 0.4])
    row = historical_periodic(values, k=3, seq_len=7)
    np.testing.assert_array_equal(row, [1, 2, 0, 1, 2, 0, 1])
    assert np.array_equal(row, historical_periodic(values, k=3, seq_len=7))


def test_phrase_sampler_matches_python_random_and_truncates() -> None:
    bank = [[1, 2, 3], [4, 5, 6]]
    rng = random.Random(7)
    expected = []
    for _ in range(2):
        unit = []
        while len(unit) < 7:
            unit.extend(rng.choice(bank))
        expected.append(unit[:7])
    assert historical_phrase_sample(bank, n_samples=2, seq_len=7, seed=7) == expected


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (historical_top_token_ids, {"k": 0}),
        (historical_topk_iid, {"k": 1, "shape": (0, 2), "seed": 1}),
        (historical_mirror, {"source_k": 1, "n_samples": 1, "seq_len": 0, "seed": 1}),
        (historical_periodic, {"k": 1, "seq_len": 0}),
        (historical_phrase_sample, {"n_samples": 1, "seq_len": 0, "seed": 1}),
    ],
)
def test_naive_inputs_fail_closed(function, kwargs) -> None:
    with pytest.raises(UnsupportedGenerationError):
        function([0.5, 0.5], **kwargs)
