"""Independent mathematical checks, separate from historical paper parity."""
import math

import numpy as np
import pytest

from dlmbench import Bank
from dlmbench.metrics import entropy
from dlmbench.metrics.jcd import BinaryCTWCoder
from dlmbench.metrics.statistics import energy_distance, frechet_squared, mmd2_from_features


@pytest.mark.parametrize("rows,within,positional", [
    ([(0, 1), (0, 1)], math.log(2), 0.0),
    ([(0, 0), (1, 1)], 0.0, math.log(2)),
    ([(0, 0), (0, 1), (1, 0), (1, 1)], math.log(2) / 2, math.log(2)),
])
def test_entropy_distinguishes_sequence_and_bank_diversity(monkeypatch, rows, within, positional):
    # Supply controlled tokenization; this test is about the entropy estimands.
    monkeypatch.setattr(entropy, "tokenizer_sequences", lambda *_: rows)
    bank = Bank(tuple(range(len(rows))), tuple("fixture" for _ in rows))
    assert entropy.token_entropy_metrics(bank, "h_uni", {}, {})["h_uni"] == pytest.approx(within)
    assert entropy.token_entropy_metrics(bank, "h_pos", {}, {})["h_pos"] == pytest.approx(positional)
    assert entropy.token_entropy_metrics(bank, "h_pool", {}, {})["h_pool"] == pytest.approx(math.log(2))


def test_unbiased_energy_distance_rejects_singletons():
    with pytest.raises(ValueError, match="two|2|sample"):
        energy_distance(np.array([[0.0]]), np.array([[1.0], [2.0]]))


def test_energy_distance_symmetry_translation_and_units():
    left = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    right = np.array([[3.0, 2.0], [5.0, 4.0]])
    distance = energy_distance(left, right)
    assert energy_distance(right, left) == pytest.approx(distance)
    assert energy_distance(left + 100, right + 100) == pytest.approx(distance)
    assert energy_distance(left * 3, right * 3) == pytest.approx(3 * distance)


def test_frechet_has_known_one_dimensional_gaussian_value():
    # Means 0 and 3; population standard deviations 1 and 2.
    left = np.array([[-1.0], [1.0]])
    right = np.array([[1.0], [5.0]])
    assert frechet_squared(left, right) == pytest.approx(3**2 + (1 - 2)**2)


def test_frechet_translation_of_identical_distribution():
    left = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 0.0], [2.0, 4.0]])
    offset = np.array([2.0, -3.0])
    assert frechet_squared(left, left + offset) == pytest.approx(13.0, abs=1e-10)


def test_cosine_mmd_does_not_change_under_per_row_positive_scaling():
    left = np.array([[1.0, 2.0], [3.0, 1.0], [4.0, 2.0]])
    right = np.array([[2.0, 1.0], [0.0, 3.0]])
    expected = mmd2_from_features(left, right)["mmd2_unbiased"]
    actual = mmd2_from_features(left * np.array([[2.0], [5.0], [0.1]]), right * 7)
    assert actual["mmd2_unbiased"] == pytest.approx(expected, abs=1e-12)


def test_frozen_depth_zero_ctw_matches_kt_prediction():
    coder = BinaryCTWCoder(max_depth=0).fit([b"\x00"])
    before = coder.counts_digest()
    code = coder.score(b"\x00\xff", 0)
    expected = -8 * math.log2(8.5 / 9) - 8 * math.log2(0.5 / 9)
    assert code.bits == pytest.approx(expected)
    assert code.n_bytes == 2
    assert coder.counts_digest() == before
