from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import issparse

from dlmbench.banks import Bank
from dlmbench.metrics import evaluate, list_metrics
from dlmbench.metrics.api import DEFAULT_PAPER_METRICS
from dlmbench.metrics.entropy import token_entropy_metrics
from dlmbench.metrics.features import (
    _modified_precision,
    char_gram_feature_pair,
    self_bleu,
    sentence_bleu,
)
from dlmbench.metrics.jcd import BinaryCTWCoder
from dlmbench.metrics.neural import _resolved_max_tokens, rank_bank_ids
from dlmbench.metrics.protocols import resolve_protocol
from dlmbench.metrics.runner import balanced_selection_indices, select_bank
from dlmbench.metrics.statistics import energy_distance, mmd2_from_features


def bank(texts: list[str], *, start: int = 0) -> Bank:
    return Bank(ids=tuple(range(start, start + len(texts))), texts=tuple(texts))


def test_aliases_resolve_before_validation() -> None:
    assert evaluate(bank(["a b", "c d"]), metrics=["gm"])["requested_metrics"] == ["gradient_moment"]
    assert set(list_metrics()) >= {"gen_ppl", "gradient_moment", "mauve"}


def test_default_selection_is_primary_paper_metrics() -> None:
    assert DEFAULT_PAPER_METRICS == (
        "gen_ppl",
        "h_uni",
        "gradient_moment",
        "jcd",
        "entity_discourse_ed",
        "char_ngram_mmd2",
        "gpt2_large_fbd",
        "mauve",
        "self_bleu",
    )
    # Dispatch only: no metric kernels or pretrained models run.
    with patch("dlmbench.metrics.runner.run_metrics", return_value={}):
        result = evaluate(bank(["a b", "c d"]))
    assert result["requested_metrics"] == list(DEFAULT_PAPER_METRICS)
    assert "encoder_ed" not in result["requested_metrics"]


def test_paper_profile_is_not_claimed_for_custom_overrides() -> None:
    result = evaluate(bank(["a b", "c d"]), metrics=["self_bleu"], self_bleu={"selection_seed": 2})
    assert result["profile"] == "custom"
    assert result["metrics"]["self_bleu"]["protocol"]["metric_version"].startswith("custom:")
    assert result["metrics"]["self_bleu"]["protocol"]["paper_protocol"] == "self-bleu.orders234-v1"
    with pytest.raises(ValueError):
        resolve_protocol("self_bleu", "custom", {"not_a_field": 1})


def test_unknown_profile_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        evaluate(bank(["a b", "c d"]), metrics=["self_bleu"], profile="paper-v2")


def test_self_bleu_reference_example() -> None:
    assert self_bleu(["the cat sat"] * 3, [2, 3])["self_bleu"] == pytest.approx(1.0)
    assert self_bleu(["a b c", "d e f", "g h i"], [2])["self_bleu"] == pytest.approx(0.0)


def test_self_bleu_historical_repeated_gram_estimator() -> None:
    from collections import Counter

    candidate = [("a", "a"), ("a", "a"), ("a", "a"), ("a", "b")]
    references = [Counter({("a", "a"): 2})]
    assert _modified_precision(candidate, references) == (pytest.approx(0.6), 10)
    assert sentence_bleu("a a a a b", ["a a a c"], [2]) == pytest.approx(0.6)


def test_char_ngram_features_and_exact_unbiased_mmd() -> None:
    protocol = {"ngram_orders": [3, 4, 5]}
    left = ["abc", "abc"]
    right = ["abc", "abc"]
    x, y = char_gram_feature_pair(left, right, protocol)
    assert issparse(x) and issparse(y)
    dense_x, dense_y = x.toarray(), y.toarray()
    estimate = mmd2_from_features(x, y)
    assert estimate["mmd2_unbiased"] == pytest.approx(0.0, abs=1e-12)
    manual = mmd2_from_features(dense_x, dense_y)
    assert estimate["mmd2_unbiased"] == pytest.approx(manual["mmd2_unbiased"], abs=1e-15)
    assert estimate["mmd2_biased_raw"] == pytest.approx(manual["mmd2_biased_raw"], abs=1e-15)

    left = np.array([[0.0, 1.0], [0.0, 1.0]])
    right = np.array([[1.0, 0.0], [1.0, 0.0]])
    estimate = mmd2_from_features(left, right)
    assert estimate["mmd2_unbiased"] == pytest.approx(2.0)
    assert estimate["mmd2_biased_raw"] == pytest.approx(2.0)
    assert estimate["negative_unbiased"] is False

    # Matched rows do not define same-sample pairs in the historical U/V
    # estimator: the cross term includes every reference/generated pair.
    left = np.array([[0.0, 1.0], [1.0, 0.0]])
    right = np.array([[1.0, 0.0], [0.0, 1.0]])
    estimate = mmd2_from_features(left, right)
    assert estimate["mmd2_unbiased"] == pytest.approx(-1.0)
    assert estimate["negative_unbiased"] is True


def test_energy_distance_reference_example() -> None:
    left = np.array([[0.0], [0.0]])
    right = np.array([[1.0], [1.0]])
    assert energy_distance(left, right) == pytest.approx(2.0)
    assert energy_distance(left, left) == pytest.approx(0.0)


def test_balanced_mmd_selection_and_self_bleu_selection() -> None:
    left, right = balanced_selection_indices(5, 3, 1729)
    assert left == [0, 1, 2]
    assert right == [2, 1, 0]
    assert balanced_selection_indices(5, 3, 1729, 2) == ([0, 1], [2, 1])
    selected = select_bank(bank([f"text {i}" for i in range(5)]), 2, 1)
    assert selected.metadata["selection"]["indices"] == [2, 3]

    historical = select_bank(bank([f"text {i}" for i in range(5)]), 2, 1729)
    assert historical.metadata["selection"]["indices"] == [4, 1]


def test_paper_profile_remains_paper_even_when_optional_metric_fails() -> None:
    result = evaluate(
        bank(["a b", "c d"]),
        metrics=["self_bleu", "char_ngram_mmd2"],
        profile="paper",
    )
    assert result["profile"] == "paper-v1"
    assert result["profile_source"].startswith(
        "Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs"
    )
    assert "7ef91976441a347c075df5db4bd06a3630ebce2bb37c7912cbbd4e175ea0bf71" in (
        result["profile_source"]
    )
    assert result["metrics"]["char_ngram_mmd2"]["status"] == "unavailable"


def test_profile_custom_override_is_labeled_and_paper_result_is_paper() -> None:
    result = evaluate(bank(["a b", "c d"]), metrics=["self_bleu"])
    assert result["profile"] == "paper-v1"
    custom = evaluate(
        bank(["a b", "c d"]),
        metrics=["self_bleu"],
        self_bleu={"selection_seed": 2},
    )
    assert custom["profile"] == "custom"


def test_ctw_frozen_reference_example() -> None:
    coder = BinaryCTWCoder(max_depth=2)
    coder.nodes[(0, 0)] = [5, 3]
    coder.nodes[(1, 0)] = [3, 1]
    coder.nodes[(2, 0)] = [2, 1]
    assert coder._kt_probability(0, 0, 0) == pytest.approx(5.5 / 9.0)
    assert coder._kt_probability(1, 0, 0) == pytest.approx(3.5 / 5.0)
    assert coder._kt_probability(2, 0, 0) == pytest.approx(2.5 / 4.0)


def test_ctw_non_ascii_and_document_boundaries() -> None:
    training = ["é" * 20, "é" * 20]
    target = "é" * 10
    coder = BinaryCTWCoder(max_depth=4).fit(text.encode("utf-8") for text in training)
    before = coder.counts_digest()
    first = coder.score(target.encode("utf-8"), 0)
    second = coder.score(target.encode("utf-8"), 0)
    assert first == second
    assert coder.counts_digest() == before
    assert first.bits > 0
    assert first.n_bytes == len(target.encode("utf-8"))


def test_jcd_declared_split_sizes_and_coder_state_provenance() -> None:
    from dlmbench.metrics.jcd import jcd

    generated = bank([f"generated text {i}" for i in range(1024)])
    reference = bank([f"reference text {i}" for i in range(1024)], start=5000)
    protocol = {
        "depth": 2,
        "fit_size_per_source": 512,
        "heldout_size_per_source": 512,
        "split_seeds": [0],
    }
    result = jcd(generated, reference, protocol)
    split = result["splits"][0]
    assert len(split["reference"]["fit_indices"]) == 512
    assert len(split["reference"]["heldout_indices"]) == 512
    assert len(split["generated"]["fit_indices"]) == 512
    assert len(split["generated"]["heldout_indices"]) == 512
    assert len(result["coder_states_sha256"]) == 1
    assert result["coder_states_sha256"][0]["reference_state_sha256"]
    assert result["coder_states_sha256"][0]["generated_state_sha256"]


def test_jcd_heldout_split_stops_at_declared_cap() -> None:
    from dlmbench.metrics.jcd import _split

    generated = bank([f"generated text {index}" for index in range(1030)])
    fit, held, split = _split(generated, 0, 512, 512)
    assert len(fit) == 512
    assert len(held) == 512
    assert split["heldout_size"] == 512
    assert len(set(fit) | set(held)) == 1024


def test_evaluate_reports_missing_reference_and_cli_exit_is_nonzero(tmp_path: Path) -> None:
    generated = bank(["a b c", "d e f", "g h i"])
    result = evaluate(generated, metrics=["mmd"])
    assert result["metrics"]["char_ngram_mmd2"]["status"] == "unavailable"
    assert result["metrics"]["char_ngram_mmd2"]["reason"] == "reference bank is required"


def test_entropy_metrics_require_matching_lengths() -> None:
    generated = bank(["a b c", "d e f"])
    result = evaluate(generated, metrics=["self_bleu"])
    assert result["metrics"]["self_bleu"]["status"] == "ok"
    assert math.isfinite(result["metrics"]["self_bleu"]["self_bleu"])


def test_h_pos_text_protocol_retokenizes_rather_than_trusting_rows() -> None:
    class TinyTokenizer:
        def __call__(self, texts, add_special_tokens=False, truncation=None):
            assert add_special_tokens is False
            assert truncation is False
            return {"input_ids": [(0, 1, 2), (3, 4, 5)]}

    generated = Bank(
        ids=(0, 1),
        texts=("a b c", "d e f"),
        token_ids=((1, 1, 2), (3, 1, 4)),
    )
    protocol = {
        "tokenizer": "unused",
        "revision": "unused",
        "input": "text",
        "add_special_tokens": False,
        "truncation": "none",
    }
    import dlmbench.metrics.entropy as entropy

    with patch.object(entropy, "_load_tokenizer", return_value=TinyTokenizer()):
        result = token_entropy_metrics(generated, "h_pos", protocol, {})
    # The tiny tokenizer's 0/1/2 and 3/4/5 rows differ at every position, so
    # the result must be ln(2).  Native rows deliberately differ and cannot be
    # the source of this value.
    assert result["h_pos"] == pytest.approx(math.log(2.0))


def test_entropy_diagnostics_have_controlled_expected_values() -> None:
    protocol = {
        "tokenizer": "unused",
        "revision": "unused",
        "input": "token_ids",
        "add_special_tokens": False,
        "truncation": "none",
    }
    shared: dict[str, object] = {}
    sequences = ((1, 1, 2), (3, 1, 4))
    generated = Bank(
        ids=(0, 1),
        texts=("a", "b"),
        metadata={"tokenizer": "unused"},
        token_ids=sequences,
    )

    # The first and second sequences have positive within-sequence entropy.
    # Position 1 is deterministic, while positions 0 and 2 each contain two
    # different tokens and therefore have one bit of cross-sample entropy.
    h_uni = token_entropy_metrics(generated, "h_uni", protocol, shared)
    expected_h_uni = sum(
        -sum(
            (row.count(token) / len(row)) * math.log(row.count(token) / len(row))
            for token in set(row)
        )
        for row in sequences
    ) / len(sequences)
    assert h_uni["h_uni"] == pytest.approx(expected_h_uni)
    h_pos = token_entropy_metrics(generated, "h_pos", protocol, shared)
    assert h_pos["h_pos"] == pytest.approx(2.0 * math.log(2.0) / 3.0)
    assert h_pos["position_entropy_min"] == pytest.approx(0.0)
    assert h_pos["position_entropy_max"] == pytest.approx(math.log(2.0))

    # Shuffling the second row does not change the active positional entropy
    # labels; the pooled distribution is (3, 2, 1), not uniform.
    generated = Bank(
        ids=(0, 1),
        texts=("a", "b"),
        metadata={"tokenizer": "unused"},
        token_ids=((1, 1, 2), (2, 1, 3)),
    )
    h_pos = token_entropy_metrics(generated, "h_pos", protocol, shared)
    assert h_pos["position_entropy_min"] == pytest.approx(0.0)
    assert h_pos["position_entropy_max"] == pytest.approx(math.log(2.0))
    assert h_pos["h_pos"] == pytest.approx(2.0 * math.log(2.0) / 3.0)
    h_pool = token_entropy_metrics(generated, "h_pool", protocol, shared)
    expected_h_pool = -(
        3.0 / 6.0 * math.log(3.0 / 6.0)
        + 2.0 / 6.0 * math.log(2.0 / 6.0)
        + 1.0 / 6.0 * math.log(1.0 / 6.0)
    )
    assert h_pool["h_pool"] == pytest.approx(expected_h_pool)


def test_input_identity_includes_root_bank_token_fields() -> None:
    generated = Bank(
        ids=(1, 2),
        texts=("a b", "c d"),
        metadata={"id": "tokens", "tokenizer": "gpt2"},
        token_ids=((1, 2), (3, 4)),
    )
    identity = evaluate(generated, metrics=["self_bleu"])["generated"]
    assert identity["ids_sha256"]
    assert identity["token_ids_sha256"]
    assert identity["token_ids"]["declared_tokenizer"] == "gpt2"
    assert identity["token_ids"]["tokenizer_identity_verified"] is False


def test_structured_tokenizer_metadata_allows_matching_native_rows() -> None:
    from dlmbench.metrics.neural import _declared_token_rows

    class Tokenizer:
        name_or_path = "gpt2"

    bank = Bank(
        ids=(1, 2),
        texts=("a b", "c d"),
        metadata={"tokenizer": {"id": "gpt2", "revision": "revision"}},
        token_ids=((1, 2), (3, 4)),
    )
    protocol = {"scorer": "gpt2", "revision": "revision", "token_id_input": "auto"}
    assert _declared_token_rows(bank, Tokenizer(), protocol) == [[1, 2], [3, 4]]
    strict = {**protocol, "token_id_input": "token_ids"}
    assert _declared_token_rows(bank, Tokenizer(), strict) == [[1, 2], [3, 4]]


def test_structured_mismatching_tokenizer_rejects_ids_or_retokenizes() -> None:
    from dlmbench.metrics.neural import _declared_token_rows

    class Tokenizer:
        name_or_path = "gpt2"

    bank = Bank(
        ids=(1, 2),
        texts=("a b", "c d"),
        metadata={"tokenizer": {"id": "bert-base-uncased", "revision": "other"}},
        token_ids=((1, 2), (3, 4)),
    )
    protocol = {"scorer": "gpt2", "revision": "revision", "token_id_input": "auto"}
    assert _declared_token_rows(bank, Tokenizer(), protocol) is None
    strict = {**protocol, "token_id_input": "token_ids"}
    with pytest.raises(ValueError, match="tokenizer space"):
        _declared_token_rows(bank, Tokenizer(), strict)


def test_matching_id_with_revision_mismatch_never_reuses_token_ids() -> None:
    from dlmbench.metrics.neural import _declared_token_rows

    class Tokenizer:
        name_or_path = "gpt2"

    bank = Bank(
        ids=(1, 2),
        texts=("a b", "c d"),
        metadata={"tokenizer": {"id": "gpt2", "revision": "old"}},
        token_ids=((1, 2), (3, 4)),
    )
    protocol = {"scorer": "gpt2", "revision": "new", "token_id_input": "auto"}
    assert _declared_token_rows(bank, Tokenizer(), protocol) is None
    with pytest.raises(ValueError, match="revision mismatch"):
        _declared_token_rows(bank, Tokenizer(), {**protocol, "token_id_input": "token_ids"})


def test_recorded_gpt2_tokenizer_alias_and_independent_revision_are_compatible() -> None:
    from dlmbench.metrics.neural import _declared_token_rows

    class Tokenizer:
        name_or_path = "openai-community/gpt2"

    bank = Bank(
        ids=(1, 2),
        texts=("a b", "c d"),
        metadata={"tokenizer": {"id": "openai-community/gpt2", "revision": "607a"}},
        token_ids=((1, 2), (3, 4)),
    )
    protocol = {
        "scorer": "gpt2-large",
        "revision": "weights-revision",
        "tokenizer_revision": "607a",
        "token_id_input": "auto",
    }
    assert _declared_token_rows(bank, Tokenizer(), protocol) == [[1, 2], [3, 4]]


def test_gen_ppl_context_resolution_and_mauve_selection() -> None:
    protocol = {"max_tokens": "bank_declared_or_model_context"}
    generated = bank(["a", "b"], start=10)
    assert _resolved_max_tokens(generated, {**protocol, **generated.metadata}, 128) == 128
    generated.metadata["seq_len"] = "100"
    assert _resolved_max_tokens(generated, protocol, 128) == 100
    assert _resolved_max_tokens(generated, protocol, 64) == 64
    generated.metadata["length_policy"] = {"sequence_length": 96}
    assert _resolved_max_tokens(generated, {**protocol, **generated.metadata}, 128) == 100

    ranked = rank_bank_ids(generated, "p", 0, 20260825)
    assert sorted(ranked) == [0, 1]
    assert rank_bank_ids(generated, "p", 0, 20260825) == ranked


def test_self_bleu_preserves_unicode_and_small_bank_shuffle():
    from dlmbench.metrics.features import sentence_bleu
    assert sentence_bleu("Ａ B", ["A B"], [2]) == 0.0
    selected = select_bank(bank([f"text {i}" for i in range(5)]), 200, 1)
    assert selected.metadata["selection"]["indices"] == [2, 3, 4, 0, 1]


def test_description_overrides_cannot_mislabel_the_estimator():
    with pytest.raises(ValueError, match="unsupported"):
        resolve_protocol("char_ngram_mmd2", "custom", {"kernel": "rbf"})
