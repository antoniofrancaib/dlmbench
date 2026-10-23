"""Immutable paper profiles and custom protocol resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PAPER_PROFILE_SOURCE = (
    "Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs "
    "New Metrics (2026-10-09), manuscript SHA-256 "
    "7ef91976441a347c075df5db4bd06a3630ebce2bb37c7912cbbd4e175ea0bf71; "
    "metric definitions in src/dlmbench/metrics/protocols.py; "
    "recorded measurements in src/dlmbench/resources/paper/"
)
PAPER_PROFILE: dict[str, dict[str, Any]] = {
    "gen_ppl": {
        "metric_version": "gen_ppl.v1",
        "scorer": "gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "tokenizer": "gpt2",
        "tokenizer_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "batch_size": 8,
        "input": "text",
        "add_special_tokens": False,
        "truncation": "first_max_tokens",
        "max_tokens": "bank_declared_or_model_context",
        "token_id_input": "auto",
        "tokenizer_identity": "required_before_token_id_use",
        "length_status": "reported",
        "aggregation": "exp(mean per-sample mean next-token NLL)",
        "units": "per-token perplexity",
    },
    "h_uni": {
        "metric_version": "entropy.token-id-v1",
        "tokenizer": "gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "aggregation": "mean per-sample token-histogram entropy",
        "units": "nats per token",
        "input": "text",
        "add_special_tokens": False,
        "truncation": "none",
    },
    "h_pos": {
        "metric_version": "entropy.token-id-v1",
        "tokenizer": "gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "aggregation": "mean over positions of plug-in cross-sample token entropy",
        "units": "nats per token",
        "input": "text",
        "add_special_tokens": False,
        "truncation": "none",
        "missing_position_policy": "active positions only; matched bank sizes required",
    },
    "h_pool": {
        "metric_version": "entropy.token-id-v1",
        "tokenizer": "gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "aggregation": "entropy of pooled token frequencies over active tokens",
        "units": "nats per token",
        "diagnostic": True,
        "input": "text",
        "add_special_tokens": False,
        "truncation": "none",
    },
    "gradient_moment": {
        "metric_version": "gradient_moment.gpt2-v1",
        "scorer": "gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "batch_size": 4,
        "max_tokens": "bank_declared_or_model_context",
        "tokenizer_identity": "required_for_token_ids",
        "truncation": "first_max_tokens",
        "gradient": "d mean per-sequence NLL / d scorer parameters",
        "token_id_input": "auto",
        "distance": "squared_l2_difference_of_mean_gradients",
    },
    "jcd": {
        "metric_version": "jcd.ctw20-frozen-v1",
        "input": "lossless UTF-8 document bytes",
        "depth": 20,
        "alphabet": "utf8_bytes",
        "bit_order": "most_significant_first",
        "prior": "krichevsky_trofimov",
        "fit_size_per_source": 512,
        "heldout_size_per_source": 512,
        "split_seeds": [0, 1, 2, 3, 4],
        "normalization": "bits per byte",
        "directions": ["coverage", "fidelity"],
        "aggregate": "mean of directional means",
    },
    "entity_discourse_ed": {
        "metric_version": "ed.entity-discourse-v1",
        "feature_count": 34,
        "features": "handcrafted entity and discourse statistics",
        "standardization": "reference mean and reference standard deviation",
        "distance": "unbiased euclidean energy distance",
    },
    "char_ngram_mmd2": {
        "metric_version": "mmd.char35-cosine-v1",
        "unicode_normalization": "NFKC",
        "whitespace": "collapse_runs_to_ascii_space_and_strip",
        "character_case": "preserve",
        "ngram_orders": [3, 4, 5],
        "feature_values": "per-document relative frequencies",
        "boundary_sentinels": ["<BOS>", "<EOS>"],
        "feature_format": "exact_scipy_sparse",
        "kernel": "cosine_linear",
        "estimator": "unbiased_u_statistic",
        "primary_score": "mmd2_unbiased",
        "sample_count": 1024,
        "selection_seed": 1729,
        "clipping": "none_for_primary_score",
    },
    "gpt2_large_fbd": {
        "metric_version": "fbd.gpt2-large-maxpool-v1",
        "featurizer": "gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "tokenizer": "gpt2",
        "tokenizer_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "max_tokens": 512,
        "pooling": "max",
        "batch_size": 16,
        "input": "text",
        "add_special_tokens": False,
        "distance": "squared_frechet_gaussian_distance",
    },
    "mauve": {
        "metric_version": "mauve.gpt2-large-v1",
        "package": "mauve-text==0.4.0",
        "featurizer": "gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "tokenizer": "gpt2",
        "tokenizer_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "max_tokens": 512,
        "batch_size": 16,
        "kmeans": "explain_90_variance_5_restarts_500_iterations",
        "scaling_factor": 5,
        "subset_seeds": [0, 1, 2, 3, 4],
        "selection_algorithm": "sha256_rank_v1",
        "selection_base_seed": 20260825,
        "evaluation_order": "canonical_sample_id",
        "internal_seed": 25,
        "subset_size": 1024,
        "minimum_subset_size": 2,
        "score": "mean over five n=1024 subset seeds",
    },
    "encoder_ed": {
        "metric_version": "encoder-ed.reference-whitening-v1",
        "encoders": [
            "gpt2-large",
            "bert-base-uncased",
            "roberta-base",
            "all-mpnet-base-v2",
            "intfloat/e5-base-v2",
        ],
        "max_tokens": 512,
        "batch_size": 16,
        "pooling": {
            "gpt2-large": "mean_non_padding",
            "bert-base-uncased": "mean_non_special",
            "roberta-base": "mean_non_special",
            "all-mpnet-base-v2": "sentence_transformers_official",
            "intfloat/e5-base-v2": "mean_attention_mask_passage_prefix",
        },
        "unit_normalize": True,
        "reference_pca_whitening": 128,
        "distance": "unbiased euclidean energy distance",
        "encoder": "all",
        "revisions": {
            "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
            "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "roberta-base": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            "all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "intfloat/e5-base-v2": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        },
        "revision_source": (
            "Model revisions pinned in src/dlmbench/metrics/protocols.py"
        ),
    },
    "encoder_gaussian_rbf_mmd2": {
        "metric_version": "encoder-mmd.gaussian-rbf-v1",
        "encoders": [
            "gpt2-large",
            "bert-base-uncased",
            "roberta-base",
            "all-mpnet-base-v2",
            "intfloat/e5-base-v2",
        ],
        "max_tokens": 512,
        "batch_size": 16,
        "pooling": {
            "gpt2-large": "mean_non_padding",
            "bert-base-uncased": "mean_non_special",
            "roberta-base": "mean_non_special",
            "all-mpnet-base-v2": "sentence_transformers_official",
            "intfloat/e5-base-v2": "mean_attention_mask_passage_prefix",
        },
        "unit_normalize": True,
        "reference_pca_whitening": 128,
        "kernel": "unweighted_average_gaussian_rbf",
        "bandwidths": [0.125, 0.25, 0.5, 1.0, 2.0],
        "estimator": "unbiased_u_statistic",
        "encoder": "all",
        "revisions": {
            "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
            "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "roberta-base": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
            "all-mpnet-base-v2": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "intfloat/e5-base-v2": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        },
        "revision_source": (
            "Model revisions pinned in src/dlmbench/metrics/protocols.py"
        ),
    },
    "self_bleu": {
        "metric_version": "self-bleu.orders234-v1",
        "tokenization": "lower_then_whitespace_split",
        "ngram_orders": [2, 3, 4],
        "references": "all other samples in the selected candidate subset",
        "precision": "historical_repeated_occurrence_clipping_per_order",
        "sample_count": 200,
        "selection_seed": 1,
        "aggregation": "mean over candidates and orders",
        "candidate_fraction": 0.5,
        "selection_order": "python_random_shuffle_then_prefix",
    },
}

# Descriptive fields define the estimator; accepting changes to them without
# changing the calculation would produce false effective-protocol metadata.
CUSTOM_PARAMETERS = {
    "gen_ppl": {"scorer", "tokenizer", "revision", "tokenizer_revision", "batch_size",
                "max_tokens", "token_id_input"},
    "gradient_moment": {"scorer", "revision", "batch_size", "max_tokens", "token_id_input"},
    "h_uni": {"tokenizer", "revision", "input", "add_special_tokens"},
    "h_pos": {"tokenizer", "revision", "input", "add_special_tokens"},
    "h_pool": {"tokenizer", "revision", "input", "add_special_tokens"},
    "jcd": {"depth", "fit_size_per_source", "heldout_size_per_source", "split_seeds"},
    "entity_discourse_ed": set(),
    "char_ngram_mmd2": {"ngram_orders", "sample_count", "selection_seed"},
    "gpt2_large_fbd": {"featurizer", "revision", "tokenizer", "tokenizer_revision",
                       "max_tokens", "batch_size"},
    "mauve": {"featurizer", "revision", "max_tokens", "batch_size", "scaling_factor",
              "subset_seeds", "selection_base_seed", "internal_seed", "subset_size",
              "minimum_subset_size"},
    "encoder_ed": {"encoders", "encoder", "revisions", "max_tokens", "batch_size",
                   "reference_pca_whitening"},
    "encoder_gaussian_rbf_mmd2": {"encoders", "encoder", "revisions", "max_tokens",
                                 "batch_size", "reference_pca_whitening", "bandwidths"},
    "self_bleu": {"ngram_orders", "sample_count", "selection_seed", "candidate_fraction"},
}


def paper_profile_source() -> str:
    """Return the immutable source provenance for the paper profile."""

    return PAPER_PROFILE_SOURCE


def paper_protocol(metric: str) -> dict[str, Any]:
    try:
        return deepcopy(PAPER_PROFILE[metric])
    except KeyError as exc:
        raise ValueError(f"unknown paper metric: {metric}") from exc


def resolve_protocol(metric: str, profile: str, overrides: dict[str, Any]) -> dict[str, Any]:
    base = paper_protocol(metric)
    if profile == "paper-v1":
        return base
    requested = dict(overrides)
    unknown = sorted(set(requested) - CUSTOM_PARAMETERS[metric])
    if unknown:
        raise ValueError(f"unknown or unsupported {metric} protocol overrides: {unknown}")
    resolved = {**base, **requested}
    resolved["metric_version"] = f"custom:{base['metric_version']}"
    resolved["paper_protocol"] = base["metric_version"]
    return resolved
