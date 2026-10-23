"""Metric dispatch without eager heavy imports."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from ..banks import Bank


def unavailable(metric: str, protocol: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"metric": metric, "status": "unavailable", "reason": reason, "protocol": protocol}


def run_metrics(
    *,
    generated: Bank,
    reference: Bank | None,
    metrics: list[str],
    profile: str,
    device: str | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    # Local imports keep the public API lightweight and avoid accidental
    # import-time torch/transformers/scipy/mauve dependencies.
    from .entropy import token_entropy_metrics
    from .features import char_gram_feature_pair, entity_discourse_matrix, self_bleu
    from .protocols import resolve_protocol
    from .statistics import energy_distance, mmd2_from_features

    output: dict[str, Any] = {}
    shared: dict[str, Any] = {}
    requested = set(metrics)
    for metric in metrics:
        protocol = resolve_protocol(metric, profile, kwargs.get(metric, {}))
        try:
            if metric == "gen_ppl":
                from .neural import gen_ppl

                output[metric] = gen_ppl(generated, protocol, device)
            elif metric in {"h_uni", "h_pos", "h_pool"}:
                output[metric] = token_entropy_metrics(generated, metric, protocol, shared)
            elif metric == "gradient_moment":
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    from .neural import gradient_moment

                    output[metric] = gradient_moment(generated, reference, protocol, device)
            elif metric == "jcd":
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    from .jcd import jcd

                    output[metric] = jcd(generated, reference, protocol)
            elif metric == "entity_discourse_ed":
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    left = entity_discourse_matrix(reference.texts)
                    right = entity_discourse_matrix(generated.texts)
                    mean = left.mean(axis=0)
                    scale = left.std(axis=0, ddof=0)
                    scale = np.where(scale > 0, scale, 1.0)
                    value = energy_distance((left - mean) / scale, (right - mean) / scale)
                    output[metric] = {
                        "metric": metric,
                        "status": "ok",
                        "entity_discourse_ed": value,
                        "protocol": protocol,
                    }
            elif metric == "char_ngram_mmd2":
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    # Build one shared reference/generated vocabulary so the
                    # two kernel matrices occupy the same exact feature space.
                    left_indices, right_indices = balanced_selection_indices(
                        len(reference),
                        len(generated),
                        protocol["selection_seed"],
                        protocol["sample_count"],
                    )
                    left, right = char_gram_feature_pair(
                        [reference.texts[index] for index in left_indices],
                        [generated.texts[index] for index in right_indices],
                        protocol,
                    )
                    output[metric] = {
                        "metric": metric,
                        "status": "ok",
                        "feature_cache": {
                            "format": "exact_scipy_sparse",
                            "shared_vocabulary": True,
                            "feature_shape": [int(left.shape[0]), int(left.shape[1])],
                        },
                        **mmd2_from_features(left, right),
                        "selection": {
                            "algorithm": "numpy_default_rng_permutation_prefix",
                            "seed": protocol["selection_seed"],
                            "reference_indices": left_indices,
                            "generated_indices": right_indices,
                            "reference_ids": [reference.ids[index] for index in left_indices],
                            "generated_ids": [generated.ids[index] for index in right_indices],
                        },
                        "protocol": protocol,
                    }
            elif metric == "gpt2_large_fbd":
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    from .neural import fbd

                    output[metric] = fbd(generated, reference, protocol, device)
            elif metric == "mauve":
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    from .neural import mauve

                    output[metric] = mauve(generated, reference, protocol, device)
            elif metric in {"encoder_ed", "encoder_gaussian_rbf_mmd2"}:
                if reference is None:
                    output[metric] = unavailable(metric, protocol, "reference bank is required")
                else:
                    from .neural import encoder_space_metrics

                    output[metric] = encoder_space_metrics(
                        metric, generated, reference, protocol, device
                    )
            elif metric == "self_bleu":
                if any(not text.split() for text in generated.texts):
                    raise ValueError("Self-BLEU requires nonempty samples")
                if protocol["sample_count"] < 2 or not 0 < protocol["candidate_fraction"] <= 1:
                    raise ValueError("Self-BLEU requires sample_count >= 2 and candidate_fraction in (0, 1]")
                selected = select_bank(
                    generated, protocol["sample_count"], protocol["selection_seed"]
                )
                selected_texts = selected.texts
                output[metric] = {
                    "metric": metric,
                    "status": "ok",
                    **self_bleu(
                        selected_texts[
                            : max(
                                2,
                                math.ceil(len(selected_texts) * protocol["candidate_fraction"]),
                            )
                        ],
                        protocol["ngram_orders"],
                    ),
                    "selected_total": len(selected_texts),
                    "candidate_fraction": protocol["candidate_fraction"],
                    "selection": selected.metadata["selection"],
                    "protocol": protocol,
                }
            else:  # pragma: no cover - guarded by API validation
                raise AssertionError(metric)
            # The Python API has the same finite, serializable result contract
            # as the JSON CLI, including unusual user-supplied inputs.
            json.dumps(output[metric], allow_nan=False)
        except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
            output[metric] = unavailable(metric, protocol, f"{type(exc).__name__}: {exc}")
        else:
            if (
                metric in output
                and output[metric].get("status") == "ok"
                and metric not in requested
            ):
                raise AssertionError("unrequested metric was produced")
    return output


def balanced_selection_indices(
    n_reference: int, n_generated: int, seed: int, sample_count: int | None = None
) -> tuple[list[int], list[int]]:
    if n_reference < 2 or n_generated < 2:
        raise ValueError("balanced MMD requires at least two samples per side")
    count = min(n_reference, n_generated)
    if sample_count is not None:
        count = min(count, sample_count)
    if count < 2:
        raise ValueError("balanced MMD requires at least two selected samples per side")
    rng = np.random.default_rng(seed)
    reference_order = rng.permutation(n_reference)
    generated_order = rng.permutation(n_generated)
    return reference_order[:count].tolist(), generated_order[:count].tolist()


def select_bank(bank: Bank, count: int, seed: int) -> Bank:
    import random

    rng = random.Random(seed)
    rows = list(enumerate(bank.texts))
    rng.shuffle(rows)
    selected = rows[:count]
    indices = [index for index, _ in selected]
    return Bank(
        ids=tuple(bank.ids[index] for index in indices),
        texts=tuple(bank.texts[index] for index in indices),
        metadata={**bank.metadata, "selection": {"count": len(indices), "seed": seed, "indices": indices}},
        token_ids=tuple(bank.token_ids[index] for index in indices) if bank.token_ids else None,
        path=bank.path,
        sha256=bank.sha256,
    )
