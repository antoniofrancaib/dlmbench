"""Tokenization-bound entropy diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from ..banks import Bank


def entropy_from_counts(counts: Counter[Any]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(
        -sum((count / total) * math.log(count / total) for count in counts.values() if count)
    )


def token_entropy_metrics(
    bank: Bank,
    metric: str,
    protocol: dict[str, Any],
    shared: dict[str, Any],
) -> dict[str, Any]:
    if protocol.get("input") == "token_ids":
        if bank.token_ids is None or any(row is None for row in bank.token_ids):
            raise ValueError("token-ID entropy requires complete token-ID rows")
        sequences = [tuple(row) for row in bank.token_ids]
    else:
        sequences = tokenizer_sequences(bank, protocol, shared)
    if not sequences:
        raise ValueError("entropy metrics require at least one non-empty token sequence")
    if metric == "h_uni":
        values = [entropy_from_counts(Counter(sequence)) for sequence in sequences]
        value = sum(values) / len(values)
        std = math.sqrt(sum((item - value) ** 2 for item in values) / len(values))
        return {
            "metric": "h_uni",
            "status": "ok",
            "h_uni": value,
            "h_uni_std": std,
            "protocol": protocol,
        }
    if metric == "h_pos" and protocol.get("input") == "token_ids":
        # Position entropy is tokenization-bound but does not need a model.
        # Native rows are consumed only after the explicit input mode selects
        # them; text input always uses the resolved retokenized sequences.
        if any(row is None for row in bank.token_ids):
            raise ValueError("h_pos requires complete token-ID rows")
        lengths = [len(row) for row in bank.token_ids]
        if len(set(lengths)) != 1:
            raise ValueError(
                "h_pos requires equal active token lengths; resample or explicitly align first"
            )
        position_entropies = [
            entropy_from_counts(Counter(row[index] for row in sequences))
            for index in range(lengths[0])
        ]
        return {
            "metric": "h_pos",
            "status": "ok",
            "h_pos": sum(position_entropies) / len(position_entropies),
            "position_entropy_min": min(position_entropies),
            "position_entropy_max": max(position_entropies),
            "protocol": protocol,
        }

    lengths = [len(sequence) for sequence in sequences]
    if metric == "h_pool":
        pooled = Counter(token for sequence in sequences for token in sequence)
        value = entropy_from_counts(pooled)
        return {
            "metric": "h_pool",
            "status": "ok",
            "h_pool": value,
            "protocol": protocol,
        }

    if metric == "h_pos":
        if len(set(lengths)) != 1:
            raise ValueError(
                "h_pos requires equal active token lengths; resample or explicitly align first"
            )
        position_entropies = [
            entropy_from_counts(Counter(sequence[index] for sequence in sequences))
            for index in range(lengths[0])
        ]
        value = sum(position_entropies) / len(position_entropies)
        return {
            "metric": "h_pos",
            "status": "ok",
            "h_pos": value,
            "position_entropy_min": min(position_entropies),
            "position_entropy_max": max(position_entropies),
            "protocol": protocol,
        }
    raise AssertionError(metric)


def tokenizer_sequences(
    bank: Bank, protocol: dict[str, Any], shared: dict[str, Any]
) -> list[tuple[int, ...]]:
    revision = protocol.get("tokenizer_revision") or protocol["revision"]
    key = (
        protocol["tokenizer"],
        revision,
        protocol["input"],
        protocol["add_special_tokens"],
        protocol["truncation"],
        bank.text_sha256,
    )
    if key in shared:
        return shared[key]
    if protocol["input"] != "text":
        raise ValueError("paper entropy protocols use exact text input")

    tokenizer = _load_tokenizer(protocol["tokenizer"], revision)
    sequences = tuple(
        tuple(int(token) for token in row)
        for row in tokenizer(
            list(bank.texts),
            add_special_tokens=protocol["add_special_tokens"],
            truncation=False,
        )["input_ids"]
    )
    if any(not sequence for sequence in sequences):
        raise ValueError("a sample tokenized to zero tokens; it is too short for this protocol")
    shared[key] = sequences
    return sequences


def _load_tokenizer(source: str, revision: Any):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, revision=revision, use_fast=True)
