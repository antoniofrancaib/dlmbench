"""Frozen CTW-20 Jeffreys Coding Divergence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ..banks import Bank


@dataclass(frozen=True)
class DocumentCode:
    bits: float
    n_bytes: int
    document_index: int


class BinaryCTWCoder:
    """Training-conditioned CTW over UTF-8 bits with frozen target scoring."""

    coder_id = "ctw_binary_frozen"

    def __init__(self, max_depth: int = 20):
        if not 0 <= max_depth <= 20:
            raise ValueError("CTW depth must be in [0, 20]")
        self.max_depth = max_depth
        self.nodes: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        self.training_bits = 0
        self.training_documents = 0
        self._log_weighted: dict[tuple[int, int], float] = {}
        self._mixture: dict[tuple[int, int], tuple[float, float]] = {}

    @staticmethod
    def bits(data: bytes):
        for value in data:
            for shift in range(7, -1, -1):
                yield (value >> shift) & 1

    @staticmethod
    def context(history: int, depth: int) -> int:
        return history & ((1 << depth) - 1)

    def fit(self, documents: Iterable[bytes]) -> BinaryCTWCoder:
        """Fit on complete UTF-8 documents; scoring never mutates this state."""

        for document in documents:
            if not isinstance(document, bytes):
                raise TypeError("CTW training documents must be bytes")
            if document:
                self.training_documents += 1
            history = 0
            for bit in self.bits(document):
                for depth in range(self.max_depth + 1):
                    self.nodes[(depth, self.context(history, depth))][bit] += 1
                self.training_bits += 1
                history = ((history << 1) | bit) & (
                    (1 << self.max_depth) - 1 if self.max_depth else 0
                )
        if not self.training_bits:
            raise ValueError("cannot fit CTW on zero bytes")
        self._log_weighted_likelihood(0, 0)
        for depth, context in tuple(self.nodes):
            if depth < self.max_depth:
                self._node_mixture(depth, context)
        return self

    @staticmethod
    def _logaddexp(left: float, right: float) -> float:
        if left == -math.inf:
            return right
        if right == -math.inf:
            return left
        if left < right:
            left, right = right, left
        return left + math.log1p(math.exp(right - left))

    @staticmethod
    def _log_kt_likelihood(counts: list[int]) -> float:
        total = sum(counts)
        result = math.lgamma(1.0) - math.lgamma(total + 1.0)
        for count in counts:
            result += math.lgamma(count + 0.5) - math.lgamma(0.5)
        return result

    def _log_weighted_likelihood(self, depth: int, context: int) -> float:
        key = (depth, context)
        if key in self._log_weighted:
            return self._log_weighted[key]
        counts = self.nodes.get(key)
        if counts is None or sum(counts) == 0:
            value = 0.0
        elif depth >= self.max_depth:
            value = self._log_kt_likelihood(counts)
        else:
            local = self._log_kt_likelihood(counts)
            child_zero = self._log_weighted_likelihood(depth + 1, context)
            child_one = self._log_weighted_likelihood(depth + 1, context | (1 << depth))
            value = self._logaddexp(math.log(0.5) + local, math.log(0.5) + child_zero + child_one)
        self._log_weighted[key] = value
        return value

    def _node_mixture(self, depth: int, context: int) -> tuple[float, float]:
        key = (depth, context)
        if key in self._mixture:
            return self._mixture[key]
        counts = self.nodes.get(key)
        if counts is None or sum(counts) == 0 or depth >= self.max_depth:
            return 1.0, 0.0
        log_stop = math.log(0.5) + self._log_kt_likelihood(counts)
        child_zero = self._log_weighted_likelihood(depth + 1, context)
        child_one = self._log_weighted_likelihood(depth + 1, context | (1 << depth))
        log_split = math.log(0.5) + child_zero + child_one
        normalizer = self._logaddexp(log_stop, log_split)
        weights = (math.exp(log_stop - normalizer), math.exp(log_split - normalizer))
        self._mixture[key] = weights
        return weights

    def _kt_probability(self, depth: int, context: int, bit: int) -> float:
        counts = self.nodes.get((depth, context))
        if counts is None:
            return 0.5
        return (counts[bit] + 0.5) / (sum(counts) + 1.0)

    def _predictive(self, depth: int, context: int, history: int, bit: int) -> float:
        local = self._kt_probability(depth, context, bit)
        if depth >= self.max_depth:
            return local
        stop, split = self._node_mixture(depth, context)
        older = (history >> depth) & 1
        child_context = context | (older << depth)
        return stop * local + split * self._predictive(depth + 1, child_context, history, bit)

    def score(self, document: bytes, index: int) -> DocumentCode:
        if not isinstance(document, bytes):
            raise TypeError("CTW target documents must be bytes")
        if not document:
            return DocumentCode(0.0, 0, index)
        history = 0
        bits = 0.0
        for bit in self.bits(document):
            probability = self._predictive(0, 0, history, bit)
            if not 0.0 < probability <= 1.0:
                raise ValueError("frozen CTW produced an invalid predictive probability")
            bits -= math.log2(probability)
            history = ((history << 1) | bit) & ((1 << self.max_depth) - 1 if self.max_depth else 0)
        return DocumentCode(bits, len(document), index)

    def state_digest(self) -> str:
        payload = {
            "coder": self.coder_id,
            "max_depth": self.max_depth,
            "prior": "kt",
            "bit_order": "msb_first",
            "history_initialization": "zero_padded_per_document",
            "target_update": "none",
            "nodes": [
                [depth, context, *counts] for (depth, context), counts in sorted(self.nodes.items())
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def counts_digest(self) -> str:
        """Backward-compatible alias for the full coder-state digest."""

        return self.state_digest()


def jcd(generated: Bank, reference: Bank, protocol: dict) -> dict:
    fit_size = int(protocol["fit_size_per_source"])
    heldout_size = int(protocol["heldout_size_per_source"])
    if len(generated) < fit_size + heldout_size:
        raise ValueError("generated bank is too small for the JCD split protocol")
    if len(reference) < fit_size + heldout_size:
        raise ValueError("reference bank is too small for the JCD split protocol")
    directional: dict[str, list[float]] = {"coverage": [], "fidelity": []}
    split_records: list[dict] = []
    coder_records: list[dict] = []
    for seed in protocol["split_seeds"]:
        reference_fit, reference_held, reference_split = _split(
            reference, seed, fit_size, heldout_size
        )
        generated_fit, generated_held, generated_split = _split(
            generated, seed, fit_size, heldout_size
        )
        reference_coder = BinaryCTWCoder(protocol["depth"]).fit(
            text.encode("utf-8") for text in reference_fit
        )
        generated_coder = BinaryCTWCoder(protocol["depth"]).fit(
            text.encode("utf-8") for text in generated_fit
        )
        coder_records.append(
            {
                "seed": seed,
                "reference_state_sha256": reference_coder.state_digest(),
                "generated_state_sha256": generated_coder.state_digest(),
            }
        )
        split_records.append(
            {
                "seed": seed,
                "reference": reference_split,
                "generated": generated_split,
            }
        )
        directional["coverage"].append(
            _direction(reference_held, reference_coder, generated_coder)
        )
        directional["fidelity"].append(
            _direction(generated_held, generated_coder, reference_coder)
        )
    coverage = sum(directional["coverage"]) / len(directional["coverage"])
    fidelity = sum(directional["fidelity"]) / len(directional["fidelity"])
    return {
        "metric": "jcd",
        "status": "ok",
        "jcd": (coverage + fidelity) / 2.0,
        "jcd_coverage": coverage,
        "jcd_fidelity": fidelity,
        "direction_values": directional,
        "splits": split_records,
        "coder_states_sha256": coder_records,
        "protocol": protocol,
    }


def _split(
    bank: Bank, seed: int, fit_size: int, heldout_size: int
) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    ranked = sorted(
        range(len(bank)),
        key=lambda index: (
            _rank(seed, bank.dataset or "", bank.id, bank.ids[index]),
            _stable_id(bank.ids[index]),
        ),
    )
    fit = ranked[:fit_size]
    held = ranked[fit_size : fit_size + heldout_size]
    return (
        tuple(bank.texts[index] for index in fit),
        tuple(bank.texts[index] for index in held),
        {
            "algorithm": "sha256_rank_v1_sample_only",
            "fit_size": fit_size,
            "heldout_size": heldout_size,
            "fit_indices": fit,
            "heldout_indices": held,
            "fit_ids": [bank.ids[index] for index in fit],
            "heldout_ids": [bank.ids[index] for index in held],
        },
    )


def _stable_id(value: str | int) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _rank(seed: int, dataset: str, bank_id: str, sample_id: str | int) -> str:
    payload = f"sample-only-split-v1\0{seed}\0{dataset}\0{bank_id}\0{_stable_id(sample_id)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _direction(
    heldout: tuple[str, ...],
    own_coder: BinaryCTWCoder,
    other_coder: BinaryCTWCoder,
) -> float:
    total_bits = 0.0
    total_bytes = 0
    for index, text in enumerate(heldout):
        data = text.encode("utf-8")
        own = own_coder.score(data, index)
        other = other_coder.score(data, index)
        total_bits += other.bits - own.bits
        total_bytes += other.n_bytes
    if total_bytes <= 0:
        raise ValueError("JCD held-out split contains no bytes")
    return total_bits / total_bytes
