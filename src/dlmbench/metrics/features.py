"""Deterministic text features and historical Self-BLEU estimator."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

BOS = "<BOS>"
EOS = "<EOS>"
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*|\d+")
CAP_WORD_RE = re.compile(r"\b[A-Z][a-z][A-Za-z']+\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def normalize_text(text: str) -> str:
    """Apply the frozen NFKC/whitespace preprocessing used by metric kernels."""

    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _character_ngram_rows(
    texts: Sequence[str], protocol: dict[str, Any], vocabulary: dict[str, int]
) -> tuple[list[list[str]], dict[str, int]]:
    """Build ordered character-gram rows while extending a shared vocabulary."""

    keys_by_row: list[list[str]] = []
    for text in texts:
        normalized = BOS + normalize_text(text) + EOS
        keys: list[str] = []
        for order in protocol["ngram_orders"]:
            keys.extend(
                f"c{order}:{normalized[index : index + order]}"
                for index in range(max(0, len(normalized) - order + 1))
            )
        keys_by_row.append(keys)
        # The complete union vocabulary is required even if only the reference
        # is being built first; generated-only grams still occupy zero columns.
        for key in keys:
            vocabulary.setdefault(key, len(vocabulary))
    return keys_by_row, vocabulary


def _sparse_frequency_matrix(
    keys_by_row: Sequence[Sequence[str]], vocabulary: dict[str, int]
) -> Any:
    from scipy.sparse import csr_matrix

    indptr = np.zeros(len(keys_by_row) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[float] = []
    for row, keys in enumerate(keys_by_row):
        counts = Counter(keys)
        denominator = float(len(keys))
        ordered = sorted((vocabulary[key], count) for key, count in counts.items())
        indices.extend(column for column, _count in ordered)
        data.extend(count / denominator for _column, count in ordered)
        indptr[row + 1] = len(indices)
    return csr_matrix(
        (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64), indptr),
        shape=(len(keys_by_row), len(vocabulary)),
    )


def char_gram_features(
    texts: Sequence[str], protocol: dict[str, Any], vocabulary: dict[str, int] | None = None
) -> Any:
    """Build exact sparse per-document relative character-n-gram frequencies."""

    owns_vocabulary = vocabulary is None
    if vocabulary is None:
        vocabulary = {}
    keys_by_row, vocabulary = _character_ngram_rows(texts, protocol, vocabulary)
    matrix = _sparse_frequency_matrix(keys_by_row, vocabulary)
    if owns_vocabulary:
        return matrix, vocabulary
    return matrix


def char_gram_feature_pair(
    reference: Sequence[str], generated: Sequence[str], protocol: dict[str, Any]
) -> tuple[Any, Any]:
    """Build reference/generated rows in one exact shared vocabulary."""

    vocabulary: dict[str, int] = {}
    reference_keys, vocabulary = _character_ngram_rows(reference, protocol, vocabulary)
    generated_keys, vocabulary = _character_ngram_rows(generated, protocol, vocabulary)
    return (
        _sparse_frequency_matrix(reference_keys, vocabulary),
        _sparse_frequency_matrix(generated_keys, vocabulary),
    )


def tokenize_lower(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(tokens: Sequence[str], order: int) -> list[tuple[str, ...]]:
    return [
        tuple(tokens[index : index + order]) for index in range(max(0, len(tokens) - order + 1))
    ]


def _modified_precision(
    candidate: Sequence[tuple[str, ...]],
    references: Sequence[Counter[tuple[str, ...]]],
) -> tuple[float, int]:
    """Return the historical repeated-gram modified precision, not unique clipping.

    The recorded ``self_bleu/runner.py`` loops over every candidate occurrence
    and adds both its clipped contribution and denominator contribution.  This
    is not the conventional NLTK-style collapse-to-unique estimator.
    """

    counts = Counter(candidate)
    clipped = 0
    total = 0
    for gram in candidate:
        maximum = max((reference[gram] for reference in references), default=0)
        count = counts[gram]
        clipped += min(count, maximum)
        total += count
    return (clipped / total if total else 0.0), total


def sentence_bleu(candidate: str, references: Sequence[str], orders: Sequence[int]) -> float:
    candidate_tokens = tokenize_lower(candidate)
    reference_tokens = [tokenize_lower(reference) for reference in references]
    log_sum = 0.0
    matched = 0
    for order in orders:
        precision, total = _modified_precision(
            _ngrams(candidate_tokens, order),
            [Counter(_ngrams(reference, order)) for reference in reference_tokens],
        )
        if total and precision > 0.0:
            log_sum += math.log(precision)
            matched += 1
    return math.exp(log_sum / matched) if matched else 0.0


def self_bleu(texts: Sequence[str], orders: Sequence[int]) -> dict[str, float | int]:
    if len(texts) < 2:
        raise ValueError("Self-BLEU requires at least two samples")
    scores = [
        sentence_bleu(text, [*texts[:index], *texts[index + 1 :]], orders)
        for index, text in enumerate(texts)
    ]
    return {
        "self_bleu": float(np.mean(scores)),
        "n_samples": len(texts),
        "ngram_orders": list(orders),
    }


STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
    "we",
    "us",
    "our",
    "ours",
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
}

DISCOURSE_MARKERS = {
    "contrast": (
        "however",
        "but",
        "although",
        "though",
        "yet",
        "nevertheless",
        "nonetheless",
        "instead",
        "whereas",
        "despite",
    ),
    "cause": (
        "because",
        "therefore",
        "thus",
        "hence",
        "so",
        "since",
        "consequently",
        "as a result",
        "due to",
    ),
    "temporal": (
        "then",
        "next",
        "after",
        "before",
        "meanwhile",
        "later",
        "finally",
        "previously",
        "subsequently",
        "during",
    ),
    "addition": (
        "also",
        "moreover",
        "furthermore",
        "additionally",
        "besides",
        "indeed",
        "in addition",
    ),
    "condition": ("if", "unless", "provided", "whether", "otherwise", "assuming"),
    "example": ("for example", "for instance", "such as", "including", "namely"),
    "conclusion": ("overall", "therefore", "in conclusion", "ultimately", "in short"),
}


def split_segments(text: str, target_words: int = 24, max_segments: int = 32) -> list[str]:
    text = text.replace("”", "\n")
    raw = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    segments: list[str] = []
    for sentence in raw:
        words = WORD_RE.findall(sentence)
        if not words:
            continue
        if len(words) > target_words * 2:
            for start in range(0, len(words), target_words):
                chunk = words[start : start + target_words]
                if chunk:
                    segments.append(" ".join(chunk))
        else:
            segments.append(sentence)
    if len(segments) < 2:
        words = WORD_RE.findall(text)
        if words:
            parts = min(max_segments, max(2, math.ceil(len(words) / max(target_words, 1))))
            size = max(1, math.ceil(len(words) / parts))
            segments = [
                " ".join(words[start : start + size]) for start in range(0, len(words), size)
            ]
    if not segments:
        return [""]
    return segments[:max_segments]


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _content_terms(words: Sequence[str]) -> list[str]:
    return [
        word.lower()
        for word in words
        if len(word) >= 4 and not word.isdigit() and word.lower() not in STOPWORDS
    ]


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _entropy_from_counts(counts: Any) -> float:
    values = [float(value) for value in counts if value]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values)


def entity_discourse_features(text: str) -> np.ndarray:
    segments = split_segments(text)
    words_raw = word_tokens(text)
    words = [word.lower() for word in words_raw]
    n_words = max(len(words), 1)
    segment_words = [word_tokens(segment) for segment in segments]
    segment_lengths = np.asarray(
        [len(words_segment) for words_segment in segment_words], dtype=np.float64
    )
    segment_terms = [set(_content_terms(words_segment)) for words_segment in segment_words]
    all_terms = _content_terms(words_raw)
    term_segment_counts: Counter[str] = Counter()
    for terms in segment_terms:
        term_segment_counts.update(terms)
    repeated_terms = {term for term, count in term_segment_counts.items() if count >= 2}
    chain_max = max(term_segment_counts.values(), default=0)
    adjacent = [
        _jaccard(segment_terms[index], segment_terms[index + 1])
        for index in range(len(segment_terms) - 1)
    ]
    if adjacent:
        adj_mean, adj_std, adj_min = (
            float(np.mean(adjacent)),
            float(np.std(adjacent)),
            float(np.min(adjacent)),
        )
    else:
        adj_mean = adj_std = adj_min = 1.0
    repeated_coverage = sum(bool(terms & repeated_terms) for terms in segment_terms) / max(
        len(segment_terms), 1
    )
    capitalized = [match.group(0).lower() for match in CAP_WORD_RE.finditer(text)]
    unique_capitalized = set(capitalized)
    repeated_capitalized = {item for item in unique_capitalized if capitalized.count(item) >= 2}
    pronouns = [word for word in words if word in PRONOUNS]
    lowered = text.lower()
    discourse_counts = {
        category: sum(
            len(re.findall(r"\b" + re.escape(marker) + r"\b", lowered)) for marker in markers
        )
        for category, markers in DISCOURSE_MARKERS.items()
    }
    segment_discourse = [
        "none"
        if not any(
            marker in segment.lower()
            for markers in DISCOURSE_MARKERS.values()
            for marker in markers
        )
        else max(
            DISCOURSE_MARKERS,
            key=lambda category: sum(
                marker in segment.lower() for marker in DISCOURSE_MARKERS[category]
            ),
        )
        for segment in segments
    ]
    transitions = Counter(pairwise(segment_discourse))
    unresolved = 0
    seen_entity = False
    for terms, segment_word_list in zip(segment_terms, segment_words):
        if terms:
            seen_entity = True
        if not seen_entity and any(word.lower() in PRONOUNS for word in segment_word_list):
            unresolved += 1
    all_markers = {marker for markers in DISCOURSE_MARKERS.values() for marker in markers}
    connective_starts = sum(
        any(segment.lower().strip().startswith(marker + " ") for marker in all_markers)
        for segment in segments
    )
    marker_rates = [discourse_counts[category] / n_words for category in DISCOURSE_MARKERS]
    values = [
        math.log1p(len(words)),
        len(segments),
        float(segment_lengths.mean()),
        float(segment_lengths.std()),
        len(all_terms) / n_words,
        len(set(all_terms)) / max(len(all_terms), 1),
        len(repeated_terms) / max(len(set(all_terms)), 1),
        chain_max / max(len(segments), 1),
        repeated_coverage,
        adj_mean,
        adj_std,
        adj_min,
        _jaccard(segment_terms[0], segment_terms[-1]) if segment_terms else 1.0,
        len(capitalized) / n_words,
        len(repeated_capitalized) / max(len(unique_capitalized), 1),
        len(pronouns) / n_words,
        len(pronouns) / max(len(set(all_terms)) + len(unique_capitalized), 1),
        unresolved / max(len(segments), 1),
        sum(discourse_counts.values()) / n_words,
        _entropy_from_counts(discourse_counts.values()),
        _entropy_from_counts(transitions.values()),
        sum(category != "none" for category in segment_discourse) / max(len(segment_discourse), 1),
        connective_starts / max(len(segments), 1),
        text.count('"') / max(len(text), 1),
        text.count(",") / n_words,
        (text.count(";") + text.count(":")) / n_words,
        text.count("\n\n") / max(len(segments), 1),
        *marker_rates,
    ]
    return np.asarray(values, dtype=np.float64)


def entity_discourse_matrix(texts: Sequence[str]) -> np.ndarray:
    matrix = np.vstack([entity_discourse_features(text) for text in texts])
    if not np.isfinite(matrix).all():
        raise ValueError("entity/discourse features produced a non-finite value")
    return matrix
