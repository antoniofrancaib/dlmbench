"""Direction-aware ranking and finite-cohort agreement calculations."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Literal

Number = int | float
Direction = Literal["lower", "higher"]


def _oriented(value: Number, direction: Direction) -> Number:
    if direction not in {"lower", "higher"}:
        raise ValueError(f"unknown metric direction {direction!r}")
    return -value if direction == "lower" else value


def tied_ranks(values: Sequence[Number], *, direction: Direction = "lower") -> list[float]:
    """Return one-based ranks with average ranks for exact ties.

    Rank 1 is always the best value under ``direction``.  Ranking is exact and
    does not perturb or sort equal values arbitrarily.
    """
    if not values:
        return []
    oriented = [_oriented(value, direction) for value in values]
    groups: dict[Number, list[int]] = defaultdict(list)
    for index, value in enumerate(oriented):
        groups[value].append(index)
    ranks = [0.0] * len(values)
    rank = 1
    for value in sorted(groups, reverse=True):
        positions = groups[value]
        average = rank + (len(positions) - 1) / 2
        for position in positions:
            ranks[position] = average
        rank += len(positions)
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("rank vectors must have the same nonzero length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = (left_ss * right_ss) ** 0.5
    if denominator == 0:
        raise ValueError("Spearman rho is undefined for a constant cohort")
    return numerator / denominator


def spearman(left: Sequence[Number], right: Sequence[Number], *,
             left_direction: Direction = "lower",
             right_direction: Direction = "lower") -> float:
    """Direction-corrected tied-rank Spearman correlation over a finite cohort."""
    if len(left) != len(right):
        raise ValueError("metric vectors have different lengths")
    left_ranks = tied_ranks(left, direction=left_direction)
    right_ranks = tied_ranks(right, direction=right_direction)
    return _pearson(left_ranks, right_ranks)


def rank_rows(rows: Iterable[dict[str, Any]], metric: str, *, direction: Direction,
              value_key: str = "values") -> dict[str, float]:
    """Rank rows by ``id``, excluding rows whose metric value is null."""
    selected = [row for row in rows if row.get(value_key, {}).get(metric) is not None]
    values = [row[value_key][metric] for row in selected]
    ranks = tied_ranks(values, direction=direction)
    return {row["id"]: rank for row, rank in zip(selected, ranks)}


def mean_metric_ranks(rows: Sequence[dict[str, Any]], metrics: Sequence[str], *,
                      directions: dict[str, Direction], dataset: str,
                      value_key: str = "values") -> dict[str, float | None]:
    """Average best-first ranks across oriented metrics within one dataset."""
    cohort = [row for row in rows if row["dataset"] == dataset and row.get("kind") != "reference"]
    if not cohort:
        raise ValueError(f"no generated rows for dataset {dataset!r}")
    result: dict[str, float | None] = {}
    for row in cohort:
        if any(row.get(value_key, {}).get(metric) is None for metric in metrics):
            result[row["id"]] = None
            continue
        ranks = []
        for metric in metrics:
            metric_rows = [item for item in cohort
                           if item.get(value_key, {}).get(metric) is not None]
            values = [item[value_key][metric] for item in metric_rows]
            metric_ranks = tied_ranks(values, direction=directions[metric])
            position = next(index for index, item in enumerate(metric_rows) if item["id"] == row["id"])
            ranks.append(metric_ranks[position])
        result[row["id"]] = sum(ranks) / len(ranks)
    return result
