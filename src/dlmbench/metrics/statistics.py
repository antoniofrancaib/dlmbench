"""Small, independently usable statistical kernels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        raise ValueError("energy distance requires two non-empty 2-D matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("feature matrices have different widths")
    if len(left) < 2 or len(right) < 2:
        raise ValueError("unbiased energy distance requires at least two samples per side")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("feature matrices contain non-finite values")

    try:
        from scipy.spatial.distance import cdist

        def distances(first, second):
            return cdist(first, second)

    except ImportError:

        def distances(first, second):
            squared = (
                (first * first).sum(axis=1)[:, None]
                + (second * second).sum(axis=1)[None, :]
                - 2.0 * (first @ second.T)
            )
            return np.sqrt(np.maximum(squared, 0.0))

    xx = distances(left, left)
    yy = distances(right, right)
    xy = distances(left, right)
    n, m = len(left), len(right)
    value = (
        2.0 * float(xy.mean())
        - float(xx.sum() - np.trace(xx)) / (n * (n - 1))
        - float(yy.sum() - np.trace(yy)) / (m * (m - 1))
    )
    return float(value)


def gaussian_rbf_kernel(
    first: np.ndarray, second: np.ndarray, bandwidths: list[float]
) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    squared = _pairwise_squared_distances(first, second)
    return np.mean(
        [np.exp(-squared / (2.0 * float(sigma) ** 2)) for sigma in bandwidths],
        axis=0,
    )


def _is_sparse(matrix: Any) -> bool:
    try:
        from scipy.sparse import issparse

        return issparse(matrix)
    except ImportError:
        return False


def _sparse_cosine_matrices(
    left: Any, right: Any
) -> tuple[Any, Any, Any, float, float, float, float, float]:
    left_norms = np.sqrt(np.asarray(left.multiply(left).sum(axis=1)).reshape(-1))
    right_norms = np.sqrt(np.asarray(right.multiply(right).sum(axis=1)).reshape(-1))
    left_scale = np.where(left_norms > 0.0, left_norms, 1.0)
    right_scale = np.where(right_norms > 0.0, right_norms, 1.0)
    cross = (left.multiply(1.0 / left_scale[:, None])) @ (
        right.multiply(1.0 / right_scale[:, None]).T
    )
    xx = (left.multiply(1.0 / left_scale[:, None])) @ (
        left.multiply(1.0 / left_scale[:, None]).T
    )
    yy = (right.multiply(1.0 / right_scale[:, None])) @ (
        right.multiply(1.0 / right_scale[:, None]).T
    )
    xx_diagonal = float(xx.diagonal().sum())
    yy_diagonal = float(yy.diagonal().sum())
    xx.setdiag(0.0)
    yy.setdiag(0.0)
    xx.eliminate_zeros()
    yy.eliminate_zeros()
    return (
        cross,
        xx,
        yy,
        float(cross.sum()),
        float(xx.sum()),
        float(yy.sum()),
        xx_diagonal,
        yy_diagonal,
    )


def mmd2_from_features(
    left: np.ndarray | Any,
    right: np.ndarray | Any,
    *,
    bandwidths: list[float] | None = None,
) -> dict[str, float | int | bool]:
    if bandwidths is None and _is_sparse(left) and _is_sparse(right):
        if left.ndim != 2 or right.ndim != 2 or not left.shape[0] or not right.shape[0]:
            raise ValueError("MMD requires two non-empty 2-D matrices")
        if left.shape[1] != right.shape[1]:
            raise ValueError("feature matrices have different widths")
        n, m = left.shape[0], right.shape[0]
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD requires at least two samples per side")
        if not np.isfinite(left.data).all() or not np.isfinite(right.data).all():
            raise ValueError("feature matrices contain non-finite values")
        (
            _cross_matrix,
            xx,
            yy,
            cross_sum,
            xx_sum,
            yy_sum,
            xx_diagonal,
            yy_diagonal,
        ) = _sparse_cosine_matrices(left, right)
        xx_value = xx_sum / (n * (n - 1))
        yy_value = yy_sum / (m * (m - 1))
        value = xx_value + yy_value - 2.0 * cross_sum / (n * m)
        return {
            "mmd2_unbiased": float(value),
            "mmd2_biased_raw": float(
                (xx_sum + xx_diagonal) / (n * n)
                + (yy_sum + yy_diagonal) / (m * m)
                - 2.0 * cross_sum / (n * m)
            ),
            "n_left": n,
            "n_right": m,
            "negative_unbiased": value < 0.0,
        }

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        raise ValueError("MMD requires two non-empty 2-D matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("feature matrices have different widths")
    n, m = len(left), len(right)
    if n < 2 or m < 2:
        raise ValueError("unbiased MMD requires at least two samples per side")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("feature matrices contain non-finite values")

    if bandwidths is None:
        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        left_unit = left / np.where(left_norm > 0, left_norm, 1.0)
        right_unit = right / np.where(right_norm > 0, right_norm, 1.0)
        similarity = left_unit @ right_unit.T
        kernel = np.clip(similarity, -1.0, 1.0)
    else:
        kernel = gaussian_rbf_kernel(left, right, bandwidths)
    # Cross-pairs include every reference/generated pairing.  Only the two
    # within-side U-statistic terms exclude their self-pairs.
    cross = float(kernel.sum()) / (n * m)

    if bandwidths is None:
        left_u, right_u = left_unit, right_unit
        xx = np.clip(left_u @ left_u.T, -1.0, 1.0)
        yy = np.clip(right_u @ right_u.T, -1.0, 1.0)
        xx_value = float(xx.sum() - np.trace(xx)) / (n * (n - 1))
        yy_value = float(yy.sum() - np.trace(yy)) / (m * (m - 1))
    else:
        xx = gaussian_rbf_kernel(left, left, bandwidths)
        yy = gaussian_rbf_kernel(right, right, bandwidths)
        xx_value = float(xx.sum() - np.trace(xx)) / (n * (n - 1))
        yy_value = float(yy.sum() - np.trace(yy)) / (m * (m - 1))
    value = xx_value + yy_value - 2.0 * cross
    return {
        "mmd2_unbiased": float(value),
        "mmd2_biased_raw": float(
            float(xx.sum()) / (n * n) + float(yy.sum()) / (m * m) - 2.0 * cross
        ),
        "n_left": n,
        "n_right": m,
        "negative_unbiased": value < 0.0,
    }


def frechet_squared(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        raise ValueError("Fréchet distance requires non-empty 2-D matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("feature matrices have different widths")
    left_mean, right_mean = left.mean(axis=0), right.mean(axis=0)
    left_cov = np.cov(left, rowvar=False, bias=True)
    right_cov = np.cov(right, rowvar=False, bias=True)
    if left_cov.ndim == 0:
        left_cov = left_cov.reshape((1, 1))
    if right_cov.ndim == 0:
        right_cov = right_cov.reshape((1, 1))
    values, vectors = np.linalg.eigh(left_cov)
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    product = (root @ right_cov @ root + (root @ right_cov @ root).T) / 2.0
    trace = float(np.sqrt(np.maximum(np.linalg.eigvalsh(product), 0.0)).sum())
    difference = left_mean - right_mean
    return float(difference @ difference + np.trace(left_cov) + np.trace(right_cov) - 2.0 * trace)


def _pairwise_squared_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.distance import cdist

        return cdist(first, second, metric="sqeuclidean")
    except ImportError:
        squared = (
            (first * first).sum(axis=1)[:, None]
            + (second * second).sum(axis=1)[None, :]
            - 2.0 * (first @ second.T)
        )
        return np.maximum(squared, 0.0)


def _safe_exp(value: float) -> float:
    return float(math.exp(min(float(value), 700.0)))
