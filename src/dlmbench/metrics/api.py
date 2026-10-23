"""Public metric API with resolved protocol provenance."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..banks import Bank
from .protocols import paper_profile_source

PROFILE_ALIASES = {"paper": "paper-v1", "paper-v1": "paper-v1"}
AVAILABLE_METRICS = (
    "gen_ppl",
    "h_uni",
    "h_pos",
    "h_pool",
    "gradient_moment",
    "jcd",
    "entity_discourse_ed",
    "char_ngram_mmd2",
    "gpt2_large_fbd",
    "mauve",
    "encoder_ed",
    "encoder_gaussian_rbf_mmd2",
    "self_bleu",
)
DEFAULT_PAPER_METRICS = (
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


def list_metrics() -> list[str]:
    """Return supported metric names in stable protocol order."""

    return list(AVAILABLE_METRICS)


def _identity(bank: Bank) -> dict[str, Any]:
    identity = bank.identity()
    if bank.token_ids is not None:
        identity["token_ids"] = {
            "present": True,
            "complete": all(row is not None for row in bank.token_ids),
            "tokenizer_identity_verified": False,
            "declared_tokenizer": bank.metadata.get("tokenizer")
            or bank.metadata.get("tokenizer_name")
            or bank.metadata.get("tokenizer_id"),
            "declared_revision": bank.metadata.get("tokenizer_revision"),
        }
    return identity


def _profile_name(profile: str | None, kwargs: dict[str, Any]) -> str:
    if profile is None and not kwargs:
        return "paper-v1"
    if profile is None:
        return "custom"
    normalized = PROFILE_ALIASES.get(str(profile).lower(), str(profile).lower())
    if normalized not in {"paper-v1", "custom"}:
        raise ValueError(f"unknown profile {profile!r}; supported: paper-v1, custom")
    if normalized == "custom" and not kwargs:
        return "custom"
    if normalized != "custom" and kwargs:
        return "custom"
    return normalized


def _paper_aliases(requested: list[str]) -> list[str]:
    """Expand public aliases into concrete protocol names."""

    aliases = {
        "gm": "gradient_moment",
        "ed": "entity_discourse_ed",
        "mmd": "char_ngram_mmd2",
        "fbd": "gpt2_large_fbd",
    }
    return [aliases.get(name, name) for name in requested]


def _resolve_metrics(metrics: Iterable[str] | None) -> list[str]:
    if metrics is None:
        return list(DEFAULT_PAPER_METRICS)
    requested = list(dict.fromkeys(_paper_aliases(str(name) for name in metrics)))
    unknown = [name for name in requested if name not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metrics {unknown}; supported metrics are {list(AVAILABLE_METRICS)}"
        )
    if not requested:
        raise ValueError("the metric list must not be empty")
    return requested


def evaluate(
    generated: Bank,
    reference: Bank | None = None,
    metrics: list[str] | None = None,
    profile: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate automatic metrics on an ordered bank.

    The return value is always JSON serializable. Successfully evaluated
    metrics have finite numeric fields and full protocol/input metadata.
    An unavailable optional dependency or missing reference is an explicit
    per-metric ``status: unavailable`` object, never a silent substitute.
    """

    if not isinstance(generated, Bank):
        raise TypeError("generated must be a dlmbench.banks.Bank")
    if reference is not None and not isinstance(reference, Bank):
        raise TypeError("reference must be a dlmbench.banks.Bank or None")

    # Import lazily so importing this package does not load neural libraries.
    from .runner import run_metrics

    profile_name = _profile_name(profile, kwargs)
    results: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile_name,
        "profile_source": paper_profile_source(),
        "requested_metrics": _resolve_metrics(metrics),
        "generated": _identity(generated),
        "reference": _identity(reference) if reference is not None else None,
        "device": device,
        "metrics": {},
    }
    results["metrics"] = run_metrics(
        generated=generated,
        reference=reference,
        metrics=results["requested_metrics"],
        profile=profile_name,
        device=device,
        kwargs=kwargs,
    )
    return results
