"""Compatibility for obsolete tokenizer state embedded in some FLM checkpoints."""

from __future__ import annotations

import tokenizers


def install() -> None:
    """Ignore redundant legacy tokenizer state while the FLM checkpoint loads."""

    if getattr(tokenizers.Tokenizer, "_dlmbench_legacy_state_ignored", False):
        return
    original = tokenizers.Tokenizer.__setstate__

    def _ignore_legacy_state(self: tokenizers.Tokenizer, state: object) -> None:
        del state

    tokenizers.Tokenizer._dlmbench_original_setstate = original  # type: ignore[attr-defined]
    tokenizers.Tokenizer.__setstate__ = _ignore_legacy_state  # type: ignore[method-assign]
    tokenizers.Tokenizer._dlmbench_legacy_state_ignored = True  # type: ignore[attr-defined]
