"""Sample-based evaluation of unconditional text generation."""

from .banks import Bank

__version__ = "0.1.0"

__all__ = ["Bank", "__version__", "evaluate", "load_bank"]


def load_bank(source, **kwargs):
    """Load an ordered sample bank without modifying its text or multiplicities."""
    from .banks import load_bank as _load_bank
    return _load_bank(source, **kwargs)


def evaluate(generated, reference=None, *, root=None, **kwargs):
    """Evaluate custom or released banks using an explicit metric protocol."""
    from .banks import Bank, load_bank
    from .metrics import evaluate as _evaluate
    generated = generated if isinstance(generated, Bank) else load_bank(generated, root=root)
    reference = (reference if reference is None or isinstance(reference, Bank)
                 else load_bank(reference, root=root))
    return _evaluate(generated, reference=reference, **kwargs)
