"""Independent small-model checks of the public neural estimators; no downloads."""

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlmbench import Bank  # noqa: E402
from dlmbench.metrics import neural  # noqa: E402

WEIGHTS = [[0.1, 0.7, -0.4], [1.0, 0.3, 0.6], [-0.2, 0.5, 1.1]]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor(WEIGHTS, dtype=torch.float64))
        self.config = SimpleNamespace(n_positions=4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.weights[input_ids])


class TinyTokenizer:
    name_or_path = "fixture"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, *, max_length=None, truncation=False, return_tensors=None, **kwargs):
        single = isinstance(texts, str)
        rows = [[int(token) for token in text.split()] for text in ([texts] if single else texts)]
        if truncation:
            rows = [row[:max_length] for row in rows]
        if return_tensors is None:
            return {"input_ids": rows[0] if single else rows}
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }


@pytest.fixture
def local_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: TinyLM()),
    ))


def protocol(batch_size):
    return {"scorer": "fixture", "revision": "local", "max_tokens": 4,
            "batch_size": batch_size, "token_id_input": "text"}


def analytic(rows):
    """NLL and score gradient from the categorical probabilities, without autodiff."""
    logits = np.asarray(WEIGHTS)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    losses, gradients, lengths = [], [], []
    for row in rows:
        row = row[:4]
        gradient = np.zeros_like(logits)
        loss = 0.0
        for source, target in zip(row[:-1], row[1:]):
            loss -= math.log(probabilities[source, target])
            gradient[source] += probabilities[source]
            gradient[source, target] -= 1
        losses.append(loss / (len(row) - 1))
        gradients.append(gradient / (len(row) - 1))
        lengths.append(len(row) - 1)
    return np.asarray(losses), np.mean(gradients, axis=0), np.asarray(lengths)


def bank(rows):
    return Bank(tuple(range(len(rows))), tuple(" ".join(map(str, row)) for row in rows))


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_neural_ppl_masks_padding_truncates_and_weights_sequences(local_model, batch_size):
    rows = [[0, 1], [2, 2, 1, 0, 2, 1], [1, 0, 2]]
    losses, _, lengths = analytic(rows)
    result = neural.gen_ppl(bank(rows), protocol(batch_size), "cpu")
    assert result["mean_nll"] == pytest.approx(losses.mean())
    assert result["gen_ppl"] == pytest.approx(math.exp(losses.mean()))
    assert result["token_weighted_nll"] == pytest.approx(np.average(losses, weights=lengths))
    assert result["n_tokens_scored"] == int(lengths.sum())
    assert result["max_length"] == 4


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_neural_gm_uses_difference_of_sequence_mean_gradients(local_model, batch_size):
    generated = [[0, 1], [2, 2, 1, 0, 2], [1, 0, 2]]
    reference = [[1, 2, 0], [0, 0]]
    _, left, _ = analytic(generated)
    _, right, _ = analytic(reference)
    result = neural.gradient_moment(bank(generated), bank(reference), protocol(batch_size), "cpu")
    assert result["gradient_moment"] == pytest.approx(float(((left - right) ** 2).sum()), rel=1e-6)
    identical = neural.gradient_moment(bank(generated), bank(generated), protocol(batch_size), "cpu")
    assert identical["gradient_moment"] == pytest.approx(0.0, abs=1e-12)
