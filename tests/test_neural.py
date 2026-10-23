"""Small-network numerical tests requiring optional neural dependencies."""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dlmbench.banks import Bank  # noqa: E402
from dlmbench.metrics.neural import _device_id, _mean_gradient  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.head = torch.nn.Linear(3, 8, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [0.5, 0.5, 0.5],
                        [-1.0, -2.0, -3.0],
                    ]
                )
            )
            self.head.weight.copy_(torch.eye(3, 8).t() * 2.0)
        self.config = type("Config", (), {"n_positions": 8})()

    def forward(self, input_ids=None, attention_mask=None):
        hidden = self.embedding(input_ids)
        return type("Output", (), {"logits": self.head(hidden)})()


class TinyTokenizer:
    pad_token_id = 7


def token_bank(rows, start=0):
    return Bank(
        ids=tuple(range(start, start + len(rows))),
        texts=tuple(f"document {row}" for row in rows),
        token_ids=tuple(tuple(row) for row in rows),
        metadata={"tokenizer": "tiny-test"},
    )


def test_gradient_moment_identical_banks_and_final_batch_weights():
    model = TinyModel().eval()
    rows = [[0, 1, 2, 3, 4], [0, 1]]
    generated = token_bank(rows)
    reference = token_bank(rows, start=10)
    generated_gradient = _mean_gradient(
        model=model,
        tokenizer=TinyTokenizer(),
        texts=generated.texts,
        token_rows=[list(row) for row in generated.token_ids],
        device="cpu",
        batch_size=1,
        maximum=8,
    )
    reference_gradient = _mean_gradient(
        model=model,
        tokenizer=TinyTokenizer(),
        texts=reference.texts,
        token_rows=[list(row) for row in reference.token_ids],
        device="cpu",
        batch_size=1,
        maximum=8,
    )
    difference = sum(
        float(((left.float() - right.float()) ** 2).sum().item())
        for left, right in zip(generated_gradient, reference_gradient)
    )
    assert difference == pytest.approx(0.0, abs=1e-12)


def test_gradient_moment_nontrivial_difference_matches_independent_mean():
    generated = token_bank([[0, 1, 2, 3, 4], [0, 1]])
    reference = token_bank([[2, 3, 4, 5], [1, 2, 3]], start=10)
    model = TinyModel().eval()
    gradients = []
    for bank in (generated, reference):
        gradients.append(
            _mean_gradient(
                model=model,
                tokenizer=TinyTokenizer(),
                texts=bank.texts,
                token_rows=[list(row) for row in bank.token_ids],
                device="cpu",
                batch_size=1,
                maximum=8,
            )
        )
    # Autodiff computes one mean gradient per bank.  Compare its nontrivial
    # value to the independent sequence-mean-of-gradient-matrices calculation.
    generated_gradients = []
    reference_gradients = []
    for rows, values in (
        (generated.token_ids, generated_gradients),
        (reference.token_ids, reference_gradients),
    ):
        for row in rows:
            model.zero_grad(set_to_none=True)
            input_ids = torch.tensor([row], dtype=torch.long)
            logits = model(input_ids).logits[:, :-1, :]
            labels = input_ids[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
            )
            (losses.mean()).backward()
            values.append(
                [
                    parameter.grad.detach().clone()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
            )
    model.zero_grad(set_to_none=True)

    def mean_gradients(rows):
        return [
            torch.stack(per_parameter).mean(dim=0)
            for per_parameter in zip(*rows, strict=True)
        ]

    independent_generated = mean_gradients(generated_gradients)
    independent_reference = mean_gradients(reference_gradients)
    expected = sum(
        float(((left.float() - right.float()) ** 2).sum().item())
        for left, right in zip(independent_generated, independent_reference, strict=True)
    )
    actual = sum(
        float(((left.float() - right.float()) ** 2).sum().item())
        for left, right in zip(gradients[0], gradients[1], strict=True)
    )
    assert actual > 0.0
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_gen_ppl_masks_padding_and_final_batches_do_not_change_per_sample_mean():
    import torch.nn.functional as F

    model = TinyModel().eval()
    long_row = [0, 1, 2, 3, 4]
    short_row = [0, 1]
    with torch.inference_mode():
        widths = []
        for row in ([long_row, short_row], [short_row]):
            width = max(len(value) for value in row)
            input_ids = torch.full((len(row), width), TinyTokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(row), width), dtype=torch.long)
            for index, values in enumerate(row):
                input_ids[index, : len(values)] = torch.tensor(values)
                attention_mask[index, : len(values)] = 1
            logits = model(input_ids, attention_mask).logits[:, :-1, :]
            labels = input_ids[:, 1:]
            mask = attention_mask[:, 1:].bool()
            losses = (
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
                )
                .reshape(labels.shape)
                .masked_fill(~mask, 0.0)
            )
            widths.extend((losses.sum(dim=1) / mask.sum(dim=1)).tolist())
    assert len(widths) == 3
    # The final short batch is a separate sequence, not an extra weight on a
    # pooled batch mean.  The same row must have the same NLL either way.
    assert widths[1] == pytest.approx(widths[2], abs=1e-12)


def test_mauve_device_indices_do_not_collapse(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _device_id(None) == 0
    assert _device_id("cpu") == -1
    assert _device_id("cuda:3") == 3
    with pytest.raises(ValueError):
        _device_id("cuda:not-number")


def test_mauve_pins_revision_and_reuses_full_bank_last_token_features(monkeypatch):
    import types

    import dlmbench.metrics.neural as neural

    calls = []
    encoded = []

    class Tokenizer:
        name_or_path = "fixture"
        eos_token_id = 0
        pad_token_id = 0

        def encode(self, text, **kwargs):
            encoded.append((text, kwargs))
            return [1, 2, 3]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((1, 2)))

        @property
        def device(self):
            return self.weight.device

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            calls.append((input_ids.tolist(), sorted(kwargs)))
            hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2)
            self.hidden_states = tuple(hidden[:, :1, :].clone() for _ in range(1))
            return types.SimpleNamespace(hidden_states=(hidden,))

    model = Model()

    def tokenizers(*args, **kwargs):
        calls.append(("tokenizer", args, kwargs))
        return Tokenizer()

    def models(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return model

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizers),
        AutoModel=types.SimpleNamespace(from_pretrained=models),
    )
    fake_mauve_utils = types.SimpleNamespace(
        featurize_tokens_from_model=lambda model, rows, batch, **kwargs: torch.stack(
            [row.view(1, -1).float().sum(dim=1).expand(1, 2) for row in rows]
        )
    )
    fake_mauve = types.ModuleType("mauve")
    fake_mauve.compute_mauve = lambda **kwargs: types.SimpleNamespace(
        mauve=float(kwargs["p_features"].shape[0] + kwargs["q_features"].shape[0]) / 2048.0
    )
    fake_mauve_utils_module = types.ModuleType("mauve.utils")
    fake_mauve_utils_module.featurize_tokens_from_model = fake_mauve_utils.featurize_tokens_from_model
    fake_mauve_utils_module.__dict__["__package__"] = "mauve"
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "mauve", fake_mauve)
    monkeypatch.setitem(sys.modules, "mauve.utils", fake_mauve_utils_module)

    texts = tuple(f"document {index}" for index in range(4))
    generated = Bank(tuple(range(4)), texts)
    reference = Bank(tuple(range(10, 14)), texts)
    protocol = {
        "featurizer": "fixture",
        "revision": "pinned-revision",
        "tokenizer": "tokenizer-fixture",
        "tokenizer_revision": "tokenizer-revision",
        "max_tokens": 512,
        "batch_size": 2,
        "minimum_subset_size": 2,
        "subset_size": 2,
        "subset_seeds": [0, 1],
        "selection_base_seed": 20260825,
        "internal_seed": 25,
        "scaling_factor": 5,
    }
    result = neural.mauve(generated, reference, protocol, "cpu")
    tokenizer_calls = [call for call in calls if call[0] == "tokenizer"]
    model_calls = [call for call in calls if call[0] == "model"]
    assert len(tokenizer_calls) == 2
    assert len(model_calls) == 2
    assert all(call[1][0] == "tokenizer-fixture" for call in tokenizer_calls)
    assert all(call[-1]["revision"] == "tokenizer-revision" for call in tokenizer_calls)
    assert all(call[-1]["revision"] == "pinned-revision" for call in model_calls)
    assert {kwargs["add_special_tokens"] for _, kwargs in encoded} == {False}
    assert {kwargs["max_length"] for _, kwargs in encoded} == {512}
    assert all(call[-1]["pad_token_id"] == 0 for call in model_calls)
    assert len(result["mauve_scores_by_seed"]) == 2
    for selection in result["selections"]:
        assert selection["p_ids"] == sorted(selection["p_ids"])
        assert selection["q_ids"] == sorted(selection["q_ids"])
    assert result["featurizer_source"].endswith("clustering/scoring")


def test_mauve_honors_scaling_factor_override(monkeypatch):
    import types

    import dlmbench.metrics.neural as neural

    captured = []

    class Tokenizer:
        name_or_path = "fixture"
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, batch, **kwargs):
            width = max(len(text) for text in batch)
            return {
                "input_ids": torch.ones((len(batch), width), dtype=torch.long),
                "attention_mask": torch.ones((len(batch), width), dtype=torch.long),
            }

        def encode(self, text, **kwargs):
            return [1, 2, 3]

    class Model(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            hidden = input_ids.to(torch.float32)[:, :, None].expand(-1, -1, 2)
            return types.SimpleNamespace(last_hidden_state=hidden)

    def tokenizers(*args, **kwargs):
        return Tokenizer()

    def models(*args, **kwargs):
        return Model()

    def compute(**kwargs):
        captured.append(kwargs["mauve_scaling_factor"])
        return types.SimpleNamespace(mauve=0.5)

    fake_mauve = types.ModuleType("mauve")
    fake_mauve.compute_mauve = compute
    fake_mauve_utils = types.ModuleType("mauve.utils")
    fake_mauve_utils.featurize_tokens_from_model = lambda model, rows, batch: torch.stack(
        [row.view(1, -1).float().sum(dim=1).expand(1, 2) for row in rows]
    )
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizers),
        AutoModel=types.SimpleNamespace(from_pretrained=models),
    )
    monkeypatch.setitem(sys.modules, "mauve", fake_mauve)
    monkeypatch.setitem(sys.modules, "mauve.utils", fake_mauve_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    protocol = {
        "featurizer": "fixture",
        "revision": "model-revision",
        "tokenizer": "tokenizer-fixture",
        "tokenizer_revision": "tokenizer-revision",
        "max_tokens": 512,
        "batch_size": 2,
        "minimum_subset_size": 2,
        "subset_size": 2,
        "subset_seeds": [0],
        "selection_base_seed": 20260825,
        "internal_seed": 25,
        "scaling_factor": 7,
    }
    generated = Bank((0, 1), ("a", "b"))
    neural.mauve(generated, generated, protocol, "cpu")
    assert captured == [7]


def test_neural_tokenizer_and_model_revisions_remain_distinct(monkeypatch):
    import types

    import dlmbench.metrics.neural as neural

    calls = []

    class Tokenizer:
        name_or_path = "fixture"
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, batch, **kwargs):
            width = max(len(text) for text in batch)
            return {
                "input_ids": torch.ones((len(batch), width), dtype=torch.long),
                "attention_mask": torch.ones((len(batch), width), dtype=torch.long),
            }

        def encode(self, text, **kwargs):
            return [1, 2, 3]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((1, 2)))

        @property
        def device(self):
            return self.weight.device

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            hidden = input_ids.to(torch.float32)[:, :, None].expand(-1, -1, 2)
            return types.SimpleNamespace(last_hidden_state=hidden)

    def tokenizers(*args, **kwargs):
        calls.append(("tokenizer", args, kwargs))
        return Tokenizer()

    def models(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return Model()

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizers),
        AutoModel=types.SimpleNamespace(from_pretrained=models),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    protocol = {
        "featurizer": "fixture",
        "revision": "model-revision",
        "tokenizer": "tokenizer-fixture",
        "tokenizer_revision": "tokenizer-revision",
        "max_tokens": 512,
        "batch_size": 2,
    }
    bank = Bank((0,), ("a b",))
    neural._features(bank, bank, protocol, "cpu")
    tokenizer_calls = [call for call in calls if call[0] == "tokenizer"]
    model_calls = [call for call in calls if call[0] == "model"]
    assert tokenizer_calls[0][2]["revision"] == "tokenizer-revision"
    assert tokenizer_calls[0][1][0] == "tokenizer-fixture"
    assert model_calls[0][2]["revision"] == "model-revision"


def test_transformer_special_mask_follows_device(monkeypatch):
    import types

    import dlmbench.metrics.neural as neural

    captured = {}

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            width = max(len(text) for text in texts)
            return {
                "input_ids": torch.ones((len(texts), width), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), width), dtype=torch.long),
            "special_tokens_mask": torch.zeros((len(texts), width), dtype=torch.long),
            }

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            captured.update(kwargs)
            shape = kwargs["input_ids"].shape
            hidden = torch.ones((shape[0], shape[1], 3))
            return types.SimpleNamespace(last_hidden_state=hidden)

    protocol = {
        "encoder": "bert-base-uncased",
        "max_tokens": 8,
        "batch_size": 2,
    }
    # CPU tensors exercise the same device-alignment code path without a GPU.
    neural._transformer_pool(Tokenizer(), Model(), ("abcd", "efgh"), protocol, torch.device("cpu"))
    assert captured["input_ids"].device == captured["attention_mask"].device
    assert captured["input_ids"].shape == captured["attention_mask"].shape


def test_sentence_transformer_constructor_receives_revision(monkeypatch):
    import types

    import dlmbench.metrics.neural as neural

    calls = []

    class FakeSentenceTransformer:
        max_seq_length = 0

        def __init__(self, model_name_or_path, revision=None, device=None):
            calls.append((model_name_or_path, revision, device))

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 2))

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    generated = Bank((0, 1), ("a", "b"))
    protocol = {
        "encoder": "all-mpnet-base-v2",
        "encoders": ["all-mpnet-base-v2"],
        "revisions": {"all-mpnet-base-v2": "mpnet-revision"},
        "max_tokens": 8,
        "batch_size": 2,
        "reference_pca_whitening": 2,
        "bandwidths": [1.0],
    }
    result = neural.encoder_features(generated, generated, protocol, "cpu")
    assert calls == [("all-mpnet-base-v2", "mpnet-revision", "cpu")]
    assert result[0].shape == (2, 2)
