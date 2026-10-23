"""Neural metrics with lazy imports and explicit unavailable statuses."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..banks import Bank
from .statistics import energy_distance, frechet_squared, mmd2_from_features

# Explicit Hub aliases for tokenizers used by the sample banks.
_CANONICAL_TOKENIZER_ALIASES = {
    "gpt2": {"openai-community/gpt2"},
    "gpt2-large": {"openai-community/gpt2-large"},
    "gpt2-xl": {"openai-community/gpt2-xl"},
}


def _canonical_tokenizer_id(value: str) -> str:
    normalized = value.rstrip("/")
    for canonical, aliases in _CANONICAL_TOKENIZER_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized


def _model(protocol: dict[str, Any], device: str | None):
    from transformers import AutoModel, AutoTokenizer

    source = protocol.get("featurizer") or protocol.get("scorer")
    revision = protocol.get("revision")
    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(source, **kwargs)
    model.eval()
    if device:
        model.to(device)
    return tokenizer, model


def _tokenizer_revision(protocol: dict[str, Any]) -> Any:
    """Return the tokenizer pin without conflating it with model weights."""

    return protocol.get("tokenizer_revision") or protocol["revision"]


def _tokenizer_source(protocol: dict[str, Any]) -> str:
    """Return the explicit tokenizer ID, falling back to the featurizer."""

    for key in ("tokenizer", "featurizer", "scorer"):
        if key in protocol:
            return str(protocol[key])
    raise ValueError("protocol has no tokenizer, featurizer, or scorer identity")


def _device_id(device: str | None) -> int:
    """Resolve the MAUVE 0.4.0 integer device contract without collapsing indices."""

    import torch

    if device is None:
        return 0 if torch.cuda.is_available() else -1
    value = str(device)
    if value == "cpu":
        return -1
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device {device!r}; expected cuda:<index>") from exc
        if index < 0:
            raise ValueError(f"invalid CUDA device index {index}")
        return index
    raise ValueError("MAUVE requires device 'cpu', 'cuda', or 'cuda:<index>'")


def _features(generated: Bank, reference: Bank, protocol: dict[str, Any], device: str | None):
    from transformers import AutoModel, AutoTokenizer

    source = protocol["featurizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(protocol), revision=_tokenizer_revision(protocol), use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(source, revision=protocol["revision"]).eval()
    if device:
        model.to(device)
    return _pool(tokenizer, model, generated.texts, protocol, device), _pool(
        tokenizer, model, reference.texts, protocol, device
    )


def _pool(tokenizer, model, texts, protocol, device):
    import torch

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), protocol["batch_size"]):
            batch = list(texts[start : start + protocol["batch_size"]])
            encoded = tokenizer(
                batch,
                add_special_tokens=False,
                truncation=True,
                max_length=protocol["max_tokens"],
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            if protocol.get("pooling") == "max":
                pooled = (
                    hidden.masked_fill(~mask.bool(), torch.finfo(hidden.dtype).min)
                    .max(dim=1)
                    .values
                )
            else:
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            outputs.append(pooled.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _empirical_entropy_nats(token_ids: list[int]) -> float:
    from math import log

    counts: dict[int, int] = {}
    for token in token_ids:
        counts[token] = counts.get(token, 0) + 1
    total = len(token_ids)
    if not total:
        return 0.0
    return -sum((count / total) * log(count / total) for count in counts.values())


def _stable_id(value: str | int) -> str:
    import json

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _tokenizer_metadata(bank: Bank) -> dict[str, Any] | None:
    """Normalize generation's flat and structured tokenizer declarations."""

    declared = (
        bank.metadata.get("tokenizer")
        or bank.metadata.get("tokenizer_name")
        or bank.metadata.get("tokenizer_id")
    )
    if declared is None:
        return None
    if isinstance(declared, dict):
        value = declared.get("id") or declared.get("name") or declared.get("tokenizer_id")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"structured tokenizer metadata has no string ID: {declared!r}")
        return {
            "id": value,
            "revision": declared.get("revision") or declared.get("tokenizer_revision"),
        }
    if isinstance(declared, str) and declared.strip():
        return {"id": declared, "revision": bank.metadata.get("tokenizer_revision")}
    raise ValueError(f"unsupported tokenizer metadata: {declared!r}")


def _declared_token_rows(bank: Bank, tokenizer, protocol: dict[str, Any]) -> list[list[int]] | None:
    mode = str(protocol.get("token_id_input", "auto"))
    if mode == "text":
        return None
    if bank.token_ids is None or any(row is None for row in bank.token_ids):
        if mode == "token_ids":
            raise ValueError("token-ID scoring requested but the bank has incomplete token rows")
        return None
    declared = _tokenizer_metadata(bank)
    if declared is None:
        raise ValueError(
            "token-ID rows require declared tokenizer identity; use text input explicitly"
        )
    name = str(getattr(tokenizer, "name_or_path", "") or "").rstrip("/")
    declared_id = declared["id"].rstrip("/")
    canonical = _canonical_tokenizer_id(declared_id)
    actual = _canonical_tokenizer_id(name)
    protocol_scorer = _canonical_tokenizer_id(str(protocol.get("scorer")))
    if actual != canonical and protocol_scorer != canonical:
        if mode == "token_ids":
            raise ValueError(
                f"token IDs are in tokenizer space {declared['id']!r}, not scorer {name!r}"
            )
        # Auto mode retokenizes the lossless text; it never borrows IDs from a
        # different vocabulary.  The resolved metric reports text input.
        return None
    # Tokenizer and model revisions are distinct production pins (for example,
    # GPT-2-large weights use the recorded GPT-2 tokenizer revision).
    declared_revision = declared.get("revision")
    protocol_revision = protocol.get("tokenizer_revision") or protocol.get("revision")
    if (
        declared_revision is not None
        and protocol_revision is not None
        and str(declared_revision) != str(protocol_revision)
    ):
        if mode == "token_ids":
            raise ValueError(
                f"tokenizer revision mismatch: bank {declared_revision!r}, scorer "
                f"{protocol_revision!r}"
            )
        return None
    return [list(row) for row in bank.token_ids]


def gen_ppl(generated: Bank, protocol: dict[str, Any], device: str | None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = protocol["scorer"]
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(protocol), revision=_tokenizer_revision(protocol), use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, revision=protocol["revision"]).eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    context = _resolved_max_tokens(
        generated, protocol, int(getattr(model.config, "n_positions", 1024) or 1024)
    )
    token_rows = _declared_token_rows(generated, tokenizer, protocol)
    sample_nll: list[float] = []
    sample_entropies: list[float] = []
    sample_lengths: list[int] = []
    token_loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        if token_rows is not None:
            for start in range(0, len(token_rows), protocol["batch_size"]):
                batch = [row[:context] for row in token_rows[start : start + protocol["batch_size"]]]
                width = max(len(row) for row in batch)
                input_ids = torch.full(
                    (len(batch), width), tokenizer.pad_token_id, dtype=torch.long
                )
                attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
                for index, row in enumerate(batch):
                    input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
                    attention_mask[index, : len(row)] = 1
                encoded = {
                    "input_ids": input_ids.to(target_device),
                    "attention_mask": attention_mask.to(target_device),
                }
                logits = model(**encoded).logits[:, :-1, :]
                labels = encoded["input_ids"][:, 1:]
                mask = encoded["attention_mask"][:, 1:].bool()
                losses = (
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
                    )
                    .reshape(labels.shape)
                    .masked_fill(~mask, 0.0)
                )
                sample_nll.extend((losses.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().tolist())
                sample_entropies.extend(
                    _empirical_entropy_nats(row) for row in batch
                )
                sample_lengths.extend(len(row) for row in batch)
                token_loss_sum += float(losses.sum().item())
                token_count += int(mask.sum().item())
        else:
            for start in range(0, len(generated.texts), protocol["batch_size"]):
                batch = list(generated.texts[start : start + protocol["batch_size"]])
                encoded = tokenizer(
                    batch,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=context,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(target_device) for key, value in encoded.items()}
                logits = model(**encoded).logits[:, :-1, :]
                labels = encoded["input_ids"][:, 1:]
                mask = encoded["attention_mask"][:, 1:].bool()
                losses = (
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
                    )
                    .reshape(labels.shape)
                    .masked_fill(~mask, 0.0)
                )
                sums = losses.sum(dim=1)
                counts = mask.sum(dim=1)
                sample_nll.extend((sums / counts.clamp(min=1)).cpu().tolist())
                valid_lengths = encoded["attention_mask"].sum(dim=1).cpu().tolist()
                sample_entropies.extend(
                    _empirical_entropy_nats(
                        [int(token) for token, valid in zip(row, valid_row) if valid]
                    )
                    for row, valid_row in zip(encoded["input_ids"].cpu(), encoded["attention_mask"].cpu())
                )
                sample_lengths.extend(int(value) for value in valid_lengths)
                token_loss_sum += float(sums.sum().item())
                token_count += int(counts.sum().item())
    if not sample_nll:
        raise RuntimeError("no scoreable samples; Gen-PPL requires at least two tokens")
    if min(sample_lengths) < 2:
        raise ValueError("Gen-PPL requires at least two tokens per sample")
    mean_nll = float(np.mean(sample_nll))
    token_weighted_nll = float(token_loss_sum / max(token_count, 1))
    return {
        "metric": "gen_ppl",
        "status": "ok",
        "gen_ppl": float(np.exp(mean_nll)),
        "mean_nll": mean_nll,
        "std_nll": float(np.std(sample_nll)),
        "token_weighted_gen_ppl": float(np.exp(token_weighted_nll)),
        "token_weighted_nll": token_weighted_nll,
        "h_emp": float(np.mean(sample_entropies)),
        "h_emp_std": float(np.std(sample_entropies)),
        "mean_token_length": float(np.mean(sample_lengths)),
        "min_token_length": int(np.min(sample_lengths)),
        "max_token_length": int(np.max(sample_lengths)),
        "n_scored": len(sample_nll),
        "n_tokens_scored": token_count,
        "scoring_input": "token_ids" if token_rows is not None else "text",
        "max_length": context,
        "protocol": protocol,
    }


def gradient_moment(generated: Bank, reference: Bank, protocol: dict[str, Any], device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = protocol["scorer"]
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(protocol), revision=_tokenizer_revision(protocol), use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, revision=protocol["revision"]).eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    maximum = _resolved_max_tokens(
        generated,
        protocol,
        int(getattr(model.config, "n_positions", 1024) or 1024),
    )
    reference_gradient = _mean_gradient(
        model=model,
        tokenizer=tokenizer,
        texts=reference.texts,
        token_rows=_declared_token_rows(reference, tokenizer, protocol),
        device=target_device,
        batch_size=protocol["batch_size"],
        maximum=maximum,
    )
    generated_gradient = _mean_gradient(
        model=model,
        tokenizer=tokenizer,
        texts=generated.texts,
        token_rows=_declared_token_rows(generated, tokenizer, protocol),
        device=target_device,
        batch_size=protocol["batch_size"],
        maximum=maximum,
    )
    value = 0.0
    generated_norm = 0.0
    reference_norm = 0.0
    with torch.no_grad():
        for generated_parameter, reference_parameter in zip(
            generated_gradient, reference_gradient
        ):
            generated_parameter = generated_parameter.float()
            reference_parameter = reference_parameter.float()
            difference = generated_parameter - reference_parameter
            value += float((difference**2).sum().item())
            generated_norm += float((generated_parameter**2).sum().item())
            reference_norm += float((reference_parameter**2).sum().item())
    return {
        "metric": "gradient_moment",
        "status": "ok",
        "gradient_moment": value,
        "gen_grad_norm_sq": generated_norm,
        "ref_grad_norm_sq": reference_norm,
        "n_reference": len(reference),
        "n_generated": len(generated),
        "max_length": maximum,
        "protocol": protocol,
    }


def _mean_gradient(
    *,
    model,
    tokenizer,
    texts: tuple[str, ...],
    token_rows: list[list[int]] | None,
    device,
    batch_size: int,
    maximum: int,
) -> list:
    import torch
    import torch.nn.functional as F

    model.zero_grad(set_to_none=True)
    if token_rows is None:
        rows = [
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=maximum,
                return_attention_mask=False,
            )["input_ids"]
            for text in texts
        ]
    else:
        rows = [row[:maximum] for row in token_rows]
    rows = [row for row in rows if len(row) >= 2]
    if not rows:
        raise ValueError("gradient moment requires at least one two-token sample")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        width = max(len(row) for row in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for index, row in enumerate(batch):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        encoded = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
        logits = model(**encoded).logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        mask = encoded["attention_mask"][:, 1:].bool()
        losses = (
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
            )
            .reshape(labels.shape)
            .masked_fill(~mask, 0.0)
        )
        sequence = losses.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # The denominator is the number of sequences, not number of batches;
        # every sequence has equal weight, including final short batches.
        (sequence.sum() / len(rows)).backward()
    gradient = []
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient.append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    return gradient


def fbd(generated: Bank, reference: Bank, protocol: dict[str, Any], device: str | None):
    # MAUVE's max-pooled feature pipeline is retained for exact compatibility.
    left, right = _features(generated, reference, protocol, device)
    return {
        "metric": "gpt2_large_fbd",
        "status": "ok",
        "fbd": frechet_squared(left, right),
        "feature_dimension": left.shape[1],
        "protocol": protocol,
    }


def _mauve_features(texts: tuple[str, ...], protocol: dict[str, Any], device: str | None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    source = protocol["featurizer"]
    revision = protocol["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(protocol), revision=_tokenizer_revision(protocol), use_fast=True
    )
    model = AutoModel.from_pretrained(
        source, revision=revision, pad_token_id=tokenizer.eos_token_id
    ).eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    tokenized = [
        tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=protocol["max_tokens"]
        )
        for text in texts
    ]
    if any(len(row) < 2 for row in tokenized):
        raise ValueError("MAUVE requires at least two tokens per selected sample")
    tokens = [torch.tensor([row], dtype=torch.long) for row in tokenized]
    from mauve.utils import featurize_tokens_from_model

    features = featurize_tokens_from_model(model, tokens, protocol["batch_size"])
    return features.detach().cpu().numpy()


def mauve(generated: Bank, reference: Bank, protocol: dict[str, Any], device: str | None):
    import mauve

    minimum = int(protocol["minimum_subset_size"])
    subset_size = int(protocol["subset_size"])
    if min(len(reference), len(generated)) < minimum:
        raise ValueError(
            f"MAUVE requires at least {minimum} samples per bank for the paper protocol"
        )
    if len(reference) < subset_size or len(generated) < subset_size:
        raise ValueError(
            f"MAUVE paper protocol requires {subset_size} samples per bank; "
            "use a custom smaller subset_size for an explicitly custom protocol"
        )
    # Extract each complete bank once.  Subset seeds then select feature rows
    # without reloading the featurizer five times.
    p_full = _mauve_features(tuple(reference.texts), protocol, device)
    q_full = _mauve_features(tuple(generated.texts), protocol, device)
    scores = []
    selections = []
    for seed in protocol["subset_seeds"]:
        p_ranked = rank_bank_ids(reference, "p", seed, protocol["selection_base_seed"])
        q_ranked = rank_bank_ids(generated, "q", seed, protocol["selection_base_seed"])
        # Hashed selection chooses membership; evaluation restores canonical
        # sample-ID order, as required by the recorded protocol.
        p_selection = sorted(p_ranked[:subset_size], key=lambda index: _stable_id(reference.ids[index]))
        q_selection = sorted(q_ranked[:subset_size], key=lambda index: _stable_id(generated.ids[index]))
        selections.append(
            {
                "seed": seed,
                "p_ids": [reference.ids[index] for index in p_selection],
                "q_ids": [generated.ids[index] for index in q_selection],
            }
        )
        result = mauve.compute_mauve(
            p_features=p_full[p_selection],
            q_features=q_full[q_selection],
            kmeans_explained_var=0.9,
            kmeans_num_redo=5,
            kmeans_max_iter=500,
            mauve_scaling_factor=protocol["scaling_factor"],
            seed=protocol["internal_seed"],
            verbose=False,
        )
        scores.append(float(result.mauve))
    return {
        "metric": "mauve",
        "status": "ok",
        "mauve": float(np.mean(scores)),
        "mauve_std": float(np.std(scores)),
        "mauve_scores_by_seed": scores,
        "selections": selections,
        "featurizer_revision": protocol["revision"],
        "featurizer_source": (
            "AutoTokenizer/AutoModel loaded at the protocol revision; "
            "mauve-text 0.4.0 featurize_tokens_from_model and clustering/scoring"
        ),
        "feature_dimension": int(p_full.shape[1]),
        "protocol": protocol,
    }


def encoder_space_metrics(metric, generated, reference, protocol, device):
    requested = (
        protocol["encoders"] if protocol.get("encoder", "all") == "all" else [protocol["encoder"]]
    )
    rows = []
    for encoder in requested:
        right, left = encoder_features(
            generated, reference, {**protocol, "encoder": encoder}, device
        )
        components = min(protocol["reference_pca_whitening"], left.shape[1], len(left) - 1)
        mean = left.mean(axis=0)
        centered = left - mean
        _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (singular_values**2) / max(1, len(left) - 1)
        tolerance = 1e-10 * max(1.0, float(eigenvalues[0])) if len(eigenvalues) else 1e-10
        components = min(components, int(np.count_nonzero(eigenvalues > tolerance)))
        retained = vectors[:components]
        scales = np.sqrt(np.maximum(eigenvalues[:components], tolerance))
        transformed_left = centered @ retained.T / scales
        transformed_right = (right - mean) @ retained.T / scales
        if metric == "encoder_ed":
            value = energy_distance(transformed_left, transformed_right)
            key = "encoder_ed"
        else:
            value = mmd2_from_features(
                transformed_left, transformed_right, bandwidths=protocol["bandwidths"]
            )["mmd2_unbiased"]
            key = "encoder_gaussian_rbf_mmd2"
        rows.append({"encoder": encoder, key: value, "n_components": int(components)})
    return {"metric": metric, "status": "ok", "by_encoder": rows, "protocol": protocol}


def encoder_features(
    generated: Bank, reference: Bank, protocol: dict[str, Any], device: str | None
):
    import torch

    encoder = protocol["encoder"]
    if encoder == "all-mpnet-base-v2":
        from sentence_transformers import SentenceTransformer

        revision = protocol["revisions"][encoder]
        model = SentenceTransformer(
            encoder,
            revision=revision,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        model.max_seq_length = protocol["max_tokens"]

        def encode(texts):
            return np.asarray(
                model.encode(
                    list(texts),
                    batch_size=protocol["batch_size"],
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                ),
                dtype=np.float32,
            )

        left, right = encode(reference.texts), encode(generated.texts)
    else:
        from transformers import AutoModel, AutoTokenizer

        revision = protocol["revisions"][encoder]
        tokenizer = AutoTokenizer.from_pretrained(encoder, revision=revision, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModel.from_pretrained(encoder, revision=revision)
            .eval()
            .to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        )
        target = next(model.parameters()).device
        prefix = "passage: " if encoder == "intfloat/e5-base-v2" else ""

        def encode(texts):
            return _transformer_pool(
                tokenizer, model, [prefix + text for text in texts], protocol, target
            )

        left, right = encode(reference.texts), encode(generated.texts)
    left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return right, left


def _transformer_pool(tokenizer, model, texts, protocol, device):
    import torch

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), protocol["batch_size"]):
            batch = texts[start : start + protocol["batch_size"]]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=True,
                max_length=protocol["max_tokens"],
                padding=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            special_tokens_mask = special_tokens_mask.to(device)
            hidden = model(**encoded).last_hidden_state
            attention = encoded["attention_mask"].bool()
            if protocol["encoder"] in {"bert-base-uncased", "roberta-base"}:
                mask = attention & ~special_tokens_mask.bool()
            else:
                mask = attention
            expanded = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)
            outputs.append(pooled.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _resolved_max_tokens(bank: Bank, protocol: dict[str, Any], model_context: int) -> int:
    value = protocol.get("max_tokens")
    if isinstance(value, int):
        return min(value, model_context)
    declared = (
        bank.metadata.get("seq_len")
        or bank.metadata.get("sequence_length")
        or (bank.metadata.get("length_policy") or {}).get("sequence_length")
    )
    return max(2, min(int(declared), model_context)) if declared is not None else model_context


def rank_bank_ids(bank: Bank, role: str, seed: int, base_seed: int) -> list[int]:
    import hashlib
    import json

    def stable(value):
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

    ranked = sorted(
        range(len(bank)),
        key=lambda index: hashlib.sha256(
            "|".join(
                (
                    "dlmbench-mauve-all-generators-sample-size-v1",
                    "mauve_all_canonical_generators_sample_size_sensitivity",
                    "reference_vs_generator",
                    role,
                    str(seed),
                    bank.id,
                    str(base_seed),
                    stable(bank.ids[index]),
                )
            ).encode("utf-8")
        ).hexdigest(),
    )
    return ranked
