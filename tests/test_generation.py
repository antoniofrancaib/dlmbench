from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dlmbench.generators import (
    CountArtifact,
    PhraseArtifact,
    UnsupportedGenerationError,
    load_count_artifact,
    load_phrase_artifact,
)
from dlmbench.generators.output import prepare_output, write_output


def count_artifact(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "token_counts",
        "dataset": "lm1b",
        "tokenizer": {"id": "gpt2", "revision": "607a30d"},
        "source": {"dataset_id": "dvruette/lm1b", "revision": "2e8a29"},
        "vocabulary_size": 3,
        "n_tokens": 10,
        "counts": [1, 7, 2],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def phrase_artifact(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "kind": "phrase_bank",
        "dataset": "lm1b",
        "tokenizer": {"id": "gpt2", "revision": "607a30d"},
        "source": {"dataset_id": "dvruette/lm1b", "revision": "2e8a29"},
        "ngram": 5,
        "phrases": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_count_artifact_rejects_synthesized_or_mismatched_values(tmp_path: Path) -> None:
    good = load_count_artifact(count_artifact(tmp_path / "counts.json"))
    assert isinstance(good, CountArtifact)
    assert good.vocabulary_size == 3
    assert good.sha256 == hashlib.sha256((tmp_path / "counts.json").read_bytes()).hexdigest()

    payload = json.loads((tmp_path / "counts.json").read_text())
    payload["n_tokens"] = 99
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnsupportedGenerationError):
        load_count_artifact(bad)


def test_phrase_artifact_requires_exact_five_token_rows(tmp_path: Path) -> None:
    good = load_phrase_artifact(phrase_artifact(tmp_path / "phrases.json"))
    assert isinstance(good, PhraseArtifact)
    assert good.ngram == 5
    assert len(good.phrases) == 2

    payload = json.loads((tmp_path / "phrases.json").read_text())
    payload["phrases"].append([1, 2, 3])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnsupportedGenerationError):
        load_phrase_artifact(bad)


def test_output_writes_manifest_and_checksums_and_rejects_conflicts(tmp_path: Path) -> None:
    rows = [{"id": index, "text": "row", "token_ids": [index]} for index in range(3)]
    manifest = {
        "generator_id": "test",
        "n_samples": 3,
        "sampling": {"seed": 1, "batch_size": 3, "precision": "token IDs"},
    }
    out = write_output(rows, tmp_path / "out", manifest)
    sample_digest = hashlib.sha256((out / "samples.jsonl").read_bytes()).hexdigest()
    stored = json.loads((out / "manifest.json").read_text())
    assert stored["n_samples"] == 3
    assert stored["output_sha256"] == sample_digest
    assert stored["runtime"]["slurm_job_id"] in {None, __import__("os").environ.get("SLURM_JOB_ID")}
    lines = (out / "checksums.sha256").read_text().splitlines()
    assert lines[0].startswith(sample_digest)
    assert lines[1].startswith(
        hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    )

    prepare_output(out, sample_digest)
    with pytest.raises(UnsupportedGenerationError):
        prepare_output(out, "0" * 64)
    with pytest.raises(UnsupportedGenerationError):
        write_output(rows, out, manifest)


def test_nonempty_output_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(UnsupportedGenerationError):
        write_output([], out, {"n_samples": 0})


def test_generation_emits_decoded_bank_compatible_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    from dlmbench.banks import load_bank
    from dlmbench.generators.cli import attach_parser
    from dlmbench.generators.tokenization import TokenizerSnapshot, provenance

    snapshot = TokenizerSnapshot(
        path=tmp_path / "tokenizer",
        identifier="gpt2",
        revision="607a30d",
        digest="a" * 64,
        tokenizer=object(),
    )
    monkeypatch.setattr(
        "dlmbench.generators.cli.load_tokenizer_snapshot",
        lambda path, *, expected_id, expected_revision: (
            snapshot if expected_id == "gpt2" and expected_revision == "607a30d"
            else pytest.fail("unexpected tokenizer provenance")
        ),
    )
    from dlmbench.generators import tokenization

    monkeypatch.setattr(
        tokenization,
        "decode_token_rows",
        lambda current_snapshot, rows: [f"decoded-{row[0]}" for row in rows],
        raising=False,
    )
    monkeypatch.setattr(
        tokenization,
        "provenance",
        lambda current_snapshot: provenance(snapshot),
        raising=False,
    )

    payload = json.loads(count_artifact(tmp_path / "counts.json").read_text())
    payload["vocabulary_size"] = 32
    payload["n_tokens"] = 32
    payload["counts"] = [1] * 32
    artifact = tmp_path / "counts32.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "tokenizer").mkdir()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    attach_parser(subparsers)
    args = parser.parse_args([
        "generate", "run", "--generator", "lm1b_topk_iid_k32", "--counts", str(artifact),
        "--out", str(tmp_path / "ids"), "--sample-count", "2", "--seed", "17",
        "--tokenizer", str(tmp_path / "tokenizer"),
    ])
    args.func(args)
    manifest = json.loads((tmp_path / "ids" / "manifest.json").read_text())
    rows = [json.loads(line) for line in (tmp_path / "ids" / "samples.jsonl").read_text().splitlines()]
    assert manifest["output_mode"] == "text_and_token_ids"
    assert manifest["n_samples"] == 2
    assert manifest["sampler"]["seed"] == 17
    assert manifest["tokenizer"]["id"] == "gpt2"
    assert manifest["tokenizer"]["revision"] == "607a30d"
    assert "user-supplied assertions" in manifest["tokenizer"]["acquisition"]
    assert [row["id"] for row in rows] == [0, 1]
    assert all(row["text"].startswith("decoded-") for row in rows)
    assert all("token_ids" in row for row in rows)

    bank = load_bank(tmp_path / "ids")
    assert bank.ids == (0, 1)
    assert len(bank.token_ids) == len(bank.texts) == 2
    assert bank.identity()["ids_sha256"] is not None
    assert bank.identity()["token_ids_sha256"] is not None
    assert bank.identity()["n_samples"] == 2


def test_zero_sample_count_fails_rather_than_falling_back(tmp_path: Path) -> None:
    import argparse

    from dlmbench.generators.cli import attach_parser

    payload = json.loads(count_artifact(tmp_path / "counts.json").read_text())
    payload["vocabulary_size"] = 32
    payload["n_tokens"] = 32
    payload["counts"] = [1] * 32
    artifact = tmp_path / "counts32.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    attach_parser(subparsers)
    args = parser.parse_args([
        "generate", "run", "--generator", "lm1b_topk_iid_k32", "--counts", str(artifact),
        "--out", str(tmp_path / "ids"), "--sample-count", "0",
        "--tokenizer", str(tmp_path / "tokenizer"),
    ])
    with pytest.raises(UnsupportedGenerationError, match="sample count must be positive"):
        args.func(args)
    assert not (tmp_path / "ids").exists()


def test_text_generation_requires_a_tokenizer(tmp_path: Path) -> None:
    import argparse

    from dlmbench.generators.cli import attach_parser

    payload = json.loads(count_artifact(tmp_path / "counts.json").read_text())
    payload["vocabulary_size"] = 32
    payload["n_tokens"] = 32
    payload["counts"] = [1] * 32
    artifact = tmp_path / "counts32.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    attach_parser(subparsers)
    args = parser.parse_args([
        "generate", "run", "--generator", "lm1b_topk_iid_k32", "--counts", str(artifact),
        "--out", str(tmp_path / "ids"), "--sample-count", "2",
    ])
    with pytest.raises(UnsupportedGenerationError, match="requires --tokenizer"):
        args.func(args)


def test_prepare_corpus_builds_count_and_phrase_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from dlmbench.generators.artifacts import load_count_artifact, load_phrase_artifact
    from dlmbench.generators.cli import attach_parser
    from dlmbench.generators.tokenization import TokenizerSnapshot


    class TinyTokenizer:
        vocab_size = 3
        name_or_path = "tiny-tokenizer"

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return [ord(character) % 3 for character in text]

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"text": "abcdef"}\n{"text": "bcdefg"}\n',
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text(
        '{"dataset_id": "my-local-corpus", "revision": "v1"}\n',
        encoding="utf-8",
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    snapshot = TokenizerSnapshot(
        path=tokenizer_dir,
        identifier="gpt2",
        revision="607a30d",
        digest="a" * 64,
        tokenizer=TinyTokenizer(),
    )
    monkeypatch.setattr(
        "dlmbench.generators.cli.load_tokenizer_snapshot",
        lambda path, *, expected_id, expected_revision: snapshot,
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    attach_parser(subparsers)
    args = parser.parse_args([
        "generate", "prepare-corpus", "--kind", "counts", "--corpus", str(corpus),
        "--tokenizer", str(tokenizer_dir), "--dataset", "my-data",
        "--tokenizer-id", "gpt2", "--tokenizer-revision", "607a30d",
        "--source", str(source), "--out", str(tmp_path / "counts.json"),
    ])
    args.func(args)
    counts = load_count_artifact(tmp_path / "counts.json")
    assert counts.dataset == "my-data"
    assert counts.tokenizer["id"] == "gpt2"
    assert counts.tokenizer["snapshot_sha256"] == "a" * 64
    assert "user-supplied assertions" in counts.tokenizer["acquisition"]
    assert counts.source["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert counts.n_tokens == 12

    args = parser.parse_args([
        "generate", "prepare-corpus", "--kind", "phrases", "--corpus", str(corpus),
        "--tokenizer", str(tokenizer_dir), "--dataset", "my-data",
        "--tokenizer-id", "gpt2", "--tokenizer-revision", "607a30d",
        "--source", str(source), "--out", str(tmp_path / "phrases.json"),
        "--max-phrases", "3",
    ])
    args.func(args)
    phrases = load_phrase_artifact(tmp_path / "phrases.json")
    assert phrases.ngram == 5
    assert len(phrases.phrases) == 3
    assert all(len(phrase) == 5 for phrase in phrases.phrases)

    args = parser.parse_args([
        "generate", "prepare-corpus", "--kind", "counts", "--corpus", str(corpus),
        "--tokenizer", str(tokenizer_dir), "--dataset", "my-data",
        "--tokenizer-id", "gpt2", "--tokenizer-revision", "607a30d",
        "--source", str(source), "--out", str(tmp_path / "counts.json"),
    ])
    with pytest.raises(UnsupportedGenerationError, match="refusing to overwrite existing"):
        args.func(args)


def test_naive_generation_rejects_wrong_dataset_or_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from dlmbench.generators.cli import attach_parser
    from dlmbench.generators.tokenization import TokenizerSnapshot

    snapshot = TokenizerSnapshot(
        path=tmp_path / "tokenizer",
        identifier="gpt2",
        revision="607a30d",
        digest="a" * 64,
        tokenizer=object(),
    )
    monkeypatch.setattr(
        "dlmbench.generators.cli.load_tokenizer_snapshot",
        lambda path, *, expected_id, expected_revision: snapshot,
    )

    payload = json.loads(count_artifact(tmp_path / "counts.json").read_text())
    payload["vocabulary_size"] = 32
    payload["n_tokens"] = 32
    payload["counts"] = [1] * 32
    payload["tokenizer"] = {"id": "gpt2", "revision": "607a30d"}
    artifact = tmp_path / "owt-counts.json"
    payload["dataset"] = "owt"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "tokenizer").mkdir()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    attach_parser(subparsers)
    args = parser.parse_args([
        "generate", "run", "--generator", "lm1b_topk_iid_k32", "--counts", str(artifact),
        "--out", str(tmp_path / "out"), "--sample-count", "2",
        "--tokenizer", str(tmp_path / "tokenizer"),
    ])
    with pytest.raises(UnsupportedGenerationError, match="does not match generator dataset"):
        args.func(args)

    payload = json.loads(count_artifact(tmp_path / "counts.json").read_text())
    payload["vocabulary_size"] = 32
    payload["n_tokens"] = 32
    payload["counts"] = [1] * 32
    payload["tokenizer"] = {"id": "bert-base-uncased", "revision": "86b5e09"}
    payload["dataset"] = "lm1b"
    artifact = tmp_path / "wrong-tokenizer-counts.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    args = parser.parse_args([
        "generate", "run", "--generator", "lm1b_topk_iid_k32", "--counts", str(artifact),
        "--out", str(tmp_path / "out"), "--sample-count", "2",
        "--tokenizer", str(tmp_path / "tokenizer"),
    ])
    with pytest.raises(UnsupportedGenerationError, match="does not match required tokenizer"):
        args.func(args)
