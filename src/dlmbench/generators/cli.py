"""CLI parser for generator inventory and portable generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import UnsupportedGenerationError
from .corpus import build_count_artifact, build_phrase_artifact
from .output import prepare_output, write_output
from .registry import catalog_sha256, generator, generators
from .tokenization import load_tokenizer_snapshot


def attach_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser("generate", help="generate samples or prepare sampler inputs")
    actions = command.add_subparsers(dest="generation_action", required=True)
    parser = actions.add_parser("list", help="list generators and their support status")
    parser.set_defaults(func=_list)
    parser.add_argument("--support", choices=["portable", "portable-unverified", "unsupported", "samples-only"])
    parser.add_argument("--dataset")
    parser.add_argument("--json", action="store_true")

    show = actions.add_parser("show", help="show one generator configuration")
    show.set_defaults(func=_show)
    show.add_argument("id")
    show.add_argument("--json", action="store_true")

    generate = actions.add_parser("run", help="generate a new sample bank")
    generate.set_defaults(func=_generate)
    generate.add_argument("--generator", required=True)
    generate.add_argument("--counts", type=Path, help="compact token-count artifact")
    generate.add_argument("--phrases", type=Path, help="compact phrase-bank artifact")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--sample-count", type=int)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--tokenizer", type=Path, help="local tokenizer snapshot directory")
    generate.add_argument(
        "--tokenizer-revision",
        help="explicit tokenizer revision assertion recorded in the manifest",
    )
    generate.add_argument("--source-root", type=Path)
    generate.add_argument("--checkpoint", type=Path)
    prepare = actions.add_parser("prepare-corpus", help="build compact naive-generator artifacts")
    prepare.set_defaults(func=_prepare_corpus)
    prepare.add_argument("--kind", choices=["counts", "phrases"], required=True)
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--tokenizer-id", required=True)
    prepare.add_argument("--tokenizer-revision")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--max-phrases", type=int)


def _list(args: argparse.Namespace) -> int:
    rows = generators()
    if args.support:
        rows = [row for row in rows if row["support"] == args.support]
    if args.dataset:
        rows = [row for row in rows if row["dataset"] == args.dataset]
    if args.json:
        print(json.dumps({"catalog_sha256": catalog_sha256(), "generators": rows}, indent=2))
    else:
        for row in rows:
            print(f"{row['id']}\t{row['dataset']}\t{row['support']}")
    return 0


def _show(args: argparse.Namespace) -> int:
    row = generator(args.id)
    if args.json:
        print(json.dumps(row, indent=2))
    else:
        for key, value in row.items():
            print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = generator(args.generator)
    sample_count = _effective_count(args, config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    if config["family"] != "naive":
        executable = {
            "flm-hydra", "langflow", "rdlm", "sdtt-official", "elf", "cobit",
        }
        if config["family"] not in executable:
            if config["support"] == "samples-only":
                message = "samples only: no checkpoint, seed, or portable sampler is available"
            elif config["support"] == "unsupported":
                message = "unsupported for fresh generation: incomplete provenance"
            else:
                message = "no executable adapter is implemented for this family"
            raise UnsupportedGenerationError(f"{config['id']}: {message}")
        if args.source_root is None or args.checkpoint is None or args.tokenizer is None:
            raise UnsupportedGenerationError(
                "learned generation requires --source-root, --checkpoint, and --tokenizer"
            )
        from .adapters import execute
        from .tokenization import load_tokenizer_snapshot, provenance

        expected_id = "bert-base-uncased" if config["dataset"] == "lm1b" else "gpt2"
        snapshot = load_tokenizer_snapshot(
            args.tokenizer,
            expected_id=expected_id,
            expected_revision=args.tokenizer_revision,
        )
        if snapshot.tokenizer is None:
            raise UnsupportedGenerationError("tokenizer snapshot was not loaded")
        prepare_output(args.out, require_empty=True)
        rows, manifest = execute(
            config,
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            tokenizer=snapshot,
            out_dir=args.out,
            sample_count=sample_count,
            seed=seed,
        )
        if len(rows) != sample_count:
            raise UnsupportedGenerationError(
                f"adapter produced {len(rows)} rows, expected {sample_count}"
            )
        manifest["tokenizer"] = provenance(snapshot)
        write_output(rows, args.out, manifest)
        print(args.out)
        return 0
    if args.sample_count is not None:
        config = {**config, "n_samples": sample_count}
    if args.seed is not None:
        config = {**config, "seed": seed}
    prepare_output(args.out, require_empty=True)
    rows, manifest = _generate_naive(config, args)
    write_output(rows, args.out, manifest)
    print(args.out)
    return 0


def _prepare_corpus(args: argparse.Namespace) -> int:
    if not str(args.tokenizer_id).strip():
        raise UnsupportedGenerationError("--tokenizer-id must be a canonical identifier")
    snapshot = load_tokenizer_snapshot(
        args.tokenizer,
        expected_id=str(args.tokenizer_id),
        expected_revision=args.tokenizer_revision,
    )
    if snapshot.tokenizer is None:
        raise UnsupportedGenerationError("tokenizer snapshot was not loaded")
    source = _read_json_object(args.source)
    if args.kind == "counts":
        output = build_count_artifact(
            corpus_path=args.corpus,
            tokenizer=snapshot,
            dataset=args.dataset,
            source=source,
            output_path=args.out,
        )
    else:
        if args.max_phrases is None:
            raise UnsupportedGenerationError("phrases requires --max-phrases")
        output = build_phrase_artifact(
            corpus_path=args.corpus,
            tokenizer=snapshot,
            dataset=args.dataset,
            source=source,
            output_path=args.out,
            max_phrases=args.max_phrases,
        )
    print(output)
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UnsupportedGenerationError(f"invalid source provenance JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise UnsupportedGenerationError(f"source provenance must be a JSON object: {path}")
    return payload


def _generate_naive(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_dataset = str(config["dataset"])
    # Historical LM1B naive baselines use GPT-2; only learned LM1B families use BERT.
    expected_tokenizer_id = "gpt2"
    if config["sampler"] == "phrase_bank":
        if args.phrases is None:
            raise UnsupportedGenerationError("phrase_bank requires --phrases")
        from .artifacts import load_phrase_artifact
        from .naive import historical_phrase_sample

        artifact = load_phrase_artifact(args.phrases)
        _validate_naive_artifact(artifact.dataset, artifact.tokenizer, expected_dataset, expected_tokenizer_id)
        sample_count = _effective_count(args, config)
        sequence_length = config["sequence_length"]
        token_rows = historical_phrase_sample(
            artifact.phrases,
            n_samples=sample_count,
            seq_len=sequence_length,
            seed=args.seed if args.seed is not None else config["seed"],
        )
        source_identity = {
            "artifact": str(Path(args.phrases).expanduser()),
            "sha256": artifact.sha256,
            "kind": "phrase_bank",
            "dataset": artifact.dataset,
            "tokenizer": artifact.tokenizer,
            "source": artifact.source,
            "ngram": artifact.ngram,
        }
    else:
        if args.counts is None:
            raise UnsupportedGenerationError(f"{config['sampler']} requires --counts")
        from .artifacts import load_count_artifact
        from .naive import (
            historical_mirror,
            historical_periodic,
            historical_topk_iid,
        )

        artifact = load_count_artifact(args.counts)
        _validate_naive_artifact(artifact.dataset, artifact.tokenizer, expected_dataset, expected_tokenizer_id)
        sample_count = _effective_count(args, config)
        sequence_length = config["sequence_length"]
        seed = args.seed if args.seed is not None else config["seed"]
        if config["sampler"] == "topk_iid":
            token_rows = historical_topk_iid(
                artifact.counts,
                k=config["k"],
                shape=(sample_count, sequence_length),
                seed=seed,
            ).tolist()
        elif config["sampler"] == "mirror":
            token_rows = historical_mirror(
                artifact.counts,
                source_k=config["source_k"],
                n_samples=sample_count,
                seq_len=sequence_length,
                seed=seed,
            ).tolist()
        else:
            row = historical_periodic(artifact.counts, k=config["k"], seq_len=sequence_length)
            token_rows = [row.tolist()] * sample_count
        source_identity = {
            "artifact": str(Path(args.counts).expanduser()),
            "sha256": artifact.sha256,
            "kind": "token_counts",
            "dataset": artifact.dataset,
            "tokenizer": artifact.tokenizer,
            "source": artifact.source,
            "n_tokens": artifact.n_tokens,
        }
    if args.tokenizer is None:
        raise UnsupportedGenerationError("generation requires --tokenizer to emit Bank-compatible text")
    from .tokenization import decode_token_rows, provenance

    expected_tokenizer = source_identity["tokenizer"]
    expected_revision = (
        args.tokenizer_revision
        if args.tokenizer_revision is not None
        else expected_tokenizer.get("revision")
    )
    snapshot = load_tokenizer_snapshot(
        args.tokenizer,
        expected_id=str(expected_tokenizer.get("id")),
        expected_revision=expected_revision,
    )
    if str(snapshot.identifier) != expected_tokenizer_id:
        raise UnsupportedGenerationError(
            f"generator expects tokenizer {expected_tokenizer_id!r}, got {snapshot.identifier!r}"
        )
    texts = decode_token_rows(snapshot, token_rows)
    tokenizer_provenance = provenance(snapshot)
    rows = [
        {"id": index, "text": text, "token_ids": token_row}
        for index, (text, token_row) in enumerate(zip(texts, token_rows, strict=True))
    ]
    return rows, {
        "generator_id": config["id"],
        "label": config.get("label"),
        "dataset": config["dataset"],
        "sequence_length": sequence_length,
        "sampler": {
            "name": config["sampler"],
            "parameters": config.get("parameters", {}),
            "seed": args.seed if args.seed is not None else config["seed"],
            "batch_size": 1,
            "precision": "integer token IDs",
            "nfe": None,
            "actual_nfe": None,
        },
        "source": source_identity,
        "tokenizer": tokenizer_provenance,
        "output_mode": "text_and_token_ids",
        "n_samples": sample_count,
        "support": "portable",
    }


def _validate_naive_artifact(
    artifact_dataset: str,
    artifact_tokenizer: dict[str, Any],
    expected_dataset: str,
    expected_tokenizer_id: str,
) -> None:
    if artifact_dataset != expected_dataset:
        raise UnsupportedGenerationError(
            f"artifact dataset {artifact_dataset!r} does not match generator dataset {expected_dataset!r}"
        )
    if str(artifact_tokenizer.get("id")) != expected_tokenizer_id:
        raise UnsupportedGenerationError(
            f"artifact tokenizer {artifact_tokenizer.get('id')!r} does not match required "
            f"tokenizer {expected_tokenizer_id!r}"
        )


def _positive(value: int) -> int:
    if value < 1:
        raise UnsupportedGenerationError("sample count must be positive")
    return value


def _effective_count(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.sample_count is None:
        return _positive(int(config["n_samples"]))
    return _positive(args.sample_count)
