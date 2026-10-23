"""Loading and verification of compact public paper evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..banks import load_catalog
from .rankings import mean_metric_ranks, spearman

PAPER_RESOURCES = {
    "rebenchmark": "rebenchmark.json",
    "archived_sweep": "archived_sweep.json",
    "judgments": "judgments.json",
    "sources": "sources.json",
    "mauve_sample_count": "mauve_sample_count.json",
    "gen_ppl_context_horizons": "gen_ppl_context_horizons.json",
    "jcd_directional": "jcd_directional.json",
    "entropy_census": "entropy_census.json",
    "entropy_controls": "entropy_controls.json",
    "encoder_rank_sensitivity": "encoder_rank_sensitivity.json",
    "encoder_rank_movement": "encoder_rank_movement.json",
    "encoder_pairwise_agreement": "encoder_pairwise_agreement.json",
    "encoder_pair_leaderboard": "encoder_pair_leaderboard.json",
    "encoder_scores_ranks": "encoder_scores_ranks.json",
}

INTEGRITY_RESOURCE = "integrity.json"

REBENCHMET_METRICS = {
    "gen_ppl": "lower",
    "h_emp": "higher",
    "self_bleu": "lower",
    "mauve": "higher",
    "gm": "lower",
    "jcd": "lower",
    "ed": "lower",
    "fbd": "lower",
    "mmd": "lower",
    "mean_rank": "lower",
}

TABLE_RESOURCES = {
    "mauve_sample_count": {
        "labels": 4,
        "value_count": 4,
        "rows": 75,
        "source_sha256": "fb294c29df035e79f56710dd3b302b6d49a5ace95e3e9ee9b8408fe99888e5cf",
    },
    "gen_ppl_context_horizons": {
        "labels": 7,
        "value_count": 7,
        "rows": 75,
        "source_sha256": "bde4d7db24c5d1fe5165bfc268fd49f1f098e950d158b0becf53dd51c27c9112",
    },
}

ARCHIVED_TABLE_RESOURCES = {
    "jcd_directional": {
        "rows": 79,
        "sections": 4,
        "labels": 67,
        "columns": 5,
        "source_sha256": "153ce94e8bab23270fa45625f07d613cf9797953538a84f229b1b92b7a77ade7",
    },
    "entropy_census": {
        "rows": 79,
        "sections": 4,
        "labels": 67,
        "columns": 3,
        "source_sha256": "3a1cc1b356c4875e7cd19dfa48bccc1ea8a0d4b16d2f002fbc0c3513f144c7f8",
    },
    "entropy_controls": {
        "rows": 8,
        "columns": 4,
        "source_sha256": "14ee76c2b80b9cf292303987643678912f9612db1214b4292fc4ee9bf752ae29",
    },
    "encoder_rank_movement": {
        "rows": 79,
        "sections": 4,
        "labels": 67,
        "columns": 4,
        "source_sha256": "3615915d4b8429b3c98e77eea7b942a1c29dae6c7be5eab9b1edfa117d440409",
    },
    "encoder_pair_leaderboard": {
        "rows": 120,
        "labels": 15,
        "columns": 5,
        "source_sha256": "b070414d9255672fb744395cc62647614640b47f79f72a0e393780e516d079c9",
    },
}

VEGA_RESOURCES = {
    "encoder_rank_sensitivity": {
        "values": 2136,
        "p0": {"Energy", "Fréchet", "MAUVE", "MMD²"},
        "p1": {"LM1B", "OWT (All)", "OWT (Short)"},
        "rows": 65,
        "encoders": 6,
        "source_sha256": "490227db5c4a829bf92f3ea371d5cfac9c2da48b170e0e52b682285891dac096",
    },
    "encoder_pairwise_agreement": {
        "values": 288,
        "p0": {"LM1B", "OpenWebText"},
        "p1": {"Energy", "Fréchet", "MAUVE", "MMD²"},
        "rows": 6,
        "cohorts": {13, 60},
        "encoders": 6,
        "source_sha256": "af0065a0c06c78ea44b98f3b1dccde3a02416152dc857b90261a4bd177014424",
    },
}


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and value == float(value) and not math.isinf(float(value)) and not math.isnan(float(value)))


def _numeric_value(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        token = value.strip().split()[0]
        if token.upper() in {"N/A", "NA", "NONE"}:
            raise ValueError(f"structural missing value must be null, not {value!r}")
        try:
            return float(token)
        except ValueError as error:
            raise ValueError(f"non-numeric paper value {value!r}") from error
    raise ValueError(f"non-numeric paper value {value!r}")


def load_resource(name: str) -> dict[str, Any]:
    if name not in PAPER_RESOURCES:
        raise ValueError(f"unknown paper resource {name!r}; available: {sorted(PAPER_RESOURCES)}")
    resource = files("dlmbench.resources.paper").joinpath(PAPER_RESOURCES[name])
    if not resource.is_file():
        raise FileNotFoundError(f"paper resource {name!r} is not bundled")
    return json.loads(resource.read_text(encoding="utf-8"))


def _resource_bytes(filename: str) -> bytes:
    resource = files("dlmbench.resources.paper").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"paper resource {filename!r} is not bundled")
    return resource.read_bytes()


def verify_packaged_integrity() -> dict[str, Any]:
    """Verify the exact packaged bytes of every public paper resource.

    This is archive integrity: it detects any changed finite value even when a
    payload keeps its shape and its internal ``source_sha256`` field. It is
    deliberately separate from manuscript display-precision acceptance.
    """
    payload = json.loads(_resource_bytes(INTEGRITY_RESOURCE))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported paper-resource integrity schema")
    recorded = payload.get("resources", {})
    if not isinstance(recorded, dict) or set(recorded) != set(PAPER_RESOURCES):
        raise ValueError("integrity manifest does not cover the packaged paper resources")
    outcomes: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for name, filename in PAPER_RESOURCES.items():
        expected = recorded.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            outcomes[name] = {"status": "mismatch", "error": "invalid checksum record"}
            failures[name] = "invalid checksum record"
            continue
        actual = hashlib.sha256(_resource_bytes(filename)).hexdigest()
        if actual != expected:
            outcomes[name] = {"status": "mismatch", "expected": expected, "actual": actual}
            failures[name] = f"expected {expected}, got {actual}"
        else:
            outcomes[name] = {"status": "verified", "sha256": actual}
    return {
        "status": "verified" if not failures else "mismatch",
        "resources": outcomes,
        **({"failures": failures} if failures else {}),
    }


def verify_rebenchmark(resource: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = resource or load_resource("rebenchmark")
    catalog = load_catalog()
    suite = set(catalog["suites"].get("paper-main", []))
    if not suite:
        raise ValueError("catalog suites are incomplete")
    rows = payload.get("rows", [])
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("rebenchmark rows contain duplicate IDs")
    if set(ids) != suite:
        missing = sorted(suite - set(ids))
        extra = sorted(set(ids) - suite)
        raise ValueError(f"rebenchmark cohort mismatch; missing={missing}, extra={extra}")
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    expected_counts = {"owt": 17, "lm1b": 14}
    counts = {name: len(items) for name, items in by_dataset.items()}
    if counts != expected_counts:
        raise ValueError(f"rebenchmark dataset counts mismatch: {counts}")
    for row in rows:
        unknown = set(row["values"]) - set(REBENCHMET_METRICS)
        if unknown:
            raise ValueError(f"unknown metrics in {row['id']}: {sorted(unknown)}")
        non_finite = {key: value for key, value in row["values"].items()
                      if value is not None and not _finite_number(value)}
        if non_finite:
            raise ValueError(f"non-finite metric values in {row['id']}: {non_finite}")
        reference = row["kind"] == "reference"
        if reference and (row["values"].get("mean_rank") is not None or row["values"].get("jcd") is not None):
            raise ValueError(f"reference row {row['id']} must not carry generated ranks or JCD")
    rank_payload = {}
    for dataset in expected_counts:
        metrics = ["mauve", "gm", "jcd", "ed", "fbd", "mmd"]
        # Rank 1 is best: higher MAUVE, lower discrepancy for all other metrics.
        directions = {metric: ("higher" if metric == "mauve" else "lower")
                      for metric in metrics}
        rank_payload[dataset] = mean_metric_ranks(by_dataset[dataset], metrics,
                                                   directions=directions, dataset=dataset)
        for row in by_dataset[dataset]:
            expected = rank_payload[dataset].get(row["id"])
            actual = row["values"].get("mean_rank")
            if expected is None and actual is None:
                continue
            if expected is None or actual is None or abs(float(expected) - float(actual)) > 5e-3:
                raise ValueError(f"mean-rank mismatch for {row['id']}: expected {expected}, got {actual}")
    return {"status": "verified", "rows": len(rows), "datasets": expected_counts,
            "metric_keys": sorted(REBENCHMET_METRICS)}


def verify_table_resource(resource: dict[str, Any], *, value_count: int,
                          labels: int | None = None, rows: int | None = None,
                          source_sha256: str | None = None) -> dict[str, Any]:
    """Validate a compact paper table and its recorded source identity."""
    if resource.get("schema_version") != 1:
        raise ValueError("unsupported paper resource schema")
    expected_labels = resource.get("labels", [])
    if labels is not None and len(expected_labels) != labels:
        raise ValueError(f"expected {labels} labels, got {len(expected_labels)}")
    if not expected_labels:
        raise ValueError("paper resource labels are empty")
    resource_rows = resource.get("rows", [])
    if not resource_rows:
        raise ValueError("paper resource rows are empty")
    if rows is not None and len(resource_rows) != rows:
        raise ValueError(f"expected {rows} rows, got {len(resource_rows)}")
    if source_sha256 and resource.get("source", {}).get("source_sha256") != source_sha256:
        raise ValueError(f"paper resource source identity mismatch: {resource.get('source', {}).get('source_sha256')}")
    seen: dict[tuple[str, str], int] = {}
    for row in resource_rows:
        label = row.get("label")
        views = row.get("views", "")
        dataset = "lm1b" if views == "lm1b" else "owt"
        key = (label, dataset)
        if not isinstance(label, str) or not label or key in seen:
            raise ValueError(f"paper resource rows require unique nonempty labels: {row}")
        seen[key] = 1
        values = [row.get(f"value_{index}") for index in range(1, value_count + 1)]
        for value in values:
            if value is None:
                continue
            number = _numeric_value(value)
            if not _finite_number(number):
                raise ValueError(f"paper resource row {label!r} has a non-finite value")
    return {"status": "verified", "rows": len(resource_rows), "labels": len(expected_labels)}


def verify_archived_table_resource(resource: dict[str, Any], *, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate compact archived tables and their source identities."""
    if resource.get("type") != "table":
        raise ValueError("archived paper resource is not a recorded table")
    source = resource.get("source", {})
    if source.get("source_sha256") != expected["source_sha256"]:
        raise ValueError("archived paper resource source identity mismatch")
    rows = resource.get("rows", [])
    if not isinstance(rows, list) or len(rows) != expected["rows"]:
        raise ValueError(f"expected {expected['rows']} archived table rows, got {len(rows) if isinstance(rows, list) else type(rows).__name__}")
    columns = resource.get("columns", [])
    if not isinstance(columns, list) or len(columns) != expected["columns"]:
        raise ValueError("archived paper resource columns mismatch")
    sections = 0
    labels = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or not row["label"]:
            raise ValueError("archived paper table rows require nonempty labels")
        if row.get("section"):
            sections += 1
            continue
        labels.add(row["label"])
        if "views" in row and (not isinstance(row["views"], list) or not row["views"]):
            raise ValueError("archived paper table views must be a nonempty list")
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(columns):
            raise ValueError(f"archived paper table value count mismatch: {row['label']!r}")
    if "sections" in expected and sections != expected["sections"]:
        raise ValueError("archived paper table section count mismatch")
    if "labels" in expected and len(labels) != expected["labels"]:
        raise ValueError(f"expected {expected['labels']} unique archived labels, got {len(labels)}")
    return {"status": "verified", "rows": len(rows), "labels": len(labels), "sections": sections}


def verify_vega_resource(resource: dict[str, Any], *, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate compact Vega-Lite payloads and their numeric value structure."""
    if resource.get("type") != "vega-lite":
        raise ValueError("archived paper resource is not a recorded Vega-Lite payload")
    if resource.get("source", {}).get("source_sha256") != expected["source_sha256"]:
        raise ValueError("archived paper resource source identity mismatch")
    spec = resource.get("spec", {})
    values = spec.get("data", {}).get("values")
    if not isinstance(values, list) or len(values) != expected["values"]:
        raise ValueError(f"expected {expected['values']} figure values, got {len(values) if isinstance(values, list) else type(values).__name__}")
    for key, expected_count in (("p0", len(expected["p0"])), ("p1", len(expected["p1"])),
                                ("r", expected["rows"]), ("c", expected["encoders"])):
        expected_values = expected.get(key, None)
        if expected_values is None:
            expected_values = set()
            for row in values:
                if key in row:
                    expected_values.add(row[key])
        actual = {row.get(key) for row in values if key in row}
        if len(actual) != expected_count:
            raise ValueError(f"figure payload {key!r} domain size mismatch")
        if not actual or actual != set(expected_values):
            raise ValueError(f"figure payload {key!r} domain mismatch")
    for row in values:
        if not _finite_number(row.get("v")):
            raise ValueError("figure payload values must be finite numbers")
        if "n" in row and (not isinstance(row["n"], int) or isinstance(row["n"], bool)):
            raise ValueError("figure payload cohort sizes must be integers")
    if "cohorts" in expected:
        cohorts = {row.get("n") for row in values if "n" in row}
        if cohorts != set(expected["cohorts"]):
            raise ValueError("figure payload cohort sizes mismatch")
    return {"status": "verified", "values": len(values)}


def verify_encoder_scores_resource(resource: dict[str, Any]) -> dict[str, Any]:
    """Validate recorded encoder scores and ranks grouped by dataset, metric, encoder."""
    if resource.get("schema_version") != 1:
        raise ValueError("unsupported encoder scores schema")
    expected_hash = "67901c255c98cb55745c4e226d2c88c0357af0b83eef0ce681291e2274e6f1bd"
    if resource.get("source", {}).get("source_sha256") != expected_hash:
        raise ValueError("archived encoder scores resource source identity mismatch")
    blocks = resource.get("blocks")
    if not isinstance(blocks, dict) or len(blocks) != 72:
        raise ValueError("encoder scores resource must contain 72 blocks")
    for key, rows in blocks.items():
        if not isinstance(key, str) or key.count(":") != 2:
            raise ValueError("encoder scores block keys must identify dataset, metric, encoder")
        if not isinstance(rows, list) or not rows or len(rows) not in {16, 18, 63}:
            raise ValueError(f"invalid encoder scores block row count: {key!r}")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("encoder scores rows must be JSON objects")
            if not isinstance(row.get("label"), str):
                raise TypeError("encoder scores rows require labels")
            cells = row.get("cells")
            if not isinstance(cells, list) or len(cells) != 2:
                raise ValueError(f"encoder scores row must have two cells: {row.get('label')!r}")
            if any(not isinstance(cell, str) for cell in cells):
                raise ValueError(f"encoder scores cells must be strings: {row.get('label')!r}")
    return {"status": "verified", "blocks": len(blocks)}


def verify_all(resources: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Verify every bundled compact paper resource and report the total coverage."""
    payload = resources or {name: load_resource(name) for name in PAPER_RESOURCES}
    outcomes: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    integrity = verify_packaged_integrity() if resources is None else {"status": "not_applicable"}
    if integrity.get("status") == "mismatch":
        failures.update(integrity.get("failures", {}))
    checks = {
        "rebenchmark": lambda: verify_rebenchmark(payload["rebenchmark"]),
        "archived_sweep": lambda: verify_archived_sweep(payload["archived_sweep"]),
        "mauve_sample_count": lambda: verify_table_resource(
            payload["mauve_sample_count"], value_count=TABLE_RESOURCES["mauve_sample_count"]["value_count"],
            **{key: value for key, value in TABLE_RESOURCES["mauve_sample_count"].items()
               if key != "value_count"}),
        "gen_ppl_context_horizons": lambda: verify_table_resource(
            payload["gen_ppl_context_horizons"], value_count=TABLE_RESOURCES["gen_ppl_context_horizons"]["value_count"],
            **{key: value for key, value in TABLE_RESOURCES["gen_ppl_context_horizons"].items()
               if key != "value_count"}),
        "judgments": lambda: verify_judgment_status(payload["judgments"]),
    }
    for name, expected in ARCHIVED_TABLE_RESOURCES.items():
        checks[name] = (lambda payload_name=name, expected=expected: verify_archived_table_resource(
            payload[payload_name], expected=expected))
    for name, expected in VEGA_RESOURCES.items():
        checks[name] = (lambda payload_name=name, expected=expected: verify_vega_resource(
            payload[payload_name], expected=expected))
    checks["encoder_scores_ranks"] = lambda: verify_encoder_scores_resource(payload["encoder_scores_ranks"])
    checks["sources"] = lambda: verify_sources_resource(payload["sources"])
    for name, check in checks.items():
        try:
            outcomes[name] = check()
            if isinstance(outcomes[name], dict) and outcomes[name].get("status") != "verified":
                failures[name] = str(outcomes[name].get("error", "resource verification returned a mismatch"))
        except (KeyError, TypeError, ValueError) as error:
            outcomes[name] = {"status": "mismatch", "error": str(error)}
            failures[name] = str(error)
    return {
        "status": "verified" if not failures else "mismatch",
        "archive_integrity": integrity,
        "resources": outcomes,
        **({"failures": failures} if failures else {}),
    }


def verify_sources_resource(resource: dict[str, Any]) -> dict[str, Any]:
    """Validate portable source identities without private filesystem locations."""
    if resource.get("schema_version") != 1:
        raise ValueError("unsupported source-map schema")
    atlas = resource.get("corpus_atlas", {})
    if atlas.get("source_file") != "6e5f1e7110fd-6e5f1e7110fd-corpus_atlas_manifest.json":
        raise ValueError("corpus atlas source identity mismatch")
    if atlas.get("sha256") != "6e5f1e7110fd526f51c286bbe71d7b920b6cb98f0caa73beca02fd2f8ed2d3c7":
        raise ValueError("corpus atlas source hash mismatch")
    catalog = resource.get("catalog", {})
    if catalog.get("bank_identities") != 82 or catalog.get("suites") != {
            "paper-main": 31, "paper-judges": 21, "paper-all": 82}:
        raise ValueError("source-map catalog contract mismatch")
    if resource.get("raw_judgments_bundled") is not False:
        raise ValueError("source map must state that raw judgments are not bundled")

    def reject_private(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                reject_private(item)
        elif isinstance(value, list):
            for item in value:
                reject_private(item)
        elif isinstance(value, str) and (value.startswith("/") or "lab/private/" in value or "/home/" in value):
            raise ValueError(f"source map contains a non-portable path: {value!r}")

    reject_private(resource)
    return {"status": "verified", "bank_identities": 82, "portable": True}


def verify_archived_sweep(resource: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete archived NFE/sweep table as a separate protocol."""
    if resource.get("type") != "table":
        raise ValueError("archived sweep is not a recorded table")
    source = resource.get("source", {})
    if source.get("source_sha256") != "4e677c003bf3c5bc8521a66754ab6243079c1330a4bb59c92427ae7e7cb5e07f":
        raise ValueError("archived sweep source identity mismatch")
    rows = resource.get("rows", [])
    columns = resource.get("columns", [])
    if len(rows) != 79 or len(columns) != 8:
        raise ValueError("archived sweep shape mismatch")
    sections = 0
    seen: set[tuple[str, str]] = set()
    labels: set[str] = set()
    for row in rows:
        label = row.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("archived sweep rows require labels")
        if row.get("section"):
            sections += 1
            continue
        views = row.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError("archived sweep rows require nonempty views")
        key = (label, views[0])
        if key in seen:
            raise ValueError(f"archived sweep contains duplicate label/view: {key}")
        seen.add(key)
        labels.add(label)
        views = row["views"]
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(columns):
            raise ValueError(f"archived sweep value count mismatch: {label!r}")
        for value in values:
            if value is None or _finite_number(value):
                continue
            _numeric_value(value)
    if sections != 4 or len(labels) != 67:
        raise ValueError("archived sweep cohort mismatch")
    return {"status": "verified", "rows": len(rows), "labels": len(labels), "sections": sections}

def verify_judgment_status(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "excluded-from-release":
        raise ValueError("judgment status must remain explicit")
    if payload.get("raw_judgments_bundled") is not False:
        raise ValueError("raw judgments must not be reported as bundled")
    expected_llm = {"dataset": "owt", "banks": 21, "comparisons": 16560, "judges": 4}
    if payload.get("llm_pairwise", {}).get("cohort") != expected_llm:
        raise ValueError("LLM pairwise cohort does not match the paper contract")
    expected_human = {
        "dataset": "owt", "trained_generators": 12, "recorded_comparisons": 304,
        "retained_comparisons": 300, "excluded_comparisons": 4, "raters": 9,
    }
    if payload.get("human", {}).get("cohort") != expected_human:
        raise ValueError("human cohort does not match the paper contract")
    return {"status": "verified", "raw_judgments_bundled": False,
            "llm_pairwise": expected_llm, "human": expected_human}


def metric_agreement(resource: dict[str, Any], dataset: str,
                    metrics: list[str], *, exclude_reference: bool = True) -> list[dict[str, Any]]:
    directions = REBENCHMET_METRICS
    rows = [row for row in resource["rows"] if row["dataset"] == dataset]
    if exclude_reference:
        rows = [row for row in rows if row["kind"] != "reference"]
    output = []
    for index, left in enumerate(metrics):
        for right in metrics[index + 1:]:
            cohort = [row for row in rows
                      if row["values"].get(left) is not None and row["values"].get(right) is not None]
            if len(cohort) < 2:
                continue
            rho = spearman([row["values"][left] for row in cohort],
                           [row["values"][right] for row in cohort],
                           left_direction=directions[left], right_direction=directions[right])
            output.append({"left": left, "right": right, "n": len(cohort), "spearman_rho": rho})
    return output


def _export_rows(payload: dict[str, Any], *, dataset: str | None = None,
                 include_reference: bool = True) -> list[dict[str, Any]]:
    rows = []
    for row in payload["rows"]:
        if dataset is not None and row["dataset"] != dataset:
            continue
        if not include_reference and row["kind"] == "reference":
            continue
        output = {"id": row["id"], "dataset": row["dataset"], "kind": row["kind"],
                  "n_samples": row.get("n_samples")}
        output.update(row["values"])
        rows.append(output)
    return rows


def _write_text(path: Path, content: str) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite existing output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def export_rebenchmark(resource: dict[str, Any], destination: Path, *, dataset: str | None = None,
                       include_reference: bool = True) -> dict[str, Any]:
    """Write the main-table evidence to a portable CSV or JSON file."""
    rows = _export_rows(resource, dataset=dataset, include_reference=include_reference)
    if not rows:
        raise ValueError("selected export cohort is empty")
    if destination.suffix.lower() == ".json":
        _write_text(destination, json.dumps(rows, indent=2, ensure_ascii=False,
                                            allow_nan=False) + "\n")
    elif destination.suffix.lower() == ".csv":
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("export destination must end in .csv or .json")
    return {"status": "written", "path": str(destination), "rows": len(rows)}


def plot_agreement(resource: dict[str, Any], destination: Path, *,
                   dataset: str, metrics: list[str]) -> dict[str, Any]:
    """Plot the finite-cohort metric agreement matrix without optional imports."""
    pairs = metric_agreement(resource, dataset, metrics)
    if len(pairs) < 2:
        raise ValueError("agreement plot requires at least two metric pairs")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise ImportError("plotting requires matplotlib and numpy from the paper extra") from error
    labels = list(dict.fromkeys([pair["left"] for pair in pairs] + [pair["right"] for pair in pairs]))
    index = {metric: position for position, metric in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan)
    for pair in pairs:
        matrix[index[pair["left"]], index[pair["right"]]] = pair["spearman_rho"]
        matrix[index[pair["right"]], index[pair["left"]]] = pair["spearman_rho"]
    figure, axis = plt.subplots(figsize=(7.2, 6.4))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    axis.set_title(f"Direction-aware tied-rank Spearman agreement ({dataset})")
    figure.colorbar(image, ax=axis, label="Spearman rho")
    figure.tight_layout()
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing output {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return {"status": "written", "path": str(destination), "pairs": len(pairs)}
