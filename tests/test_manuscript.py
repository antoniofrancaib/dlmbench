"""Independent manuscript-precision acceptance of the released main table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dlmbench.experiments.evidence import load_resource


def verify_manuscript_display(display: dict[str, Any], rebenchmark: dict[str, Any],
                               *, tolerance: float = 5e-1) -> dict[str, Any]:
    """Compare source rows to the frozen manuscript's displayed precision.

    This is an independent acceptance check, never a source of raw evidence.
    Reference JCD/mean-rank values are structural nulls in both payloads.
    """
    if display.get("schema_version") != 1:
        raise ValueError("unsupported manuscript display schema")
    if display.get("source", {}).get("limitation") != (
            "Manuscript display precision; acceptance target only, not raw evidence."):
        raise ValueError("manuscript display target must record its precision limitation")
    columns = display.get("columns", [])
    expected_columns = ["gen_ppl", "h_emp", "self_bleu", "mauve", "gm", "jcd", "ed",
                        "fbd", "mmd", "mean_rank"]
    if columns != expected_columns:
        raise ValueError(f"manuscript columns mismatch: {columns}")
    display_rows = display.get("rows", [])
    source_rows = rebenchmark.get("rows", [])
    if len(display_rows) != 31 or len(source_rows) != 31:
        raise ValueError("manuscript acceptance requires the exact 31-row main cohort")
    by_id = {row["id"]: row for row in source_rows}
    if {row["id"] for row in display_rows} != set(by_id):
        missing = sorted(set(by_id) - {row["id"] for row in display_rows})
        extra = sorted({row["id"] for row in display_rows} - set(by_id))
        raise ValueError(f"manuscript cohort mismatch; missing={missing}, extra={extra}")
    mismatches = []
    for row in display_rows:
        source = by_id[row["id"]]
        for metric in columns:
            target = row["values"].get(metric)
            value = source["values"].get(metric)
            if target == "null":
                target = None
            if target is None:
                if value is not None:
                    mismatches.append({"id": row["id"], "metric": metric,
                                       "reason": "source value where manuscript is null",
                                       "source": value})
                continue
            target_number = float(target)
            if value is None:
                mismatches.append({"id": row["id"], "metric": metric,
                                   "reason": "source value missing", "target": target_number})
                continue
            if abs(float(value) - target_number) > tolerance * 10 ** -_decimal_places(target):
                mismatches.append({"id": row["id"], "metric": metric, "source": value,
                                   "target": target_number})
    if mismatches:
        return {"status": "mismatch", "rows": len(display_rows), "mismatches": mismatches}
    return {"status": "verified", "rows": len(display_rows), "columns": len(columns)}



def _decimal_places(value: str) -> int:
    text = str(value).strip().lower()
    if "e" in text:
        return 0
    return len(text.split(".")[1]) if "." in text else 0


def test_main_table_matches_manuscript_display():
    display = json.loads((Path(__file__).parent / "fixtures/manuscript_display.json").read_text())
    source = load_resource("rebenchmark")
    assert verify_manuscript_display(display, source)["status"] == "verified"
    display["rows"][1]["values"]["mauve"] = "0.900"
    assert verify_manuscript_display(display, source)["status"] == "mismatch"
