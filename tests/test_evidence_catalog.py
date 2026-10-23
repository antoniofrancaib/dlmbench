import json
from collections import Counter

import pytest

from dlmbench.experiments import (
    load_resource,
    metric_agreement,
    verify_all,
    verify_rebenchmark,
)
from dlmbench.experiments.evidence import verify_table_resource


def test_release_catalog_has_exact_paper_cohorts():
    catalog = load_resource_catalog()
    rows = catalog["banks"]
    assert catalog["schema_version"] == 1
    assert len(rows) == 82
    assert len({row["id"] for row in rows}) == 82
    assert Counter(row["dataset"] for row in rows) == {"owt": 68, "lm1b": 14}
    assert Counter(row["kind"] for row in rows) == {
        "trained": 72, "constructed": 8, "reference": 2,
    }
    assert {suite: len(ids) for suite, ids in catalog["suites"].items()} == {
        "paper-main": 31, "paper-judges": 21, "paper-all": 82,
    }
    assert Counter(row["dataset"] for row in rows
                   if "paper-main" in row["suites"]) == {"owt": 17, "lm1b": 14}
    assert Counter(row["dataset"] for row in rows
                   if "paper-judges" in row["suites"]) == {"owt": 21}
    assert all(row["path"] == f"samples/{row['dataset']}/{row['id']}/samples.jsonl"
               for row in rows if row["kind"] != "reference")


def test_unavailable_real_references_are_explicit_and_never_downloadable():
    catalog = load_resource_catalog()
    references = [row for row in catalog["banks"] if row["kind"] == "reference"]
    assert len(references) == 2
    for row in references:
        assert row["bytes"] is None
        assert row["path"] is None
        assert row["download_urls"] == []
        assert row["provenance"]["status"] == "unavailable-reference"
        assert row["provenance"]["limitations"]
        assert "unavailable" in " ".join(row["provenance"]["limitations"]).lower()
    assert all(row["path"] is not None and row["bytes"] > 0
               for row in catalog["banks"] if row["kind"] != "reference")


def test_catalog_aliases_are_unique_across_all_banks():
    catalog = load_resource_catalog()
    names = []
    for row in catalog["banks"]:
        names.append(row["id"])
        names.extend(row["aliases"])
    assert len(names) == len(set(names))


def test_judgment_contract_reports_exact_cohorts_and_excluded_raw_inputs():
    judgments = load_resource("judgments")
    assert judgments["status"] == "excluded-from-release"
    assert judgments["raw_judgments_bundled"] is False
    assert judgments["llm_pairwise"]["cohort"] == {
        "dataset": "owt", "banks": 21, "comparisons": 16560, "judges": 4,
    }
    assert judgments["human"]["cohort"] == {
        "dataset": "owt", "trained_generators": 12,
        "recorded_comparisons": 304, "retained_comparisons": 300,
        "excluded_comparisons": 4, "raters": 9,
    }


def test_bundled_rebenchmark_recomputes_exact_mean_ranks_and_validates_all_resources():
    outcome = verify_rebenchmark()
    assert outcome == {
        "status": "verified", "rows": 31, "datasets": {"owt": 17, "lm1b": 14},
        "metric_keys": [
            "ed", "fbd", "gen_ppl", "gm", "h_emp", "jcd", "mauve",
            "mean_rank", "mmd", "self_bleu",
        ],
    }
    assert verify_all()["status"] == "verified"
    rows = load_resource("rebenchmark")["rows"]
    assert next(row for row in rows if row["id"] == "owt_ar_base")["values"]["mean_rank"] > 2.58
    assert next(row for row in rows if row["id"] == "owt_ar_base")["values"]["mean_rank"] < 2.59


def test_wrong_rebenchmark_cohort_is_rejected(tmp_path):
    payload = load_resource("rebenchmark")
    payload["rows"] = payload["rows"][:-1]
    with pytest.raises(ValueError, match="cohort mismatch"):
        verify_rebenchmark(payload)


def test_direction_aware_agreement_changes_with_orientation_and_reference_cohort():
    payload = load_resource("rebenchmark")
    metrics = ["gen_ppl", "mauve"]
    generated = metric_agreement(payload, "owt", metrics)
    with_reference = metric_agreement(payload, "owt", metrics, exclude_reference=False)
    assert generated[0]["n"] == 16
    assert with_reference[0]["n"] == 17
    assert generated[0]["spearman_rho"] != with_reference[0]["spearman_rho"]


def test_compact_figure_tables_have_complete_finite_values():
    mauve = load_resource("mauve_sample_count")
    gen_ppl = load_resource("gen_ppl_context_horizons")
    assert verify_table_resource(mauve, value_count=4, labels=4)["rows"] == len(mauve["rows"])
    assert verify_table_resource(gen_ppl, value_count=7, labels=7)["rows"] == len(gen_ppl["rows"])


def test_rebenchmark_exports_exact_csv_and_json_cohorts(tmp_path):
    from dlmbench.experiments.evidence import export_rebenchmark
    payload = load_resource("rebenchmark")
    csv_path = tmp_path / "main.csv"
    json_path = tmp_path / "owt.json"
    assert export_rebenchmark(payload, csv_path)["rows"] == 31
    assert export_rebenchmark(payload, json_path, dataset="owt",
                              include_reference=False)["rows"] == 16
    import csv as csv_module
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 31
    assert rows[0]["id"] == "owt_data_train"
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["id"] for row in loaded] == [
        row["id"] for row in payload["rows"]
        if row["dataset"] == "owt" and row["kind"] != "reference"
    ]


def test_agreement_plot_replays_from_bundled_evidence(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from dlmbench.experiments.evidence import plot_agreement
    destination = tmp_path / "agreement.png"
    outcome = plot_agreement(load_resource("rebenchmark"), destination, dataset="owt",
                              metrics=["gen_ppl", "mauve", "gm", "jcd", "ed", "fbd", "mmd"])
    assert outcome == {"status": "written", "path": str(destination), "pairs": 21}
    assert destination.stat().st_size > 10_000


def load_resource_catalog():
    # The catalog is intentionally not one of the compact paper resources.
    from importlib.resources import files

    from dlmbench.banks import load_catalog
    resource = files("dlmbench.resources").joinpath("banks.json")
    assert resource.is_file()
    assert json.loads(resource.read_text(encoding="utf-8")) == load_catalog()
    return load_catalog()
