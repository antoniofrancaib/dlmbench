import json
import subprocess
import sys

import pytest

from dlmbench.banks import load_catalog
from dlmbench.experiments import PAPER_RESOURCES, load_resource, reproduce_run, verify_all
from dlmbench.experiments.evidence import verify_packaged_integrity


def test_unified_replay_cli_writes_a_complete_manifest(tmp_path):
    destination = tmp_path / "cli-replay"
    result = subprocess.run(
        [sys.executable, "-m", "dlmbench", "reproduce", "all", "--out", str(destination)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((destination / "replay-manifest.json").read_text())
    assert len(manifest["outputs"]) == 21
    assert sum(row["path"].endswith(".png") for row in manifest["outputs"]) == 5


def test_catalog_scientific_metadata_is_recorded_without_inference():
    catalog = load_catalog()
    fields = {"sequence_length", "tokenizer_id", "tokenizer_revision",
              "source_manifest_sha256", "checkpoint_digest", "seed"}
    for row in catalog["banks"]:
        metadata = row["provenance"]["scientific_metadata"]
        assert set(metadata) == fields
        assert all(value is None for value in metadata.values()) == (row["path"] is None)
    lm1b = {row["id"]: row for row in catalog["banks"] if row["dataset"] == "lm1b"}
    assert lm1b["lm1b_mdlm_1024_nfe"]["provenance"]["scientific_metadata"] == {
        "sequence_length": 128,
        "tokenizer_id": "bert-base-uncased",
        "tokenizer_revision": "local-sha256:7fad9202709f07f817e9ffebbbadcb1d2a6b798d13cdc8919325b677b3636c58",
        "source_manifest_sha256": "1bae2d03ef173039794ecbdb811f0bfe400518e80b218114d70a3d0889553dd6",
        "checkpoint_digest": "d42739e5a99bbd7c0f604dff7fe4c9ab12348b475e30f1d807e70ef32249e514",
        "seed": 1,
    }
    assert lm1b["lm1b_phrase_bank_1000"]["provenance"]["scientific_metadata"]["tokenizer_id"] == "gpt2"
    for row in catalog["banks"]:
        if row["dataset"] == "lm1b" and row["path"] is not None:
            assert row["provenance"]["scientific_metadata"]["sequence_length"] == 128


def test_integrity_detects_changed_resource_bytes(monkeypatch, tmp_path):
    import copy

    from dlmbench.experiments.evidence import load_resource as original
    payload = copy.deepcopy(original("rebenchmark"))
    ar = next(row for row in payload["rows"] if row["id"] == "owt_ar_base")
    ar["values"]["mauve"] = 0.812
    def load_resource(name):
        return payload if name == "rebenchmark" else original(name)
    monkeypatch.setattr("dlmbench.experiments.evidence.load_resource", load_resource)
    outcome = verify_all()
    assert outcome["archive_integrity"]["status"] == "verified"
    assert outcome["resources"]["rebenchmark"]["status"] == "mismatch"


def test_reproduce_run_writes_verified_manifest_and_external_limits(tmp_path):
    destination = tmp_path / "replay"
    outcome = reproduce_run(destination)
    assert outcome["status"] == "written"
    manifest = json.loads((destination / "replay-manifest.json").read_text())
    assert manifest["limitations"]["raw_judgments_bundled"] is False
    assert len(manifest["limitations"]["missing_exact_references"]) == 2
    paths = {row["path"] for row in manifest["outputs"]}
    assert {"tables/main.csv", "tables/main.json", "tables/owt-generated.csv",
            "tables/lm1b-generated.csv", "tables/metric_agreement.json",
            "figures/agreement-owt.png", "figures/agreement-lm1b.png",
            "figures/mauve-sample-count.png", "figures/jcd-directional.png",
            "figures/entropy-diagnostics.png",
            "data/mauve_sample_count.json", "data/jcd_directional.json",
            "data/entropy_census.json", "data/entropy_controls.json",
            "specs/encoder_rank_sensitivity.vega.json",
            "data/sampling_sweeps.json"} <= paths
    assert (destination / "tables/main.csv").read_text(encoding="utf-8").startswith("id,dataset,kind")
    for relative in ("figures/mauve-sample-count.png", "figures/jcd-directional.png",
                      "figures/entropy-diagnostics.png"):
        assert (destination / relative).stat().st_size > 10_000
    verification = manifest["verification"]
    assert verification["archive_integrity"]["status"] == "verified"
    assert verification["status"] == "verified"
    assert load_resource("archived_sweep")["source"]["protocol_note"]


def test_reproduce_run_refuses_nonempty_existing_output(tmp_path):
    destination = tmp_path / "replay"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        reproduce_run(destination)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_packaged_integrity_has_one_checksum_per_public_resource():
    outcome = verify_packaged_integrity()
    assert outcome["status"] == "verified"
    assert set(outcome["resources"]) == set(PAPER_RESOURCES)


def test_named_main_table_run_exports_only_its_results(tmp_path):
    destination = tmp_path / "main"
    result = subprocess.run(
        [sys.executable, "-m", "dlmbench", "reproduce", "main-table", "--out", str(destination)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((destination / "replay-manifest.json").read_text())
    assert manifest["experiments"] == ["main-table"]
    assert {row["path"] for row in manifest["outputs"]} == {
        "tables/main.csv", "tables/main.json", "tables/owt-generated.csv", "tables/lm1b-generated.csv",
    }
    assert not (destination / "figures").exists()
    rows = json.loads((destination / "tables/main.json").read_text())
    assert len(rows) == 31


def test_unknown_experiment_does_not_create_output(tmp_path):
    destination = tmp_path / "unknown"
    with pytest.raises(ValueError, match="unknown experiment"):
        reproduce_run(destination, experiment="unknown")
    assert not destination.exists()
