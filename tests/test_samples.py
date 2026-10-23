"""The checked-in samples must remain identical to the paper's catalog."""
from pathlib import Path

import pytest

from dlmbench.artifacts import fetch_bank, verify_bank
from dlmbench.banks import load_bank, load_catalog


@pytest.fixture(scope="module")
def sample_root():
    root = Path(__file__).resolve().parents[1]
    if not (root / "samples").is_dir():
        pytest.skip("samples are distributed in the Git repository")
    return root


def test_all_sample_files_match_the_catalog(sample_root):
    rows = [row for row in load_catalog()["banks"] if row["path"]]
    assert len(rows) == 80
    actual = {path.relative_to(sample_root).as_posix()
              for path in (sample_root / "samples").rglob("*") if path.is_file()}
    assert actual == {row["path"] for row in rows}
    total = 0
    for row in rows:
        path = sample_root / row["path"]
        assert not path.is_symlink()
        assert path.stat().st_size == row["bytes"], row["id"]
        bank = load_bank(row["id"], root=sample_root)
        assert bank.sha256 == row["sha256"], row["id"]
        assert len(bank) == row["n_samples"] == 1024, row["id"]
        total += len(bank)
    assert total == 81920


def test_samples_work_from_a_clone_without_fetching(sample_root, monkeypatch):
    monkeypatch.chdir(sample_root)
    monkeypatch.delenv("DLMBENCH_DATA_DIR", raising=False)
    bank = load_bank("lm1b_topk_iid_k32")
    assert bank.path.resolve() == sample_root / "samples/lm1b/lm1b_topk_iid_k32/samples.jsonl"
    assert verify_bank(bank.id)["status"] == "verified"


def test_sample_directory_import_preserves_bytes(sample_root, tmp_path):
    bank_id = "lm1b_topk_iid_k32"
    result = fetch_bank(bank_id, root=tmp_path, source=sample_root)
    original = load_bank(bank_id, root=sample_root)
    imported = load_bank(bank_id, root=tmp_path)
    assert result["status"] == "verified"
    assert imported.identity() == original.identity()
    assert imported.path.read_bytes() == original.path.read_bytes()
