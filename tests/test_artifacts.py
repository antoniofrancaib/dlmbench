import hashlib
import io
import tarfile

import pytest

from dlmbench import artifacts, banks


@pytest.fixture
def catalog(monkeypatch):
    payload = b'{"id":0,"text":"same"}\n{"id":1,"text":"same"}\n'
    row = {"id": "example", "aliases": ["owt/example"], "dataset": "owt",
           "kind": "generated", "n_samples": 2, "bytes": len(payload),
           "sha256": hashlib.sha256(payload).hexdigest(),
           "path": "samples/owt/example/samples.jsonl", "download_urls": []}
    data = {"schema_version": 1, "banks": [row], "suites": {"paper-main": ["example"]}}
    monkeypatch.setattr(banks, "load_catalog", lambda: data)
    monkeypatch.setattr(artifacts, "load_catalog", lambda: data)
    return row, payload


def source_directory(tmp_path, row, payload):
    source = tmp_path / "source"
    path = source / row["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return source


def test_import_and_reload_by_alias_are_lossless(tmp_path, catalog):
    row, payload = catalog
    source = source_directory(tmp_path, row, payload)
    root = tmp_path / "downloaded"
    result = artifacts.fetch_bank("owt/example", root=root, source=source)
    assert result["status"] == "verified"
    bank = banks.load_bank("example", root=root)
    assert bank.texts == ("same", "same")
    assert bank.sha256 == row["sha256"]
    # Repeating acquisition is safe and can work without the original source.
    assert artifacts.fetch_bank("example", root=root, source="absent") == result


def test_sample_root_defaults_to_cwd_and_respects_overrides(tmp_path, catalog, monkeypatch):
    row, payload = catalog
    source = source_directory(tmp_path, row, payload)
    monkeypatch.chdir(source)
    monkeypatch.delenv("DLMBENCH_DATA_DIR", raising=False)
    assert banks.load_bank("example").texts == ("same", "same")
    assert artifacts.verify_bank("example")["status"] == "verified"
    monkeypatch.setenv("DLMBENCH_DATA_DIR", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        banks.load_bank("example")
    assert banks.load_bank("example", root=source).texts == ("same", "same")


def test_corrupt_download_never_publishes_partial_bank(tmp_path, catalog):
    row, payload = catalog
    source = source_directory(tmp_path, row, payload[:-1])
    root = tmp_path / "downloaded"
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.fetch_bank("example", root=root, source=source)
    assert not (root / row["path"]).exists()
    assert not list(root.rglob(".download-*"))


def test_existing_corrupt_data_is_preserved_for_investigation(tmp_path, catalog):
    row, payload = catalog
    source = source_directory(tmp_path, row, payload)
    root = tmp_path / "downloaded"
    destination = root / row["path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"user data")
    with pytest.raises(ValueError):
        artifacts.fetch_bank("example", root=root, source=source)
    assert destination.read_bytes() == b"user data"


@pytest.mark.parametrize("prefix", ["samples/", ""])
def test_archive_import_does_not_extract_untrusted_paths(tmp_path, catalog, prefix):
    row, payload = catalog
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        name = prefix + row["path"].removeprefix("samples/")
        for name, content in [("../../escape", b"bad"), (name, payload)]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            handle.addfile(member, io.BytesIO(content))
    result = artifacts.fetch_banks(["example"], root=tmp_path / "downloaded", source=archive)
    assert result[0]["status"] == "verified"
    assert not (tmp_path / "escape").exists()
    single = artifacts.fetch_bank("example", root=tmp_path / "single", source=archive)
    assert single["status"] == "verified"


def test_symlink_cannot_escape_destination(tmp_path, catalog):
    row, payload = catalog
    source = source_directory(tmp_path, row, payload)
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "samples").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        artifacts.fetch_bank("example", root=root, source=source)
    assert not list(outside.iterdir())


def test_missing_public_host_is_an_actionable_error(tmp_path, catalog):
    with pytest.raises(ValueError, match="no public download"):
        artifacts.fetch_bank("example", root=tmp_path)


def test_missing_archive_member_reports_failure(tmp_path, catalog):
    archive = tmp_path / "empty.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    result = artifacts.fetch_banks(["example"], root=tmp_path / "root", source=archive)
    assert result[0]["status"] == "error"


def test_catalog_counts_checked_even_with_correct_byte_hash(tmp_path, catalog):
    row, payload = catalog
    row["n_samples"] = 3
    source = source_directory(tmp_path, row, payload)
    with pytest.raises(ValueError, match="count mismatch"):
        artifacts.fetch_bank("example", root=tmp_path / "root", source=source)


def test_catalog_path_traversal_rejected(catalog):
    row, _ = catalog
    row["path"] = "../outside/samples.jsonl"
    with pytest.raises(ValueError, match="safe relative"):
        artifacts.fetch_bank("example")
