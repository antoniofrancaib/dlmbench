import gzip
import hashlib
import json

import pytest

from dlmbench.banks import Bank, load_bank


def test_bank_roundtrip_preserves_observations(tmp_path):
    texts = ["  café\r\n\t", "repeat", "repeat", ""]
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps({"id": i, "text": text})
                              for i, text in enumerate(texts)), encoding="utf-8")
    bank = load_bank(path)
    assert bank.texts == tuple(texts)
    assert bank.ids == (0, 1, 2, 3)
    assert bank.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(bank) == 4


def test_text_identity_is_ordered_and_unambiguous():
    def identity(texts):
        return Bank(tuple(range(len(texts))), tuple(texts)).text_sha256
    assert identity(["ab", "c"]) != identity(["a", "bc"])
    assert identity(["one", "two"]) != identity(["two", "one"])
    assert identity(["one"]) != identity(["one", "one"])


def test_gzip_has_same_text_identity_but_distinct_file_identity(tmp_path):
    payload = b'{"text":"hello"}\n'
    plain, compressed = tmp_path / "samples.jsonl", tmp_path / "samples.jsonl.gz"
    plain.write_bytes(payload)
    compressed.write_bytes(gzip.compress(payload, mtime=0))
    a, b = load_bank(plain), load_bank(compressed)
    assert a.texts == b.texts
    assert a.text_sha256 == b.text_sha256
    assert a.sha256 != b.sha256


@pytest.mark.parametrize("payload", [
    '{"text":null}', '{"text":"ok"}\nnot-json',
    '{"text":"a","id":0}\n{"text":"a","id":0}',
    '{"text":"a","token_ids":[true]}', '{"text":"a","token_ids":[-1]}', "",
])
def test_invalid_observations_are_not_silently_filtered(tmp_path, payload):
    path = tmp_path / "samples.jsonl"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_bank(path)


def test_manifest_count_and_checksum_are_enforced(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"text":"a"}\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"n_samples": 2}))
    with pytest.raises(ValueError, match="count mismatch"):
        load_bank(path)
    manifest.write_text(json.dumps({"samples_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bank(path)


def test_token_rows_and_original_ids_survive(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"id":"example","text":"a","token_ids":[12,9]}\n'
                    '{"id":2,"text":"b"}\n')
    bank = load_bank(path)
    assert bank.ids == ("example", 2)
    assert bank.token_ids == ((12, 9), None)


def test_catalog_provenance_preserves_scoring_length_and_tokenizer(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"text":"a","token_ids":[12,9]}\n')
    fields = {"sequence_length": 128, "tokenizer_id": "bert-base-uncased",
              "tokenizer_revision": "recorded-snapshot", "seed": 1}
    manifest = {"provenance": {"scientific_metadata": fields}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    bank = load_bank(path)
    assert all(bank.metadata[key] == value for key, value in fields.items())
    manifest["sequence_length"] = 1024
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="conflicting recorded bank metadata"):
        load_bank(path)
