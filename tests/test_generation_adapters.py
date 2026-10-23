from __future__ import annotations

import subprocess
from importlib import import_module
from pathlib import Path

import pytest

from dlmbench.generators import UnsupportedGenerationError
from dlmbench.generators.adapters.cobit import _stage_checkpoint
from dlmbench.generators.adapters.langflow import LANGFLOW_DECODING
from dlmbench.generators.adapters.provenance import (
    adapter_scratch,
    make_scratch,
    remove_scratch,
    source_identity,
)
from dlmbench.generators.output import prepare_output, write_output
from dlmbench.generators.registry import generator


def test_scratch_is_created_outside_output_and_removed(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    parent_before = {path.name for path in tmp_path.iterdir()}
    scratch = make_scratch(output)
    try:
        assert scratch.is_dir()
        assert scratch.parent == tmp_path
        assert scratch.name not in parent_before
        assert output.is_dir()
        assert not any(output.iterdir())
    finally:
        remove_scratch(scratch)
    assert not scratch.exists()
    assert output.is_dir()
    assert not any(output.iterdir())


def test_scratch_rejects_missing_output_parent(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedGenerationError, match="output parent does not exist"):
        make_scratch(tmp_path / "missing" / "out")


def test_adapter_scratch_success_removes_files_and_failure_retains_them(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    output.mkdir()

    with adapter_scratch(output) as scratch:
        (scratch / "nested").mkdir()
        (scratch / "nested" / "upstream.json").write_text("intermediate", encoding="utf-8")
    assert not scratch.exists()
    assert not scratch.with_name(f"{scratch.name}.retained").exists()
    assert output.is_dir()
    assert not any(output.iterdir())

    with (
        pytest.raises(RuntimeError, match="upstream failed"),
        adapter_scratch(output) as scratch,
    ):
        (scratch / "upstream.log").write_text("failure diagnostics", encoding="utf-8")
        raise RuntimeError("upstream failed")
    assert scratch.is_dir()
    assert (scratch / "upstream.log").read_text(encoding="utf-8") == "failure diagnostics"
    marker = scratch.with_name(f"{scratch.name}.retained")
    assert marker.is_file()
    assert "retained after a failed run" in marker.read_text(encoding="utf-8")


def test_cobit_stages_supplied_checkpoint_atomically(tmp_path: Path) -> None:
    source = tmp_path / "cobit-m.pt"
    source.write_bytes(b"checkpoint")
    destination = tmp_path / "runs" / "experiment" / "checkpoints" / "step=000750000.pt"
    _stage_checkpoint(source, destination)
    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()
    assert destination.read_bytes() == b"checkpoint"

    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"replacement")
    _stage_checkpoint(replacement, destination)
    assert destination.resolve() == replacement.resolve()
    assert destination.read_bytes() == b"replacement"


def test_langflow_manifest_records_historical_decoder_settings() -> None:
    config = generator("lm1b_langflow_1024_nfe")
    manifest = LANGFLOW_DECODING
    assert manifest == {
        "method": "transformers.batch_decode",
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": None,
        "note": "tokenizer default",
    }
    assert config["dataset"] == "lm1b"


@pytest.mark.parametrize(("adapter_name", "generator_id", "required_paths"), [
    ("flm", "lm1b_flm_1024_nfe", ("main.py",)),
    ("langflow", "lm1b_langflow_1024_nfe", (
        "inference.py", "langflow/model.py", "langflow/config.py", "langflow/config.json",
    )),
    ("rdlm", "lm1b_rdlm_1000_nfe", ("main.py", "run_sample.py", "sampling.py", "data.py")),
    ("sdtt", "owt_sdtt_mdlm_teacher_1024_nfe", ("src/sdtt/main.py",)),
    ("elf", "owt_v2_elf_m_64_nfe", (
        "src/eval.py", "src/configs/training_configs/train_owt_ELF-M.yml",
    )),
    ("cobit", "owt_v2_cobit_m_256_nfe", (
        "evaluation/run_eval.py", "configs/owt/eval_cobit_m_750K.py",
    )),
])
def test_adapter_validates_source_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    adapter_name: str, generator_id: str, required_paths: tuple[str, ...],
) -> None:
    adapter = import_module(f"dlmbench.generators.adapters.{adapter_name}")

    source = tmp_path / "upstream"
    source.mkdir()
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    for relative in required_paths:
        source.joinpath(relative).parent.mkdir(parents=True, exist_ok=True)
        source.joinpath(relative).write_text(relative, encoding="utf-8")
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise UnsupportedGenerationError("untrusted upstream source")

    monkeypatch.setattr(adapter, "source_identity", fail)

    with pytest.raises(UnsupportedGenerationError, match="untrusted upstream source"):
        adapter.execute(
            generator(generator_id),
            source_root=source,
            checkpoint=checkpoint,
            tokenizer=object(),
            out_dir=tmp_path / "out",
            sample_count=4,
            seed=17,
        )
    assert calls


def test_source_identity_rejects_wrong_revision_and_modifications(tmp_path: Path) -> None:
    repository = tmp_path / "langflow"
    repository.mkdir()
    _run_git(repository, "init")
    _run_git(repository, "config", "user.email", "test@example.invalid")
    _run_git(repository, "config", "user.name", "test")
    repository.joinpath("inference.py").write_text("entrypoint\n", encoding="utf-8")
    _run_git(repository, "add", ".")
    _run_git(repository, "commit", "-m", "initial")
    _run_git(repository, "rev-parse", "HEAD")
    with pytest.raises(UnsupportedGenerationError, match="source revision mismatch"):
        source_identity("langflow", repository, entrypoint="inference.py")
    repository.joinpath("dirty").write_text("modified", encoding="utf-8")
    with pytest.raises(UnsupportedGenerationError, match="upstream source checkout is modified"):
        source_identity("langflow", repository, entrypoint="inference.py")


def test_prepare_output_rejects_existing_completed_bank_before_generation(
    tmp_path: Path,
) -> None:
    prepare_output(tmp_path / "missing")
    out = write_output(
        [{"id": 0, "text": "row", "token_ids": [0]}],
        tmp_path / "out",
        {"generator_id": "test", "n_samples": 1},
    )
    with pytest.raises(UnsupportedGenerationError, match="already contains a completed bank"):
        try:
            prepare_output(out, require_empty=True)
        except UnsupportedGenerationError as error:
            if "is not empty" in str(error):
                pytest.fail("completed bank should be recognized before generic emptiness rejection")
            raise


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()
