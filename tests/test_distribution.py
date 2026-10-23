"""Public entry points must work without optional frameworks or a source checkout."""
import json
import subprocess
import sys


def test_import_does_not_load_optional_frameworks():
    result = subprocess.run(
        [sys.executable, "-c", ("import sys, dlmbench; "
         "assert not {'torch','transformers','scipy','matplotlib'} & set(sys.modules)")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_help_and_catalog_work_offline():
    for arguments in (["--help"], ["samples", "list", "--json"]):
        result = subprocess.run([sys.executable, "-m", "dlmbench", *arguments],
                                check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        if arguments == ["--help"]:
            for command in ("samples", "evaluate", "generate", "reproduce"):
                assert command in result.stdout
        if arguments[-1] == "--json":
            rows = json.loads(result.stdout)
            assert len(rows) > 0
            assert len({row["id"] for row in rows}) == len(rows)


def test_cli_construction_does_not_import_neural_frameworks():
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; from dlmbench.cli import build_parser; "
         "build_parser(); assert not {'torch','transformers','scipy','matplotlib'} & set(sys.modules)")],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_discovery_works_through_the_four_primary_commands():
    for arguments, field in [
        (["generate", "list", "--json"], "generators"),
        (["generate", "show", "lm1b_topk_iid_k32", "--json"], "id"),
        (["evaluate", "--list", "--json"], "metrics"),
    ]:
        result = subprocess.run([sys.executable, "-m", "dlmbench", *arguments],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)[field]
    result = subprocess.run([sys.executable, "-m", "dlmbench", "reproduce", "--list"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "main-table" in result.stdout
