"""Checks for accidentally committed credentials and local files."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from dlmbench.banks import load_catalog

PRIVATE_DIRECTORIES = {"artifacts", "runs", "lab", ".codex", "assets", "third_party",
                       "tokenizers", ".ssh", ".aws", ".azure", ".kube"}
PRIVATE_NAMES = {".env", ".netrc", ".pypirc", "AGENTS.md", "CLAUDE.md",
                 "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}
PRIVATE_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".pkl", ".pickle",
                    ".sqlite", ".sqlite3", ".db", ".pem", ".key", ".p12", ".pfx"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}"),
    "credential-bearing URL": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}
PRIVATE_PATH = re.compile(
    r"/(?:mnt/(?:storage\d+/home|localdata/scratch)|scratch|Users|home)/[\w.-]+"
)


def inspect_file(relative: Path, path: Path, *, sample: bool = False) -> list[str]:
    """Check one release file without following symlinks or printing its content."""
    problems = []
    if any(part in PRIVATE_DIRECTORIES or part.startswith(".venv") for part in relative.parts):
        problems.append(f"private artifact directory: {relative}")
    if relative.name in PRIVATE_NAMES or relative.name.startswith(".env."):
        problems.append(f"local configuration or credential file: {relative}")
    if relative.suffix.lower() in PRIVATE_SUFFIXES:
        problems.append(f"private/binary artifact tracked: {relative}")
    if path.is_symlink():
        return [*problems, f"symlink is not allowed in the release payload: {relative}"]
    if not path.is_file():
        return [*problems, f"tracked file is absent: {relative}"]
    if path.stat().st_size > (100 if sample else 5) * 1024 * 1024:
        return [*problems, f"file exceeds size limit: {relative}"]
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [*problems, f"unexpected binary file in the source release: {relative}"]
    for rule, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(content):
            line = content.count("\n", 0, match.start()) + 1
            problems.append(f"{rule} in {relative}:{line}")
    if sample:
        # Generated text can contain paths. Check metadata for local paths,
        # while still scanning the complete file for credentials above.
        try:
            rows = [json.loads(line) for line in content.split("\n") if line.strip()]
            content = "\n".join(json.dumps({k: v for k, v in row.items() if k != "text"})
                                for row in rows)
        except (ValueError, AttributeError):
            return [*problems, f"invalid sample JSONL: {relative}"]
    for match in PRIVATE_PATH.finditer(content):
        line = content.count("\n", 0, match.start()) + 1
        problems.append(f"private absolute path in {relative}:{line}")
    return problems


def test_tracked_files_do_not_expose_credentials_or_local_files():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("requires a Git checkout")
    samples = {row["path"] for row in load_catalog()["banks"] if row["path"]}
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                            capture_output=True, check=True)
    problems = []
    for name in result.stdout.decode().split("\0"):
        if name:
            problems.extend(inspect_file(Path(name), root / name, sample=name in samples))
    assert not problems, "\n".join(problems)


def test_sample_text_paths_are_preserved_but_metadata_paths_are_rejected(tmp_path):
    path = tmp_path / "samples.jsonl"
    private_path = "/home/" + "fixture-user/experiment"
    row = {"id": 0, "text": f"Example path: {private_path}"}
    path.write_text(json.dumps(row) + "\n")
    assert inspect_file(Path(path.name), path, sample=True) == []
    row["source"] = private_path
    path.write_text(json.dumps(row) + "\n")
    assert inspect_file(Path(path.name), path, sample=True) == [
        "private absolute path in samples.jsonl:1"
    ]


def test_sample_text_is_still_checked_for_credentials(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"text": "hf_" + "C" * 34}) + "\n")
    assert inspect_file(Path(path.name), path, sample=True) == [
        "Hugging Face token in samples.jsonl:1"
    ]



@pytest.mark.parametrize("name", [
    ".env.production", "nested/.venv-model/config", "nested/.aws/credentials",
    "key.pem", "id_ed25519", "state.pkl", "participants.sqlite3",
])
def test_private_files_are_rejected(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private fixture", encoding="utf-8")
    assert inspect_file(Path(name), path)


@pytest.mark.parametrize("secret", [
    "ghp_" + "A" * 36,
    "github_pat_" + "B" * 82,
    "hf_" + "C" * 34,
    "sk-proj-" + "D" * 48,
    "AKIA" + "E" * 16,
    "xoxb-" + "F" * 32,
    "-----BEGIN " + "RSA PRIVATE KEY-----",
    "https://" + "user:credential@example.invalid/file",
])
def test_credentials_are_rejected_without_disclosure(tmp_path, secret):
    path = tmp_path / "config.txt"
    path.write_text(f"first line\n{secret}\n", encoding="utf-8")
    problems = inspect_file(Path(path.name), path)
    assert problems
    assert all("config.txt:2" in problem for problem in problems)
    assert all(secret not in problem for problem in problems)


def test_symlink_is_rejected_without_reading_its_target(tmp_path):
    path = tmp_path / "outside-link"
    path.symlink_to(tmp_path / "missing-target")
    assert any("symlink" in problem for problem in
               inspect_file(Path(path.name), path))


def test_public_code_and_hashes_are_allowed(tmp_path):
    path = tmp_path / "public.py"
    path.write_text('model = "gpt2"\nrevision = "' + "a" * 40 + '"\n', encoding="utf-8")
    assert inspect_file(Path(path.name), path) == []


def test_private_paths_are_reported_without_echoing_them(tmp_path):
    private = "/home/" + "fixture-user/experiment"
    path = tmp_path / "source.json"
    path.write_text(private, encoding="utf-8")
    problems = inspect_file(Path(path.name), path)
    assert problems == ["private absolute path in source.json:1"]
    assert private not in problems[0]
