"""Fetch and verify samples from GitHub, local directories, or tar archives."""
from __future__ import annotations

import hashlib
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .banks import bank_relative_path, catalog_entry, data_root, load_bank, load_catalog
from .provenance import atomic_json, sha256_file


def select_banks(ids: list[str] | None = None, *, suite: str | None = None,
                 dataset: str | None = None) -> list[dict[str, Any]]:
    catalog = load_catalog()
    if suite is not None and suite not in catalog["suites"]:
        raise ValueError(f"unknown suite {suite!r}; available: {', '.join(catalog['suites'])}")
    rows = [catalog_entry(value) for value in ids] if ids else catalog["banks"]
    members = set(catalog["suites"][suite]) if suite else None
    unique = {}
    for row in rows:
        if (members is None or row["id"] in members) and (
            dataset is None or row["dataset"] == dataset
        ):
            unique[row["id"]] = row
    return list(unique.values())


def _destination(entry: dict, root: str | Path | None) -> Path:
    base = data_root(root).resolve()
    destination = base / bank_relative_path(entry)
    if not destination.resolve().is_relative_to(base):
        raise ValueError("sample destination escapes the artifact directory")
    return destination


def _expected(entry: dict) -> tuple[str, int]:
    checksum, size = entry.get("sha256"), entry.get("bytes")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError(f"bank {entry['id']} has no verified release checksum")
    if type(size) is not int or size <= 0:
        raise ValueError(f"bank {entry['id']} has no verified byte count")
    return checksum, size


def _validate(path: Path, entry: dict) -> dict:
    checksum, size = _expected(entry)
    if path.stat().st_size != size:
        raise ValueError(f"byte count mismatch for {entry['id']}")
    bank = load_bank(path, expected_sha256=checksum)
    if len(bank) != entry["n_samples"]:
        raise ValueError(f"sample count mismatch for {entry['id']}")
    return {"id": entry["id"], "status": "verified", "n_samples": len(bank),
            "bytes": size, "sha256": checksum, "text_sha256": bank.text_sha256}


def verify_bank(bank_id: str, *, root: str | Path | None = None) -> dict:
    entry = catalog_entry(bank_id)
    _expected(entry)
    return _validate(_destination(entry, root), entry)


def _install(stream: BinaryIO, entry: dict, destination: Path) -> None:
    checksum, size = _expected(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=destination.suffix,
                                dir=destination.parent)
    temporary = Path(name)
    try:
        digest, received = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as output:
            while chunk := stream.read(min(1024 * 1024, size - received + 1)):
                received += len(chunk)
                if received > size:
                    raise ValueError(f"download exceeds expected byte count for {entry['id']}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if received != size or digest.hexdigest() != checksum:
            raise ValueError(f"download checksum/size mismatch for {entry['id']}")
        _validate(temporary, entry)
        try:
            # Publish atomically without replacing existing data, even in a race.
            os.link(temporary, destination)
        except FileExistsError:
            _validate(destination, entry)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_bank(bank_id: str, *, root: str | Path | None = None,
               source: str | Path | None = None, base_url: str | None = None) -> dict:
    """Acquire exact catalog bytes. Never overwrite a corrupt or different bank.

    ``source`` is a local artifact directory or a tar archive containing the
    catalog paths at its root. ``base_url`` is an explicit HTTP(S) mirror root.
    Without either, the catalog's published per-bank URLs are tried in order.
    """
    if source is not None and base_url is not None:
        raise ValueError("use either a local source or a URL mirror")
    entry = catalog_entry(bank_id)
    _expected(entry)
    destination = _destination(entry, root)
    if destination.exists():
        return _validate(destination, entry)
    relative = bank_relative_path(entry)
    if source is not None:
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            with (source_path / relative).open("rb") as handle:
                _install(handle, entry, destination)
        elif source_path.is_file():
            with tarfile.open(source_path, mode="r|*") as archive:
                found = False
                for member in archive:
                    name = member.name.removeprefix("./")
                    if name not in {relative.as_posix(), relative.as_posix().removeprefix("samples/")}:
                        continue
                    if not member.isfile() or member.size != entry["bytes"]:
                        raise ValueError(f"invalid archive member for {bank_id}")
                    with archive.extractfile(member) as handle:
                        _install(handle, entry, destination)
                    found = True
                    break
                if not found:
                    raise FileNotFoundError(f"archive does not contain {relative}")
        else:
            raise FileNotFoundError(source_path)
    else:
        urls = ([base_url.rstrip("/") + "/" + quote(relative.as_posix(), safe="/")]
                if base_url else entry.get("download_urls", []))
        if not urls:
            raise ValueError(f"no public download is configured for {bank_id}; "
                             "use --from with a local release bundle/directory or --base-url")
        failures = []
        for url in urls:
            if urlsplit(url).scheme not in {"https", "http"}:
                raise ValueError("sample mirrors must use HTTP or HTTPS")
            try:
                request = Request(url, headers={"User-Agent": "dlmbench/0.1"})
                with urlopen(request, timeout=60) as response:
                    _install(response, entry, destination)
                break
            except (OSError, ValueError) as error:
                failures.append(str(error))
        else:
            raise OSError(f"could not fetch {bank_id}: {'; '.join(failures)}")
    # A compact release manifest supplies provenance when loading a file directly.
    manifest = destination.parent / "manifest.json"
    if not manifest.exists():
        atomic_json(manifest, {**entry, "samples_sha256": sha256_file(destination)})
    return _validate(destination, entry)


def fetch_banks(bank_ids: list[str], *, root: str | Path | None = None,
                source: str | Path | None = None, base_url: str | None = None) -> list[dict]:
    """Fetch a selection, reading a compressed release bundle only once."""
    if not bank_ids:
        return []
    if source is not None and base_url is not None:
        raise ValueError("use either a local source or a URL mirror")
    entries = select_banks(bank_ids)
    outcomes = {}
    archive_path = Path(source).expanduser() if source is not None else None
    if archive_path is None or not archive_path.is_file():
        for entry in entries:
            try:
                outcomes[entry["id"]] = fetch_bank(entry["id"], root=root, source=source,
                                                    base_url=base_url)
            except (OSError, ValueError, tarfile.TarError) as error:
                outcomes[entry["id"]] = {"id": entry["id"], "status": "error", "error": str(error)}
    else:
        pending = {}
        for entry in entries:
            try:
                _expected(entry)
                destination = _destination(entry, root)
                if destination.exists():
                    outcomes[entry["id"]] = _validate(destination, entry)
                else:
                    pending[bank_relative_path(entry).as_posix()] = (entry, destination)
            except (OSError, ValueError) as error:
                outcomes[entry["id"]] = {"id": entry["id"], "status": "error", "error": str(error)}
        if pending:
            try:
                with tarfile.open(archive_path, mode="r|*") as archive:
                    for member in archive:
                        name = member.name.removeprefix("./")
                        # The v1 bundle starts at <dataset>/<bank>; catalog
                        # paths and newer mirrors include the samples/ prefix.
                        key = name if name in pending else "samples/" + name
                        item = pending.pop(key, None)
                        if item is None:
                            continue
                        entry, destination = item
                        try:
                            if not member.isfile() or member.size != entry["bytes"]:
                                raise ValueError(f"invalid archive member for {entry['id']}")
                            with archive.extractfile(member) as handle:
                                _install(handle, entry, destination)
                            manifest = destination.parent / "manifest.json"
                            if not manifest.exists():
                                atomic_json(manifest, {**entry, "samples_sha256": entry["sha256"]})
                            outcomes[entry["id"]] = _validate(destination, entry)
                        except (OSError, ValueError, tarfile.TarError, EOFError) as error:
                            outcomes[entry["id"]] = {"id": entry["id"], "status": "error", "error": str(error)}
                        if not pending:
                            break
            except (OSError, tarfile.TarError, EOFError) as error:
                for entry, _ in pending.values():
                    outcomes[entry["id"]] = {"id": entry["id"], "status": "error", "error": str(error)}
        for entry, _ in pending.values():
            outcomes.setdefault(entry["id"], {"id": entry["id"], "status": "error",
                                               "error": "bank is absent from the release bundle"})
    return [outcomes[entry["id"]] for entry in entries]
