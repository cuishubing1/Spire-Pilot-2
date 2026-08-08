from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import zstandard as zstd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def state_hash(observation: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(observation).encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(str(tmp), str(path))


def write_zstd_json(path: Path, value: Any, *, exclusive: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value).encode("utf-8")
    compressed = zstd.ZstdCompressor(level=10).compress(payload)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(compressed)
    return sha256_bytes(compressed)


def read_zstd_json(path: Path) -> Any:
    data = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return json.loads(data.decode("utf-8"))


def iter_jsonl_zst(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as reader:
            text = __import__("io").TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                if line.strip():
                    yield json.loads(line)


def command_version(command: Iterable[str]) -> str:
    result = subprocess.run(list(command), text=True, capture_output=True, timeout=30)
    if result.returncode:
        return f"ERROR({result.returncode}): {result.stderr.strip()}"
    return result.stdout.strip()


def platform_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }

