from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import zstandard as zstd

from . import SCHEMA_VERSION
from .util import canonical_json, sha256_file, utc_now


class RawRunWriter:
    """Append-only JSONL writer; only sealed files are considered dataset inputs."""

    def __init__(self, target: Path, run_id: str):
        self.target = target
        self.partial = target.with_suffix(target.suffix + ".partial")
        self.run_id = run_id
        self.sequence = 0
        self._raw = None
        self._compressor = None
        self._text = None

    def __enter__(self) -> "RawRunWriter":
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists():
            raise FileExistsError(f"Sealed raw run already exists: {self.target}")
        if self.partial.exists():
            raise FileExistsError(f"Partial run exists; inspect it before retrying: {self.partial}")
        self._raw = self.partial.open("xb")
        self._compressor = zstd.ZstdCompressor(level=8).stream_writer(self._raw, closefd=False)
        self._text = io.TextIOWrapper(self._compressor, encoding="utf-8", newline="\n")
        return self

    def write(self, record_type: str, **payload: Any) -> dict[str, Any]:
        if self._text is None:
            raise RuntimeError("Writer is not open")
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "run_id": self.run_id,
            "sequence_no": self.sequence,
            "timestamp_utc": utc_now(),
            **payload,
        }
        self._text.write(canonical_json(record) + "\n")
        self.sequence += 1
        return record

    def seal(self) -> tuple[Path, str]:
        if self._text is None:
            raise RuntimeError("Writer is not open")
        self._text.flush()
        self._text.detach().close()
        self._raw.close()
        self._text = self._compressor = self._raw = None
        os.replace(str(self.partial), str(self.target))
        return self.target, sha256_file(self.target)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._text is not None:
            try:
                self._text.flush()
                self._text.detach().close()
            finally:
                if self._raw is not None:
                    self._raw.close()
            self._text = self._compressor = self._raw = None

