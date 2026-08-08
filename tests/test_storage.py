from sts2_dataset.storage import RawRunWriter
from sts2_dataset.util import iter_jsonl_zst


def test_raw_writer_seals_atomically(tmp_path):
    path = tmp_path / "run.jsonl.zst"
    with RawRunWriter(path, "run") as writer:
        writer.write("run_start", seed="s")
        writer.write("run_end", terminal=True)
        sealed, digest = writer.seal()
    assert sealed == path
    assert len(digest) == 64
    records = list(iter_jsonl_zst(path))
    assert [r["sequence_no"] for r in records] == [0, 1]

