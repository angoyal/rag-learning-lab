from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from src.ingest.metrics import IngestMetrics


def _sample_record(**overrides: object) -> dict:
    base = {
        "source": "test.pdf",
        "file_size_bytes": 1024,
        "read_time_s": 0.1,
        "num_sentences": 10,
        "chunk_time_s": 0.2,
        "num_chunks": 5,
        "embed_time_s": 0.3,
        "store_time_s": 0.05,
        "total_time_s": 0.65,
        "error": None,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestIngestMetrics:
    def test_creates_log_dir_and_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "sub" / "logs"
        m = IngestMetrics(log_dir=str(log_dir))
        assert log_dir.exists()
        assert m.log_path.exists()
        assert m.log_path.suffix == ".jsonl"
        assert m.log_path.name.startswith("ingest_")
        m.close()

    def test_record_and_flush(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        rec = _sample_record()
        m.record(rec)
        m.flush()
        lines = m.log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == rec
        m.close()

    def test_flush_multiple_records(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        for i in range(3):
            m.record(_sample_record(source=f"doc{i}.pdf"))
        m.flush()
        lines = m.log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["source"] for line in lines] == [
            "doc0.pdf",
            "doc1.pdf",
            "doc2.pdf",
        ]
        m.close()

    def test_flush_empty_buffer(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        m.flush()
        assert m.log_path.read_text() == ""
        m.close()

    def test_close_flushes_remaining(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        m.record(_sample_record())
        m.close()
        lines = m.log_path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_close_prints_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        m.close()
        assert str(m.log_path) in capsys.readouterr().out

    def test_context_manager(self, tmp_path: Path) -> None:
        with IngestMetrics(log_dir=str(tmp_path)) as m:
            m.record(_sample_record())
            log_path = m.log_path
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_context_manager_on_exception(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="boom"):
            with IngestMetrics(log_dir=str(tmp_path)) as m:
                m.record(_sample_record())
                log_path = m.log_path
                raise ValueError("boom")
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_record_with_error(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        rec = _sample_record(error="parse failed")
        m.record(rec)
        m.flush()
        written = json.loads(m.log_path.read_text().strip())
        assert written["error"] == "parse failed"
        m.close()

    def test_thread_safety(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        num_threads = 8
        records_per_thread = 50

        def writer(tid: int) -> None:
            for i in range(records_per_thread):
                m.record(_sample_record(source=f"t{tid}_doc{i}.pdf"))

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        m.close()

        lines = m.log_path.read_text().strip().splitlines()
        assert len(lines) == num_threads * records_per_thread

    def test_log_path_property(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        assert isinstance(m.log_path, Path)
        assert m.log_path.parent == tmp_path
        m.close()

    def test_incremental_flush(self, tmp_path: Path) -> None:
        m = IngestMetrics(log_dir=str(tmp_path))
        m.record(_sample_record(source="a.pdf"))
        m.flush()
        m.record(_sample_record(source="b.pdf"))
        m.flush()
        lines = m.log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["source"] == "a.pdf"
        assert json.loads(lines[1])["source"] == "b.pdf"
        m.close()
