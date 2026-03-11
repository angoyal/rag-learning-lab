from __future__ import annotations

import json
import queue
from datetime import datetime
from pathlib import Path


class IngestMetrics:
    """Thread-safe collector for per-document ingestion timing metrics.

    Buffers metric dicts in a queue and writes them to a JSONL log file.
    """

    def __init__(self, log_dir: str = "debug/logs") -> None:
        """Initialize metrics collector and open the JSONL log file.

        Args:
            log_dir: Directory for log files. Created if it does not exist.
        """
        dir_path = Path(log_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = dir_path / f"ingest_{timestamp}.jsonl"
        self._file = open(self._log_path, "a")
        self._buffer: queue.Queue[dict] = queue.Queue()

    @property
    def log_path(self) -> Path:
        """Return the Path to the JSONL log file."""
        return self._log_path

    def record(self, data: dict) -> None:
        """Push a metrics dict to the internal buffer (thread-safe).

        Args:
            data: Dict with keys like source, file_size_bytes, read_time_s,
                  num_sentences, chunk_time_s, num_chunks, embed_time_s,
                  store_time_s, total_time_s, and error.
        """
        self._buffer.put(data)

    def flush(self) -> None:
        """Write all buffered records to the JSONL file, one JSON object per line."""
        while True:
            try:
                data = self._buffer.get_nowait()
            except queue.Empty:
                break
            self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush remaining records, close the file, and print the log path."""
        self.flush()
        self._file.close()
        print(f"Ingest metrics log: {self._log_path}")

    def __enter__(self) -> IngestMetrics:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager, ensuring close is called."""
        self.close()
