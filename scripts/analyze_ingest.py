"""CLI script to analyze JSONL ingestion metrics logs and produce statistics and charts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


STAGES = ["read_time_s", "chunk_time_s", "embed_time_s", "store_time_s", "total_time_s"]
LOG_DIR = Path("debug/logs")
CHART_DIR = Path("debug/charts")


def load_records(path: Path) -> list[dict]:
    """Load all records from a JSONL file.

    Args:
        path: Path to the JSONL log file.

    Returns:
        List of dicts, one per line in the file.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def successful_records(records: list[dict]) -> list[dict]:
    """Filter records to only those without errors.

    Args:
        records: All loaded records.

    Returns:
        Records where the error field is None or absent.
    """
    return [r for r in records if r.get("error") is None]


def compute_stage_stats(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute mean, median, p50, p95, and stddev for each timing stage.

    Args:
        records: Successful (non-error) records.

    Returns:
        Dict mapping stage name to a dict of stat_name -> value.
    """
    stats = {}
    for stage in STAGES:
        values = np.array([r[stage] for r in records if stage in r])
        if len(values) == 0:
            continue
        stats[stage] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "stddev": float(np.std(values)),
        }
    return stats


def print_stats_table(stats: dict[str, dict[str, float]]) -> None:
    """Print a formatted table of stage timing statistics.

    Args:
        stats: Output of compute_stage_stats.
    """
    header = f"{'stage':<16} {'mean':>8} {'median':>8} {'p50':>8} {'p95':>8} {'stddev':>8}"
    print(header)
    print("-" * len(header))
    for stage, s in stats.items():
        print(
            f"{stage:<16} {s['mean']:>8.3f} {s['median']:>8.3f} "
            f"{s['p50']:>8.3f} {s['p95']:>8.3f} {s['stddev']:>8.3f}"
        )
    print()


def print_top_slowest(records: list[dict], n: int = 10) -> None:
    """Print the top N slowest files by total_time_s.

    Args:
        records: Successful records.
        n: Number of files to show.
    """
    sorted_recs = sorted(records, key=lambda r: r.get("total_time_s", 0), reverse=True)
    print(f"Top {n} slowest files:")
    print(f"  {'file':<50} {'total_time_s':>12}")
    print(f"  {'-'*50} {'-'*12}")
    for r in sorted_recs[:n]:
        source = Path(r.get("source", "unknown")).name
        print(f"  {source:<50} {r.get('total_time_s', 0):>12.3f}")
    print()


def print_summary_counts(all_records: list[dict], ok_records: list[dict]) -> None:
    """Print count of total and failed documents.

    Args:
        all_records: All records including errors.
        ok_records: Records without errors.
    """
    failed = len(all_records) - len(ok_records)
    print(f"Total documents processed: {len(all_records)}")
    print(f"Failed documents: {failed}")
    print()


def save_stage_breakdown_chart(stats: dict[str, dict[str, float]], out_dir: Path) -> Path:
    """Save a stacked bar chart showing mean time per stage.

    Args:
        stats: Stage statistics from compute_stage_stats.
        out_dir: Directory to save the chart.

    Returns:
        Path to the saved chart file.
    """
    stages = [s for s in STAGES if s != "total_time_s" and s in stats]
    means = [stats[s]["mean"] for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = 0.0
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for i, (stage, mean) in enumerate(zip(stages, means)):
        ax.bar("Mean Time", mean, bottom=bottom, label=stage, color=colors[i % len(colors)])
        bottom += mean

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Mean Time per Ingestion Stage")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = out_dir / "stage_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_total_time_histogram(records: list[dict], out_dir: Path) -> Path:
    """Save a histogram of total_time_s across all documents.

    Args:
        records: Successful records.
        out_dir: Directory to save the chart.

    Returns:
        Path to the saved chart file.
    """
    values = [r["total_time_s"] for r in records if "total_time_s" in r]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(30, max(5, len(values) // 3)), color="#2196F3", edgecolor="white")
    ax.set_xlabel("Total Time (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Total Ingestion Time per Document")
    fig.tight_layout()

    path = out_dir / "total_time_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_time_vs_chunks_chart(records: list[dict], out_dir: Path) -> Path:
    """Save a scatter plot of num_chunks vs total_time_s.

    Args:
        records: Successful records.
        out_dir: Directory to save the chart.

    Returns:
        Path to the saved chart file.
    """
    data = [(r["num_chunks"], r["total_time_s"]) for r in records
            if "num_chunks" in r and "total_time_s" in r]
    if not data:
        return out_dir / "time_vs_chunks.png"

    chunks, times = zip(*data)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(chunks, times, alpha=0.6, color="#4CAF50", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Number of Chunks")
    ax.set_ylabel("Total Time (seconds)")
    ax.set_title("Total Ingestion Time vs Number of Chunks")
    fig.tight_layout()

    path = out_dir / "time_vs_chunks.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_time_vs_filesize_chart(records: list[dict], out_dir: Path) -> Path:
    """Save a scatter plot of file_size_bytes vs total_time_s.

    Args:
        records: Successful records.
        out_dir: Directory to save the chart.

    Returns:
        Path to the saved chart file.
    """
    data = [(r["file_size_bytes"], r["total_time_s"]) for r in records
            if "file_size_bytes" in r and "total_time_s" in r]
    if not data:
        return out_dir / "time_vs_filesize.png"

    sizes, times = zip(*data)
    sizes_kb = [s / 1024 for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sizes_kb, times, alpha=0.6, color="#FF9800", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Total Time (seconds)")
    ax.set_title("Total Ingestion Time vs File Size")
    fig.tight_layout()

    path = out_dir / "time_vs_filesize.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_summary(records: list[dict]) -> None:
    """Run summary mode: print stats and generate charts.

    Args:
        records: All loaded records from the JSONL file.
    """
    ok = successful_records(records)
    if not ok:
        print("No successful records found.")
        return

    stats = compute_stage_stats(ok)
    print_stats_table(stats)
    print_top_slowest(ok)
    print_summary_counts(records, ok)

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths = [
        save_stage_breakdown_chart(stats, CHART_DIR),
        save_total_time_histogram(ok, CHART_DIR),
        save_time_vs_chunks_chart(ok, CHART_DIR),
        save_time_vs_filesize_chart(ok, CHART_DIR),
    ]
    print("Charts saved:")
    for p in chart_paths:
        print(f"  {p}")


def run_xray(records: list[dict], filename: str) -> None:
    """Run x-ray mode: deep-dive analysis of a single file.

    Args:
        records: All loaded records from the JSONL file.
        filename: The filename to analyze (matched against source field).
    """
    match = [r for r in records if filename in r.get("source", "")]
    if not match:
        print(f"No record found matching '{filename}'")
        sys.exit(1)

    rec = match[0]
    ok = successful_records(records)

    print(f"X-ray for: {rec.get('source', 'unknown')}")
    print("-" * 60)

    # Print all metrics
    for key, value in rec.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    if rec.get("error") is not None:
        print("This document had an error — no timing breakdown available.")
        return

    # Stage breakdown percentages
    total = rec.get("total_time_s", 0)
    per_stage_stages = [s for s in STAGES if s != "total_time_s"]
    if total > 0:
        print("Stage breakdown:")
        for stage in per_stage_stages:
            val = rec.get(stage, 0)
            pct = val / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {stage:<16} {val:>8.3f}s  ({pct:>5.1f}%)  {bar}")
        print()

    # Comparison to median
    if ok:
        medians = {}
        for stage in STAGES:
            values = [r[stage] for r in ok if stage in r]
            if values:
                medians[stage] = float(np.median(values))

        print("Comparison to median:")
        for stage in STAGES:
            if stage in rec and stage in medians and medians[stage] > 0:
                ratio = rec[stage] / medians[stage]
                print(f"  {stage}: {rec[stage]:.3f}s ({ratio:.1f}x median)")
        print()

    # Insight
    if total > 0:
        slowest_stage = max(per_stage_stages, key=lambda s: rec.get(s, 0))
        slowest_val = rec.get(slowest_stage, 0)
        slowest_pct = slowest_val / total * 100

        reasons = []
        if slowest_stage == "chunk_time_s" and "num_sentences" in rec:
            sent_vals = [r.get("num_sentences", 0) for r in ok if "num_sentences" in r]
            median_sentences = float(np.median(sent_vals)) if sent_vals else 0
            reasons.append(
                f"This file has {rec['num_sentences']} sentences "
                f"(median: {median_sentences:.0f}). "
                f"Chunking took {slowest_pct:.0f}% of total time — "
                f"likely due to high sentence count requiring more similarity computations."
            )
        elif slowest_stage == "embed_time_s" and "num_chunks" in rec:
            chunk_vals = [r.get("num_chunks", 0) for r in ok if "num_chunks" in r]
            median_chunks = float(np.median(chunk_vals)) if chunk_vals else 0
            reasons.append(
                f"This file produced {rec['num_chunks']} chunks "
                f"(median: {median_chunks:.0f}). "
                f"Embedding took {slowest_pct:.0f}% of total time — "
                f"likely due to the large number of chunks to embed."
            )
        elif slowest_stage == "read_time_s" and "file_size_bytes" in rec:
            size_kb = rec["file_size_bytes"] / 1024
            size_vals = [r.get("file_size_bytes", 0) for r in ok if "file_size_bytes" in r]
            median_size = float(np.median(size_vals)) / 1024 if size_vals else 0
            reasons.append(
                f"File size is {size_kb:.0f} KB (median: {median_size:.0f} KB). "
                f"Reading took {slowest_pct:.0f}% of total time — "
                f"likely due to large file size or complex PDF parsing."
            )
        elif slowest_stage == "store_time_s" and "num_chunks" in rec:
            reasons.append(
                f"This file has {rec.get('num_chunks', 0)} chunks. "
                f"Storing took {slowest_pct:.0f}% of total time — "
                f"likely due to vector store write overhead for many chunks."
            )

        if reasons:
            print("Insight:")
            for reason in reasons:
                print(f"  {reason}")
        else:
            print(
                f"Insight: {slowest_stage} was the dominant stage "
                f"at {slowest_pct:.0f}% of total time."
            )
        print()


def export_to_mlflow(
    records: list[dict],
    log_path: Path,
    experiment_name: str = "ingestion_runs",
) -> None:
    """Export ingestion metrics to MLflow as a tracked run.

    Creates an MLflow run with:
    - Parameters: chunker strategy, file count, log file name
    - Per-stage summary metrics: mean, median, p95, stddev
    - Per-document metrics logged as step-indexed values
    - Charts and the JSONL log saved as artifacts

    Args:
        records: All loaded records from the JSONL file.
        log_path: Path to the source JSONL file (logged as artifact).
        experiment_name: MLflow experiment name.
    """
    import mlflow

    ok = successful_records(records)
    if not ok:
        print("No successful records to export.")
        return

    stats = compute_stage_stats(ok)
    failed_count = len(records) - len(ok)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=log_path.stem):
        # Log parameters
        mlflow.log_param("log_file", log_path.name)
        mlflow.log_param("total_documents", len(records))
        mlflow.log_param("failed_documents", failed_count)
        mlflow.log_param("successful_documents", len(ok))

        # Log summary metrics
        for stage, stage_stats in stats.items():
            for stat_name, value in stage_stats.items():
                mlflow.log_metric(f"{stage}_{stat_name}", value)

        # Log per-document metrics as step-indexed values
        for step, rec in enumerate(ok):
            for stage in STAGES:
                if stage in rec:
                    mlflow.log_metric(stage, rec[stage], step=step)
            if "num_chunks" in rec:
                mlflow.log_metric("num_chunks", rec["num_chunks"], step=step)
            if "num_sentences" in rec:
                mlflow.log_metric("num_sentences", rec["num_sentences"], step=step)
            if "file_size_bytes" in rec:
                mlflow.log_metric(
                    "file_size_kb",
                    rec["file_size_bytes"] / 1024,
                    step=step,
                )

        # Log the JSONL file as artifact
        mlflow.log_artifact(str(log_path))

        # Generate and log charts as artifacts
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        chart_paths = [
            save_stage_breakdown_chart(stats, CHART_DIR),
            save_total_time_histogram(ok, CHART_DIR),
            save_time_vs_chunks_chart(ok, CHART_DIR),
            save_time_vs_filesize_chart(ok, CHART_DIR),
        ]
        for chart_path in chart_paths:
            if chart_path.exists():
                mlflow.log_artifact(str(chart_path))

        run_id = mlflow.active_run().info.run_id
        print(f"Exported to MLflow experiment '{experiment_name}'")
        print(f"  Run ID: {run_id}")
        print("  View: mlflow ui  (then open http://localhost:5000)")


def list_log_files() -> None:
    """List available JSONL log files in the debug/logs directory."""
    if not LOG_DIR.exists():
        print(f"Log directory not found: {LOG_DIR}")
        return

    files = sorted(LOG_DIR.glob("ingest_*.jsonl"))
    if not files:
        print("No ingestion log files found.")
        return

    print("Available log files:")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f}  ({size_kb:.1f} KB)")


def main() -> None:
    """Entry point for the ingestion metrics analyzer CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze JSONL ingestion metrics logs and produce statistics and charts."
    )
    parser.add_argument("logfile", nargs="?", help="Path to the JSONL log file")
    parser.add_argument("--file", help="Deep-dive into a specific file (x-ray mode)")
    parser.add_argument("--list", action="store_true", help="List available log files")
    parser.add_argument(
        "--export-mlflow",
        action="store_true",
        help="Export metrics to MLflow as a tracked run",
    )

    args = parser.parse_args()

    if args.list:
        list_log_files()
        return

    if not args.logfile:
        parser.error("logfile is required unless using --list")

    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    records = load_records(log_path)
    if not records:
        print("No records found in log file.")
        sys.exit(1)

    if args.export_mlflow:
        export_to_mlflow(records, log_path)
    elif args.file:
        run_xray(records, args.file)
    else:
        run_summary(records)


if __name__ == "__main__":
    main()
