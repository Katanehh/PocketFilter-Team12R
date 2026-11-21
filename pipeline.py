"""End-to-end PocketFilter read evaluation and CSV decision export."""

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from preprocess import preprocess_signal
from sdtw_matcher import window_constrained_sdtw


@dataclass
class MatchResult:
    read_id: str
    best_distance: float
    decision: str
    match_start: int
    match_end: int
    window_width: int
    threshold: float
    runtime_ms: float
    note: str = "Research-only target/background screening prototype"


def evaluate_read(
    read_id: str,
    raw_signal: np.ndarray,
    target_ref: np.ndarray,
    window_width: int,
    threshold: float,
    median_kernel_size: int = 3,
) -> MatchResult:
    """Run preprocessing, window-constrained sDTW, and threshold classification."""
    t0 = time.perf_counter()

    clean_signal = preprocess_signal(raw_signal, kernel_size=median_kernel_size)
    clean_ref = preprocess_signal(target_ref, kernel_size=median_kernel_size)

    best_distance, match_start, match_end = window_constrained_sdtw(
        signal=clean_signal,
        target_ref=clean_ref,
        window_width=window_width,
    )

    decision = "KEEP" if best_distance <= threshold else "REJECT"
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return MatchResult(
        read_id=read_id,
        best_distance=round(best_distance, 4),
        decision=decision,
        match_start=match_start,
        match_end=match_end,
        window_width=window_width,
        threshold=threshold,
        runtime_ms=round(runtime_ms, 2),
    )


def export_results_csv(results: list[MatchResult], csv_path: Path) -> None:
    """Export ID-linked screening decisions to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))
