"""Generate controlled synthetic nanopore tests, run golden model, and save plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (
    MATCH_THRESHOLD,
    MEDIAN_KERNEL_SIZE,
    OUTPUT_DIR,
    TEST_INPUT_DIR,
    WINDOW_WIDTH,
)
from pipeline import MatchResult, evaluate_read, export_results_csv


def make_squiggle(levels: np.ndarray, dwells: np.ndarray) -> np.ndarray:
    """Expand discrete k-mer current levels into a 1D raw electrical signal."""
    return np.repeat(levels, dwells).astype(np.float64)


def add_noise_and_spikes(
    signal: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 1.5,
    num_spikes: int = 4,
) -> np.ndarray:
    """Add realistic Gaussian sensor noise and isolated impulse spikes."""
    noisy = signal + rng.normal(0.0, noise_std, size=signal.shape[0])
    spike_indices = rng.choice(signal.shape[0], size=num_spikes, replace=False)
    spike_signs = rng.choice([-1.0, 1.0], size=num_spikes)
    noisy[spike_indices] += spike_signs * 55.0
    return noisy


def generate_controlled_tests() -> tuple[np.ndarray, list[dict]]:
    """Build 3 controlled synthetic tests and save inputs to test_inputs/."""
    rng = np.random.default_rng(42)

    # Target reference: 20 k-mer states x 5 samples/state = 100 samples (~65 to 115 pA)
    target_levels = rng.uniform(65.0, 115.0, size=20)
    target_ref = make_squiggle(target_levels, np.full(20, 5))

    # Test 1: Embedded target match (starts at index 110, length 100)
    prefix_1 = make_squiggle(rng.uniform(65.0, 115.0, size=22), np.full(22, 5))
    suffix_1 = make_squiggle(rng.uniform(65.0, 115.0, size=26), np.full(26, 5))
    read_1 = add_noise_and_spikes(np.concatenate([prefix_1, target_ref, suffix_1]), rng)

    # Test 2: Time-warped target match (variable translocation speed, dwells 3-7 samples)
    warped_dwells = np.array([4, 6, 5, 7, 3, 6, 4, 5, 7, 4, 6, 5, 4, 6, 5, 3, 7, 5, 6, 4])
    warped_target = make_squiggle(target_levels, warped_dwells)
    prefix_2 = make_squiggle(rng.uniform(65.0, 115.0, size=24), np.full(24, 5))
    suffix_2 = make_squiggle(rng.uniform(65.0, 115.0, size=24), np.full(24, 5))
    read_2 = add_noise_and_spikes(np.concatenate([prefix_2, warped_target, suffix_2]), rng)

    # Test 3: Background / non-target read (unrelated k-mer sequence)
    bg_levels = rng.uniform(65.0, 115.0, size=68)
    read_3 = add_noise_and_spikes(make_squiggle(bg_levels, np.full(68, 5)), rng)

    # Save inputs as CSV so teammates can inspect and reuse them for HLS simulation
    TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(TEST_INPUT_DIR / "target_ref.csv", target_ref, delimiter=",", fmt="%.4f")

    tests = [
        {
            "read_id": "read_001_target_match",
            "title": "Test 1: Embedded Target Match (Expected: KEEP, true span 110-209)",
            "expected": "KEEP",
            "signal": read_1,
        },
        {
            "read_id": "read_002_warped_match",
            "title": (
                f"Test 2: Time-Warped Target Match (Expected: KEEP, "
                f"true span 120-{120 + len(warped_target) - 1})"
            ),
            "expected": "KEEP",
            "signal": read_2,
        },
        {
            "read_id": "read_003_background_reject",
            "title": "Test 3: Background / Non-Target Read (Expected: REJECT)",
            "expected": "REJECT",
            "signal": read_3,
        },
    ]

    for t in tests:
        np.savetxt(
            TEST_INPUT_DIR / f"{t['read_id']}.csv",
            t["signal"],
            delimiter=",",
            fmt="%.4f",
        )

    return target_ref, tests


def save_summary_plot(
    target_ref: np.ndarray,
    tests: list[dict],
    results: list[MatchResult],
    plot_path: Path,
) -> None:
    """Save a clean multi-panel figure for presentation slides."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    fig.suptitle(
        "PocketFilter Golden Model — Controlled Synthetic Verification Tests\n"
        f"(Window-Constrained sDTW | Band Width = {WINDOW_WIDTH} | "
        f"Threshold = {MATCH_THRESHOLD} | Research-Only Screening Prototype)",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 0: Target reference waveform
    axes[0].plot(target_ref, color="#1f77b4", linewidth=1.6)
    axes[0].set_title(f"Target Reference Signal (Length = {len(target_ref)} samples)")
    axes[0].set_ylabel("Current (pA)")
    axes[0].grid(True, alpha=0.3)

    # Panels 1-3: Controlled test reads with detected match window highlighted
    for idx, (test, res) in enumerate(zip(tests, results), start=1):
        ax = axes[idx]
        sig = test["signal"]
        ax.plot(sig, color="#333333", linewidth=1.0, label="Raw read (with noise & spikes)")

        span_color = "#2ca02c" if res.decision == "KEEP" else "#d62728"
        ax.axvspan(
            res.match_start,
            res.match_end,
            color=span_color,
            alpha=0.22,
            label=f"Best aligned span [{res.match_start}, {res.match_end}]",
        )

        ax.set_title(
            f"{test['title']}  |  Distance = {res.best_distance:.2f}  ->  {res.decision}",
            color=span_color,
            fontweight="bold",
            fontsize=10,
        )
        ax.set_ylabel("Current (pA)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample Index")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> None:
    target_ref, tests = generate_controlled_tests()
    results: list[MatchResult] = []

    print("\n=== PocketFilter Golden Model Verification (Controlled Synthetic Tests) ===")
    print(f"Window Width: {WINDOW_WIDTH} | Threshold: {MATCH_THRESHOLD} | Median Kernel: {MEDIAN_KERNEL_SIZE}\n")
    print(f"{'Read ID':<28} {'Expected':<9} {'Distance':<10} {'Decision':<9} {'Match Span':<14} {'Time (ms)'}")
    print("-" * 86)

    for test in tests:
        res = evaluate_read(
            read_id=test["read_id"],
            raw_signal=test["signal"],
            target_ref=target_ref,
            window_width=WINDOW_WIDTH,
            threshold=MATCH_THRESHOLD,
            median_kernel_size=MEDIAN_KERNEL_SIZE,
        )
        results.append(res)
        span_str = f"[{res.match_start}, {res.match_end}]"
        print(
            f"{res.read_id:<28} {test['expected']:<9} {res.best_distance:<10.2f} "
            f"{res.decision:<9} {span_str:<14} {res.runtime_ms:.2f}"
        )

    csv_path = OUTPUT_DIR / "results.csv"
    plot_path = OUTPUT_DIR / "pocketfilter_test_results.png"
    export_results_csv(results, csv_path)
    save_summary_plot(target_ref, tests, results, plot_path)

    print(f"\nSaved test inputs to: {TEST_INPUT_DIR.resolve()}")
    print(f"Saved decision log to: {csv_path.resolve()}")
    print(f"Saved summary plot to: {plot_path.resolve()}\n")


if __name__ == "__main__":
    main()
