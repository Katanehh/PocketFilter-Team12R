"""Core window-constrained subsequence Dynamic Time Warping (sDTW) matcher."""

import numpy as np


def window_constrained_sdtw(
    signal: np.ndarray,
    target_ref: np.ndarray,
    window_width: int,
) -> tuple[float, int, int]:
    """
    Compare a target reference against a longer incoming read using
    window-constrained sDTW.

    For each candidate start index s in the read, the alignment path (i, j)
    is restricted to |(j - s) - i| <= window_width.

    Returns:
        best_distance: Minimum cumulative L1 alignment distance (lower = better).
        match_start:   Start sample index of the best match in `signal`.
        match_end:     Inclusive end sample index of the best match in `signal`.
    """
    q = np.asarray(signal, dtype=np.float64)
    r = np.asarray(target_ref, dtype=np.float64)
    n = q.shape[0]
    m = r.shape[0]
    w = int(window_width)

    if n == 0 or m == 0:
        raise ValueError("Input signal and target reference must be non-empty.")
    if w < 0:
        raise ValueError("window_width must be non-negative.")

    band_size = 2 * w + 1
    starts = np.arange(n)

    # prev_row[s, d_idx] holds cumulative cost for row (i - 1) at offset d = d_idx - w
    prev_row = np.full((n, band_size), np.inf, dtype=np.float64)

    for i in range(m):
        curr_row = np.full((n, band_size), np.inf, dtype=np.float64)
        r_val = r[i]

        for d_idx in range(band_size):
            d = d_idx - w

            # Row 0 starts at j = s (d = 0) and may only advance right (d > 0)
            if i == 0 and d < 0:
                continue

            j = starts + i + d
            valid = (j >= 0) & (j < n)
            if not np.any(valid):
                continue

            step_cost = np.abs(r_val - q[j[valid]])

            if i == 0 and d == 0:
                curr_row[valid, d_idx] = step_cost
            else:
                # Diagonal predecessor: (i - 1, j - 1) -> same offset d_idx
                best_prev = prev_row[valid, d_idx]

                # Vertical predecessor: (i - 1, j) -> offset d + 1
                if d_idx + 1 < band_size:
                    best_prev = np.minimum(best_prev, prev_row[valid, d_idx + 1])

                # Horizontal predecessor: (i, j - 1) -> offset d - 1
                if d_idx - 1 >= 0:
                    best_prev = np.minimum(best_prev, curr_row[valid, d_idx - 1])

                curr_row[valid, d_idx] = step_cost + best_prev

        prev_row = curr_row

    best_flat = int(np.argmin(prev_row))
    best_s, best_d_idx = np.unravel_index(best_flat, prev_row.shape)
    best_distance = float(prev_row[best_s, best_d_idx])

    if not np.isfinite(best_distance):
        return float("inf"), -1, -1

    match_start = int(best_s)
    match_end = int(best_s + (m - 1) + (best_d_idx - w))
    return best_distance, match_start, match_end
