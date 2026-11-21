"""Shared configuration for the PocketFilter Python/NumPy golden model."""

from pathlib import Path

# Preprocessing (Alternative 1: median filtering + z-score normalization)
MEDIAN_KERNEL_SIZE = 3

# Window-constrained sDTW parameters
WINDOW_WIDTH = 15       # Band half-width around each candidate start
MATCH_THRESHOLD = 30.0  # Keep if best_distance <= MATCH_THRESHOLD

# Directories
TEST_INPUT_DIR = Path("test_inputs")
OUTPUT_DIR = Path("outputs")
