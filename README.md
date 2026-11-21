# PocketFilter (Team 12-R)

Python model for our senior design project. It cleans up a raw nanopore-style signal, compares it to a target reference using window-constrained sDTW, and outputs a keep or reject decision.

Right now this is just the software version running on synthetic test signals. We will test on real recorded nanopore data and move the matching code to the AUP-ZU3 FPGA later. (Research prototype only, not for diagnostic use.)

## How to run it

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_tests.py
```

Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_tests.py
```

This creates sample CSV files in `test_inputs/`, runs three basic tests (normal match, time-warped match, and non-match), and saves the results CSV and signal plot in `outputs/`.

## Inputs and Outputs
* **Inputs:** `raw_signal` (1D array), `target_ref` (1D array), `window_width` (int), `threshold` (float), `read_id` (str)
* **Outputs:** `best_distance` (float, lower = better match), `decision` (`KEEP` or `REJECT`), `match_start` & `match_end` (sample indices in read), `runtime_ms`

## Files

- `config.py`: settings for window width, threshold, and filter size
- `preprocess.py`: median filter to remove spikes, then normalization
- `sdtw_matcher.py`: window-constrained sDTW matcher
- `pipeline.py`: runs a read through the steps and writes the CSV log
- `run_tests.py`: generates the 3 test signals, runs them, and saves the plot