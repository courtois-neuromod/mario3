# Mario 3 Replay Processing

Processes `.bk2` replay files to generate video, metadata, game variables, and low-level features.

## Prerequisites & Installation

1.  **Environment**: Python 3.8+, Mario 3 dataset (with `.bk2` replays), and ROMs in `stimuli/`.
2.  **Setup**:
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r code/replays/requirements.txt
    ```

## Usage

```bash
python code/replays/generate_replays.py --datapath . --output . [options]
```

### Arguments (Brief)
-   `--datapath`: Root directory of the dataset.
-   `--output`: Output directory.
-   `--skip_videos`, `--skip_variables`, `--skip_lowlevel`: Skip specific outputs.
-   `--n_jobs`: Number of parallel jobs (default: all cores).
-   `--subjects`, `--sessions`: Filter processing.
-   `--stimuli`: Custom path for ROMs.
-   `--verbose`: Enable detailed logging.

## Generated Files

For each replay (e.g., `sub-01_ses-001_..._rep-000`):
1.  `*_recording.mp4`: Video recording.
2.  `*_variables.json`: Frame-by-frame RAM variables (if extracted).
3.  `*_lowlevel.npy`: Luminance, optical flow, and audio features.
4.  `*_summary.json`: Summary metadata (BIDS sidecar).

## Summary Variables (in sidecar JSON)

All variables rely on RAM addresses defined in `stimuli/SuperMarioBros3-Nes/data.json`.

| Variable | Source / Logic |
| :--- | :--- |
| **Duration** | Total replay duration in seconds. |
| **Outcome** | `cleared` (level completed & !killed), `failed/timeout` (timer=0), `failed/fall` (X pos check), `failed/killed` (other deaths). |
| **X_traveled** | Max distance reached from start (`max(page * 256 + x) - start`). |
| **Enemies_stomped** | Count of `stomp_counter` transitions from 0 to 1. |
| **Hits_taken** | Count of ANY decrement in `powerup` value + 1 if outcome is `failed/killed`. |
| **Bricks_smashed** | Count of `score` increments of exactly 1. |
| **Coins** | Difference between start and end coin count (requires `coins` variable). |
| **Powerups_collected** | Count of ANY increment in `powerup` value. |
| **Stars_collected** | Count of starman activations (duration > 0). |
| **Flights_activated** | Count of flight timer activations (duration > 0). |
| **Phase** | `discovery` (level repeats) or `practice` (sequential progression) - determined *per subject*. |
