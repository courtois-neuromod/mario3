# Mario 3 Replay Processing

This script processes `.bk2` replay files from the Super Mario Bros 3 dataset and generates various outputs including videos, metadata, game variables, and low-level psychophysical features.

## Prerequisites

- Python 3.8 or higher
- The Mario 3 dataset with `.bk2` replay files
- ROM files in the `stimuli/` directory

## Installation

### 1. Create a Python virtual environment

From the root directory of the mario3 repository:

```bash
python -m venv env
```

### 2. Activate the environment

```bash
source env/bin/activate  # On Linux/Mac
# OR
env\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r code/replays/requirements.txt
```

This will install all required packages including:
- numpy, pandas
- retro, stable-retro (for replay processing)
- joblib, tqdm (for parallel processing)
- videogames_utils (from github.com/courtois-neuromod/videogames_utils - includes moviepy for video generation)

## Usage

### Basic Usage

From the root directory of the mario3 repository:

```bash
python code/replays/generate_replays.py --datapath . --output .
```

This will:
- Scan all `*_events.tsv` files in the dataset
- Process all `.bk2` replay files referenced in those events
- Generate all output files by default in `sub-XX/ses-XXX/gamelogs/` directories

### Output Files

For each `.bk2` replay file, the following files are generated (all BIDS-compliant naming):

1. **`*_recording.mp4`** - Video playback of the replay with audio
2. **`.json`** - Metadata sidecar with Mario 3-specific statistics:
   - Duration, World, Level
   - Score gained, distance traveled, average speed
   - Lives lost, hits taken, enemies killed
   - Powerups collected, bricks destroyed, coins gained
   - Level cleared status
3. **`*_variables.json`** - Frame-by-frame game variables (when available in data.json)
4. **`*_lowlevel.npy`** - Low-level psychophysical features:
   - Luminance
   - Optical flow
   - Audio envelope per frame

**IMPORTANT NOTE**: Many game variables expected in Super Mario Bros 3 are not yet available in the current `data.json` file. The script handles missing variables gracefully by setting metrics that depend on unavailable variables to `None` in the JSON metadata. This allows the script to run without crashing while producing partial metadata until a complete `data.json` file becomes available.

### Skipping Specific Outputs

If you want to skip certain outputs (e.g., to save time/space), use the `--skip_*` flags:

```bash
# Skip video generation (fastest, saves most space)
python code/replays/generate_replays.py --datapath . --output . --skip_videos

# Skip multiple outputs
python code/replays/generate_replays.py --datapath . --output . --skip_videos --skip_variables

# Only generate JSON metadata
python code/replays/generate_replays.py --datapath . --output . --skip_videos --skip_variables --skip_lowlevel
```

Available skip flags:
- `--skip_videos` - Skip video generation
- `--skip_variables` - Skip game variables extraction
- `--skip_lowlevel` - Skip low-level features computation

### Mario 3-Specific Features

#### Level Naming Convention

Mario 3 uses varied level naming: `level-w{world}l{levelname}`
- Examples: `w1lFortress`, `w7lPiranhaPlant1`, `w3lFortress1`
- World numbers range from 1-8
- Level names can be numbers or descriptive names (Fortress, PiranhaPlant, etc.)

#### Repetition Naming

Mario 3 uses 3-digit repetition indices: `rep-000`, `rep-001`, etc.

#### Discovery vs Practice Phases

The script automatically detects:
- **Discovery**: Single level repeated multiple times
- **Practice**: Multiple different levels in sequence

### Advanced Options

```bash
# Use parallel processing with multiple jobs (default is all cores)
python code/replays/generate_replays.py --datapath . --output . --n_jobs 4

# Use all available CPU cores (default)
python code/replays/generate_replays.py --datapath . --output . --n_jobs -1

# Use single-threaded processing
python code/replays/generate_replays.py --datapath . --output . --n_jobs 1

# Verbose output
python code/replays/generate_replays.py --datapath . --output . --verbose

# Custom stimuli path (if ROMs are in a different location)
python code/replays/generate_replays.py --datapath . --output . --stimuli /path/to/stimuli

# Filter by subject
python code/replays/generate_replays.py --datapath . --output . --subjects sub-01 sub-02

# Filter by session
python code/replays/generate_replays.py --datapath . --output . --sessions ses-001 ses-002

# Filter by both
python code/replays/generate_replays.py --datapath . --output . --subjects sub-01 --sessions ses-001
```

## How It Works

1. **Discovery**: The script walks through the dataset directory and finds all `*_events.tsv` files
2. **Extraction**: For each events file, it extracts the list of `.bk2` replay files
3. **Ordering**: Replays are sorted and assigned:
   - Global index (across all replays for a subject)
   - Level-specific index (for each world-level combination)
4. **Phase Detection**: Determines if each run is discovery (single level) or practice (multiple levels)
5. **Smart Processing**: For each replay, the script checks which outputs already exist and only regenerates missing files
6. **Processing**: Replays are processed in parallel by default (use `--n_jobs 1` for sequential processing)

## Mario 3-Specific Event Detection

The script computes high-level statistics for Mario 3 gameplay.

**IMPORTANT**: The current `data.json` file for Super Mario Bros 3 is incomplete. Many variables required for detailed event detection are not yet available. The script uses safe error handling to avoid crashes:

### Currently Available Variables (in data.json)
- `lives` - Player lives count
- `score` - Current score
- `world` - Current world number
- `killed` - Death state
- `mario_form` - Mario's current power-up form
- `complete_level` - Level completion status
- `time` / `timer_*` - Time remaining
- `x_pos_map`, `y_pos_map` - Map position (for world map)

### Variables NOT Yet Available (needed for full event detection)
The following are handled gracefully with `None` values when missing:

#### Enemy Kills
- Would require: `enemy_kill30-35` variables (6 enemy slots)
- Expected values: 4 (stomp), 34 (impact), 132 (kick)
- Currently returns: `None` for Enemies_killed metric

#### Brick Destruction
- Would require: Score tracking + `jump_airborne` state
- Currently returns: `None` for Bricks_destroyed metric

#### Hits Taken
- Would require: `powerstate` for powerup loss detection
- Uses `lives` for death detection (this works!)
- Currently returns: Partial data (only deaths, not powerup losses)

#### Powerup Collection
- Would require: `player_state` values [9, 12, 13] for powerup animation
- Currently returns: `None` for Powerups_collected metric

#### Coin Collection
- Would require: `coins` counter
- Currently returns: `None` for CoinsGained metric

#### Distance & Speed
- Would require: `xscrollHi` and `xscrollLo` for position tracking
- Currently returns: `None` for X_Traveled and Average_speed metrics

#### Level Cleared Detection
- Would require: `player_y_screen` and `player_state` for completion detection
- Currently returns: `None` for Cleared metric

The script will automatically populate these metrics once the `data.json` file is updated with the required variables. Until then, it produces partial metadata without crashing.

## File Structure

```
mario3/
├── sub-01/
│   ├── ses-001/
│   │   ├── func/
│   │   │   └── sub-01_ses-001_task-mario3_run-01_events.tsv
│   │   └── gamelogs/
│   │       ├── sub-01_ses-001_task-mario3_run-01_level-w1lFortress_rep-000.bk2
│   │       ├── sub-01_ses-001_task-mario3_run-01_level-w1lFortress_rep-000.json
│   │       ├── sub-01_ses-001_task-mario3_run-01_level-w1lFortress_rep-000_recording.mp4
│   │       ├── sub-01_ses-001_task-mario3_run-01_level-w1lFortress_rep-000_variables.json
│   │       └── sub-01_ses-001_task-mario3_run-01_level-w1lFortress_rep-000_lowlevel.npy
│   └── ...
├── stimuli/
│   └── SuperMarioBros3-Nes/
├── code/
│   └── replays/
│       ├── generate_replays.py
│       ├── requirements.txt
│       └── README.md
└── env/  # Created by you
```

## Troubleshooting

### "File not found" errors for .bk2 files
- Ensure the `.bk2` files exist in the paths specified in the `*_events.tsv` files
- The paths in events.tsv should be relative to the dataset root

### ROM/stimuli errors
- Make sure the `stimuli/` directory exists in the dataset root
- Verify that `stimuli/SuperMarioBros3-Nes/` contains the ROM and game data files

### Memory issues
- Use fewer parallel jobs: `--n_jobs 2`
- Skip videos: `--skip_videos`
- Process one subject at a time: `--subjects sub-01`

### Already processed files
- The script automatically detects existing outputs and skips them
- To force regeneration, delete the existing output files

### Many metrics are None in JSON files
- This is expected! The current `data.json` file is incomplete
- The script safely handles missing variables by setting dependent metrics to `None`
- Once the `data.json` file is updated with required variables, re-run the script to populate these metrics

## Performance Tips

- **Fastest**: `--skip_videos --skip_lowlevel` (only JSON + variables)
- **Balanced**: `--skip_videos`
- **Full processing**: No skip flags (default - generates everything)

Processing time per replay (approximate):
- JSON only: ~1-2 seconds
- With video: ~10-30 seconds
- With all outputs: ~30-60 seconds

For parallel processing, expect roughly linear speedup up to the number of physical CPU cores.

## Next Steps

To get full event detection working:

1. Update the `data.json` file in `stimuli/SuperMarioBros3-Nes/` with all required game variables
2. Re-run the replay processing script to populate the missing metrics
3. The script will automatically use newly available variables without code changes

## Questions or Issues?

If you encounter any problems or have questions about the script, please check:
1. That all dependencies are installed correctly
2. That the virtual environment is activated
3. That you're running with correct `--datapath` and `--output` arguments
4. The verbose output for detailed error messages: `--verbose`
