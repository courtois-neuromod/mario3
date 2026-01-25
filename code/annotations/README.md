# Mario 3 Annotations Generator

This script generates BIDS-compatible annotated event files (`*_desc-annotated_events.tsv`) for the Super Mario Bros 3 dataset. It reads pre-processed game variables and computes detailed annotations for all gameplay events including button presses, enemy kills, hits taken, item collection, and more.

**IMPORTANT NOTE**: This script contains placeholder logic that needs to be updated once the `data.json` file for Super Mario Bros 3 is completed. Many game-specific variables are currently missing from `data.json`, so most event detection features will not work until the data file is updated. The script handles missing variables gracefully to avoid crashes.

## Prerequisites

- Python 3.8 or higher
- The Mario 3 dataset with `.bk2` replay files
- **Replays must be processed first** using `code/replays/create_replays.py` to generate `*_variables.json` files
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
pip install -r code/annotations/requirements.txt
```

This will install:
- numpy
- pandas
- stable-retro

## Usage

### Basic Usage

From the root directory of the mario3 repository:

```bash
python code/annotations/generate_annotations.py --datapath .
```

This will:
- Scan all `*_events.tsv` files in the dataset
- Load corresponding replay variables from `gamelogs/*_variables.json`
- Generate `*_desc-annotated_events.tsv` files with detailed event annotations

### Options

```bash
# Specify a custom data path
python code/annotations/generate_annotations.py --datapath /path/to/mario3

# Custom output path
python code/annotations/generate_annotations.py --datapath . --output_path /path/to/output

# Filter by subject
python code/annotations/generate_annotations.py --datapath . --subjects sub-01 sub-02

# Filter by session
python code/annotations/generate_annotations.py --datapath . --sessions ses-001 ses-002
```

## Generated Annotations

The script produces `*_desc-annotated_events.tsv` files with the following structure:

### Column Order

| Column | Description |
|--------|-------------|
| trial_type | Type of event (see below) |
| rep_index | Repetition index within the run (integer) |
| level | Level identifier (e.g., "w1lFortress", "w7lPiranhaPlant1") |
| onset | Time in seconds from the start of the run (3 decimal places) |
| duration | Duration of the event in seconds (3 decimal places) |
| frame_start | Frame index where event starts (integer) |
| frame_stop | Frame index where event ends (integer) |
| phase | "discovery" or "practice" (see below) |

### Event Types

**Note**: Button press events come from the replay file and are always available. Game state events require variables from `data.json` which are mostly missing.

#### Repetition Events
- `gym-retro_game` - Base repetition events from the original events file

#### Button Press Events ✅ AVAILABLE
Continuous events with onset and duration:
- `UP`, `DOWN`, `LEFT`, `RIGHT` - D-pad directions
- `A` - Jump button
- `B` - Run/fireball button
- `START` - Pause
- `SELECT` - Mode select

**Status**: Button inputs come from the .bk2 replay file itself (not data.json) and are always extracted automatically. These events ARE generated!

#### Enemy Kill Events (Available)
Instantaneous events (duration=0):
- `Kill/stomp` - Jumping on enemy (detected via `stomp_counter` 0->1 transition)

**Status**: Implemented using `stomp_counter`.

#### Hit Events (Available)
Instantaneous events (duration=0):
- `Hit/powerup_lost` - Lost powerup state (detected via `powerup` DECREMENT)
- `Hit/killed` - Death via enemy collision (detected via outcome logic + `killed` flag)
- `Hit/fall` - Death via falling into a pit (detected via outcome logic: last 100 frames X position static)

**Status**: Implemented using `powerup` and outcome logic.

#### Item Collection / Activation Events (Available)
- `Powerup_collected` (Instant, duration=0): Any increase in `powerup` value.
- `Star_activated` (Duration): Period where `invincibility_timer` > 0.
- `Flight_activated` (Duration): Period where `flight_timer` > 0.
- `P-Switch_activated` (Duration): Period where `p_switch_timer` > 0.
- `Brick_smashed` (Instant): Brick destroyed (detected via `score` increment of 1)
- `Coin_collected` (Instant): Coin counter increases (requires `coins` variable - Not Implemented)

**Status**: Powerups, Star, Flight, Bricks implemented. Coins pending.

### Phase Information

Each run is classified as:
- **discovery**: Single level repeated multiple times (practice/training)
- **practice**: Multiple different levels in sequence (testing)

## Current Data.json Status

### Button Inputs ✅ ALWAYS AVAILABLE
Button inputs come from the **replay file (.bk2)** not from data.json:
- ✅ `UP`, `DOWN`, `LEFT`, `RIGHT`, `A`, `B`, `START`, `SELECT`
- These are automatically extracted by the replay processing and stored in _variables.json
- Button press events WILL be generated!

### RAM Variables Available in data.json
The current `data.json` file for Super Mario Bros 3 includes only:
- ✅ `lives` - Player lives count (allows life loss detection)
- ✅ `score` - Current score
- ✅ `world` - Current world number
- ✅ `killed` - Death state
- ✅ `mario_form` - Mario's current power-up form
- ✅ `powerup` - Mario's powerup state (0=small, 1=super, etc.)
- ✅ `complete_level` - Level completion status
- ✅ `time` / `timer_*` - Time remaining
- ✅ `x_pos_map`, `y_pos_map` - Map position (for world map)
- ✅ `invincibility_timer` - Star power timer
- ✅ `flight_timer` - Flight timer
- ✅ `p_switch_timer` - P-Switch timer

### Missing RAM Variables (Required for Full Event Detection)

The following RAM variables are needed but NOT currently in data.json:

1. **Enemy Tracking** (for kill events)
   - `enemy_kill30` through `enemy_kill35` (6 enemy slots)
   - Values should indicate kill types (stomp, impact, kick)

2. **Game State** (for specific events)
   - `jump_airborne` - For brick smashing detection
   - `player_y_screen` - For level completion detection

3. **Item Counters** (for collection events)
   - `coins` - Coin count

4. **Position Tracking** (for distance/speed metrics)
   - `xscrollHi`, `xscrollLo` - Horizontal scroll position

## Logic Status

### 1. Enemy Kill Detection
- **Location**: `generate_kill_events()` function
- **Current Status**: Returns empty dataframe if enemy_kill variables missing
- **TODO**: Verify enemy slot count, variable names, and kill type values for SMB3

### 2. Hit Detection
- **Location**: `generate_hits_taken_events()` function
- **Current Status**: Fully implemented using `powerup` decrements and outcome logic.

### 3. Brick Destruction
- **Location**: `generate_bricks_smashed_events()` function
- **Current Status**: Implemented using score increments, but `jump_airborne` check is skipped if missing.
- **TODO**: Verify score increment value and jump_airborne detection

### 4. Powerup Collection
- **Location**: `generate_powerup_events()` function
- **Current Status**: Fully implemented using `powerup` increments.

### 5. Coin Collection
- **Location**: `generate_coin_events()` function
- **Current Status**: Returns empty dataframe if coins variable missing
- **TODO**: Add coins variable to data.json

### 5. Coin Collection
- **Location**: `generate_coin_events()` function
- **Current Status**: Returns empty dataframe if coins variable missing
- **TODO**: Add coins variable to data.json

## Dependencies

This script requires that replays have been processed first:

```bash
# First, process replays to generate variables
python code/replays/create_replays.py --datapath .

# Then run annotations
python code/annotations/generate_annotations.py --datapath .
```

## Troubleshooting

### "Variables file not found" errors
- Ensure you've run `code/replays/create_replays.py` first
- Check that `gamelogs/*_variables.json` files exist for each .bk2 file

### "No bk2 files available for this run"
- Normal if a run has no valid .bk2 files (all marked as "Missing file")

### ROM/stimuli errors
- Verify that `stimuli/SuperMarioBros3-Nes/` contains the ROM files

### Already annotated files
- The script skips files that already have annotated versions
- To force regeneration, delete existing `*_desc-annotated_events.tsv` files

### Very few game state events detected
- **This is expected!** The current data.json is incomplete
- Button press events WILL be detected (from replay file)
- Life loss events WILL be detected (uses `lives` from data.json)
- Most other game state events (kills, coins, powerups, bricks) cannot be detected until data.json is updated

### Annotated files contain mostly button presses and repetition events
- This is normal with the incomplete data.json
- You should see many button press events (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT)
- You should see life loss events (Hit/life_lost)
- Other game state events require data.json variables that don't exist yet
- Once data.json is updated, re-run the script to generate full annotations

## Next Steps

To get full event detection working:

1. **Update data.json** in `stimuli/SuperMarioBros3-Nes/` with all required RAM variables:
   - Enemy tracking variables (enemy_kill30-35)
   - Player state variables (player_state, powerstate, jump_airborne, player_y_screen)
   - Item counters (coins)
   - Position tracking (xscrollHi, xscrollLo)

   Note: Button inputs don't need to be added - they come from the replay file automatically!

2. **Re-process replays** to extract the new variables:
   ```bash
   # Remove old variables files first
   find . -name "*_variables.json" -delete

   # Re-run replay processing
   python code/replays/create_replays.py --datapath .
   ```

3. **Re-run annotation generation**:
   ```bash
   # Remove old annotated files
   find . -name "*_desc-annotated_events.tsv" -delete

   # Generate new annotations with full event detection
   python code/annotations/generate_annotations.py --datapath .
   ```

4. **Verify event detection** on a small subset of data to ensure the placeholder values are correct for SMB3

## Questions or Issues?

If you encounter any problems or have questions about the script, please check:
1. That all dependencies are installed correctly
2. That the virtual environment is activated
3. That you've run `create_replays.py` first to generate variables files
4. That the missing variables are documented above and expected behavior
