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

## Usage

### Basic Usage

From the root directory of the mario3 repository:

```bash
python code/annotations/generate_annotations.py
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
python code/annotations/generate_annotations.py --output_path /path/to/output

# Filter by subject
python code/annotations/generate_annotations.py --subjects sub-01 sub-02

# Filter by session
python code/annotations/generate_annotations.py --sessions ses-001 ses-002
```

## Generated Annotations

The script produces BIDS-compatible `*_desc-annotated_events.tsv` files with the following structure:

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

#### Repetition Events
- `gym-retro_game` - Base repetition events from the original events file

#### Button Press Events
Continuous events with onset and duration:
- `UP`, `DOWN`, `LEFT`, `RIGHT` - D-pad directions
- `JUMP` - Jump button
- `RUN/THROW` - Run/fireball button
- `START` - Pause
- `SELECT` - Mode select

#### Enemy Kill Events
Instantaneous events (duration=0):
- `Kill/stomp` - Jumping on enemy (detected via `stomp_counter` 0->1 transition)

#### Hit Events
Instantaneous events (duration=0):
- `Hit/powerup_lost` - Lost powerup state (detected via `powerup` DECREMENT)
- `Hit/killed` - Death via enemy collision (detected via outcome logic + `killed` flag)
- `Hit/fall` - Death via falling into a pit (detected via outcome logic: last 100 frames X position static)

#### Item Collection / Activation Events
- `Powerup_collected` (Instant, duration=0): Any increase in `powerup` value.
- `Star_activated` (Variable duration): Period where `invincibility_timer` > 0.
- `Flight_activated` (Variable duration): Period where `flight_timer` > 0.
- `P-Switch_activated` (Variable duration): Period where `p_switch_timer` > 0.
- `Brick_smashed` (Instant): Brick destroyed (detected via `score` increment of 1)
- `Coin_collected` (Instant): Coin counter increases

### Phase Information

Each run was performed in one of these two phases:
- **discovery**: Single level repeated multiple times (practice/training)
- **practice**: Multiple different levels in sequence (testing)

### RAM Variables Available in data.json
The current `data.json` file for Super Mario Bros 3 includes:
- `lives` - Player lives count (allows life loss detection)
- `score` - Current score
- `coins` - Coin count
- `world` - Current world number
- `killed` - Death state
- `mario_form` - Mario's current power-up form
- `powerup` - Mario's powerup state (0=small, 1=super, etc.)
- `complete_level` - Level completion status
- `time` / `timer_*` - Time remaining
- `x_pos_map`, `y_pos_map` - Map position (for world map)
- `invincibility_timer` - Star power timer
- `flight_timer` - Flight timer
- `p_switch_timer` - P-Switch timer