#!/usr/bin/env python
"""
Generate replay outputs for the Mario 3 dataset.

By default, all files are generated:
  - JSON sidecar file with metadata
  - MP4 video file
  - Variables JSON file with game variables
  - Low-level features NPY file (luminance, optical flow, audio envelope)

Use the flags below to skip specific outputs:
  --skip_videos      : Skip generating video files (_recording.mp4).
  --skip_variables   : Skip generating variables files (_variables.json).
  --skip_lowlevel    : Skip generating low-level features (_lowlevel.npy).

Use the -v/--verbose flag to display verbose output.
"""

import argparse
import os
import os.path as op
import stable_retro
import pandas as pd
import json
import numpy as np
import gc
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import logging
from videogames_utils.replay import get_variables_from_replay
from videogames_utils.video import make_mp4
from videogames_utils.psychophysics import (
    compute_luminance,
    compute_optical_flow,
    audio_envelope_per_frame,
)


# ============================================================================
# Mario 3-specific utility functions (using data.json variables)
# ============================================================================

def _calculate_world_and_level(level_str):
    """Extract world and level identifiers from level string."""
    # Mario 3 uses naming like "w1lFortress", "w7lPiranhaPlant2", etc.
    try:
        if level_str and level_str.startswith('w') and 'l' in level_str:
            parts = level_str.split('l')
            world = parts[0][1:]  # Remove 'w' prefix
            level = parts[1] if len(parts) > 1 else None
            return world, level
    except:
        pass
    return None, None


def _calculate_distance_traveled(repetition_variables):
    """
    Calculate maximum X distance traveled using player level position.
    
    Combines player_x_level_high (high byte) and player_x_level_low (low byte) for full position.
    Returns the furthest point reached minus the starting position.
    """
    try:
        player_x = repetition_variables["player_x_level_low"]
        # Use player_x_level_high as the high byte (page number)
        # Note: address 117 (0x75) tracks horizontal position in 256-pixel increments
        level_page = repetition_variables["player_x_level_high"]
        
        # Calculate full X position for each frame: page * 256 + x
        full_x_positions = [
            level_page[i] * 256 + player_x[i] 
            for i in range(len(player_x))
        ]
        
        start_x = full_x_positions[0]
        max_x = max(full_x_positions)
        
        # Return maximum distance traveled from start
        return max_x - start_x
        
    except (KeyError, IndexError):
        return None


def _determine_outcome(repetition_variables):
    """
    Determine how the replay ended: 'cleared' or 'failed/*'.

    Outcomes:
    - cleared: Level completed successfully (any of goal_cards_p1_1/2/3 increases)
    - failed/fall: Last 100 frames of player_x_level_low are all 0
    - failed/timeout: Timer reaches 0
    - failed/killed: Killed by enemy (default failure if not fall/timeout)
    """
    try:
        # Check if level was cleared: any goal_cards_p1 variable increases
        for card_var in ["goal_cards_p1_1", "goal_cards_p1_2", "goal_cards_p1_3"]:
            cards = repetition_variables.get(card_var, [])
            if len(cards) > 1 and any(cards[i] > cards[i - 1] for i in range(1, len(cards))):
                return "cleared"

        # If no completion detected, fall back to end-of-replay checks

        # Check for Timeout: time variable reaches 0, confirmed by all timer digits being 0
        time_val = repetition_variables.get("time", [])
        all_timer_zero = (
            repetition_variables.get("timer_100", [1])[-1] == 0
            and repetition_variables.get("timer_10", [1])[-1] == 0
            and repetition_variables.get("timer_1", [1])[-1] == 0
        )
        if (time_val and time_val[-1] == 0) or all_timer_zero:
            return "failed/timeout"

        # Check for Fall vs Killed
        x_low = repetition_variables.get("player_x_level_low", [])
        last_segment = x_low[-100:] if len(x_low) > 0 else []

        if last_segment and all(v == 0 for v in last_segment):
            return "failed/fall"

        return "failed/killed"
        
    except (KeyError, IndexError):
        return "unknown"


def count_stomps(repetition_variables):
    """
    Count total enemies stomped using stomp_counter.
    Counts the number of times stomp_counter goes from 0 to 1.
    """
    try:
        stomps = repetition_variables["stomp_counter"]
        stomp_count = 0
        
        # Count transitions from 0 to 1
        for idx in range(1, len(stomps)):
            if stomps[idx] == 1 and stomps[idx - 1] == 0:
                stomp_count += 1
                
        return stomp_count
    except (KeyError, IndexError):
        return None


def count_bricks_smashed(repetition_variables):
    """
    Count bricks smashed.
    Counts the number of times score increments by 1.
    """
    try:
        if "score" not in repetition_variables:
            return None
        
        score_increments = list(np.diff(repetition_variables["score"]))
        # Count score increments of 1
        return sum(1 for inc in score_increments if inc == 1)
    except (KeyError, IndexError):
        return None


def count_hits_taken(repetition_variables, outcome):
    """
    Count total hits taken. 
    Counts ANY decrement in powerup value.
    Adds 1 if the outcome is 'failed/killed'.
    """
    try:
        powerups = repetition_variables["powerup"]
        hit_count = 0
        
        # Count transitions where value decreases (e.g., 3->1, 1->0)
        for idx in range(1, len(powerups)):
            if powerups[idx] < powerups[idx - 1]:
                hit_count += 1
                
        if outcome == "failed/killed":
            hit_count += 1
            
        return hit_count
    except (KeyError, IndexError):
        return None


def count_powerups_collected(repetition_variables):
    """
    Count powerups collected by detecting powerup value increases.
    """
    try:
        if "powerup" not in repetition_variables:
            return None
        
        powerups = repetition_variables["powerup"]
        powerup_count = 0
        
        for idx in range(1, len(powerups)):
            if powerups[idx] > powerups[idx - 1]:
                powerup_count += 1
        
        return powerup_count
    except (KeyError, IndexError):
        return None


def count_star_power_activations(repetition_variables):
    """Count times star power was activated (invincibility_timer goes from 0 to >0)."""
    try:
        if "invincibility_timer" not in repetition_variables:
            return None
        
        timer = repetition_variables["invincibility_timer"]
        activations = 0
        
        for idx in range(1, len(timer)):
            if timer[idx - 1] == 0 and timer[idx] > 0:
                activations += 1
        
        return activations
    except (KeyError, IndexError):
        return None


def count_flight_activations(repetition_variables):
    """Count times flight was activated (flight_timer goes from 0 to >0)."""
    try:
        if "flight_timer" not in repetition_variables:
            return None
        
        timer = repetition_variables["flight_timer"]
        activations = 0
        
        for idx in range(1, len(timer)):
            if timer[idx - 1] == 0 and timer[idx] > 0:
                activations += 1
        
        return activations
    except (KeyError, IndexError):
        return None


def _count_kuribo_shoe_uses(repetition_variables):
    """Count times Kuribo's shoe was acquired."""
    try:
        if "kuribo_shoe" not in repetition_variables:
            return None
        
        shoe = repetition_variables["kuribo_shoe"]
        acquisitions = 0
        
        for idx in range(1, len(shoe)):
            if shoe[idx - 1] == 0 and shoe[idx] == 1:
                acquisitions += 1
        
        return acquisitions
    except (KeyError, IndexError):
        return None


def _safe_get_first(variables, key):
    """Safely get first element of a variable, returns None if unavailable."""
    try:
        return variables[key][0]
    except (KeyError, IndexError):
        return None


def _safe_get_last(variables, key):
    """Safely get last element of a variable, returns None if unavailable."""
    try:
        return variables[key][-1]
    except (KeyError, IndexError):
        return None


def _safe_diff(variables, key):
    """Safely compute difference between first and last elements, returns None if unavailable."""
    first = _safe_get_first(variables, key)
    last = _safe_get_last(variables, key)
    if first is not None and last is not None:
        return last - first
    return None


def _get_final_timer(repetition_variables):
    """Get the final timer value as a combined integer (e.g., 245 for 2:45)."""
    try:
        hundreds = _safe_get_last(repetition_variables, "timer_100") or 0
        tens = _safe_get_last(repetition_variables, "timer_10") or 0
        ones = _safe_get_last(repetition_variables, "timer_1") or 0
        return hundreds * 100 + tens * 10 + ones
    except:
        return None


def _get_final_powerup_state(repetition_variables):
    """
    Get the final powerup state as a human-readable string.
    
    powerup values:
    0 = Small, 1 = Big, 2 = Fire, 3 = Raccoon, 4 = Frog, 5 = Tanooki, 6 = Hammer
    """
    try:
        powerup = _safe_get_last(repetition_variables, "powerup")
        if powerup is not None:
            powerup_names = {
                0: "small", 1: "big", 2: "fire", 3: "raccoon",
                4: "frog", 5: "tanooki", 6: "hammer"
            }
            return powerup_names.get(powerup, f"unknown_{powerup}")
    except:
        pass
    return None


def _get_max_p_meter(repetition_variables):
    """Get the maximum P-meter value reached during the replay."""
    try:
        if "p_meter" not in repetition_variables:
            return None
        return max(repetition_variables["p_meter"])
    except (KeyError, ValueError):
        return None


def create_sidecar_dict(repetition_variables):
    """
    Create JSON sidecar metadata from replay variables.

    Extracts high-level statistics from frame-by-frame game data.
    Metrics that require unavailable variables are set to None.

    Args:
        repetition_variables: Dictionary with per-frame game variables

    Returns:
        Dictionary with comprehensive summary statistics for the replay
    """
    # Calculate duration based on score frames
    try:
        n_frames = len(repetition_variables["score"])
        duration = n_frames / 60
    except KeyError:
        n_frames = None
        duration = None

    # Calculate distance and speed
    distance = _calculate_distance_traveled(repetition_variables)
    average_speed = None
    if distance is not None and duration is not None and duration > 0:
        average_speed = distance / duration

    # Determine outcome
    outcome = _determine_outcome(repetition_variables)

    # Calculate score (multiply by 10 to get correct value)
    score_gained = _safe_diff(repetition_variables, "score")
    score = score_gained * 10 if score_gained is not None else None




    # Build comprehensive result dict
    result = {
        # === Timing ===
        "Duration_seconds": round(duration, 3) if duration else None,
        "Frame_count": n_frames,
        "Timer_final": _get_final_timer(repetition_variables),
        
        # === Outcome ===
        "Outcome": outcome,  # 'cleared', 'death', 'timeout', 'unknown'
        
        # === Score & Progression ===
        "Score": score,
        
        # === Movement ===
        "X_traveled": distance,
        "Average_speed": round(average_speed, 2) if average_speed else None,
        "Max_p_meter": _get_max_p_meter(repetition_variables),
        
        # === Lives ===

        
        # === Combat ===
        "Hits_taken": count_hits_taken(repetition_variables, outcome),
        "Enemies_stomped": count_stomps(repetition_variables),
        
        # === Items ===
        "Coins": _safe_diff(repetition_variables, "coins_p1"),
        "Powerups_collected": count_powerups_collected(repetition_variables),
        "Stars_collected": count_star_power_activations(repetition_variables),
        "Bricks_smashed": count_bricks_smashed(repetition_variables),
        
        # === Player State ===
        "Player_form_final": _get_final_powerup_state(repetition_variables),
        
        # === SMB3-Specific ===
        "Flights_activated": count_flight_activations(repetition_variables),
        "Kuribo_shoe_acquired": _count_kuribo_shoe_uses(repetition_variables),
    }

    return result


# ============================================================================
# Main replay processing functions
# ============================================================================

def _extract_subject_from_bk2(bk2_file):
    """Extract subject ID from bk2 filename."""
    return bk2_file.split("/")[-1].split("_")[0]


def _extract_session_from_bk2(bk2_file):
    """Extract session ID from bk2 filename."""
    return bk2_file.split("/")[-1].split("_")[1]


def _extract_level_from_bk2(bk2_file):
    """Extract level ID from bk2 filename."""
    return bk2_file.split("/")[-1].split("_")[4].split("-")[1]


def get_passage_order(bk2_df):
    """
    Sort replays and assign global and level-specific indices.

    Args:
        bk2_df: DataFrame with replay data including 'bk2_file' column

    Returns:
        DataFrame with added subject, session, level, global_idx, and level_idx columns
    """
    bk2_df["subject"] = [
        _extract_subject_from_bk2(x) for x in bk2_df["bk2_file"].values
    ]
    bk2_df["session"] = [
        _extract_session_from_bk2(x) for x in bk2_df["bk2_file"].values
    ]
    bk2_df["level"] = [_extract_level_from_bk2(x) for x in bk2_df["bk2_file"].values]

    bk2_df = bk2_df.sort_values(["subject", "session", "run", "idx_in_run"]).assign(
        global_idx=lambda x: x.groupby("subject").cumcount()
    )
    bk2_df = bk2_df.sort_values(
        ["subject", "level", "session", "run", "idx_in_run"]
    ).assign(level_idx=lambda x: x.groupby(["subject", "level"]).cumcount())
    return bk2_df.sort_values(["subject", "global_idx"])


def _setup_stimuli_path(args, data_path):
    """Set up and register stimuli path with retro."""
    if args.stimuli is None:
        stimuli_path = op.abspath(op.join(data_path, "stimuli"))
    else:
        stimuli_path = op.abspath(args.stimuli)
    logging.debug(f"Adding stimuli path: {stimuli_path}")
    stable_retro.data.Integrations.add_custom_path(stimuli_path)


def _validate_bk2_file(bk2_file, bk2_path):
    """Check if bk2 file is valid and exists."""
    if bk2_file == "Missing file" or isinstance(bk2_path, float):
        return False
    if not op.exists(bk2_path):
        logging.error(f"File not found: {bk2_path}")
        return False
    return True


def _check_outputs_exist(paths, args):
    """
    Check which output files already exist.

    Returns:
        tuple: (all_exist, missing_outputs) where all_exist is bool and
               missing_outputs is list of output types that need to be generated
    """
    missing = []

    # JSON is always required
    if not op.exists(paths["json"]):
        missing.append("json")

    # Check optional outputs (if not skipped)
    if not args.skip_videos and not op.exists(paths["mp4"]):
        missing.append("mp4")
    if not args.skip_variables and not op.exists(paths["variables"]):
        missing.append("variables")
    if not args.skip_lowlevel and not op.exists(paths["lowlevel"]):
        missing.append("lowlevel")

    return len(missing) == 0, missing


def _build_output_paths(output_folder, bk2_file, subject, session):
    """Build all output file paths for replay processing using flat gamelogs/ structure."""
    entities = bk2_file.split("/")[-1].split(".")[0]
    gamelogs_folder = op.join(output_folder, subject, session, "gamelogs")

    return {
        "mp4": op.join(gamelogs_folder, f"{entities}_recording.mp4"),
        "json": op.join(gamelogs_folder, f"{entities}_summary.json"),
        "variables": op.join(gamelogs_folder, f"{entities}_variables.json"),
        "lowlevel": op.join(gamelogs_folder, f"{entities}_lowlevel.npy"),
        "entities": entities,
    }


def _save_optional_outputs(
    args,
    paths,
    replay_frames,
    repetition_variables,
    audio_track,
    audio_rate,
):
    """Save video, variables, and lowlevel files if not skipped."""
    if not args.skip_videos:
        os.makedirs(os.path.dirname(paths["mp4"]), exist_ok=True)
        make_mp4(replay_frames, paths["mp4"], audio=audio_track, sample_rate=audio_rate)
        logging.info(f"Video saved to: {paths['mp4']}")

    if not args.skip_variables:
        os.makedirs(os.path.dirname(paths["variables"]), exist_ok=True)
        with open(paths["variables"], "w") as f:
            json.dump(repetition_variables, f)
        logging.info(f"Variables saved to: {paths['variables']}")

    if not args.skip_lowlevel:
        os.makedirs(os.path.dirname(paths["lowlevel"]), exist_ok=True)
        # Compute psychophysical low-level features (luminance, optical flow, audio envelope)
        luminance = compute_luminance(replay_frames)
        optical_flow = compute_optical_flow(replay_frames)
        audio_envelope = audio_envelope_per_frame(
            audio_track,
            sample_rate=audio_rate,
            frame_rate=60.0,
            frame_count=len(replay_frames),
        )

        lowlevel_dict = {
            "luminance": luminance,
            "optical_flow": optical_flow,
            "audio_envelope": audio_envelope,
        }
        np.save(paths["lowlevel"], lowlevel_dict)
        logging.info(f"Low-level features saved to: {paths['lowlevel']}")


def _create_and_save_sidecar(repetition_variables, task_metadata, paths):
    """Create and save JSON sidecar with replay metadata."""
    info_dict = create_sidecar_dict(repetition_variables)
    info_dict.update(
        {
            "IndexInRun": task_metadata["idx_in_run"],
            "Run": task_metadata["run"],
            "IndexGlobal": task_metadata["global_idx"],  # 0-indexed
            "IndexLevel": task_metadata["level_idx"],  # 0-indexed
            "Phase": task_metadata["phase"],
        }
    )

    os.makedirs(os.path.dirname(paths["json"]), exist_ok=True)
    with open(paths["json"], "w") as f:
        json.dump(info_dict, f)
    logging.info(f"JSON saved for: {paths['json']}")


def process_bk2_file(task, args):
    """
    Process a single .bk2 replay file.

    Extracts game data and creates JSON metadata sidecar.
    Optionally saves video, variables, and low-level features.

    Args:
        task: Tuple of (bk2_file, run, idx_in_run, phase, subject,
              session, level, global_idx, level_idx)
        args: Command-line arguments with processing options
    """
    game_name = "SuperMarioBros3-Nes"
    data_path = op.abspath(args.datapath)
    output_folder = op.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)
    # Set up stimuli path in each worker process for parallel processing
    _setup_stimuli_path(args, data_path)

    bk2_file, run, idx_in_run, phase, subject, session, level, global_idx, level_idx = (
        task
    )
    bk2_path = op.abspath(op.join(data_path, bk2_file))

    if not _validate_bk2_file(bk2_file, bk2_path):
        return

    paths = _build_output_paths(output_folder, bk2_file, subject, session)

    # Check if all required outputs already exist - skip if so
    all_exist, missing_outputs = _check_outputs_exist(paths, args)
    if all_exist:
        logging.info(f"Skipping (all outputs exist): {paths['entities']}")
        return
    else:
        logging.info(
            f"Processing {paths['entities']} (missing: {', '.join(missing_outputs)})"
        )

    # Get replay data with audio
    repetition_variables, _, replay_frames, audio_track, audio_rate = (
        get_variables_from_replay(
            op.join(data_path, bk2_file),
            skip_first_step=(idx_in_run == 0),
            game=game_name,
            inttype=stable_retro.data.Integrations.CUSTOM_ONLY,
        )
    )

    _save_optional_outputs(
        args,
        paths,
        replay_frames,
        repetition_variables,
        audio_track,
        audio_rate,
    )

    task_metadata = {
        "idx_in_run": idx_in_run,
        "run": run,
        "global_idx": global_idx,
        "level_idx": level_idx,
        "phase": phase,
        "level": level,
    }
    _create_and_save_sidecar(repetition_variables, task_metadata, paths)

    # Explicitly clear large data structures to free memory
    del replay_frames
    del repetition_variables
    if audio_track is not None:
        del audio_track
    # Force garbage collection to release memory immediately
    gc.collect()


def _configure_logging(verbose):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", force=True)


def _assign_phases_per_subject(bk2_df):
    """
    Assign discovery/practice phase to each replay based on level progression.
    
    Discovery phase: Levels are played in sequential order (first playthrough).
    Practice phase: Levels are played in random order (revisiting levels).
    
    The transition from discovery to practice occurs when we encounter a level
    that we've already visited AND it's not an immediate retry of the same level.
    
    Args:
        bk2_df: DataFrame sorted by subject and global_idx, with 'level' column
        
    Returns:
        DataFrame with 'phase' column added
    """
    phases = []
    
    for subject in bk2_df["subject"].unique():
        subject_df = bk2_df[bk2_df["subject"] == subject].sort_values("global_idx")
        
        visited_levels = set()
        previous_level = None
        current_phase = "discovery"
        
        for _, row in subject_df.iterrows():
            level = row["level"]
            
            if current_phase == "discovery":
                # Discovery continues if:
                # 1. This is a new level we haven't seen before, OR
                # 2. This is the same level as previous (immediate retry)
                if level in visited_levels and level != previous_level:
                    # We're revisiting an old level non-sequentially -> switch to practice
                    current_phase = "practice"
            
            phases.append(current_phase)
            visited_levels.add(level)
            previous_level = level
    
    bk2_df["phase"] = phases
    return bk2_df


def _extract_run_from_filename(filename):
    """Extract run ID from events file name."""
    return filename.split("_")[-2]


def _collect_bk2_info_from_events(run_events_file):
    """Collect bk2 file info from a single events file."""
    run = _extract_run_from_filename(op.basename(run_events_file))
    logging.info(f"Processing events file: {run_events_file}")

    try:
        events_df = pd.read_table(run_events_file)
    except Exception as e:
        logging.error(f"Cannot read {run_events_file}: {e}")
        return []

    # Filter to only rows with valid .bk2 stim_files BEFORE enumerating
    # This ensures idx_in_run correctly counts only actual game repetitions
    valid_bk2_mask = events_df["stim_file"].apply(
        lambda x: isinstance(x, str) and ".bk2" in x
    )
    bk2_files = events_df.loc[valid_bk2_mask, "stim_file"].values.tolist()

    bk2_list = []
    for idx_in_run, bk2_file in enumerate(bk2_files):
        bk2_list.append(
            {
                "bk2_file": bk2_file,
                "run": run,
                "idx_in_run": idx_in_run,
            }
        )
    return bk2_list


def _collect_all_bk2_files(data_path, subjects=None, sessions=None):
    """
    Walk dataset and collect all bk2 file information.

    Parameters
    ----------
    data_path : str
        Path to the mario3 dataset root directory
    subjects : list of str, optional
        List of subject IDs to process (e.g., ['sub-01', 'sub-02']).
        If None, processes all subjects.
    sessions : list of str, optional
        List of session IDs to process (e.g., ['ses-001', 'ses-002']).
        If None, processes all sessions.

    Returns
    -------
    list
        List of dicts containing bk2 file information
    """
    bk2_list = []
    for root, _, files in sorted(os.walk(data_path)):
        for file in files:
            if "events.tsv" in file and "annotated" not in file:
                # Check if this file matches subject filter
                if subjects is not None:
                    if not any(sub in root for sub in subjects):
                        continue

                # Check if this file matches session filter
                if sessions is not None:
                    if not any(ses in root for ses in sessions):
                        continue

                run_events_file = op.join(root, file)
                bk2_list.extend(_collect_bk2_info_from_events(run_events_file))
    return bk2_list


def _run_parallel_processing(tasks, args):
    """Process tasks in parallel using joblib."""
    with tqdm_joblib(tqdm(desc="Processing files", total=len(tasks))):
        Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
            delayed(process_bk2_file)(task, args) for task in tasks
        )


def _run_sequential_processing(tasks, args):
    """Process tasks sequentially with progress bar."""
    for task in tqdm(tasks, desc="Processing files"):
        process_bk2_file(task, args)


def main(args):
    """
    Main entry point for replay processing.

    Scans dataset for events files, collects bk2 file info,
    and processes each replay in parallel or sequentially.

    Args:
        args: Parsed command-line arguments
    """
    _configure_logging(args.verbose)
    data_path = op.abspath(args.datapath)

    # Set up stimuli path once before parallel processing to avoid race conditions
    _setup_stimuli_path(args, data_path)

    # Get subject/session filters if provided
    subjects = getattr(args, "subjects", None)
    sessions = getattr(args, "sessions", None)

    if subjects:
        logging.info(f"Filtering subjects: {', '.join(subjects)}")
    if sessions:
        logging.info(f"Filtering sessions: {', '.join(sessions)}")

    bk2_list = _collect_all_bk2_files(data_path, subjects=subjects, sessions=sessions)

    if not bk2_list:
        logging.warning("No bk2 files found to process. Check your datapath and ensure events.tsv files exist.")
        return

    bk2_df = pd.DataFrame(bk2_list)
    bk2_df = get_passage_order(bk2_df)
    bk2_df = _assign_phases_per_subject(bk2_df)
    
    # Reorder columns to match process_bk2_file expected order
    bk2_df = bk2_df[["bk2_file", "run", "idx_in_run", "phase", "subject", "session", "level", "global_idx", "level_idx"]]

    tasks = [tuple(row) for row in bk2_df.values]
    logging.info(f"Found {len(tasks)} bk2 files to process.")

    n_jobs = os.cpu_count() if args.n_jobs == -1 else args.n_jobs
    logging.info(f"Using {n_jobs} parallel jobs")

    if n_jobs != 1:
        _run_parallel_processing(tasks, args)
    else:
        _run_sequential_processing(tasks, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datapath",
        default=".",
        type=str,
        help="Data path to look for events.tsv and .bk2 files. Should be the root of the mario3 dataset.",
    )
    parser.add_argument(
        "-s",
        "--stimuli",
        default=None,
        type=str,
        help="Data path to look for the stimuli files (rom, state files, data.json etc...).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=".",
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        default=1,
        type=int,
        help="Number of parallel jobs to run. Use -1 to use all available cores.",
    )
    parser.add_argument(
        "--skip_videos",
        action="store_true",
        help="Skip generating the playback video file (_recording.mp4).",
    )
    parser.add_argument(
        "--skip_variables",
        action="store_true",
        help="Skip generating the variables file (_variables.json) that contains game variables.",
    )
    parser.add_argument(
        "--skip_lowlevel",
        action="store_true",
        help="Skip generating low-level features (_lowlevel.npy) - luminance, optical flow, audio envelope.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display verbose output.",
    )
    parser.add_argument(
        "--subjects",
        "-sub",
        nargs="+",
        default=None,
        help="List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.",
    )
    parser.add_argument(
        "--sessions",
        "-ses",
        nargs="+",
        default=None,
        help="List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.",
    )

    args = parser.parse_args()

    # Main loop
    main(args)
