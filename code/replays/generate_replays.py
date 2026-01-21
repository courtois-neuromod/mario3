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
# Mario 3-specific utility functions
# ============================================================================

def _calculate_world_and_level(level_str):
    """Extract world and level identifiers from level string."""
    # Mario 3 uses naming like "w1lFortress", "w7lPiranhaPlant2", etc.
    # Extract world number and level identifier
    try:
        if level_str.startswith('w') and 'l' in level_str:
            parts = level_str.split('l')
            world = parts[0][1:]  # Remove 'w' prefix
            level = parts[1] if len(parts) > 1 else None
            return world, level
    except:
        pass
    return None, None


def _calculate_distance_traveled(repetition_variables):
    """Calculate total X distance traveled. Returns None if required variables missing."""
    try:
        start_x = (repetition_variables["xscrollLo"][0] +
                   (256 * repetition_variables["xscrollHi"][0]))
        end_x = (repetition_variables["xscrollLo"][-1] +
                 (256 * repetition_variables["xscrollHi"][-1]))
        return end_x - start_x
    except KeyError:
        return None


def _check_level_cleared(repetition_variables):
    """Determine if level was successfully cleared. Returns None if required variables missing."""
    try:
        if repetition_variables["player_y_screen"][-1] > 1:
            return False
        if repetition_variables["lives"][-1] == -1:
            return False
        if repetition_variables["player_state"][-1] in [6, 11]:
            return False
        return True
    except KeyError:
        return None


def _count_enemy_kills_for_slot(repetition_variables, slot_idx):
    """Count kills for a specific enemy slot. Returns 0 if required variables missing."""
    try:
        kill_count = 0
        enemy_key = f"enemy_kill3{slot_idx}"

        for idx, val in enumerate(repetition_variables[enemy_key][:-1]):
            if val in [4, 34, 132]:
                if repetition_variables[enemy_key][idx + 1] != val:
                    if slot_idx == 5 and repetition_variables["powerup_yes_no"] == 0:
                        kill_count += 1
                    elif slot_idx != 5:
                        kill_count += 1
        return kill_count
    except KeyError:
        return 0


def count_kills(repetition_variables):
    """Count total enemies killed during replay. Returns None if no enemy slots available."""
    kills = sum(_count_enemy_kills_for_slot(repetition_variables, i) for i in range(6))
    # If all slots returned 0 due to missing keys, we can't determine kills
    # Check if at least one enemy key exists
    if not any(f"enemy_kill3{i}" in repetition_variables for i in range(6)):
        return None
    return kills


def count_bricks_destroyed(repetition_variables):
    """Count bricks destroyed by jumping. Returns None if required variables missing."""
    try:
        score_increments = list(np.diff(repetition_variables["score"]))
        bricks_destroyed = 0

        for idx, inc in enumerate(score_increments):
            if inc == 5 and repetition_variables["jump_airborne"][idx] == 1:
                bricks_destroyed += 1
        return bricks_destroyed
    except KeyError:
        return None


def _count_powerstate_hits(repetition_variables):
    """Count hits from powerstate changes. Returns None if required variables missing."""
    try:
        diff_state = list(np.diff(repetition_variables["powerstate"]))
        return sum(1 for val in diff_state if val < -10000)
    except KeyError:
        return None


def _count_life_losses(repetition_variables):
    """Count hits from life losses. Returns None if required variables missing."""
    try:
        diff_lives = list(np.diff(repetition_variables["lives"]))
        return sum(1 for val in diff_lives if val < 0)
    except KeyError:
        return None


def count_hits_taken(repetition_variables):
    """Count total hits taken (damage + deaths). Returns None if required variables missing."""
    powerstate_hits = _count_powerstate_hits(repetition_variables)
    life_losses = _count_life_losses(repetition_variables)
    if powerstate_hits is None and life_losses is None:
        return None
    return (powerstate_hits or 0) + (life_losses or 0)


def count_powerups_collected(repetition_variables):
    """Count powerups collected during replay. Returns None if required variables missing."""
    try:
        powerup_count = 0

        for idx, val in enumerate(repetition_variables["player_state"][:-1]):
            if val in [9, 12, 13]:
                if repetition_variables["player_state"][idx + 1] != val:
                    powerup_count += 1
        return powerup_count
    except KeyError:
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


def create_sidecar_dict(repetition_variables):
    """
    Create JSON sidecar metadata from replay variables.

    Extracts high-level statistics from frame-by-frame game data.
    Metrics that require unavailable variables are set to None.

    Args:
        repetition_variables: Dictionary with per-frame game variables

    Returns:
        Dictionary with summary statistics for the replay
    """
    # Calculate world/level - these should always be available
    try:
        world, level = _calculate_world_and_level(repetition_variables["level"])
    except KeyError:
        world, level = None, None

    # Calculate duration based on score frames (fallback to None)
    try:
        duration = len(repetition_variables["score"]) / 60
    except KeyError:
        duration = None

    # Calculate distance and speed
    distance = _calculate_distance_traveled(repetition_variables)
    average_speed = None
    if distance is not None and duration is not None and duration > 0:
        average_speed = distance / duration

    # Build result dict with all available metrics
    result = {
        "Subject": repetition_variables.get("subject"),
        "World": world,
        "Level": level,
        "Duration": duration,
        "Cleared": _check_level_cleared(repetition_variables),
        "ScoreGained": _safe_diff(repetition_variables, "score"),
        "X_Traveled": distance,
        "Average_speed": average_speed,
        "Lives_lost": None,
        "Hits_taken": count_hits_taken(repetition_variables),
        "Enemies_killed": count_kills(repetition_variables),
        "Powerups_collected": count_powerups_collected(repetition_variables),
        "Bricks_destroyed": count_bricks_destroyed(repetition_variables),
        "CoinsGained": _safe_diff(repetition_variables, "coins"),
    }

    # Calculate lives lost if available
    lives_first = _safe_get_first(repetition_variables, "lives")
    lives_last = _safe_get_last(repetition_variables, "lives")
    if lives_first is not None and lives_last is not None:
        result["Lives_lost"] = lives_first - lives_last

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
        "json": op.join(gamelogs_folder, f"{entities}.json"),
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
            "IndexGlobal": task_metadata["global_idx"],
            "IndexLevel": task_metadata["level_idx"],
            "Phase": task_metadata["phase"],
            "LevelFullName": task_metadata["level"],
            "Bk2File": paths["entities"],
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


def _determine_phase(events_dataframe):
    """Determine if replay is discovery or practice phase."""
    unique_levels = len(np.unique(events_dataframe["level"].dropna()))
    return "discovery" if unique_levels == 1 else "practice"


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

    phase = _determine_phase(events_df)
    bk2_files = events_df["stim_file"].values.tolist()

    bk2_list = []
    for idx_in_run, bk2_file in enumerate(bk2_files):
        if isinstance(bk2_file, str) and ".bk2" in bk2_file:
            bk2_list.append(
                {
                    "bk2_file": bk2_file,
                    "run": run,
                    "idx_in_run": idx_in_run,
                    "phase": phase,
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
