#!/usr/bin/env python
"""
Generate annotated event files for the Mario 3 dataset from replay variables.

This script reads game variables from replay processing and generates detailed
BIDS-compatible event files containing:
  - Button press events (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT)
  - Kill events (stomp) via stomp_counter
  - Hit events (life lost, powerup lost, timeout) via lives and powerup
  - Brick smashing events via score increments
  - Coin collection events via coins variable
  - Powerup collection events via powerup variable increases
  - Star power events via invincibility_timer
  - Flight events via flight_timer
  - P-Switch events via p_switch_timer

Usage:
    python generate_annotations.py

    Or with explicit paths:
    python generate_annotations.py --datapath /path/to/mario3

Note: Requires replay files (_variables.json) in gamelogs/ folders.
      Run create_replays.py first if they don't exist.
"""

import argparse
import os
import os.path as op
import stable_retro
import pandas as pd
import numpy as np
import json


def create_runevents(runvars, run_id, events_dataframe, FS=60):
    """Create a BIDS compatible events dataframe from game variables and start/duration info of repetitions

    Parameters
    ----------
    runvars : list
        A list of repvars dicts, corresponding to the different repetitions of a run. Each repvar must have it's own duration and onset.
    events_dataframe : pandas.DataFrame
        A BIDS-formatted DataFrame specifying the onset and duration of each repetition.
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format.
    """
    all_df = [events_dataframe]
    for idx, repvars in enumerate(runvars):
        n_frames_total = len(repvars["START"])
        repvars["rep_onset"] = [events_dataframe["onset"][idx]]
        repvars["rep_duration"] = n_frames_total / FS
        rep_index = events_dataframe['rep_index'].iloc[idx]

        if len(repvars.keys()) > 0:  # Check if repetition logs are available
            # Actions - button inputs are always available from replay file
            ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
            for act in ACTIONS:
                temp_df = generate_key_events(repvars, act, FS=FS)
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Kills
            temp_df = generate_kill_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Hits taken
            temp_df = generate_hits_taken_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Bricks smashed
            temp_df = generate_bricks_smashed_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Coins collected
            temp_df = generate_coin_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Powerups
            temp_df = generate_powerup_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Star power (SMB3-specific)
            temp_df = generate_star_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # Flight (SMB3-specific)
            temp_df = generate_flight_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

            # P-Switch (SMB3-specific)
            temp_df = generate_pswitch_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                temp_df["rep_index"] = rep_index
                all_df.append(temp_df)

    try:
        events_df = pd.concat(all_df).sort_values(by="onset").reset_index(drop=True)

        # Round onset and duration to 3 decimal places
        events_df['onset'] = events_df['onset'].round(3)
        events_df['duration'] = events_df['duration'].round(3)

        # Ensure integer types for frame columns and rep_index
        for col in ['frame_start', 'frame_stop', 'rep_index']:
            if col in events_df.columns:
                events_df[col] = events_df[col].astype('Int64')  # nullable integer

        # Reorder columns: trial_type, rep_index, level, onset, duration, frame_start, frame_stop, phase
        cols = events_df.columns.tolist()
        priority_cols = ['trial_type', 'rep_index', 'level', 'onset', 'duration', 'frame_start', 'frame_stop', 'phase']
        priority_cols = [c for c in priority_cols if c in cols]  # only include existing columns
        other_cols = [c for c in cols if c not in priority_cols]
        events_df = events_df[priority_cols + other_cols]

    except ValueError:
        print("No bk2 files available for this run. Returning empty df.")
        events_df = pd.DataFrame()
    return events_df


def generate_key_events(repvars, key, FS=60):
    """Create a BIDS compatible events dataframe containing key (actions) events

    Parameters
    ----------
    repvars : list
        A dict containing all the variables of a single repetition
    key : string
        Name of the action variable to process
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format containing the
        corresponding action events.
    """

    var = np.multiply(repvars[key], 1)
    # always keep the first and last value as 0 so diff will register the state transition
    var[0] = 0
    var[-1] = 0

    var_bin = [int(val) for val in var]
    diffs = list(np.diff(var_bin, n=1))
    presses = [round(i / FS, 3) for i, x in enumerate(diffs) if x == 1]
    releases = [round(i / FS, 3) for i, x in enumerate(diffs) if x == -1]
    frame_start = [i for i, x in enumerate(diffs) if x == 1]
    frame_stop = [i for i, x in enumerate(diffs) if x == -1]
    onset = presses
    level = [repvars["level"] for x in onset]
    duration = [round(releases[i] - presses[i], 3) for i in range(len(presses))]
    trial_type = ["{}".format(key) for i in range(len(presses))]
    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_kill_events(repvars, FS=60):
    """Create a BIDS compatible events dataframe containing kill events.

    Super Mario Bros 3 uses the stomp_counter variable to track enemy kills.
    The stomp_counter increments when Mario stomps on enemies.

    Parameters
    ----------
    repvars : dict
        A dict containing all the variables of a single repetition.
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df :
        An events DataFrame in BIDS-compatible format containing the
        kill events.
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    # Check if stomp_counter is available
    if "stomp_counter" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    stomps = repvars["stomp_counter"]
    
    # Detect when stomp_counter increases
    for frame_idx in range(1, len(stomps)):
        if stomps[frame_idx] > stomps[frame_idx - 1]:
            # Each increment is a stomp kill
            num_kills = stomps[frame_idx] - stomps[frame_idx - 1]
            for _ in range(num_kills):
                onset.append(frame_idx / FS)
                duration.append(0)
                trial_type.append("Kill/stomp")
                level.append(repvars["level"])
                frame_start.append(frame_idx)
                frame_stop.append(frame_idx)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_hits_taken_events(repvars, FS=60):
    """Generate events for when Mario takes damage or loses a life.

    Super Mario Bros 3 hit detection:
    - Powerup lost: powerup variable decreases (e.g., 3→1 means lost raccoon)
    - Life lost: lives counter decreases
    - Timeout: timer reaches 000 (detected via timer_100/10/1)

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    # Track frames where we detected a life loss (to avoid double-counting)
    life_loss_frames = set()

    # Powerup lost (powerup value decreased)
    if "powerup" in repvars:
        powerups = repvars["powerup"]
        for frame_idx in range(1, len(powerups)):
            if powerups[frame_idx] < powerups[frame_idx - 1]:
                onset.append(frame_idx / FS)
                duration.append(0)
                trial_type.append("Hit/powerup_lost")
                level.append(repvars["level"])
                frame_start.append(frame_idx)
                frame_stop.append(frame_idx)

    # Life lost
    if "lives" in repvars:
        diff_lives = list(np.diff(repvars["lives"]))
        for idx_val, val in enumerate(diff_lives):
            if val < 0:
                frame_idx = idx_val + 1  # diff shifts by 1
                
                # Check if this is a timeout death
                is_timeout = False
                if "timer_100" in repvars and "timer_10" in repvars and "timer_1" in repvars:
                    # Look at timer a few frames before death
                    check_frame = max(0, frame_idx - 5)
                    timer_h = repvars["timer_100"][check_frame]
                    timer_t = repvars["timer_10"][check_frame]
                    timer_o = repvars["timer_1"][check_frame]
                    if timer_h == 0 and timer_t == 0 and timer_o <= 1:
                        is_timeout = True
                
                if is_timeout:
                    onset.append(frame_idx / FS)
                    duration.append(0)
                    trial_type.append("Hit/timeout")
                    level.append(repvars["level"])
                    frame_start.append(frame_idx)
                    frame_stop.append(frame_idx)
                else:
                    onset.append(frame_idx / FS)
                    duration.append(0)
                    trial_type.append("Hit/life_lost")
                    level.append(repvars["level"])
                    frame_start.append(frame_idx)
                    frame_stop.append(frame_idx)
                
                life_loss_frames.add(frame_idx)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_bricks_smashed_events(repvars, FS=60):
    """Generate events for when Mario smashes bricks.

    In Super Mario Bros 3, brick breaking gives 10 points.
    We detect score increments of 10 while in_air flag is set.

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """

    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    # Check if required variables are available
    if "score" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    score_increments = list(np.diff(repvars["score"]))
    
    # In SMB3, brick breaking gives 10 points
    for idx_val, inc in enumerate(score_increments):
        if inc == 10:
            # Check if in_air flag is set (if available)
            if "in_air" in repvars:
                if repvars["in_air"][idx_val] != 0:
                    onset.append(idx_val / FS)
                    duration.append(0)
                    trial_type.append("Brick_smashed")
                    level.append(repvars["level"])
                    frame_start.append(idx_val)
                    frame_stop.append(idx_val)
            else:
                # If in_air not available, still record the event
                onset.append(idx_val / FS)
                duration.append(0)
                trial_type.append("Brick_smashed")
                level.append(repvars["level"])
                frame_start.append(idx_val)
                frame_stop.append(idx_val)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_coin_events(repvars, FS=60):
    """Generate events for coin collection.

    Detected by increase in coins counter.

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    # Check if coins variable is available
    if "coins" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    diff_coins = np.diff(repvars["coins"])
    for idx_val, val in enumerate(diff_coins):
        if val > 0:
            onset.append(idx_val / FS)
            duration.append(0)
            trial_type.append("Coin_collected")
            level.append(repvars["level"])
            frame_start.append(idx_val)
            frame_stop.append(idx_val)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_powerup_events(repvars, FS=60):
    """Generate events for powerup collection.

    Super Mario Bros 3 powerup detection:
    - Powerup gained when 'powerup' variable increases
    - Types: 0=small, 1=big, 2=fire, 3=raccoon, 4=frog, 5=tanooki, 6=hammer

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    # Check if powerup variable is available
    if "powerup" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    powerup_names = {
        0: "small", 1: "mushroom", 2: "fire_flower", 3: "leaf",
        4: "frog_suit", 5: "tanooki_suit", 6: "hammer_suit"
    }

    powerups = repvars["powerup"]
    
    for idx in range(1, len(powerups)):
        prev_val = powerups[idx - 1]
        curr_val = powerups[idx]
        
        if curr_val > prev_val:
            # Determine powerup type based on new value
            powerup_type = powerup_names.get(curr_val, f"unknown_{curr_val}")
            onset.append(idx / FS)
            duration.append(0)
            trial_type.append(f"Powerup/{powerup_type}")
            level.append(repvars["level"])
            frame_start.append(idx)
            frame_stop.append(idx)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_star_events(repvars, FS=60):
    """Generate events for star power activation.

    Detected by invincibility_timer going from 0 to >0.

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    if "invincibility_timer" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    timer = repvars["invincibility_timer"]
    
    for idx in range(1, len(timer)):
        if timer[idx - 1] == 0 and timer[idx] > 0:
            onset.append(idx / FS)
            duration.append(0)
            trial_type.append("Powerup/star")
            level.append(repvars["level"])
            frame_start.append(idx)
            frame_stop.append(idx)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_flight_events(repvars, FS=60):
    """Generate events for flight activation (Raccoon/Tanooki Mario).

    Detected by flight_timer going from 0 to >0.

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    if "flight_timer" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    timer = repvars["flight_timer"]
    
    for idx in range(1, len(timer)):
        if timer[idx - 1] == 0 and timer[idx] > 0:
            onset.append(idx / FS)
            duration.append(0)
            trial_type.append("Flight_started")
            level.append(repvars["level"])
            frame_start.append(idx)
            frame_stop.append(idx)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def generate_pswitch_events(repvars, FS=60):
    """Generate events for P-Switch activation.

    Detected by p_switch_timer going from 0 to >0.

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file

    Returns
    -------
    events_df : pandas.DataFrame
        Events DataFrame in BIDS-compatible format
    """
    onset = []
    duration = []
    trial_type = []
    level = []
    frame_start = []
    frame_stop = []

    if "p_switch_timer" not in repvars:
        return pd.DataFrame(
            data={
                "onset": onset,
                "duration": duration,
                "trial_type": trial_type,
                "level": level,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
            }
        )

    timer = repvars["p_switch_timer"]
    
    for idx in range(1, len(timer)):
        if timer[idx - 1] == 0 and timer[idx] > 0:
            onset.append(idx / FS)
            duration.append(0)
            trial_type.append("P-Switch_activated")
            level.append(repvars["level"])
            frame_start.append(idx)
            frame_stop.append(idx)

    events_df = pd.DataFrame(
        data={
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "level": level,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
        }
    )
    return events_df


def main(args):
    FS = 60

    # Get datapath
    DATA_PATH = args.datapath
    if DATA_PATH == ".":
        print("No data path specified. Searching files in this folder.")
    print(f"Generating annotations for the mario3 dataset in : {DATA_PATH}")
    # Import stimuli
    stimuli_path = op.join(DATA_PATH, "stimuli")
    stable_retro.data.Integrations.add_custom_path(stimuli_path)

    OUTPUT_PATH = args.output_path

    # Get subject/session filters
    subjects = args.subjects
    sessions = args.sessions

    if subjects:
        print(f"Filtering subjects: {', '.join(subjects)}")
    if sessions:
        print(f"Filtering sessions: {', '.join(sessions)}")

    # Walk through all folders looking for events.tsv files
    for root, folder, files in sorted(os.walk(DATA_PATH)):
        if not "sourcedata" in root:
            # Check if this path matches subject filter
            if subjects is not None:
                if not any(sub in root for sub in subjects):
                    continue

            # Check if this path matches session filter
            if sessions is not None:
                if not any(ses in root for ses in sessions):
                    continue

            for file in files:
                if "events.tsv" in file and not "annotated" in file:
                    run_events_file = op.join(root, file)
                    run_id = file.split("_")[3]
                    if OUTPUT_PATH is not None:
                        sub = file.split("_")[0]
                        ses = file.split("_")[1]
                        events_annotated_fname = op.join(
                            OUTPUT_PATH,
                            sub,
                            ses,
                            "func",
                            file.replace("_events.", "_desc-annotated_events."),
                        )
                        os.makedirs(op.dirname(events_annotated_fname), exist_ok=True)
                    else:
                        events_annotated_fname = run_events_file.replace(
                            "_events.", "_desc-annotated_events."
                        )
                    if not op.isfile(events_annotated_fname):
                        print(f"Processing : {file}")
                        events_dataframe = pd.read_table(run_events_file, index_col=0)
                        events_dataframe = events_dataframe[
                            events_dataframe["trial_type"] == "gym-retro_game"
                        ]  # select only repetition events
                        events_dataframe = events_dataframe[
                            ["trial_type", "onset", "level", "stim_file"]
                        ].reset_index()  # select only relevant columns
                        bk2_files = events_dataframe["stim_file"].values.tolist()
                        runvars = []
                        for bk2_idx, bk2_file in enumerate(bk2_files):
                            if bk2_file != "Missing file" and type(bk2_file) != float:
                                print("Adding : " + bk2_file)
                                sub = bk2_file.split("/")[0]
                                ses = bk2_file.split("/")[1]
                                filename = bk2_file.split("/")[-1]
                                # Look for variables file in gamelogs/ within datapath
                                variables_sidecar_fname = op.join(
                                    DATA_PATH,
                                    sub,
                                    ses,
                                    "gamelogs",
                                    filename.replace(".bk2", "_variables.json"),
                                )
                                if op.exists(variables_sidecar_fname):
                                    with open(variables_sidecar_fname, "r") as f:
                                        repvars = json.load(f)

                                    # Add info to repetition event
                                    events_dataframe.loc[
                                        events_dataframe["stim_file"] == bk2_file,
                                        "level",
                                    ] = repvars[
                                        "level"
                                    ]  # replace level value in the dataframe by the one in the repvars dict
                                    events_dataframe.loc[
                                        events_dataframe["stim_file"] == bk2_file,
                                        "frame_start",
                                    ] = int(0)

                                    # Safely get score length or use default
                                    if "score" in repvars:
                                        frame_count = int(len(repvars["score"]))
                                    elif "START" in repvars:
                                        frame_count = int(len(repvars["START"]))
                                    else:
                                        frame_count = 0

                                    events_dataframe.loc[
                                        events_dataframe["stim_file"] == bk2_file,
                                        "frame_stop",
                                    ] = frame_count
                                    events_dataframe.loc[
                                        events_dataframe["stim_file"] == bk2_file,
                                        "duration",
                                    ] = frame_count / FS

                                    # rename index column to rep_index
                                    events_dataframe.rename(
                                        columns={"index": "rep_index"}, inplace=True
                                    )

                                    runvars.append(repvars)
                                else:
                                    print(f"\nError: Variables file not found: {variables_sidecar_fname}")
                                    print("Please run create_replays.py first to generate the required files.")
                                    return
                            else:
                                print("Missing file, skipping")
                                runvars.append({})

                        # Add phase (discovery VS practice)
                        if (
                            events_dataframe["level"].values[0]
                            == events_dataframe["level"].values[1]
                        ):
                            phase = "discovery"
                        else:
                            phase = "practice"
                        events_dataframe["phase"] = phase
                        events_df = create_runevents(
                            runvars, run_id, events_dataframe, FS=FS
                        )

                        events_df.to_csv(events_annotated_fname, sep="\t", index=False)
                        print(f"Saved annotated events to: {events_annotated_fname}")


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
        "-o",
        "--output_path",
        default=None,
        type=str,
        help="Path to save the annotated events files. If not specified, saves in the same folder as the input events.tsv files.",
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
    main(args)
