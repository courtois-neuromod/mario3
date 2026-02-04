#!/usr/bin/env python
"""
Generate annotated event files for the Mario 3 dataset from replay variables.

This script reads game variables from replay processing and generates detailed
BIDS-compatible event files containing:
  - Button press events (UP, DOWN, LEFT, RIGHT, A, B, START, SELECT)
  - Kill events (stomp) via stomp_counter
  - Hit events (powerup lost, killed) via powerup/outcome
  - Brick smashing events via score increments (1 point)
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
      Run generate_replays.py first if they don't exist.
"""

import argparse
import os
import os.path as op
import stable_retro
import pandas as pd
import numpy as np
import json



def _determine_outcome(repetition_variables):
    """
    Determine how the replay ended: 'cleared' or 'failed/*'.
    
    Outcomes:
    - cleared: Level completed successfully (complete_level hits 1 AND killed is 0 at that frame)
    - failed/fall: Last 100 frames of player_x_level_low are all 0
    - failed/timeout: Timer reaches 0
    - failed/killed: Killed by enemy (default failure if not fall/timeout)
    """
    try:
        # Check if level was ever completed
        if "complete_level" in repetition_variables:
            complete_indices = [i for i, x in enumerate(repetition_variables["complete_level"]) if x == 1]
            
            if complete_indices:
                idx = complete_indices[0]
                
                # Check Killed at the frame of completion FIRST
                # If killed is 0 when complete_level becomes 1, the level was cleared
                if "killed" in repetition_variables:
                    is_killed = (repetition_variables["killed"][idx] == 1)
                    
                    if not is_killed:
                        # complete_level=1 and killed=0 means level was successfully cleared
                        return "cleared"
                    
                    # If killed=1 at completion, determine failure type
                    # Check Timer at the frame of completion
                    t_h = repetition_variables["timer_100"][idx]
                    t_t = repetition_variables["timer_10"][idx]
                    t_o = repetition_variables["timer_1"][idx]
                    timer_at_completion = t_h * 100 + t_t * 10 + t_o
                    
                    if timer_at_completion == 0:
                        return "failed/timeout"
                    
                    # Determine if Fall or Killed
                    if "player_x_level_low" in repetition_variables:
                        x_low = repetition_variables["player_x_level_low"]
                        last_segment = x_low[-100:] if len(x_low) > 0 else []
                        if last_segment and all(v == 0 for v in last_segment):
                            return "failed/fall"
                    return "failed/killed"
                
                return "cleared"
        
        # If no completion detected, fall back to end-of-replay checks
        
        # Check for Timeout (at end)
        if "timer_100" in repetition_variables:
            timer_h = repetition_variables["timer_100"][-1]
            timer_t = repetition_variables["timer_10"][-1]
            timer_o = repetition_variables["timer_1"][-1]
            timer = timer_h * 100 + timer_t * 10 + timer_o
                
            if timer == 0:
                return "failed/timeout"
            
        # Check for Fall vs Killed
        if "player_x_level_low" in repetition_variables:
            x_low = repetition_variables["player_x_level_low"]
            last_segment = x_low[-100:] if len(x_low) > 0 else []
            
            if last_segment and all(v == 0 for v in last_segment):
                return "failed/fall"

        return "failed/killed"
        
    except (KeyError, IndexError):
        return "unknown"


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

        if len(repvars.keys()) > 0:  # Check if repetition logs are available
            # Actions - button inputs are always available from replay file
            ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]
            for act in ACTIONS:
                temp_df = generate_key_events(repvars, act, FS=FS)
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Kills
            temp_df = generate_kill_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Hits taken
            temp_df = generate_hits_taken_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Bricks smashed
            temp_df = generate_bricks_smashed_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Coins collected
            temp_df = generate_coin_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Powerups
            temp_df = generate_powerup_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Star power (SMB3-specific)
            temp_df = generate_star_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Flight (SMB3-specific)
            temp_df = generate_flight_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # P-Switch (SMB3-specific)
            temp_df = generate_pswitch_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
                all_df.append(temp_df)

            # Level complete
            temp_df = generate_level_complete_events(repvars, FS=FS)
            if not temp_df.empty:
                temp_df["onset"] = temp_df["onset"] + repvars["rep_onset"]
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

        # Reorder columns: trial_type, level, onset, duration, frame_start, frame_stop, phase, rep_index, stim_file
        cols = events_df.columns.tolist()
        priority_cols = ['trial_type', 'level', 'onset', 'duration', 'frame_start', 'frame_stop', 'phase', 'rep_index', 'stim_file']
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
    duration = [round(releases[i] - presses[i], 3) for i in range(len(presses))]
    
    event_name = key
    if key == "A":
        event_name = "JUMP"
    elif key == "B":
        event_name = "RUN/THROW"
        
    trial_type = ["{}".format(event_name) for i in range(len(presses))]
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
    We detect kills when stomp_counter transitions from 0 to 1.

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
    # Detect when stomp_counter increases from 0 to 1
    for frame_idx in range(1, len(stomps)):
        if stomps[frame_idx] == 1 and stomps[frame_idx - 1] == 0:
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
    - Powerup lost: powerup variable 1->0
    - Killed: outcome is failed/killed

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


    # Powerup lost (powerup 1 -> 0)
    if "powerup" in repvars:
        powerups = repvars["powerup"]
        for frame_idx in range(1, len(powerups)):
            # Count any decrement in powerup value (e.g. 3->1, 1->0)
            if powerups[frame_idx] < powerups[frame_idx - 1]:
                onset.append(frame_idx / FS)
                duration.append(0)
                trial_type.append("Hit/powerup_lost")
                level.append(repvars["level"])
                frame_start.append(frame_idx)
                frame_stop.append(frame_idx)

    # Check outcome for "failed/killed" or "failed/fall"
    outcome = _determine_outcome(repvars)
    
    if outcome == "failed/killed":
        # Add a hit event at the end or appropriate time
        # We'll use the last frame as the onset for the kill hit
        last_frame = len(repvars.get("lives", [])) - 1
        if last_frame > 0:
            onset.append(last_frame / FS)
            duration.append(0)
            trial_type.append("Hit/killed")
            level.append(repvars.get("level", 0))
            frame_start.append(last_frame)
            frame_stop.append(last_frame)
            
    elif outcome == "failed/fall":
        # Add a hit event for fall
        last_frame = len(repvars.get("lives", [])) - 1
        if last_frame > 0:
            onset.append(last_frame / FS)
            duration.append(0)
            trial_type.append("Hit/fall")
            level.append(repvars.get("level", 0))
            frame_start.append(last_frame)
            frame_stop.append(last_frame)

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

    We detect score increments of 1.

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
    
    # In SMB3, brick breaking usually gives 10 points
    # But user requested to count score increments of 1
    for idx_val, inc in enumerate(score_increments):
        if inc == 1:
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
            # Note: We just report Powerup_collected as requested
            onset.append(idx / FS)
            duration.append(0)
            trial_type.append("Powerup_collected")
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
    
    # Detect contiguous blocks where timer > 0
    in_event = False
    event_start_idx = 0
    
    for idx in range(len(timer)):
        val = timer[idx]
        
        if val > 0 and not in_event:
            # Event started
            in_event = True
            event_start_idx = idx
            
        elif val == 0 and in_event:
            # Event ended
            in_event = False
            onset.append(event_start_idx / FS)
            dur = (idx - event_start_idx) / FS
            duration.append(dur)
            trial_type.append("Star_activated")
            level.append(repvars["level"])
            frame_start.append(event_start_idx)
            frame_stop.append(idx)
            
    # Handle case where event goes until end of replay
    if in_event:
        idx = len(timer)
        onset.append(event_start_idx / FS)
        dur = (idx - event_start_idx) / FS
        duration.append(dur)
        trial_type.append("Star_activated")
        level.append(repvars["level"])
        frame_start.append(event_start_idx)
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
    
    # Detect contiguous blocks where timer > 0
    in_event = False
    event_start_idx = 0
    
    for idx in range(len(timer)):
        val = timer[idx]
        
        if val > 0 and not in_event:
            # Event started
            in_event = True
            event_start_idx = idx
            
        elif val == 0 and in_event:
            # Event ended
            in_event = False
            onset.append(event_start_idx / FS)
            dur = (idx - event_start_idx) / FS
            duration.append(dur)
            trial_type.append("Flight_activated")
            level.append(repvars["level"])
            frame_start.append(event_start_idx)
            frame_stop.append(idx)
            
    # Handle case where event goes until end of replay
    if in_event:
        idx = len(timer)
        onset.append(event_start_idx / FS)
        dur = (idx - event_start_idx) / FS
        duration.append(dur)
        trial_type.append("Flight_activated")
        level.append(repvars["level"])
        frame_start.append(event_start_idx)
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


def generate_level_complete_events(repvars, FS=60):
    """Generate events for level completion.

    Super Mario Bros 3 level completion is detected when complete_level
    transitions to 1 and killed is 0 at that frame.

    Parameters
    ----------
    repvars : dict
        Dictionary containing all the variables of a single repetition
    FS : int
        The sampling rate of the .bk2 file (default: 60)

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

    if "complete_level" not in repvars:
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

    complete_level = repvars["complete_level"]
    killed = repvars.get("killed", [0] * len(complete_level))

    # Detect when complete_level becomes 1 and killed is 0
    LOOKBACK_SECONDS = 5
    LOOKBACK_FRAMES = int(LOOKBACK_SECONDS * FS)  # 300 frames at 60fps
    
    for idx in range(1, len(complete_level)):
        if complete_level[idx] == 1 and complete_level[idx - 1] == 0:
            if killed[idx] == 0:
                # Adjust onset to 5 seconds earlier (when the actual completion happened)
                adjusted_frame = max(0, idx - LOOKBACK_FRAMES)
                onset.append(adjusted_frame / FS)
                duration.append(0)
                trial_type.append("Level_complete")
                level.append(repvars["level"])
                frame_start.append(adjusted_frame)
                frame_stop.append(adjusted_frame)
                break  # Only one level complete event per repetition

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
                        ].reset_index(drop=True)  # select only relevant columns
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

                                    # Set rep_index to 0-based sequential index for gym-retro_game events
                                    events_dataframe["rep_index"] = range(len(events_dataframe))

                                    runvars.append(repvars)
                                else:
                                    print(f"\nError: Variables file not found: {variables_sidecar_fname}")
                                    print("Please run create_replays.py first to generate the required files.")
                                    return
                            else:
                                print("Missing file, skipping")
                                runvars.append({})

                        # Add phase (discovery VS practice)
                        # Discovery: until the first random level replay (which implies practice starts)
                        # We track visited levels. If we see a level that we've seen before, AND it's not the
                        # immediate previous one (in case of a direct replay/restart), it implies random access -> practice.
                        # Simple heuristic: "practice" starts when we see a level again (after playing others) or jump non-sequentially?
                        # User request: "discovery phase so long as the successive repetitions correspond to the same or successive levels,
                        # and it should become practice as soon as the repetitions are played on random levels"
                        
                        visited_levels = set()
                        phases = []
                        current_phase = "discovery"
                        previous_level = None
                        
                        # Assuming levels roughly follow a sequence (names might not sort perfectly but we can track sets)
                        # Actually simpler: Discovery ends when we jump to a Random/Already Visited level that isn't the next expected one?
                        # Let's rely on the User's rule: "practice as soon as repetitions are played on random levels (i.e. after the last level of the game has been repeated)"
                        
                        # Let's implement robust tracking:
                        # 1. Start in 'discovery'.
                        # 2. Iterate. If we are in 'discovery':
                        #    Check if this level is "expected" (same as prev or "next" in some sense). 
                        #    Since "next" is hard to define without a map, let's detect the "random" condition.
                        #    The user said: "practice as soon as the repetitions are played on random levels (i.e. after the last level of the game has been repeated)"
                        #    Actually, easier logic:
                        #    If we encounter a level that we visited "long ago" (not just now), we are in practice.
                        #    Or if the level sequence jumps "backwards".
                        
                        # Let's try:
                        # Iterate through reps.
                        # If current_phase is 'discovery':
                        #   If level is in visited_levels AND level != previous_level:
                        #       current_phase = 'practice'
                        # phases.append(current_phase)
                        # visited_levels.add(level)
                        
                        for idx, row in events_dataframe.iterrows():
                            level = row["level"]
                            if current_phase == "discovery":
                                if level in visited_levels and level != previous_level:
                                    current_phase = "practice"
                            
                            phases.append(current_phase)
                            visited_levels.add(level)
                            previous_level = level

                        events_dataframe["phase"] = phases
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
