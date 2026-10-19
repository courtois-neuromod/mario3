"""The Super Mario Bros. 3 parts of the replay pipeline (:mod:`videogames_utils.replays`): the
game's own ``_summary.json`` fields and the protocol phases.

The counting functions are those of the former ``code/replays/generate_replays.py``.
"""

from __future__ import annotations

from .generator import FRAME_RATE


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

def _count(events, *trial_types) -> int:
    """Rows of the repetition's events with one of ``trial_types``."""
    return int(events["trial_type"].isin(trial_types).sum()) if len(events) else 0


def hits_taken(events) -> int:
    """Hits that cost a power-up or a life. A fall or a timeout is not a hit."""
    return _count(events, "PlayerDamaged", "PlayerDied/Enemy")


def bricks_smashed(events) -> int:
    """Bricks broken, as in the events (``BlockSmashed``)."""
    return _count(events, "BlockSmashed")


def summary(repvars: dict, events) -> dict:
    """The game's fields of a repetition's ``_summary.json``."""
    try:
        duration = len(repvars["score"]) / FRAME_RATE
    except KeyError:
        duration = None
    distance = _calculate_distance_traveled(repvars)
    average_speed = None
    if distance is not None and duration is not None and duration > 0:
        average_speed = distance / duration
    # Calculate score (multiply by 10 to get correct value)
    score_gained = _safe_diff(repvars, "score")
    return {
        "ScoreGained": score_gained * 10 if score_gained is not None else None,
        "HitsTaken": hits_taken(events),
        "EnemiesStomped": count_stomps(repvars),
        "CoinsCollected": _safe_diff(repvars, "coins_p1"),
        "PowerupsCollected": count_powerups_collected(repvars),
        "StarsCollected": count_star_power_activations(repvars),
        "BricksSmashed": bricks_smashed(events),
        "DistanceTraveled": distance,
        "AverageSpeed": round(average_speed, 2) if average_speed else None,
        "FinalTimer": _get_final_timer(repvars),
        "FinalForm": _get_final_powerup_state(repvars),
        "FlightsActivated": count_flight_activations(repvars),
        "KuriboShoeUses": _count_kuribo_shoe_uses(repvars),
        "MaxPMeter": _get_max_p_meter(repvars),
    }


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


def phases(repetitions):
    """See :func:`_assign_phases_per_subject`."""
    return _assign_phases_per_subject(repetitions.copy())["phase"]
