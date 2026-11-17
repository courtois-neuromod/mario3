"""How a repetition ended, decided from the game engine rather than from heuristics.

The labels, shared by the CNeuroMod videogame datasets:

    cleared
    failed/killed, failed/fall, failed/timeout
    incomplete/warp, incomplete/interrupted
    unknown

**mario3.** The shipped logic inferred death from a timer freeze plus the player's X
position, and fell through to ``failed/killed`` as a catch-all -- which is why 91% of
mario3 repetitions carry that label. ``Player_IsDying`` ($00F1) states the cause outright.

"""

from __future__ import annotations

#: Player_IsDying values in SMB3.
SMB3_DEATH_CAUSE = {1: "failed/killed", 2: "failed/fall", 3: "failed/timeout"}




def smb3_outcome(repvars: dict) -> str:
    """Outcome for Super Mario Bros 3.

    Clearing is a goal card entering the inventory; the cause of failure comes straight
    from ``Player_IsDying``. Falls back to the previous goal-card-only logic when the new
    variable is absent, so it still works on sidecars generated before the RAM addresses
    were added.
    """
    try:
        for card_var in ("goal_cards_p1_1", "goal_cards_p1_2", "goal_cards_p1_3"):
            cards = repvars.get(card_var) or []
            if len(cards) > 1 and any(cards[i] > cards[i - 1] for i in range(1, len(cards))):
                return "cleared"

        dying = repvars.get("player_is_dying")
        if dying:
            for value in dying:
                if value in SMB3_DEATH_CAUSE:
                    return SMB3_DEATH_CAUSE[value]
            return "incomplete/interrupted"

        # Legacy path: no Player_IsDying available.
        time_series = repvars.get("time") or []
        all_zero = all(
            (repvars.get(name) or [1])[-1] == 0
            for name in ("timer_100", "timer_10", "timer_1"))
        if (time_series and time_series[-1] == 0) or all_zero:
            return "failed/timeout"
        x_low = repvars.get("player_x_level_low") or []
        if x_low[-100:] and all(v == 0 for v in x_low[-100:]):
            return "failed/fall"
        return "failed/killed"
    except (KeyError, IndexError, TypeError):
        return "unknown"




outcome = smb3_outcome
