"""Hand-written documentation of the mario3 events, rendered into the dataset README."""

#: Why a vocabulary entry produces no row in the dataset. Entries absent from here that
#: have zero rows are reported as "never occurred". Keep this honest: "not implemented"
#: means the pipeline has no detector for it.
NOT_EMITTED = {
    "PlayerState/Hammer": "never obtained",
    "ItemCollected/Powerup": "not implemented; the pickup is the first frame of the "
                             "`PlayerState/*` row it starts",
    "EnemyAttack/{enemy_type}": "not implemented (no verified RAM signal)",
    "EnemyDefeated/Shell/{enemy_type}": "the object state table does not attribute a "
                                        "shell kill to the shell; kicks are reported "
                                        "as `ShellStartedMoving`",
    "PipeEntered": "not implemented; pipe and door travel is the `Screen/Transition` "
                   "row",
    "TimerWarningStarted": "not implemented for SMB3",
    "Action/Other": "the NES pad has no extra buttons",
    "Action/Start": "never pressed",
    "Action/Select": "never pressed",
}

#: Accuracy notes, in Markdown. Reviewed by hand when the generator changes.
NOTES = """\
- `PlayerState/*` rows are read straight from the RAM: `Player_Suit` ($00ED,
  `powerup`) for the form, `Player_StarInv` (`invincibility_timer`) for `Star`,
  `Player_FlashInv` (`invisibility_timer`) for `HitRecovery`, `Player_FlyTime`
  (`flight_timer`) for `Flying`, plus `statue_timer` and `kuribo_shoe`. A hit as Fire or
  in a suit drops the player to Super (verified: every 2 -> 1 transition coincides with
  `player_suit_lost`), so one form row ends and the next starts on the same frame.
- `Screen/*`: there is no title card (a repetition opens in the level with a short
  fade-in). The death is `Player_IsDying` != 0; area transitions are `Player_HaltGame`
  >= 128 while the next area loads, plus the pipe-entry hold before it; the end of a
  level runs from the goal card to the end of the recording (card animation and COURSE
  CLEAR). All checked on rendered frames.
- **148 of 4063 recordings run on past a game-over death** into the world map with the
  GAME OVER dialog (median 18 s, up to 4 min), and one (sub-06 ses-010 w6l4 rep-000)
  returns to the map and records a second attempt. These are `Screen/GameOver` and
  `Screen/Map` rows, excluded from the alive frames, so they are easy to find and drop.
  They are recording overruns, not gameplay.
- Object ids come from `Level_ObjectID` ($0671), decoded through the 170-entry `OBJ_*`
  table generated from captainsouthbird's SMB3 disassembly. Visibility uses the game's
  own `Objects_SprHVis` / `Objects_SprVVis` flags, validated against the object's level
  position minus the scroll: horizontally inside the band on 99.9% of accepted frames,
  vertically on 100.0%.
- Cause of death comes from `Player_IsDying` ($00F1), which states it directly
  (1 = enemy, 2 = dropped off screen, 3 = time up).
- `GoalCardVisible` is the roulette card object (`OBJ_ENDLEVELCARD`, 0x41), which is
  on screen 0.5-4 s before it is taken. It is untyped because its face cycles; the type
  taken is in `GoalCardCollected`.
- A level attempt spans up to three one-life `.bk2` files. The first gets
  `LevelStarted`, the rest `LevelRestarted`.
"""
