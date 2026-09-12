# Annotated event files - mario3

`generate_annotations.py` turns the per-frame RAM dumps written by
`code/replays/generate_replays.py` (`gamelogs/*_variables.json`) into
BIDS event files at `sub-*/ses-*/func/*_desc-annotated_events.tsv`.

The event logic is shared by all four CNeuroMod videogame datasets and lives in
[`videogames_utils.events`](https://github.com/courtois-neuromod/videogames_utils);
this directory only holds the command-line front end. The machine-readable
vocabulary is published as `task-mario3_events.json` at the dataset root.

## Usage

```bash
python code/annotations/generate_annotations.py -d . --overwrite --validate
```

## Columns

| Column | Description |
|---|---|
| `trial_type` | Event type, from the controlled vocabulary below. |
| `level` | Level of the repetition the event belongs to. |
| `onset` | Seconds from the start of the run. |
| `duration` | Seconds. 0 for point events. |
| `frame_start` | First emulator frame of the event, relative to its repetition. |
| `frame_stop` | Last emulator frame of the event, relative to its repetition. |
| `button` | Raw controller button behind an `Action/*` event. |
| `stim_file` | Path to the repetition's `.bk2` replay. |

Onsets are computed at the console's true frame rate (60.099827 Hz, read from the
emulator core), not the 60.0 Hz the pipeline previously assumed. That correction
shifts onsets by up to ~0.6 s in the longest repetitions.

## Event types

`{...}` is replaced at generation time with a decoded name.

| Event | Description |
|---|---|
| `Player_damaged` | The player is hit and loses the current power-up state (Mario games) or health (Shinobi). |
| `Player_died/Enemy` | The player dies after being hit by an enemy or another damaging object. |
| `Player_died/Fall` | The player falls into a pit and dies. `onset` is `player_is_dying == 2`, which the game sets at the drop, 210 frames (3.5 s) before the life is lost. |
| `Player_died/Timeout` | The player dies because the level timer ran out. |
| `Life_gained` | The player collects or earns an extra life. |
| `Player_state/Super` | The player is Super (big) Mario, from the frame the mushroom is collected until hit, death or the end of the repetition. |
| `Player_state/Fire` | The player is Fire Mario (can throw fireballs). |
| `Player_state/Raccoon` | The player is Raccoon Mario. |
| `Player_state/Frog` | The player wears the Frog suit. |
| `Player_state/Tanooki` | The player wears the Tanooki suit. |
| `Player_state/Hammer` | The player wears the Hammer Brothers suit. |
| `Player_state/Star` | Star invincibility is active. |
| `Player_state/Hit_recovery` | Post-hit recovery: the player has just been damaged and blinks. In the Mario games nothing can hurt the player until it ends; in Shinobi it is the game's post-hit counter. |
| `Player_state/Flying` | The player is flying with a flight-capable suit. |
| `Player_state/Statue` | Tanooki Mario is in statue form. |
| `Player_state/Kuribo_shoe` | The player rides Kuribo's Shoe. |
| `Item_on_screen/{item_type}` | A coin, mushroom, flower, star or extra life is visible on screen. `duration` spans the time it is visible. |
| `Item_collected/Coin` | The player collects a coin. |
| `Item_collected/Powerup` | The player collects a mushroom, flower, star or other power-up item. |
| `Block_smashed` | The player destroys a breakable brick block from below. |
| `Enemy_on_screen/{enemy_type}` | A specific enemy type is visible on screen. `duration` runs from the frame it becomes visible until it leaves the screen or is defeated; an enemy that leaves and returns produces two separate events. |
| `Enemy_attack/{enemy_type}` | An enemy begins an attack, such as firing a projectile or emerging from a pipe. |
| `Enemy_defeated/Stomp/{enemy_type}` | The player defeats an enemy by jumping on it. |
| `Enemy_defeated/Projectile/{enemy_type}` | The player defeats an enemy with a fireball or other projectile. |
| `Enemy_defeated/Shell/{enemy_type}` | The player defeats an enemy using a moving shell. |
| `Projectile_on_screen/{projectile_type}` | A fireball, Bullet Bill, hammer or other moving projectile is visible on screen. `duration` spans the time it is visible. |
| `Shell_started_moving` | A shell begins moving after being kicked or otherwise activated. |
| `Pipe_entered` | The player enters a pipe. |
| `Timer_warning_started` | The game begins warning the player that little time remains. |
| `P-Switch_started` | The player activates a P-Switch, temporarily changing nearby bricks and coins. |
| `P-Switch_expired` | The temporary P-Switch effect ends. |
| `Goal_card_visible/{card_type}` | The mushroom, flower or star goal card becomes visible at the end of the level. |
| `Goal_card_collected/{card_type}` | The player touches and collects the end-of-level goal card. |
| `Auto_scroll_started` | The level begins scrolling independently of the player. |
| `Level_started` | A new level or gameplay attempt begins. |
| `Level_restarted` | The level restarts after the player dies. |
| `Level_completed` | The player successfully finishes the level. |
| `Action/Left` | The player holds the left direction. |
| `Action/Right` | The player holds the right direction. |
| `Action/Up` | The player holds the up direction. |
| `Action/Down` | The player holds the down direction (duck / crouch). |
| `Action/Jump` | The player presses the jump button. |
| `Action/Run` | The player presses the run / throw button (Mario games). |
| `Action/Other` | The player presses a button with no documented function in this game (e.g. L/R on the SNES pad in Super Mario All-Stars). The raw button is in the `button` column. |
| `Action/Start` | The player presses START (pauses the game). |
| `Action/Select` | The player presses SELECT / MODE. |
| `gym-retro_game` | One repetition of gameplay (one .bk2 file). This is the container row that carries `stim_file`; all other events fall inside its window. |

## Renamed from the previous vocabulary

This release renames every event type. Old analyses filtering on the former
names need updating; the mapping is:

`Enemy_disappeared` was **dropped**: it was a point event on the last visible frame
of an `Enemy_on_screen` track that no defeat claimed, so it is exactly
`onset + duration` of that row (verified on 6620 of 6620 rows, matching on integer
frames). Read it as an `Enemy_on_screen` row with no `Enemy_defeated` at its end.

| Former | Now |
|---|---|
| `Brick_smashed` | `Block_smashed` |
| `Enemy_appeared/{enemy_type}` | `Enemy_on_screen/{enemy_type}` |
| `Enemy_disappeared/{enemy_type}` | *(dropped)* |
| `Item_appeared/{item_type}` | `Item_on_screen/{item_type}` |
| `Projectile_appeared/{projectile_type}` | `Projectile_on_screen/{projectile_type}` |
| `Coin_collected` | `Item_collected/Coin` |
| `DOWN` | `Action/Down` |
| `Flight_activated` | `Player_state/Flying` |
| `HealthLoss` | `Player_damaged` |
| `Hit/fall` | `Player_died/Fall` |
| `Hit/killed` | `Player_died/Enemy` |
| `Hit/life_lost` | `Player_died/Enemy` |
| `Hit/powerup_lost` | `Player_damaged` |
| `Hit/timeout` | `Player_died/Timeout` |
| `JUMP` | `Action/Jump` |
| `Kill/impact` | `Enemy_defeated/Projectile/{enemy_type}` |
| `Kill/kick` | `Enemy_defeated/Shell/{enemy_type}` |
| `Kill/stomp` | `Enemy_defeated/Stomp/{enemy_type}` |
| `LEFT` | `Action/Left` |
| `Level_complete` | `Level_completed` |
| `MODE` | `Action/Select` |
| `P-Switch_activated` | `P-Switch_started` |
| `Powerup_collected` | `Item_collected/Powerup` |
| `RIGHT` | `Action/Right` |
| `RUN/THROW` | `Action/Run` |
| `SELECT` | `Action/Select` |
| `START` | `Action/Start` |
| `Star_activated` | `Player_state/Star` |
| `UP` | `Action/Up` |


The previous release's `Powerup_started/*` and `Powerup_expired/*` point events (and `Flight_started` / `Flight_expired`, now `Player_state/Flying`) are
replaced by the durational `Player_state/*` rows: the onset of `Player_state/Super` is the
old `Powerup_started/Super`, the end of `Player_state/Star` is the old
`Powerup_expired/Star`, and so on. `Powerup_started/Small` (emitted on a hit) has no
successor: Small has no state row, and the hit is already `Player_damaged`.

### Accuracy notes

- `Player_state/*` rows are read straight from the RAM: `Player_Suit` ($00ED,
  `powerup`) for the form, `Player_StarInv` (`invincibility_timer`) for `Star`,
  `Player_FlashInv` (`invisibility_timer`) for `Hit_recovery`, `Player_FlyTime`
  (`flight_timer`) for `Flying`, plus `statue_timer` and `kuribo_shoe`. A hit as Fire or
  in a suit drops the player to Super (verified: every 2 -> 1 transition coincides with
  `player_suit_lost`), so one form row ends and the next starts on the same frame. Rows
  are cut at a death that falls inside them. No new RAM variable was needed.
- Object ids come from `Level_ObjectID` ($0671), decoded through the 170-entry `OBJ_*`
  table generated from captainsouthbird's SMB3 disassembly. Verified: a World 1-1 replay
  decodes to Goomba, ParaGoomba, RedTroopa and VenusFireTrap, matching that level's
  actual roster, and a `SOBJ_PIRANHAFIREBALL` appears exactly while a VenusFireTrap is
  live.
- Cause of death comes from `Player_IsDying` ($00F1), which states it directly
  (1 = enemy, 2 = dropped off screen, 3 = time up). This replaces the previous
  timer-freeze / y-position heuristic and is expected to correct some of the shipped
  `Outcome` labels, of which 91% were the catch-all `failed/killed`.
- **Provisional:** `Enemy_on_screen` / `Enemy_counter` use object
  slot occupancy as the visibility proxy. Unlike SMB1 there is no verified per-slot
  on-screen flag -- `Objects_SprHVis` / `SprVVis` could not be validated because
  `Objects_SpriteX/Y` hold stale values while an object is not being drawn. SMB3 only
  loads objects near the screen, so occupancy is a reasonable proxy, but the accuracy of
  these three event types must be established by the video review rather than argued
  from the RAM map.
- A level attempt spans up to three one-life `.bk2` files. The first gets
  `Level_started`, the rest `Level_restarted`.

## Validation

```bash
python -m videogames_utils.events.validate_cli check . mario3
python -m videogames_utils.events.validate_cli cross-port ../mario ../mariostars
```

`check` runs two layers: the schema and controlled-vocabulary checks (V0), and
invariants recomputed straight from `_variables.json` by a different route than
the generator used (V1) -- coin counts against the coin counter, deaths against
the lives counter, level completion against the summary outcome, enemy track
bookkeeping, and frame-range bounds.

A human video review measures precision and recall per event type:

```python
from videogames_utils.events import review
review.build_review_set('.', 'mario3', out_dir='/tmp/review', per_type=50)
# rate the clips in /tmp/review/review.html, then:
review.score_reviews('/tmp/review/ratings.json', '/tmp/review/manifest.json')
```
