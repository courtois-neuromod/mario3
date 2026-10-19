Object ids come from `Level_ObjectID` (`$0671`, 8 slots), decoded through the 170-entry
`OBJ_*` table generated from the disassembly. Verified: a World 1-1 replay decodes to
Goomba, ParaGoomba, RedTroopa and VenusFireTrap — that level's actual roster.

### On-screen visibility

| Dataset | Predicate | Validation |
|---|---|---|
| mario3 | `Objects_State != 0` **and** `Objects_SprHVis == 0` **and** `Objects_SprVVis == 0` | `SprHVis == 0` is horizontally inside the band on 99.9% of frames (n=9571) vs 11.7% when non-zero; among frames passing that test, `SprVVis == 0` is vertically inside the band on 100.0% (n=37436) vs 5.1% when non-zero |

SMB3 is the game where the two notions diverge most: it scrolls vertically, so a slot can
be horizontally on screen while sitting entirely above or below it. The vertical flag
removes 21% of the frames the horizontal test alone would have called visible.

### Screens

| Row | Meaning | Detection |
|---|---|---|
| `Screen/Gameplay` | The player controls the character in the level. | everything not below |
| `Screen/Death` | The death sequence, no control. | `player_is_dying` ≠ 0, ~200 frames |
| `Screen/LevelEnd` | The end-of-level sequence, no control. | from the goal card to the end of the recording: card animation and COURSE CLEAR, ~340 frames |
| `Screen/Transition` | Between areas, no control. | `player_halt_game` ≥ 128 while the next area loads, plus the pipe-entry hold before it |
| `Screen/GameOver` | The GAME OVER screen. | `level_music` = 0 with `lives` < 0 after the death: the map with the GAME OVER dialog. **148 of 4063 recordings** ran on into this instead of stopping at the death, for a median 18 s and up to 4 min. |
| `Screen/Map` | The SMB3 world map. | `level_music` = 0 with `lives` ≥ 0 after the death: one recording (sub-06 ses-010 w6l4 rep-000) returned to the map and recorded a second attempt at the level |

### Timing a fall death

| Dataset | How the fall is dated | Accuracy |
|---|---|---|
| mario3 | unchanged: `player_is_dying == 2` already fires at the drop, 210 frames (3.5 s) *before* the life is lost | the game's own flag |

### Player

| Event | Description | Detection |
|---|---|---|
| `PlayerDamaged` | Hit, losing the current suit. | `player_suit_lost` (the suit-lost poof counter) becomes non-zero, falling back to `invisibility_timer` (`Player_FlashInv`, the post-hit blink). Suppressed within 120 frames of a death. |
| `PlayerDied/Enemy` | Killed by an enemy. | `player_is_dying` (`$00F1`) becomes **1**. |
| `PlayerDied/Fall` | Dropped off the bottom of the screen. | `player_is_dying` becomes **2**. |
| `PlayerDied/Timeout` | Time ran out. | `player_is_dying` becomes **3**. |
| `LifeGained` | Gained an extra life. | `lives` increases. |
| `PlayerState/{Small,Super,Fire,Raccoon,Frog,Tanooki,Hammer}` | The suit the player wears, one row per continuous stretch. Together they tile the frames on which the player is alive in the level. | Maximal runs of a constant `powerup` (`Player_Suit`, `$00ED`) over the alive frames (outside `Screen/Death`, `Screen/GameOver`, `Screen/Map`). A hit as Fire or in a suit drops the player to Super, so one row ends and the next starts on the same frame. |
| `PlayerState/Star` | Star invincibility. | `invincibility_timer` (`Player_StarInv`) non-zero run, ~448 frames. |
| `PlayerState/HitRecovery` | Post-hit invulnerability while blinking, ~113 frames. | `invisibility_timer` (`Player_FlashInv`) non-zero run. |
| `PlayerState/Flying` | Flying with a flight-capable suit. | `flight_timer` (`Player_FlyTime`) non-zero run. |
| `PlayerState/Statue`, `PlayerState/KuriboShoe` | Tanooki statue form; riding Kuribo's Shoe. | `statue_timer` / `kuribo_shoe` non-zero run. Both are rare (1 and 5 repetitions in the corpus). |
| `PSwitchStarted`, `PSwitchExpired` | A P-Switch is active. | `p_switch_timer` (`Level_PSwitchCnt`) non-zero run. |

> `player_is_dying` states the cause of death outright. It replaces the previous
> timer-freeze plus X-position heuristic, and moved 942 repetitions from the catch-all
> `failed/killed` to `failed/fall`.

### Objects

| Event | Description | Detection |
|---|---|---|
| `EnemyOnScreen/{type}` | An enemy became visible. | A visible track opens in an object slot (`Objects_State != 0` and both `Objects_SprHVis` and `Objects_SprVVis` clear); type from `object_id_*`. |
| `EnemyDefeated/Stomp/{type}` | Stomped. | `object_state_*` enters **7** (`OBJSTATE_SQUASHED`), or enters **3** (`OBJSTATE_SHELLED`) from a non-shell state — a stomp that shells an enemy. |
| `EnemyDefeated/Projectile/{type}` | Killed by a projectile or tail. | `object_state_*` enters **6** (`OBJSTATE_KILLED`) or **8** (`OBJSTATE_POOFDEATH`). |
| `ShellStartedMoving` | A shelled enemy was kicked. | `object_state_*` enters **5** (`OBJSTATE_KICKED`). |
| `ProjectileOnScreen/{type}` | A hammer, fireball, cannonball etc. became visible. | `special_obj_id_*` (`SpecialObj_ID`) non-zero, decoded through the 22-entry `SOBJ_*` table. |
| `ItemOnScreen/{1Up,Starman,Mushroom,FireFlower,SuperLeaf}` | A power-up became visible. | Object ids `0x0B`, `0x0C`, `0x0D`, `0x19`, `0x1E`. |
| `ItemCollected/Coin` | Collected a coin. | `coins_p1` increases, or wraps at 100. |
| `BlockSmashed` | Destroyed a brick. | `score` increases by exactly **1** (10 points) while `in_air` is set. |

Sixty object ids are excluded from `Enemy_*` as scenery, platforms, items or engine
control objects. The exclusion list is derived from the disassembly's own `OBJ_*` names
rather than hand-written, so it stays correct if the table is regenerated.

### Level

| Event | Description | Detection |
|---|---|---|
| `LevelStarted` | First attempt at this level in the run. | Frame 0 of a `rep-000` file. |
| `LevelRestarted` | A retry after dying. | Frame 0 of a `rep-001`+ file. In SMB3 one level attempt spans up to three one-life `.bk2` files. |
| `GoalCardVisible` | The end-of-level roulette card is on screen, 0.5-4 s before it is taken. Untyped: its face cycles until touched. | A visible track of object id **`0x41`** (`OBJ_ENDLEVELCARD`). An earlier release keyed on the card items `0x21`-`0x23` (`PowerUpMushCard` etc.), which never appear at a level end, so the event never fired. |
| `LevelCompleted` | Finished the level. | A goal card enters the inventory (`goal_cards_p1_{1,2,3}` increases). |
| `GoalCardCollected/{Mushroom,Flower,Star,…}` | Which card was collected. | The same transition; the card value names it. |
| `AutoScrollStarted` | The level began scrolling on its own. | `level_hautoscroll` (`Level_HAutoScroll`) becomes non-zero. |

### Controller actions

| Event | NES |
|---|---|
| `Action/Left`, `Action/Right`, `Action/Up`, `Action/Down` | D-pad |
| `Action/Jump` | `A` |
| `Action/Run` | `B` |
| `Action/Start` | `START` |
| `Action/Select` | `SELECT` |

### Repetition outcomes

The cause of death comes from `player_is_dying`, which dropped the catch-all
`failed/killed` from 91% to 67% of repetitions.

### Provenance

RAM addresses and value tables are taken from the published disassembly and each was
verified against an independent signal before being used:

- **Super Mario Bros. 3** — [captainsouthbird's disassembly](https://github.com/captainsouthbird/smb3)
