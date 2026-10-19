"""RAM variables of Super Mario Bros. 3 (NES): the candidates verified before they were
added to ``stimuli/SuperMarioBros3-Nes/data.json``.

Source: captainsouthbird's SMB3 disassembly (smb3.asm), https://github.com/captainsouthbird/smb3
"""

from __future__ import annotations

from typing import List

from videogames_utils.events.ram import Candidate, candidate_array


# --- Super Mario Bros 3 (NES) -----------------------------------------------------

SMB3_CANDIDATES: List[Candidate] = [
    *candidate_array("object_id", 0x0671, 8, "|u1", "Level_ObjectID", "OBJ_* actor id per slot"),
    *candidate_array("object_state", 0x0661, 8, "|u1", "Objects_State", "OBJSTATE_* per slot"),
    # Objects_DetStat is COLLISION bits, not visibility: $01 hit wall right, $02 hit wall
    # left, $04 hit ground, $08 hit ceiling, $80 on a 32px partition floor. The trailing
    # "on screen" in its .ds comment is vestigial. Kept for EnemyAttack heuristics only.
    *candidate_array("object_detstat", 0x00D9, 8, "|u1", "Objects_DetStat",
            "collision bits: 1 wall-R, 2 wall-L, 4 ground, 8 ceiling, 0x80 partition"),
    # These are the actual visibility flags.
    *candidate_array("object_sprhvis", 0x0651, 8, "|u1", "Objects_SprHVis",
            "bits set when each 8x16 sprite is horizontally off screen"),
    *candidate_array("object_sprvvis", 0x0681, 8, "|u1", "Objects_SprVVis",
            "bits set when each 8x16 sprite is vertically off screen"),
    *candidate_array("object_sprite_x", 0x00AC, 8, "|u1", "Objects_SpriteX", "screen X per slot"),
    *candidate_array("object_sprite_y", 0x00B5, 8, "|u1", "Objects_SpriteY", "screen Y per slot"),
    *candidate_array("object_x", 0x0091, 8, "|u1", "Objects_X", "level X low per slot"),
    *candidate_array("object_x_hi", 0x0076, 8, "|u1", "Objects_XHi", "level X high per slot"),
    *candidate_array("object_y", 0x00A3, 8, "|u1", "Objects_Y", "level Y low per slot"),
    *candidate_array("object_y_hi", 0x0088, 8, "|u1", "Objects_YHi", "level Y high per slot"),
    Candidate("player_is_dying", 0x00F1, "|u1", "Player_IsDying",
              "0 no, 1 dying, 2 dropped off screen, 3 TIME UP"),
    *candidate_array("object_player_hitstat", 0x0796, 8, "|u1", "Objects_PlayerHitStat",
            "which object hit the player"),
    Candidate("level_hautoscroll", 0x0580, "|u1", "Level_HAutoScroll",
              "-> AutoScrollStarted"),
    Candidate("player_halt_game", 0x00CE, "|u1", "Player_HaltGame",
              "dying / growing / shrinking"),
    Candidate("player_suit_lost", 0x0554, "|u1", "Player_SuitLost",
              "suit-lost poof counter"),
    Candidate("kill_tally", 0x05F4, "|u1", "Kill_Tally",
              "stomp chain (already shipped as stomp_counter)"),
    # Screen scroll. Needed as an INDEPENDENT geometric reference for validating the
    # per-slot visibility flags: Objects_SpriteX/Y hold stale values while an object is
    # not being drawn, so they cannot adjudicate. With the scroll position the object's
    # screen-relative X can be computed from Objects_X/XHi, exactly as is done for SMB1.
    Candidate("horz_scroll", 0x00FD, "|u1", "Horz_Scroll",
              "horizontal scroll of the name table"),
    Candidate("horz_scroll_hi", 0x0012, "|u1", "Horz_Scroll_Hi",
              "high byte of the horizontal scroll (current screen)"),
    Candidate("vert_scroll", 0x00FC, "|u1", "Vert_Scroll", "vertical scroll"),
    Candidate("vert_scroll_hi", 0x0013, "|u1", "Vert_Scroll_Hi",
              "high byte of the vertical scroll (vertical levels only)"),
]

# Addresses at/above $6000 in smb3.asm come from overlapping .org contexts and cannot be
# resolved unambiguously by static parsing. They are proposed separately and only ship if
# the harness confirms them.
SMB3_UNVERIFIED: List[Candidate] = [
    # VERIFIED: decodes to known SOBJ_* constants, and a PiranhaFireball appears exactly
    # while a VenusFireTrap is live in an adjacent slot.
    *candidate_array("special_obj_id", 0x7FC6, 8, "|u1", "SpecialObj_ID", "SOBJ_* projectile ids"),
    # REJECTED: Inventory_Cards at $7C74 reads all-zero even on a cleared repetition, so
    # the static resolution of that overlapping .org context is wrong. The shipped
    # goal_cards_p1_{1,2,3} already detect level clearing correctly, so nothing is lost;
    # do NOT add this to data.json until a working address is found.
    # *candidate_array("inventory_cards", 0x7C74, 3, "|u1", "Inventory_Cards", "goal cards"),
]


#: Existing data.json keys whose names are wrong, with the disassembly's actual symbol.
#: The addresses are correct, so the keys are kept as aliases; this only records the truth
#: for documentation.
MISLABELLED = {
    "killed": "Map_ReturnStatus ($0713) - NOT a kill flag; unused by the generators",
    "mario_form": "Player_QueueSuit ($0578) - queued suit change, not the current form",
    "invisibility_timer": "Player_FlashInv ($0552) - post-hit flashing, not invisibility",
    "invincibility_timer": "Player_StarInv ($0553) - correct meaning, star invincibility",
    "stomp_counter": "Kill_Tally ($05F4)",
    "timer_subframe": "Level_TimerTick ($05F1)",
    "object_set": "Level_Tileset ($070A)",
    "p_meter": "Player_Power ($03DD)",
    "powerup": "Player_Suit ($00ED) - the actual current form",
}
