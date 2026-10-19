"""Hand-written documentation of the mario3 events, rendered into the dataset README.

The reference itself is ``reference.md``, next to this file.
"""

import os.path as op

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


#: How each event is read from the RAM and how it was validated, in Markdown. Embedded in
#: the README and in the common event reference.
with open(op.join(op.dirname(__file__), "reference.md")) as _handle:
    REFERENCE = _handle.read()
