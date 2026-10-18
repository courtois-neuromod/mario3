"""What :mod:`videogames_utils.events` needs to know about Super Mario Bros. 3 (NES).

See :class:`videogames_utils.events.game.Game`.
"""

import re

from videogames_utils.events.game import Game

from . import docs, generator, outcome, ram, validation
from .vocabulary import VOCABULARY


def level_from_events(level: str) -> str:
    """The events.tsv level in the annotated form.

    ``1Player.World1.Level5`` -> ``w1l5``, ``1Player.World6.Airship`` -> ``w6lAirship``.
    """
    m = re.fullmatch(r"1Player\.World(\w+)\.(?:Level)?(\w+)", level)
    return f"w{m.group(1)}l{m.group(2)}" if m else level


GAME = Game(
    task="mario3",
    name="Super Mario Bros. 3 (NES)",
    game_id="SuperMarioBros3-Nes",
    frame_rate=generator.FRAME_RATE,
    vocabulary=VOCABULARY,
    generate=generator.generate,
    outcome=outcome.outcome,
    level_from_events=level_from_events,
    form_states=frozenset({"PlayerState/Small", "PlayerState/Super", "PlayerState/Fire",
                           "PlayerState/Raccoon", "PlayerState/Frog",
                           "PlayerState/Tanooki", "PlayerState/Hammer"}),
    form_var="powerup",
    state_timers=(("invincibility_timer", "PlayerState/Star"),
                  ("invisibility_timer", "PlayerState/HitRecovery"),
                  ("flight_timer", "PlayerState/Flying"),
                  ("statue_timer", "PlayerState/Statue"),
                  ("kuribo_shoe", "PlayerState/KuriboShoe")),
    coin_var="coins_p1",
    check_repetition=validation.check_repetition,
    notes=docs.NOTES,
    not_emitted=docs.NOT_EMITTED,
    golden_sessions=[("sub-03", "ses-013")],
    ram_candidates=ram.SMB3_CANDIDATES + ram.SMB3_UNVERIFIED,
    mislabelled=ram.MISLABELLED,
)
