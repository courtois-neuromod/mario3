"""What :mod:`videogames_utils.events` needs to know about Super Mario Bros. 3 (NES).

See :class:`videogames_utils.events.game.Game`.
"""

from videogames_utils.events.game import Game

from . import docs, generator, outcome, ram, replays, validation
from .vocabulary import VOCABULARY


GAME = Game(
    task="mario3",
    name="Super Mario Bros. 3 (NES)",
    game_id="SuperMarioBros3-Nes",
    frame_rate=generator.FRAME_RATE,
    vocabulary=VOCABULARY,
    generate=generator.generate,
    outcome=outcome.outcome,
    summary=replays.summary,
    phases=replays.phases,
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
    reference=docs.REFERENCE,
    not_emitted=docs.NOT_EMITTED,
    golden_sessions=[("sub-03", "ses-013")],
    ram_candidates=ram.SMB3_CANDIDATES + ram.SMB3_UNVERIFIED,
    mislabelled=ram.MISLABELLED,
)
