"""RAM invariants specific to Super Mario Bros. 3, run on every repetition after the shared
ones."""

from __future__ import annotations

import collections


def check_repetition(events, repvars, summary, path, report) -> None:
    """One .bk2 is one life, so a failed repetition has exactly one death."""
    counts = collections.Counter(events["trial_type"].astype(str))
    deaths = sum(n for t, n in counts.items()
                 if t == "PlayerDied" or t.startswith("PlayerDied/"))
    outcome = (summary or {}).get("Outcome", "")
    if outcome.startswith("failed") and deaths != 1:
        report.add("V1", "failed repetition without exactly one death", path,
                   f"outcome={outcome}, {deaths} PlayerDied events")
    if outcome == "cleared" and deaths:
        report.add("V1", "cleared repetition with a death", path,
                   f"{deaths} PlayerDied events")
