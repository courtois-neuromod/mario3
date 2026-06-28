#!/usr/bin/env python3
"""
mark_missing_bk2s.py

Mark the *missing* duplicate repetitions in a Mario3 BIDS dataset's events.tsv
files.

Background
----------
A bug in the BizHawk recording pipeline caused some `.bk2` replay files to be
saved with duplicate content: when a level was played several times in a
session, the later replays were overwritten with the data of the *first* play.
`bk2_duplicates.csv` lists these duplicate groups.

Empirically (verified against the PsychoPy logs for every group whose logs
survive), the single surviving recording in each group always corresponds to
the *first* play, i.e. the lowest-numbered rep of the group. The real
recordings of every higher rep are gone.

This script therefore, for each duplicate group:
  - leaves the lowest-numbered rep untouched (its recording genuinely exists);
  - rewrites the `stim_file` field of every higher rep in the matching
    events.tsv row to a placeholder string ("Missing File" by default),
    because the .bk2 those reps point to does not actually exist.

Only the `stim_file` column is touched. All other columns, the tab layout, and
line endings are preserved byte-for-byte.

Validation
----------
After marking (or on its own via --validate-only), the script verifies that the
kept rep is the *most probable* survivor. The `.bk2` duration of the kept rep
(n_frames / fps) is compared against the play's time slot derived from the
onset columns: each play's length is measured as the gap from its
`gym-retro_game` onset to the onset of the *next* `fixation_dot` (the marker
that ends the play). This boundary is reliable across all sessions, unlike the
unreliable `duration` column. The kept (lowest) rep should be the one whose slot
is closest to its `.bk2` duration; groups where it is not, or where the residual
is large, are FLAGGED for manual review. A per-group report is written to
`bk2_kept_validation.csv`.

Removing the duplicate .bk2 files
---------------------------------
With --remove-bk2 the script also deletes the duplicate `.bk2` recordings (the
higher reps), keeping only the good one (the lowest rep). The dataset's `.bk2`
are git-annex symlinks; the *original* recordings use the SHA256E annex backend
while the duplicate copies created by the bug use MD5E. Removal therefore only
deletes a higher rep when it is an MD5E symlink AND the kept rep is a present
SHA256E original — so genuine data is never deleted. Groups that don't fit this
pattern (e.g. group 161, whose reps span two sessions with differing content and
where the kept rep is itself MD5E) are FLAGGED and left untouched. Removal uses
`git rm` (stages deletions); commit afterwards with `datalad save`.

Usage
-----
    python mark_missing_bk2s.py                 # apply (./mario3, ./bk2_duplicates.csv) + validate
    python mark_missing_bk2s.py --dry-run       # preview, change nothing, still validate
    python mark_missing_bk2s.py --unlock --remove-bk2   # mark events.tsv AND remove duplicate .bk2
    python mark_missing_bk2s.py --validate-only # only run the coherence validation
    python mark_missing_bk2s.py --backup        # write a .bak next to each modified file
    python mark_missing_bk2s.py --dataset /path/to/mario3 --duplicates /path/to/bk2_duplicates.csv

The script is idempotent: rows already set to the placeholder are left as-is and
already-removed .bk2 are skipped, so it is safe to run more than once.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict


REP_RE = re.compile(r'_rep-(\d+)\.bk2$')


def load_missing_reps(duplicates_csv):
    """Return the set of BIDS bk2 paths (stim_file values) whose real recording
    is missing.

    For each duplicate group, the lowest-numbered rep is the survivor and is
    kept; every other rep in the group is considered missing.
    """
    groups = defaultdict(list)
    with open(duplicates_csv, newline='') as f:
        reader = csv.DictReader(f)
        if 'group_id' not in reader.fieldnames or 'bk2_path' not in reader.fieldnames:
            raise SystemExit(
                f"{duplicates_csv}: expected columns 'group_id' and 'bk2_path', "
                f"got {reader.fieldnames}")
        for row in reader:
            path = row['bk2_path'].strip()
            m = REP_RE.search(path)
            if not m:
                # Not a rep-numbered bk2 path; skip defensively.
                continue
            rep = int(m.group(1))
            groups[row['group_id']].append((rep, path))

    missing = set()
    kept = set()
    for members in groups.values():
        members.sort(key=lambda t: t[0])
        kept.add(members[0][1])           # lowest rep -> survivor, keep
        for _rep, path in members[1:]:    # higher reps -> missing
            missing.add(path)
    return missing, kept, len(groups)


def load_groups(duplicates_csv):
    """Return {group_id: [(rep, path, n_frames), ...]} sorted by rep."""
    groups = defaultdict(list)
    with open(duplicates_csv, newline='') as f:
        for row in csv.DictReader(f):
            m = REP_RE.search(row['bk2_path'].strip())
            if not m:
                continue
            try:
                n_frames = int(row.get('n_frames', '') or 0)
            except ValueError:
                n_frames = 0
            groups[row['group_id']].append(
                (int(m.group(1)), row['bk2_path'].strip(), n_frames))
    for members in groups.values():
        members.sort(key=lambda t: t[0])
    return groups


def parse_events_games(path):
    """Parse one events.tsv into an ordered list of gym-retro_game events.

    For each game, the duration of the play is estimated as the time from its
    onset to the onset of the *next fixation_dot* (which marks the end of the
    play / start of the inter-trial period). This boundary is reliable even in
    sessions where many keypress/questionnaire rows are logged between events,
    and unlike the `duration` column it tracks the real .bk2 length.

    Returns list of dicts: {level, onset, est_duration, stim_file}, in file order.
    """
    with open(path, 'r', newline='') as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split('\t')
    try:
        i_tt = header.index('trial_type')
        i_on = header.index('onset')
        i_lv = header.index('level')
        i_sf = header.index('stim_file')
    except ValueError:
        return []

    rows = []
    for ln in lines[1:]:
        c = ln.split('\t')
        def get(i):
            return c[i] if i < len(c) else ''
        onset = get(i_on)
        try:
            onset = float(onset)
        except ValueError:
            onset = None
        rows.append((get(i_tt), onset, get(i_lv), get(i_sf)))

    games = []
    for i, (tt, onset, level, stim) in enumerate(rows):
        if tt != 'gym-retro_game':
            continue
        next_fix = None
        if onset is not None:
            for j in range(i + 1, len(rows)):
                if (rows[j][0] == 'fixation_dot' and rows[j][1] is not None
                        and rows[j][1] > onset):
                    next_fix = rows[j][1]
                    break
        est = (next_fix - onset) if (next_fix is not None) else None
        games.append(dict(level=level, onset=onset, est_duration=est, stim_file=stim))
    return games


def build_event_index(dataset):
    """Index every events.tsv: locate each survivor stim_file and keep the
    ordered per-(file, level) game list so missing reps can be found by position.

    Returns (loc, bylevel):
        loc[stim_file] = (file, level, position_in_level_list)
        bylevel[(file, level)] = [game_dict, ...] in onset/file order
    """
    loc = {}
    bylevel = {}
    tsvs = sorted(glob.glob(os.path.join(dataset, 'sub-*', 'ses-*', 'func',
                                         '*_events.tsv')))
    for tsv in tsvs:
        per_level = defaultdict(list)
        for game in parse_events_games(tsv):
            per_level[game['level']].append(game)
        for level, lst in per_level.items():
            bylevel[(tsv, level)] = lst
            for pos, game in enumerate(lst):
                sf = game['stim_file']
                if sf and sf != 'Missing File':
                    loc[sf] = (tsv, level, pos)
    return loc, bylevel


def validate_coherence(duplicates_csv, dataset, fps=60.0, tol=3.0,
                       weak_margin=2.0, report_csv=None):
    """Check, for every duplicate group, that the kept (lowest) rep is the most
    probable survivor: its .bk2 duration (n_frames / fps) should match the kept
    rep's onset->next-fixation time slot better than any other rep's slot.

    Prints a report and (optionally) writes a per-group CSV. Returns a dict of
    counts.
    """
    groups = load_groups(duplicates_csv)
    loc, bylevel = build_event_index(dataset)

    results = []
    counts = defaultdict(int)
    for gid, members in groups.items():
        krep, kpath, kframes = members[0]
        k = len(members)
        D = kframes / fps if kframes else None

        rec = dict(group_id=gid, kept_rep=krep, kept_bk2_dur=D,
                   verdict='', kept_slot=None, all_slots=None,
                   closest_rep=None, residual=None, margin=None)

        if kpath not in loc:
            rec['verdict'] = 'skip:kept-rep-not-in-events'
            counts['skipped'] += 1
            results.append(rec)
            continue
        tsv, level, pos = loc[kpath]
        lst = bylevel[(tsv, level)]
        if pos + k > len(lst):
            rec['verdict'] = 'skip:group-reps-exceed-event-list'
            counts['skipped'] += 1
            results.append(rec)
            continue

        ests = [lst[pos + j]['est_duration'] for j in range(k)]
        if D is None or any(e is None for e in ests):
            rec['verdict'] = 'skip:missing-duration-or-boundary'
            counts['skipped'] += 1
            rec['all_slots'] = ests
            results.append(rec)
            continue

        diffs = [abs(e - D) for e in ests]
        best_j = min(range(k), key=lambda j: diffs[j])
        ordered = sorted(diffs)
        margin = (ordered[1] - ordered[0]) if k > 1 else None
        rec.update(kept_slot=ests[0], all_slots=ests, closest_rep=members[best_j][0],
                   residual=diffs[0], margin=margin)

        if best_j == 0 and diffs[0] <= tol:
            if margin is not None and margin < weak_margin:
                rec['verdict'] = 'coherent-weak'  # kept is closest, but a sibling is nearly as close
                counts['coherent'] += 1
                counts['weak'] += 1
            else:
                rec['verdict'] = 'coherent'
                counts['coherent'] += 1
        else:
            rec['verdict'] = 'FLAGGED'
            counts['flagged'] += 1
        results.append(rec)

    # ---- report ----
    total = len(groups)
    print("=" * 64)
    print("VALIDATION: is the kept (lowest) rep the most probable survivor?")
    print("  signal = .bk2 duration (n_frames/%.1f) vs onset->next-fixation slot" % fps)
    print("=" * 64)
    print(f"  groups                       : {total}")
    print(f"  coherent (kept = best match) : {counts['coherent']}"
          f"   ({counts['weak']} with a near-tie sibling)")
    print(f"  FLAGGED (needs review)       : {counts['flagged']}")
    print(f"  skipped (no usable boundary) : {counts['skipped']}")
    residuals = [r['residual'] for r in results if r['residual'] is not None]
    if residuals:
        residuals_sorted = sorted(residuals)
        med = residuals_sorted[len(residuals_sorted) // 2]
        print(f"  residual |kept_slot - bk2_dur|: median={med:.2f}s "
              f"max={max(residuals):.2f}s")

    flagged = [r for r in results if r['verdict'] == 'FLAGGED']
    if flagged:
        print("\n  FLAGGED groups (kept rep is NOT the closest, or residual > "
              f"{tol:.0f}s):")
        for r in flagged:
            slots = [round(s, 1) for s in (r['all_slots'] or [])]
            print(f"    group {r['group_id']}: kept rep-{r['kept_rep']:03d} "
                  f"bk2={r['kept_bk2_dur']:.1f}s, rep slots={slots}, "
                  f"closest=rep-{r['closest_rep']:03d}")

    skipped = [r for r in results if r['verdict'].startswith('skip')]
    if skipped:
        print(f"\n  {len(skipped)} group(s) skipped (could not verify):")
        reasons = defaultdict(int)
        for r in skipped:
            reasons[r['verdict']] += 1
        for reason, n in sorted(reasons.items()):
            print(f"    {n:3d}  {reason}")

    if report_csv:
        with open(report_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['group_id', 'kept_rep', 'verdict', 'bk2_duration_s',
                        'kept_onset_slot_s', 'closest_rep', 'residual_s',
                        'margin_to_2nd_s', 'all_rep_slots_s'])
            for r in results:
                w.writerow([
                    r['group_id'], r['kept_rep'], r['verdict'],
                    f"{r['kept_bk2_dur']:.2f}" if r['kept_bk2_dur'] is not None else '',
                    f"{r['kept_slot']:.2f}" if r['kept_slot'] is not None else '',
                    r['closest_rep'] if r['closest_rep'] is not None else '',
                    f"{r['residual']:.2f}" if r['residual'] is not None else '',
                    f"{r['margin']:.2f}" if r['margin'] is not None else '',
                    ';'.join(f"{s:.2f}" for s in r['all_slots']) if r['all_slots'] else '',
                ])
        print(f"\n  Per-group report written to: {report_csv}")

    return counts


def is_annex_locked(path):
    """True if `path` is a git-annex locked file (a symlink into .git/annex)."""
    if not os.path.islink(path):
        return False
    target = os.readlink(path)
    return '.git/annex/' in target or '/annex/objects/' in target


def annex_unlock(dataset, rel_paths):
    """git annex unlock the given paths (relative to `dataset`). Returns True on success."""
    if not rel_paths:
        return True
    try:
        subprocess.run(['git', '-C', dataset, 'annex', 'unlock', '--'] + rel_paths,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = getattr(e, 'stderr', b'')
        if isinstance(msg, bytes):
            msg = msg.decode('utf-8', 'replace')
        print(f"  git annex unlock failed: {msg.strip() or e}", file=sys.stderr)
        return False


def annex_key(dataset, rel_path):
    """Return the git-annex key (symlink target basename) for a tracked file,
    or None if it is not a present annex symlink."""
    p = os.path.join(dataset, rel_path)
    if os.path.islink(p):
        return os.path.basename(os.readlink(p))
    return None


def annex_backend(key):
    """Backend prefix of an annex key, e.g. 'SHA256E' or 'MD5E'."""
    return key.split('-', 1)[0] if key else None


def git_rm(dataset, rel_paths):
    """git rm the given paths (relative to dataset). Returns True on success."""
    if not rel_paths:
        return True
    try:
        subprocess.run(['git', '-C', dataset, 'rm', '--quiet', '--'] + rel_paths,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        msg = getattr(e, 'stderr', b'')
        if isinstance(msg, bytes):
            msg = msg.decode('utf-8', 'replace')
        print(f"  git rm failed: {msg.strip() or e}", file=sys.stderr)
        return False


def remove_duplicate_bk2s(duplicates_csv, dataset, dry_run):
    """Remove the duplicate `.bk2` files (the higher reps of each group),
    keeping only the good one (the lowest rep).

    Safety gate (per group): a higher rep's `.bk2` is removed only when
      - the kept (lowest) rep's `.bk2` is a present annex symlink whose key uses
        the SHA256E backend (the *original* dataset files use SHA256E; the
        duplicates created by the bug use MD5E), AND
      - the higher rep's `.bk2` is an MD5E annex symlink (a duplicate).
    Groups that don't fit this pattern (e.g. group 161: kept rep is itself an
    MD5E file, reps span two sessions with differing content) are FLAGGED and
    left completely untouched.

    Returns a dict of counts.
    """
    groups = load_groups(duplicates_csv)
    to_remove = []          # rel paths to git rm
    kept_paths = []         # rel paths of survivors (left in place)
    flagged = []            # (group_id, reason)
    already_gone = 0

    for gid, members in sorted(groups.items(), key=lambda kv: kv[0]):
        kept_rep, kept_path, _ = members[0]
        kkey = annex_key(dataset, kept_path)
        if kkey is None:
            flagged.append((gid, f"kept rep-{kept_rep:03d} .bk2 is not a present "
                                 f"annex symlink — skipping group"))
            continue
        if annex_backend(kkey) != 'SHA256E':
            flagged.append((gid, f"kept rep-{kept_rep:03d} is {annex_backend(kkey)} "
                                 f"(not the SHA256E original) — skipping group for safety"))
            continue

        group_removals = []
        group_ok = True
        for rep, path, _ in members[1:]:
            full = os.path.join(dataset, path)
            k = annex_key(dataset, path)
            if k is None:
                if not os.path.lexists(full):
                    already_gone += 1            # already removed -> idempotent
                else:
                    flagged.append((gid, f"rep-{rep:03d} .bk2 is a regular file, "
                                         f"not an annex symlink — not removing"))
                    group_ok = False
                continue
            if annex_backend(k) != 'MD5E':
                flagged.append((gid, f"rep-{rep:03d} is {annex_backend(k)} "
                                     f"(not an MD5E duplicate) — not removing"))
                group_ok = False
                continue
            group_removals.append(path)

        if group_removals:
            to_remove.extend(group_removals)
            kept_paths.append(kept_path)

    print("-" * 64)
    print("REMOVE DUPLICATE .bk2 (keep only the good/lowest rep)")
    print("-" * 64)
    print(f"  groups handled                : {len(kept_paths)}")
    print(f"  duplicate .bk2 to remove      : {len(to_remove)}")
    if already_gone:
        print(f"  already removed (skipped)     : {already_gone}")
    if flagged:
        print(f"  FLAGGED groups (left untouched): {len({g for g, _ in flagged})}")
        for gid, reason in flagged:
            print(f"      group {gid}: {reason}")

    if to_remove and not dry_run:
        ok = git_rm(dataset, to_remove)
        if not ok:
            raise SystemExit("git rm failed; no .bk2 files were removed.")
        print(f"\n  Removed {len(to_remove)} duplicate .bk2 (staged as deletions).")
    elif to_remove:
        print(f"\n  [dry-run] would remove {len(to_remove)} duplicate .bk2.")

    return dict(removed=len(to_remove), kept=len(kept_paths),
                already_gone=already_gone, flagged=len({g for g, _ in flagged}))


def process_events_file(path, missing, placeholder, dry_run):
    """Rewrite missing-rep stim_file values in a single events.tsv.

    Returns (n_rows_changed, n_rows_already_marked).
    """
    with open(path, 'r', newline='') as f:
        text = f.read()

    lines = text.splitlines(keepends=True)
    if not lines:
        return 0, 0

    # Locate the stim_file column from the header.
    header = lines[0].rstrip('\r\n').split('\t')
    try:
        stim_idx = header.index('stim_file')
    except ValueError:
        return 0, 0  # no stim_file column -> nothing to do

    changed = 0
    already = 0
    changed_paths = set()
    out_lines = [lines[0]]
    for line in lines[1:]:
        # Separate the line body from its newline so we can restore it exactly.
        nl = ''
        body = line
        if body.endswith('\r\n'):
            nl, body = '\r\n', body[:-2]
        elif body.endswith('\n'):
            nl, body = '\n', body[:-1]
        elif body.endswith('\r'):
            nl, body = '\r', body[:-1]

        fields = body.split('\t')
        if len(fields) > stim_idx:
            value = fields[stim_idx]
            if value in missing:
                fields[stim_idx] = placeholder
                changed += 1
                changed_paths.add(value)
                body = '\t'.join(fields)
            elif value == placeholder:
                already += 1
        out_lines.append(body + nl)

    if changed and not dry_run:
        with open(path, 'w', newline='') as f:
            f.write(''.join(out_lines))

    return changed, already, changed_paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mark missing duplicate reps as 'Missing File' in a Mario3 "
                    "BIDS dataset's events.tsv files.")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--dataset', default=os.path.join(here, 'mario3'),
                        help="Path to the mario3 BIDS dataset (default: ./mario3)")
    parser.add_argument('--duplicates', default=os.path.join(here, 'bk2_duplicates.csv'),
                        help="Path to bk2_duplicates.csv (default: ./bk2_duplicates.csv)")
    parser.add_argument('--placeholder', default='Missing File',
                        help="String to write into stim_file for missing reps "
                             "(default: 'Missing File')")
    parser.add_argument('--dry-run', action='store_true',
                        help="Report what would change without modifying any file.")
    parser.add_argument('--backup', action='store_true',
                        help="Write a '<file>.bak' copy before modifying each file.")
    parser.add_argument('--unlock', action='store_true',
                        help="git annex unlock locked events.tsv files before "
                             "editing (the dataset is a DataLad/git-annex repo, so "
                             "its files are read-only symlinks by default). After "
                             "running, commit with: datalad save -m '...'")
    parser.add_argument('--validate-only', action='store_true',
                        help="Only run the coherence validation (no file edits).")
    parser.add_argument('--no-validate', action='store_true',
                        help="Skip the coherence validation that normally runs "
                             "after marking.")
    parser.add_argument('--fps', type=float, default=60.0,
                        help="Emulator frame rate used to convert .bk2 n_frames "
                             "to seconds during validation (default: 60.0).")
    parser.add_argument('--remove-bk2', action='store_true',
                        help="Also remove the duplicate .bk2 files (the higher "
                             "reps of each group), keeping only the good/lowest "
                             "rep. Uses 'git rm' on the annex symlinks; commit "
                             "afterwards with datalad save. Safe: only MD5E "
                             "duplicates are removed and only when the kept rep "
                             "is the SHA256E original.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset):
        raise SystemExit(f"Dataset directory not found: {args.dataset}")
    if not os.path.isfile(args.duplicates):
        raise SystemExit(f"Duplicates CSV not found: {args.duplicates}")

    report_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'bk2_kept_validation.csv')

    if args.validate_only:
        validate_coherence(args.duplicates, args.dataset, fps=args.fps,
                           report_csv=report_csv)
        return 0

    missing, kept, n_groups = load_missing_reps(args.duplicates)
    print(f"Duplicate groups            : {n_groups}")
    print(f"Survivors kept (lowest rep) : {len(kept)}")
    print(f"Missing reps to mark        : {len(missing)}")
    print(f"Placeholder                 : {args.placeholder!r}")
    print(f"Mode                        : {'DRY-RUN (no files written)' if args.dry_run else 'APPLY'}")
    print()

    tsv_files = sorted(glob.glob(os.path.join(args.dataset, 'sub-*', 'ses-*',
                                              'func', '*_events.tsv')))
    if not tsv_files:
        raise SystemExit(f"No events.tsv files found under {args.dataset}")

    # Figure out which files actually need changes (dry pass), so we only
    # unlock / back up the ones we will touch.
    files_to_change = []
    for tsv in tsv_files:
        n, _, _ = process_events_file(tsv, missing, args.placeholder, dry_run=True)
        if n:
            files_to_change.append(tsv)

    # This is a DataLad/git-annex dataset: events.tsv are read-only symlinks.
    locked = [t for t in files_to_change if is_annex_locked(t)]
    if locked and not args.dry_run:
        if args.unlock:
            print(f"Unlocking {len(locked)} annexed file(s) with git annex unlock...")
            rels = [os.path.relpath(t, args.dataset) for t in locked]
            if not annex_unlock(args.dataset, rels):
                raise SystemExit("Unlock failed; aborting without modifying files.")
            print()
        else:
            raise SystemExit(
                f"{len(locked)} target events.tsv file(s) are git-annex locked "
                f"(read-only symlinks).\n"
                f"Re-run with --unlock to unlock them automatically, or unlock "
                f"manually first, e.g.:\n"
                f"    datalad unlock -d {args.dataset} <files>\n"
                f"Then commit afterwards with: datalad save -d {args.dataset} "
                f"-m 'mark missing bk2 reps'")

    total_changed = 0
    total_already = 0
    files_modified = 0
    matched_paths = set()

    for tsv in tsv_files:
        # Back up only files we will actually change.
        if args.backup and not args.dry_run and tsv in files_to_change:
            with open(tsv, 'rb') as src, open(tsv + '.bak', 'wb') as dst:
                dst.write(src.read())

        changed, already, changed_paths = process_events_file(
            tsv, missing, args.placeholder, args.dry_run)
        total_changed += changed
        total_already += already
        matched_paths |= changed_paths
        if changed:
            files_modified += 1
            rel = os.path.relpath(tsv, args.dataset)
            print(f"  {'would mark' if args.dry_run else 'marked'} {changed:3d} "
                  f"rep(s)  {rel}")

    print()
    print(f"events.tsv files scanned    : {len(tsv_files)}")
    print(f"events.tsv files modified   : {files_modified}")
    print(f"rows {'that would be ' if args.dry_run else ''}marked Missing File : {total_changed}")
    if total_already:
        print(f"rows already marked         : {total_already} (left unchanged)")

    # A missing rep is "accounted for" if we just marked it, or it was already
    # marked on a previous run. Anything left over never appeared in any
    # events.tsv (possible CSV/dataset mismatch worth surfacing).
    accounted = total_changed + total_already
    if accounted < len(missing):
        print()
        print(f"WARNING: {len(missing) - accounted} missing-rep path(s) from the "
              f"CSV were not found in any events.tsv (neither marked now nor "
              f"already marked).")
        if total_already == 0:
            not_found = missing - matched_paths
            for p in sorted(not_found)[:20]:
                print(f"    {p}")
            if len(not_found) > 20:
                print(f"    ... and {len(not_found) - 20} more")

    removed_count = 0
    if args.remove_bk2:
        print()
        rm_counts = remove_duplicate_bk2s(args.duplicates, args.dataset,
                                          args.dry_run)
        removed_count = rm_counts['removed']

    if not args.no_validate:
        print()
        validate_coherence(args.duplicates, args.dataset, fps=args.fps,
                           report_csv=report_csv)

    if not args.dry_run and (total_changed or removed_count):
        print()
        print("Done. The dataset is a DataLad repo — commit the changes with:")
        msg = ('mark missing duplicate bk2 reps in events.tsv'
               + (' and remove duplicate .bk2 files' if removed_count else ''))
        print(f"    datalad save -d {args.dataset} -m '{msg}'")

    return 0


if __name__ == '__main__':
    sys.exit(main())
