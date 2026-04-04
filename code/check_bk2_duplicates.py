"""
Check for duplicate action sequences across all .bk2 files in the dataset.

Two replays are considered duplicates if:
  - their action sequences are identical, OR
  - one sequence equals the other with its first frame removed
    (i.e. they are the same replay but one has one extra leading frame)

Strategy: index each file under both hash(frames) and hash(frames[1:]).
Two files sharing any hash key are candidate duplicates; union-find merges
transitive groups.

Outputs a CSV listing all files that share an action sequence with at least
one other file. Each row is one file; files in the same duplicate group share
the same group_id.

Usage:
    python code/check_bk2_duplicates.py [dataset_root] [--output path/to/out.csv]

    dataset_root defaults to the directory containing this script's parent.
    output      defaults to <dataset_root>/code/bk2_duplicates.csv
"""

import csv
import hashlib
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


def parse_actions(bk2_path):
    """Return list of frozensets of pressed buttons, one per frame.

    BizHawk NES input log format:
      header:  P1 A|P1 Right|P1 Left|P1 Down|P1 Up|P1 Start|P1 Select|P1 B|
      data:    |..|........|
                  ^^         field 0: 2 system buttons (not in header, ignored)
                    ^^^^^^^^ field 1: 8 P1 buttons, one char per position;
                             '.' = not pressed, anything else = pressed
    """
    with zipfile.ZipFile(bk2_path) as z:
        with z.open("Input Log.txt") as log:
            lines = log.read().decode().strip().split("\n")
    header = lines[1]
    buttons = [b.strip() for b in header.split("|") if b.strip()]
    frames = []
    for line in lines[2:]:
        parts = line.strip("|").split("|")
        p1_field = parts[1] if len(parts) > 1 else ""
        pressed = frozenset(
            buttons[i] for i, c in enumerate(p1_field)
            if i < len(buttons) and c != "."
        )
        frames.append(pressed)
    return frames


def seq_hash(frames):
    raw = "|".join(",".join(sorted(f)) for f in frames)
    return hashlib.sha256(raw.encode()).hexdigest()


# --- Union-Find ---

def make_uf(keys):
    return {k: k for k in keys}


def find(uf, x):
    while uf[x] != x:
        uf[x] = uf[uf[x]]
        x = uf[x]
    return x


def union(uf, x, y):
    px, py = find(uf, x), find(uf, y)
    if px != py:
        uf[px] = py


def main():
    args = sys.argv[1:]
    output_path = None
    if "--output" in args:
        idx = args.index("--output")
        output_path = Path(args.pop(idx + 1))
        args.pop(idx)

    dataset_root = Path(args[0]) if args else Path(__file__).parent.parent

    if output_path is None:
        output_path = Path(__file__).parent / "bk2_duplicates.csv"

    bk2_files = sorted(
        p for p in dataset_root.rglob("*.bk2")
        if ".git" not in p.parts
    )
    print(f"Found {len(bk2_files)} .bk2 files. Hashing action sequences...")

    # For each file store: full hash and body hash (frames[1:])
    # Two files are duplicates if they share either hash.
    full_hash = {}   # path -> hash(frames)
    body_hash = {}   # path -> hash(frames[1:])
    n_frames  = {}   # path -> len(frames)
    errors    = []

    full_map = defaultdict(list)  # hash -> [path]  (full sequence)
    body_map = defaultdict(list)  # hash -> [path]  (sequence without first frame)

    for bk2 in bk2_files:
        try:
            frames = parse_actions(bk2)
            fh = seq_hash(frames)
            bh = seq_hash(frames[1:]) if len(frames) > 1 else fh
            full_hash[bk2] = fh
            body_hash[bk2] = bh
            n_frames[bk2]  = len(frames)
            full_map[fh].append(bk2)
            body_map[bh].append(bk2)
        except Exception as e:
            errors.append((bk2, str(e)))

    # Build union-find over all file paths
    uf = make_uf(bk2_files)

    # Rule 1: identical full sequences
    for paths in full_map.values():
        for p in paths[1:]:
            union(uf, paths[0], p)

    # Rule 2: one file's full sequence == another's body (extra leading frame)
    #   full_map[h] contains files whose full sequence hashes to h
    #   body_map[h] contains files whose frames[1:] hashes to h
    #   If h appears in both maps, the full_map files are 1-frame-shorter
    #   duplicates of the body_map files.
    for h, full_paths in full_map.items():
        if h in body_map:
            body_paths = body_map[h]
            all_paths = full_paths + body_paths
            for p in all_paths[1:]:
                union(uf, all_paths[0], p)

    # Collect groups with more than one member
    groups = defaultdict(list)
    for p in bk2_files:
        groups[find(uf, p)].append(p)

    duplicate_groups = {root: paths for root, paths in groups.items() if len(paths) > 1}
    total = sum(len(v) for v in duplicate_groups.values())
    print(f"Found {len(duplicate_groups)} duplicate groups involving {total} files.")

    if errors:
        print(f"\n{len(errors)} files could not be parsed:")
        for path, err in errors:
            print(f"  {path.relative_to(dataset_root)}: {err}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "bk2_path", "n_frames"])
        for group_id, (_, paths) in enumerate(sorted(duplicate_groups.items()), start=1):
            for p in sorted(paths):
                writer.writerow([group_id, str(p.relative_to(dataset_root)), n_frames[p]])

    print(f"CSV written to: {output_path}")


if __name__ == "__main__":
    main()
