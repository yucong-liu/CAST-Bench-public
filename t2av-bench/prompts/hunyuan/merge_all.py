#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge 9 categories' per-category CSVs into two aggregate CSVs.

Output (in this same directory):
    all_single_10s.csv   -> concat of <cat>/<cat>_single_10s.csv
    all_multi_10s.csv    -> concat of <cat>/<cat>_multi_10s.csv

- Category order: alphabetical (action, ambient, animal, bgm, instrument,
  non_verbal_vocal, object, sfx, speech).
- Header from the first file is kept; others' headers are skipped.
- The `index` column is re-numbered globally starting from 0.
- All other columns are preserved as-is.
- Encoding: utf-8-sig (matches the source files).
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = [
    "action",
    "ambient",
    "animal",
    "bgm",
    "instrument",
    "non_verbal_vocal",
    "object",
    "sfx",
    "speech",
]


def merge(shot: str, out_name: str) -> None:
    out_path = os.path.join(HERE, out_name)
    header = None
    rows = []

    for cat in CATEGORIES:
        src = os.path.join(HERE, cat, f"{cat}_{shot}_10s.csv")
        if not os.path.exists(src):
            print(f"[WARN] missing: {src}", file=sys.stderr)
            continue

        with open(src, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            cur_header = next(reader)
            if header is None:
                header = cur_header
            elif cur_header != header:
                print(
                    f"[WARN] header mismatch in {src}\n  expected: {header}\n  got:      {cur_header}",
                    file=sys.stderr,
                )
            n = 0
            for row in reader:
                if not row:
                    continue
                rows.append(row)
                n += 1
            print(f"  {cat:<18s} {shot:<6s} +{n} rows")

    # Re-number the `index` column globally, starting from 0.
    try:
        idx_col = header.index("index")
    except ValueError:
        print("[ERR] no 'index' column in header", file=sys.stderr)
        sys.exit(1)

    for i, row in enumerate(rows):
        row[idx_col] = str(i)

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"=> wrote {out_path}  (total {len(rows)} rows, index 0..{len(rows)-1})\n")


def main() -> None:
    print(">>> merging SINGLE-shot CSVs")
    merge("single", "all_single_10s.csv")
    print(">>> merging MULTI-shot CSVs")
    merge("multi", "all_multi_10s.csv")


if __name__ == "__main__":
    main()
