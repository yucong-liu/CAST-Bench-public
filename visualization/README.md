# visualization/

`build_index.py` is the scoring module used by
[`evaluation/aggregate_18_metrics.py`](../evaluation/aggregate_18_metrics.py).

It defines `SUMMARY_METRICS` (the 18 metrics, their groups, directions and
source CSVs) and `build_score_summary()`, which normalises every metric to
0-100 and computes group and overall scores.

Only those two symbols are used by the aggregator. The module also contains the
index builder for the project's web viewer; that entry point expects the full
internal workspace layout (`t2av-bench/`, model video roots, `runs/`) and is not
needed to reproduce the paper's tables.

The workspace root defaults to the repository root and can be overridden with
the `CAST_BENCH_WS` environment variable.
