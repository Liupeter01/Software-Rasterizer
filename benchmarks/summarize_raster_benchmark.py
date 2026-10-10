#!/usr/bin/env python3
"""Recompute frame-time percentiles from raw CSV; never average percentiles.

Usage: python3 benchmarks/summarize_raster_benchmark.py run1.csv run2.csv ...
Only combine runs of the same workload/build. Matching JSON metadata is checked.
"""
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def percentile(values, q):
    values = sorted(values)
    rank = q * (len(values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def summarize(paths):
    groups = {}
    baseline = None
    keys = ("obj", "width", "height", "input_triangles", "covered_pixels",
            "primitive", "shader", "compiler", "build_type", "tbb_enabled", "scope")
    for path in paths:
        metadata = json.loads(path.with_suffix(".json").read_text())
        workload = {key: metadata[key] for key in keys}
        if baseline is not None and workload != baseline:
            raise ValueError("cannot pool different workloads/builds")
        baseline = workload
        local = {}
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                mode, value = row["mode"], float(row["draw_ms"])
                if mode not in ("scalar", "simd") or not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid measured sample")
                local.setdefault(mode, []).append(value)
                groups.setdefault(mode, []).append(value)
        for mode in local:
            if len(local[mode]) != metadata["frames_per_mode"]:
                raise ValueError("CSV sample count disagrees with metadata")
        for result in metadata["results"]:
            values = local[result["mode"]]
            for q in (.10, .50, .90, .95, .99):
                key = f"p{round(q * 100)}_ms"
                if not math.isclose(percentile(values, q), result[key], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"CSV and JSON disagree: {key}")
    results = []
    for mode, values in sorted(groups.items()):
        result = {"mode": mode, "samples": len(values), "mean_ms": statistics.mean(values),
                  "min_ms": min(values), "max_ms": max(values)}
        result.update({f"p{round(q * 100)}_ms": percentile(values, q)
                       for q in (.10, .50, .90, .95, .99)})
        result["draws_per_second"] = 1000 / result["mean_ms"]
        results.append(result)
    return {"files": [str(p) for p in paths], "workload": baseline,
            "percentile": "pooled raw samples, linear interpolation at q*(N-1), type 7",
            "results": results}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("provide one or more benchmark CSV files")
    print(json.dumps(summarize([Path(p) for p in sys.argv[1:]]), indent=2))
