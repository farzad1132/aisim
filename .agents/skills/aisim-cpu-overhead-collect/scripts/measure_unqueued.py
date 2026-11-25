# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure host time outside the engine per request at one lane, and fit the unqueued overhead.

For each successful profiling request, the client gives the start time, client TTFT and input
tokens (an AIPerf ``profile_export.jsonl``). The engine gives the request's prefill steps (FPM step
records carrying ``host_scheduled_at_s`` and ``host_emitted_at_s`` on the server monotonic clock,
plus the recorder's ``received_unix_s``/``received_monotonic_s`` pair that maps it to wall time).

    before engine = first prefill step scheduled - request start
    after engine  = client first token - last prefill step emitted
    outside       = before + after

It fits ``outside = base_ms + total_us_per_token * input_tokens`` and subtracts the queued tokenizer
slope to give the unqueued ``engine.extra_cpu_overhead``. Only one request may be in flight, so run
it on a one-lane capture.

Usage:
    python measure_unqueued.py --client profile_export.jsonl --steps fpm.jsonl[.gz] \
        --tokenizer-us 1.217 [--output result.json]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def client_requests(path: Path, phase: str) -> list[tuple[float, float, int]]:
    rows = []
    with path.open() as source:
        for line in source:
            record = json.loads(line)
            meta, metrics = record["metadata"], record["metrics"]
            if meta.get("benchmark_phase") != phase or meta.get("was_cancelled"):
                continue
            if "time_to_first_token" not in metrics or "input_sequence_length" not in metrics:
                continue
            rows.append(
                (
                    meta["request_start_ns"] / 1e9,
                    metrics["time_to_first_token"]["value"] / 1e3,
                    int(metrics["input_sequence_length"]["value"]),
                )
            )
    return sorted(rows)


def prefill_steps(path: Path) -> np.ndarray:
    offsets, steps = [], []
    with _open(path) as source:
        for line in source:
            record = json.loads(line)
            offsets.append(record["received_unix_s"] - record["received_monotonic_s"])
            scheduled = record["fpm"]["scheduled_requests"]
            if scheduled["num_prefill_requests"] > 0 and "host_scheduled_at_s" in scheduled:
                steps.append((scheduled["host_scheduled_at_s"], scheduled["host_emitted_at_s"]))
    if not steps:
        raise SystemExit("no prefill steps with host timestamps: capture with the FPM step hooks")
    return np.asarray(steps) + float(np.median(offsets))


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.c_[np.ones(len(x)), x]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    r2 = 1 - ((y - design @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"base_ms": float(coef[0]), "us_per_token": float(coef[1]), "r2": float(r2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client", type=Path, required=True, help="AIPerf profile_export.jsonl")
    parser.add_argument("--steps", type=Path, required=True, help="FPM step records (.jsonl or .jsonl.gz)")
    parser.add_argument("--tokenizer-us", type=float, required=True, help="Queued tokenizer slope, us/token")
    parser.add_argument("--phase", default="profiling")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    requests = client_requests(args.client, args.phase)
    steps = prefill_steps(args.steps)
    rows = []
    for start, ttft, tokens in requests:
        end = start + ttft
        inside = steps[(steps[:, 0] >= start) & (steps[:, 0] <= end)]
        if len(inside):
            before = (inside[:, 0].min() - start) * 1e3
            after = (end - inside[:, 1].max()) * 1e3
            rows.append((tokens, before, after))
    if len(rows) < 30:
        raise SystemExit(f"only {len(rows)} requests matched a prefill span; need at least 30")
    data = np.asarray(rows, dtype=float)
    tokens_k = data[:, 0] / 1e3
    outside = fit(tokens_k, data[:, 1] + data[:, 2])
    result = {
        "requests": len(requests),
        "matched": len(rows),
        "median_input_tokens": float(np.median(data[:, 0])),
        "before_engine_median_ms": float(np.median(data[:, 1])),
        "after_engine_median_ms": float(np.median(data[:, 2])),
        "outside_engine_fit": outside,
        "tokenizer_us_per_token": args.tokenizer_us,
        "extra_cpu_overhead": {
            "base_ms": max(outside["base_ms"], 0.0),
            "per_input_token_us": max(outside["us_per_token"] - args.tokenizer_us, 0.0),
        },
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
