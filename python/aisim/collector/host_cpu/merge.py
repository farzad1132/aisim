# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge independent runs of one entry: per-request median across runs.

A shared GPU node's host CPU is noisy from run to run. Independent jobs time the same requests; the median
of each request's component times across jobs removes most of that noise. Runs must have timed the same
request set (same workload hash) with the same software.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Facts that must agree for two runs to be one entry.
_IDENTITY = (
    "framework",
    "framework_version",
    "tokenizer_key",
    "model",
    "tokenizer_class",
    "workload_sha256",
    "requests",
)


def merge_runs(run_dirs: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    """Write ``rows.jsonl`` and ``environment.json`` for the merged entry; return its environment."""
    runs = [Path(path) for path in run_dirs]
    if not runs:
        raise ValueError("need at least one run")
    environments = [json.loads((run / "environment.json").read_text()) for run in runs]
    for key in _IDENTITY:
        if len({json.dumps(env.get(key), sort_keys=True) for env in environments}) != 1:
            raise ValueError(f"runs disagree on {key}; they are not the same entry")
    tables = [
        [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines() if line.strip()] for run in runs
    ]
    merged = []
    for rows in zip(*tables, strict=True):
        if len({row["tokens"] for row in rows}) != 1:
            raise ValueError(f"request {rows[0]['request']} tokenized differently across runs")
        row = {"request": rows[0]["request"], "tokens": rows[0]["tokens"], "body_kb": rows[0]["body_kb"]}
        for key in rows[0]:
            if key.endswith("_ms"):
                row[key] = statistics.median(r[key] for r in rows)
        merged.append(row)
    environment = dict(environments[0])
    hosts = [env["host"] for env in environments]
    environment["merged_runs"] = len(runs)
    environment["host"] = {
        **hosts[0],
        "hostnames": [host.get("hostname") for host in hosts],
        "slurm_job_ids": [host.get("slurm_job_id") for host in hosts],
        "load_average_1m": [host["load_average_1m"] for host in hosts],
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged))
    (out / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    return environment
