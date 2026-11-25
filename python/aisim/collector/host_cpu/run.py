# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time one (framework, tokenizer) entry on this host and write per-request rows plus its environment.

Usage (inside the framework image, on the serving host CPU)::

    PYTHONPATH=python/aisim python -m collector.host_cpu.run \\
        --framework vllm --tokenizer qwen2 --model Qwen/Qwen3.5-397B-A17B \\
        --payloads raw.jsonl --count 60 --repeats 3 --output out/vllm-qwen2

Each request is timed ``repeats`` times after ``warmup`` untimed passes; the median per component is
written, so one slow pass does not enter the fit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS, Adapter
from .workload import load_payloads, workload_digest

SCHEMA = "aisim-host-cost-run/v1"


def host_environment() -> dict[str, Any]:
    """The host facts a fit depends on."""
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
    fields = dict(line.split(":", 1) for line in lscpu.splitlines() if ":" in line)
    fields = {key.strip(): value.strip() for key, value in fields.items()}
    return {
        "cpu_model": fields.get("Model name"),
        "cpu_sockets": fields.get("Socket(s)"),
        "cpu_threads": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "load_average_1m": os.getloadavg()[0],
        "hostname": platform.node(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
    }


def time_entry(adapter: Adapter, bodies: list[dict[str, Any]], warmup: int, repeats: int) -> list[dict[str, Any]]:
    """Median-of-``repeats`` per-request component times for every body."""
    for body in bodies[: max(warmup, 0)]:
        adapter.time_request(body)
    rows = []
    gc.collect()
    gc.disable()
    try:
        for index, body in enumerate(bodies):
            samples = [adapter.time_request(body) for _ in range(repeats)]
            row = {"request": index, "tokens": samples[0]["tokens"], "body_kb": samples[0]["body_kb"]}
            if any(sample["tokens"] != row["tokens"] for sample in samples):
                raise RuntimeError(f"request {index} tokenized to different lengths across repeats")
            for key in samples[0]:
                if key.endswith("_kb") or key == "tokens":
                    continue
                row[key + "_ms"] = statistics.median(sample[key] for sample in samples)
            rows.append(row)
    finally:
        gc.enable()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--framework", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--tokenizer", required=True, help="tokenizer family key, e.g. qwen2")
    parser.add_argument("--model", required=True, help="HF id or local path holding the tokenizer files")
    parser.add_argument("--payloads", type=Path, required=True, help="AgentX raw.jsonl with request payloads")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    bodies = list(load_payloads(args.payloads, args.count))
    if not bodies:
        print("no profiling-phase payloads found", file=sys.stderr)
        return 2
    started = time.time()
    adapter = ADAPTERS[args.framework](args.model)
    load_seconds = time.time() - started
    started = time.time()
    rows = time_entry(adapter, bodies, args.warmup, args.repeats)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    environment = {
        "schema": SCHEMA,
        "framework": args.framework,
        "tokenizer_key": args.tokenizer,
        "model": args.model,
        "requests": len(rows),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "workload_sha256": workload_digest(bodies),
        "tokens_min": min(row["tokens"] for row in rows),
        "tokens_max": max(row["tokens"] for row in rows),
        "load_seconds": load_seconds,
        "timing_seconds": time.time() - started,
        **adapter.environment(),
        "host": host_environment(),
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    print(f"{args.framework}/{args.tokenizer}: {len(rows)} requests, {environment['timing_seconds']:.1f}s timed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
