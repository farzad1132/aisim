# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-hour ledger of a collection: what each entry cost and what the whole campaign cost.

A job holds its GPUs for its whole wall time, so an entry's cost is the wall time it ran inside the job
times the job's GPU count. The ledger adds the unattributed remainder (container start, image load,
idle) as its own row, so the per-entry costs plus the remainder equal the Slurm allocation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aisim_core.data.gpu_time import GpuTime, gpu_time_from_sacct


@dataclass(frozen=True)
class EntryCost:
    system: str
    framework: str
    tokenizer: str
    exit_status: int
    wall_seconds: float
    gpu_seconds: float


@dataclass(frozen=True)
class JobCost:
    system: str
    job_id: str
    gpu_time: GpuTime
    entries: tuple[EntryCost, ...]
    note: str = ""  # e.g. "superseded": the job's cost is kept in the ledger but its data is not used

    @property
    def unattributed_gpu_seconds(self) -> float:
        return float(self.gpu_time.gpu_seconds) - sum(entry.gpu_seconds for entry in self.entries)


def read_entries(entries_txt: str | Path, *, system: str, gpu_count: int) -> tuple[EntryCost, ...]:
    """Parse ``entries.txt`` written by ``run_entries.sh``: ``framework key exit_status start end`` per line."""
    entries = []
    for line in Path(entries_txt).read_text().splitlines():
        if not line.strip():
            continue
        framework, tokenizer, status, start, end = line.split()
        wall = float(end) - float(start)
        entries.append(EntryCost(system, framework, tokenizer, int(status), wall, wall * gpu_count))
    return tuple(entries)


def job_cost(
    system: str, job_id: str, sacct_text: str, entries_txt: str | Path, *, gpu_type: str, note: str = ""
) -> JobCost:
    """Cost of one job from its ``sacct --parsable2`` text and its ``entries.txt``."""
    gpu_time = gpu_time_from_sacct(sacct_text, gpu_type=gpu_type)
    if gpu_time.gpu_count is None:
        raise ValueError(f"job {job_id}: sacct text carries no GPU allocation")
    entries = read_entries(entries_txt, system=system, gpu_count=gpu_time.gpu_count)
    return JobCost(system, job_id, gpu_time, entries, note)


def render_markdown(jobs: list[JobCost]) -> str:
    """The cost ledger: one row per entry, one per job remainder, and the campaign total."""
    lines = [
        "| System | Job | Framework | Tokenizer | Exit | Wall (s) | GPU-hours |",
        "|---|---|---|---|---|---|---|",
    ]
    total = 0.0
    for job in jobs:
        label = f" ({job.note})" if job.note else ""
        for entry in job.entries:
            lines.append(
                f"| {entry.system} | {job.job_id} | {entry.framework}{label} | {entry.tokenizer} | {entry.exit_status} "
                f"| {entry.wall_seconds:.0f} | {entry.gpu_seconds / 3600:.4f} |"
            )
        rest = job.unattributed_gpu_seconds
        lines.append(
            f"| {job.system} | {job.job_id} | (unattributed: start-up, image load, idle) | | | | {rest / 3600:.4f} |"
        )
        total += float(job.gpu_time.gpu_seconds)
    lines.append(f"| **Total** | {len(jobs)} jobs | | | | | **{total / 3600:.3f}** |")
    return "\n".join(lines) + "\n"
