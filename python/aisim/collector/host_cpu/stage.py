# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a campaign directory into ``host-cost`` dataset entries plus the GPU-hour ledger and a fit table.

Campaign layout (one directory per Slurm job, fetched from the cluster)::

    <campaign>/<system>/<framework>-<n>/{job.txt, entries.txt, sacct.txt, <framework>/<tokenizer>/...}

Independent jobs of the same (system, framework) are merged per request (median) into one entry.

The entries under ``--output`` are public. With ``--internal``, each entry also gets
``evidence/<entry directory>/`` there: its host names, job IDs and node load (``host.json``) and its jobs'
Slurm records, listed in an ``evidence-manifest.json``. That root belongs in ``aisim-data-internal``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from aisim_core.data import Manifest
from aisim_core.data.evidence import EVIDENCE_MANIFEST, EVIDENCE_ROOT, EVIDENCE_SCHEMA
from aisim_core.data.gpu_time import GpuTime
from aisim_core.data.host_cost import CATEGORY, internal_record, read_fit, write_entry
from aisim_core.host_cost.fit import fit_rows

from .config import TOKENIZER_MODELS
from .ledger import JobCost, job_cost, read_entries, render_markdown
from .merge import merge_runs

GPU_TYPES = {"b200_sxm": "B200", "b300_sxm": "B300", "h100_sxm": "H100", "h200_sxm": "H200"}


def _job_id(job_dir: Path) -> str:
    match = re.search(r"job=(\d+)", (job_dir / "job.txt").read_text())
    if match is None:
        raise ValueError(f"{job_dir}: job.txt has no job id")
    return match.group(1)


MIN_R = 0.98
MAX_DROPPED_FRACTION = 0.05


def _quality_failure(rows: list[dict]) -> str | None:
    """Why an entry is rejected (recorded, never averaged in), or ``None``."""
    try:
        fit = fit_rows(rows)
    except ValueError as exc:
        return str(exc)
    problems = []
    if fit.r < MIN_R:
        problems.append(f"r {fit.r:.3f} < {MIN_R}")
    if fit.n_dropped > MAX_DROPPED_FRACTION * (fit.n + fit.n_dropped):
        problems.append(f"{fit.n_dropped} of {fit.n + fit.n_dropped} requests dropped as outliers")
    return "; ".join(problems) or None


_JOB_RECORDS = ("job.txt", "entries.txt", "sacct.txt")


def _write_internal(internal: Path, manifest_path: str, configuration_id: str, record: dict, jobs: list[Path]) -> None:
    """Write the internal evidence of one public entry: its host record and its jobs' Slurm records."""
    config_path = manifest_path.rsplit("/", 1)[0]
    directory = internal / EVIDENCE_ROOT / config_path
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    files = {"host.json": ("host-record", json.dumps(record, indent=2, sort_keys=True) + "\n")}
    for job in jobs:
        for name in _JOB_RECORDS:
            files[f"jobs/{job.name}/{name}"] = ("slurm-job-record", (job / name).read_text())
    artifacts = []
    for rel, (kind, text) in sorted(files.items()):
        target = directory / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode()
        target.write_bytes(data)
        artifacts.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "kind": kind})
    manifest = {
        "schema_version": EVIDENCE_SCHEMA,
        "configuration_id": configuration_id,
        "category": CATEGORY,
        "config_path": config_path,
        "artifacts": artifacts,
    }
    (directory / EVIDENCE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def stage(
    campaign: Path, root: Path, evidence: Path, superseded: Path | None = None, internal: Path | None = None
) -> list[str]:
    """Write every public entry under ``root``, its cluster records under ``internal``, and the ledger and fit
    table under ``evidence``; return manifest paths."""
    manifests: list[str] = []
    failed: list[str] = []
    jobs: list[JobCost] = []
    fit_rows_md = [
        "| System | Framework | Tokenizer | Tokenizer class | µs/token | base ms | r | n | CI95 µs/token | Runs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system_dir in sorted(p for p in campaign.iterdir() if p.is_dir()):
        system = system_dir.name
        job_dirs = sorted(p for p in system_dir.iterdir() if p.is_dir())
        for job_dir in job_dirs:
            jobs.append(
                job_cost(
                    system,
                    _job_id(job_dir),
                    (job_dir / "sacct.txt").read_text(),
                    job_dir / "entries.txt",
                    gpu_type=GPU_TYPES[system],
                )
            )
        by_framework: dict[str, list[Path]] = {}
        for job_dir in job_dirs:
            by_framework.setdefault(job_dir.name.rsplit("-", 1)[0], []).append(job_dir)
        for framework, dirs in sorted(by_framework.items()):
            tokenizers = sorted({p.name for d in dirs for p in (d / framework).iterdir() if p.is_dir()})
            for tokenizer in tokenizers:
                runs = [d / framework / tokenizer for d in dirs if (d / framework / tokenizer / "rows.jsonl").is_file()]
                wall = 0.0
                job_ids = []
                job_dirs_used = []
                for d in dirs:
                    for entry in read_entries(d / "entries.txt", system=system, gpu_count=1):
                        if entry.framework == framework and entry.tokenizer == tokenizer and entry.exit_status == 0:
                            wall += entry.wall_seconds
                            job_ids.append(_job_id(d))
                            job_dirs_used.append(d)
                gpu_time = GpuTime(
                    authority="measured",
                    gpu_type=GPU_TYPES[system],
                    gpu_count=1,
                    wall_seconds=wall,
                    # Job IDs identify the cluster; they go to the internal record only.
                    source={
                        "source": "slurm",
                        "attribution": "entry wall time inside the job allocation, one GPU held",
                    },
                )
                with tempfile.TemporaryDirectory() as tmp:
                    merge_runs(runs, tmp)
                    merged = [json.loads(line) for line in (Path(tmp) / "rows.jsonl").read_text().splitlines()]
                    failure = _quality_failure(merged)
                    if failure is not None:
                        failed.append(f"| {system} | {framework} | {tokenizer} | {len(runs)} | {failure} |")
                        continue
                    manifests.append(
                        write_entry(root, tmp, system=system, model=TOKENIZER_MODELS[tokenizer], gpu_time=gpu_time)
                    )
                    manifest = Manifest.model_validate_json((root / manifests[-1]).read_bytes())
                    if internal is not None:
                        record = internal_record(tmp, job_ids=job_ids)
                        _write_internal(internal, manifests[-1], manifest.configuration_id, record, job_dirs_used)
                fit = read_fit(manifest)
                env = manifest.provenance["host_cost"]["environment"]
                fit_rows_md.append(
                    f"| {system} | {framework} | {tokenizer} | {env['tokenizer_class']} | {fit.per_input_token_us:.3f} "
                    f"| {fit.base_ms:.2f} | {fit.r:.3f} | {fit.n} | "
                    f"{fit.per_input_token_us_ci95[0]:.2f}-{fit.per_input_token_us_ci95[1]:.2f} | {len(runs)} |"
                )
    for system_dir in sorted(p for p in (superseded or Path()).iterdir() if p.is_dir()) if superseded else []:
        for job_dir in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            jobs.append(
                job_cost(
                    system_dir.name,
                    _job_id(job_dir),
                    (job_dir / "sacct.txt").read_text(),
                    job_dir / "entries.txt",
                    gpu_type=GPU_TYPES[system_dir.name],
                    note="superseded run; data not used",
                )
            )
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "cost.md").write_text(render_markdown(jobs))
    (evidence / "fits.md").write_text("\n".join(fit_rows_md) + "\n")
    if failed:
        (evidence / "failed.md").write_text(
            "| System | Framework | Tokenizer | Runs | Reason |\n|---|---|---|---|---|\n" + "\n".join(failed) + "\n"
        )
    (evidence / "entries.json").write_text(json.dumps(sorted(manifests), indent=2) + "\n")
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="dataset root to write data/host-cost/ under")
    parser.add_argument("--evidence", type=Path, required=True, help="directory for cost.md and fits.md")
    parser.add_argument(
        "--internal", type=Path, help="aisim-data-internal root for each entry's host and Slurm job records"
    )
    parser.add_argument(
        "--superseded", type=Path, help="campaign-layout directory of runs whose cost is ledgered but not used"
    )
    args = parser.parse_args(argv)
    manifests = stage(args.campaign, args.output, args.evidence, args.superseded, args.internal)
    print(f"{len(manifests)} entries written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
