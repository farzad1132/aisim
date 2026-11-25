<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Queued tokenizer cost

Produce `host-cost` entries: per-request component times for one
(framework, framework version, system, tokenizer) on that system's host CPU, plus the
regression `base_ms + per_input_token_us × tokens`. The harness is `collector/host_cpu/`; the
fit is `aisim_core.host_cost.fit`; storage is the `host-cost` category
(`aisim_core.data.host_cost`).

## What is measured, and what is not

- The harness runs the framework's own request code from its pinned image (never a reimplementation),
  on a real chat payload set shared by every tokenizer so tokenizers see the same text.
- Queued replay cost is `template + encode + hash` (`engine.tokenization`). `json` and `validate` are
  recorded but not fitted; they belong to the unqueued overhead, which is collected separately.
- Idle single-thread cost only. Contention under load is not measured. Say so in any report.
- A GPU is held only because the host CPU must be the serving node's. The measurement uses no GPU kernels.

## Before the run

1. Freeze scope: frameworks and versions (Dynamo 1.6.0: vLLM 0.31.0, SGLang 0.5.21, TRT-LLM 1.3.0rc29),
   tokenizer families, systems. Read `collector/host_cpu/config.py` and `slurm/clusters.json`.
2. Audit each framework's request path from the installed source in its image: where the request
   arrival is stamped relative to chat-template rendering and tokenization, tokenizer worker defaults,
   and whether prompt hashing happens before scheduling. Write the map to
   `benchmarks/evidence/<campaign>/request-path-<framework>.md`. Do not write an adapter from memory.
3. Prefetch tokenizer files only (`python -m collector.host_cpu.prefetch`). Gated repos need a
   mirror with the same `tokenizer.json`; record the mirror and the file hashes.
4. Run one canary entry on one system. Reproduce a known number or explain the difference before
   spending more allocation.

## Run

One Slurm job per system, on one GPU node: `slurm/collect.sbatch` runs the frameworks in order, each
in its pinned image, looping every tokenizer. Keep the job inside the partition's wall limit and
record each job's id. `entries.txt` gives each entry's wall seconds for the ledger.

## After the run

1. Stage: `python -m collector.host_cpu.stage --campaign <dir> --output <public root> --evidence <reports>
   --internal <internal root>` merges the runs, fits, gates and writes each entry. Rejected entries (fewer
   than 60 requests, r below 0.98, more than 5% dropped) are recorded in `failed.md`, never averaged in.
2. Keep cluster facts internal. The public entry names the tokenizer by its Hugging Face repository and
   keeps only CPU facts; host names, Slurm job IDs, node load, local paths and the Slurm records go to
   `evidence/<entry>/` under `--internal`, for `aisim-data-internal`. `write_entry` refuses a local path
   as the model and job IDs in `gpu_time`.
3. Ledger: `collector/host_cpu/ledger.py` combines `sacct` and `entries.txt`. Per-entry cost plus the
   unattributed remainder must equal the Slurm allocation. Record the grand total GPU-hours.
4. Validate the public tree with `aisim_core.data.mirror.validate_tree(..., profile="public")`: the
   redaction scan refuses host names, job IDs, node load and local paths. Validate the internal tree's
   `evidence/` with `aisim_core.data.evidence.check_evidence`.
5. Rows go to `aisim-data`; the run reports and Slurm records go to `aisim-data-internal`, never to
   `docs/` or the public dataset.

## Using a result

Feed `lookup` output into `engine.tokenization`. A missing (framework, version, tokenizer, CPU)
means no default, not a nearby one. A missing framework version may reuse the closest collected
version only through an explicit declaration that names the donor version.
