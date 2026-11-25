<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Unqueued request overhead

Produce `base_ms` and `per_input_token_us` for
[`engine.extra_cpu_overhead`](../../../../docs/replay/engine/README.md#extra-cpu-overhead) for one
deployment key, and show what it does to TTFT. Replay applies it before the engine, after the
queued `engine.tokenization`, and it never queues. Collect the tokenizer slope first
([tokenizer.md](tokenizer.md)): this measurement subtracts it.

## What is measured

Per request at **one lane**, where nothing queues in front of the engine:

| Term | From |
|---|---|
| Request start, client TTFT, input tokens | The load generator's per-request export (AIPerf `profile_export.jsonl`) |
| Engine prefill span | The first prefill step's schedule time to the last prefill step's emit time |
| Outside the engine | Client TTFT minus the engine prefill span (before plus after) |

- Fit `outside = base_ms + total_us × input tokens` with **total** input tokens, cached included.
  Agentic prompts are mostly cached, so the cost follows prompt length, not prefill work.
- Unqueued `per_input_token_us` = `total_us` minus the queued tokenizer slope for the same
  framework, tokenizer and host CPU ([tokenizer.md](tokenizer.md)). `base_ms` is kept as fitted.
- Expect almost all of it before the engine. After the engine is about 2 ms on SGLang; a large
  after-engine term means a detokenization or streaming cost, which belongs elsewhere.

## Before the run

1. Pick the deployment exactly as the E2E points ran it: image, server arguments, endpoint
   (chat or completions), load generator version and corpus. Record the host CPU model.
2. Confirm engine step timestamps exist for each step. The SGLang FPM step hooks record
   `host_scheduled_at_s` and `host_emitted_at_s`; the recorder keeps the clock pair that maps
   them to wall time. For an engine without per-step host timestamps, use engine-side per-request
   timestamps (arrival, first scheduled) instead and say so.
3. Use capture runs that are **not** the E2E points you will score. Fitting on the scored points
   makes the TTFT check in-sample; if unavoidable, mark those points in-sample in every report.

## Run

- Serve one lane for at least 15 minutes of profiling, with the load generator's warm-up excluded.
  Two independent runs (different seeds or a fresh server) are the minimum. Lane 1 of a step
  capture campaign already qualifies.
- Measure each run:

  ```bash
  python .agents/skills/aisim-cpu-overhead-collect/scripts/measure_unqueued.py \
      --client <run>/c1/aiperf_artifacts/profile_export.jsonl \
      --steps <run>/fpm.jsonl --tokenizer-us <queued tokenizer slope> --output <run>/unqueued.json
  ```

- Gate each run: at least 30 matched requests and R² of at least 0.8. Report runs that fail; do
  not average them in.

## Result

- Use the midpoint of the runs for `base_ms` and `per_input_token_us`, and report the spread.
- Record the per-run JSON, the run identities and the host CPU with the campaign evidence,
  following the evidence rules in [SKILL.md](../SKILL.md).

## Validate

Replay the E2E points twice, tokenizer only and then tokenizer plus `engine.extra_cpu_overhead`,
and report as [SKILL.md](../SKILL.md) describes. Name what is left. A gap that stays at high lane counts while one lane closes is queueing or
   cache behaviour, not request overhead.

Reference results, B300, both measured this way or from client TTFT minus engine time:

| Deployment | Tokenizer (queued) | Unqueued | TTFT e2e error, before → after |
|---|---|---|---|
| vLLM 0.31.0, Qwen3.5, chat | 1.436 µs/token | 29.5 ms + 0.245 µs/token | about 40% → 0.6–11% (learned FPM; lanes 1 and 4 in-sample) |
| SGLang 0.5.20, GLM-5.3 MTP, chat | 1.217 µs/token | 12.6 ms + 0.342 µs/token | 24.7% → 13.0% average (learned FPM; out of sample) |
