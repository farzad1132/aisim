---
name: aisim-cpu-overhead-collect
description: Collect the host-CPU time a serving deployment spends on each request before the engine, which replay models as CPU overhead. Covers the queued tokenizer cost (engine.tokenization, from the host-cost harness) and the unqueued request-path cost (engine.extra_cpu_overhead, from client TTFT minus engine time at one lane). Use when collecting or refreshing CPU-overhead data for vLLM, SGLang or TRT-LLM on a host CPU, or when TTFT is predicted too fast by a roughly constant fraction even at one lane; not for GPU kernel or step timing, response detokenization or replay accuracy campaigns as a whole.
license: Apache-2.0
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Collect CPU overhead

A serving deployment spends host-CPU time on every request before the scheduler sees it. Replay
prices it in front of the engine, separately from the forward-pass model:

| Part | Replay field | Behaviour | How to collect |
|---|---|---|---|
| Tokenizer (chat template, encode, prefix-cache hashing) | `engine.tokenization` | Queued: a tokenizer server with `workers` workers | [references/tokenizer.md](references/tokenizer.md) |
| Everything else on the request path (HTTP receive, parsing, input processing, first-chunk work) | `engine.extra_cpu_overhead` | Unqueued: every request pays it whatever the load | [references/unqueued.md](references/unqueued.md) |

Collect the tokenizer first: the unqueued measurement subtracts its slope.

This skill does not authorize a GPU allocation beyond the campaign's recorded plan, an upload or a
dataset publish. Ask before spending more.

## Rules for every part

- **One deployment key.** Coefficients belong to one (framework, framework version, endpoint,
  host CPU, tokenizer). Record all five with every result.
- **No default from a neighbour.** A deployment that has not been measured has no coefficient. A
  missing framework version may reuse the closest collected version only through an explicit
  declaration that names the donor version.
- **Evidence stays private.** Fitted numbers and CPU facts may be public. Host names, Slurm job
  IDs, node load and local paths go to `aisim-data-internal`, never to `docs/` or the public dataset.
- **Validate on replay.** Replay the E2E points without and with the new coefficients and report,
  per lane, TTFT e2e error (request-average) and per-request quantile error (p5–p95), unsigned with
  bootstrap standard error, plus TPOT e2e error to show it did not move. Mark any point whose own
  data produced a coefficient as in-sample.
