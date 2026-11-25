# Tokenization cost: vLLM, SGLang and TRT-LLM on Blackwell and Hopper hosts (2026-11-18)

Measured host-CPU time per request that a serving framework spends before the engine sees it (JSON, validation,
chat template, tokenization, and for vLLM prefix-cache block hashing), fitted as
`base_ms + per_input_token_us × tokens`. 84 entries: 3 frameworks × 4 systems × 7 tokenizers.

## Scope

| Item | Value |
|---|---|
| Frameworks (Dynamo 1.6.0 versions) | vLLM 0.31.0, SGLang 0.5.21, TRT-LLM 1.3.0rc29, each in its pinned image |
| Systems and host CPUs | `b200_sxm` Xeon 8568Y+, `b300_sxm` Xeon 8559C, `h200_sxm` Xeon 8468, `h100_sxm` AMD EPYC 7R13 |
| Tokenizers | Qwen2 (`Qwen/Qwen3.5-397B-A17B`), Llama 3 (ungated mirror `NousResearch/Meta-Llama-3.1-70B-Instruct`), DeepSeek (`DeepSeek-V3`), Kimi (`Kimi-K2.5`), GLM (`GLM-5`), MiniMax (`MiniMax-M2.5`), gpt-oss (`gpt-oss-120b`) |
| Workload | 60 real AgentX Weka chat requests, 2,950 to 178,116 tokens (Qwen2), identical text for every tokenizer |
| Method | Median of 3 timed passes per request per run; 2 independent Slurm jobs per (system, framework), merged by per-request median (4 runs for three entries that failed the first gate) |
| Gate | An entry needs r ≥ 0.98 and at most 5% of requests dropped as outliers; failures are recorded, not averaged in. Three B200 entries failed on two runs and passed after two more runs |

Request paths are in [`request-path.md`](https://huggingface.co/datasets/ai-dynamo/aisim-data-internal/discussions/2); every fit is in [`fits.md`](https://huggingface.co/datasets/ai-dynamo/aisim-data-internal/discussions/2); the data is staged in
[`dataset/`](https://huggingface.co/datasets/ai-dynamo/aisim-data/discussions/6) in `aisim-host-cost/v1` layout (`data/host-cost/<framework>/<version>/<system>/<tokenizer>/`).

## GPU hours

Each job holds 1 GPU for its whole wall time, because the host CPU must be the serving node's. The measurement runs no
GPU kernels. Full per-entry and per-job table: [`cost.md`](https://huggingface.co/datasets/ai-dynamo/aisim-data-internal/discussions/2).

| Phase | Jobs | GPU-hours |
|---|---|---|
| vLLM (final) | 10 | 0.677 |
| SGLang | 10 | 0.666 |
| TRT-LLM | 8 | 0.822 |
| vLLM first pass (superseded: used `AutoTokenizer`, not vLLM's own loader) | 8 | 0.590 |
| Canaries and adapter debugging | 5 | 0.136 |
| **Total** | **41** | **2.890** |

- **Cost per entry:** one entry-run takes 36 s of wall time on average (vLLM 35 s, SGLang 30 s, TRT-LLM 44 s, at most 96 s),
  so an entry with 2 runs costs about 0.02 GPU-hours. TRT-LLM's Kimi tokenizer is the slow one.
- Framework audits (reading installed source in each image) ran on CPU nodes and cost no GPU time.

## Findings

- **Hosts** (per-token cost relative to B300, over the 21 framework-tokenizer pairs): B200 0.86 to 1.08 (median 0.96), H200 1.01 to 1.60 (median 1.26), H100 1.20 to 1.48 (median 1.37).
- **Run-to-run noise** on the shared GPU nodes is real: across the two first-pass vLLM runs the per-entry slope differed by a
  median 2.4% (p90 8.2%, max 12.6%), which is why every entry merges at least two jobs.
- **Same text, same tokens:** token counts are identical across frameworks for every tokenizer (Kimi differs by at most 2 tokens on
  SGLang), so differences are framework cost, not different prompts.
- **Frameworks differ** (B300, µs/token): vLLM 1.53 to 1.89 (includes block hashing, about 0.3 µs of it), SGLang 1.14 to 1.42 and TRT-LLM 1.21 to 1.50 (Kimi excluded from both).
- **Kimi** is the outlier: SGLang 0.36 (direct `encode`), vLLM 1.56 (the `__call__` path vLLM uses), TRT-LLM 4.65 (its tokenizer wrapper).
- **Reproduction of the earlier number:** vLLM Qwen2 on B300 is 1.53 µs/token here versus 1.436 in the earlier single-pass run, within
  the observed run-to-run spread; the earlier script also used `AutoTokenizer`, not vLLM's own loader.
- **Negative intercepts:** many fitted `base_ms` are slightly negative (to about -20 ms on H200 Qwen2 for SGLang); replay clamps
  `base_ms` at 0 in `LookupResult.engine_tokenization()`.

## Replay check

AgentX Weka l1 and l4 (Qwen3.5-397B-A17B BF16, vLLM 0.31.0, B300 TP8) replayed with the frozen learned forward-pass model and
`engine.tokenization` set from the vLLM / Qwen2 / B300 entry (1.532 µs/token, `base_ms` 0, 1 worker). Errors are unsigned,
from the AgentX scorer. The 35 ms unqueued overhead is the earlier hand-measured value (client TTFT minus vLLM server TTFT on l1,
so in-sample); it is not part of this data.

| Replay | l1 point TTFT | l4 point TTFT | l1 per-request TTFT MAPE | l4 per-request TTFT MAPE |
|---|---|---|---|---|
| No host cost (earlier run) | 39.1% | 40.9% | 39.2% | 40.9% |
| Tokenization 1.46 µs/token (earlier benchmark) | 10.9% | 11.4% | 14.3% | 14.8% |
| **Tokenization 1.532 µs/token (this data)** | **9.5%** | **9.4%** | 13.4% | 14.4% |
| This data + 35 ms unqueued overhead | 1.8% | 1.8% | 11.8% | 12.9% |

The measured coefficient alone brings the point TTFT error under 10% on both lanes. Per-request TTFT MAPE stays above 10%.

## Limits

- Idle, single-thread cost. Contention under load, HTTP receive, streaming and detokenization are not measured.
- TRT-LLM tokenizer worker count is not verifiable in 1.3.0rc29 (`num_input_processor_workers` does not exist); entries store `null` and
  replay uses 1.
- One workload (agentic coding text); characters per token differ for other text.
- The Llama tokenizer is from an ungated mirror (same `tokenizer.json` content as the Meta repo is assumed, not verified;
  file hashes are in [`tokenizers/manifest.json`](tokenizers/manifest.json)).
- Tokenizer families are inferred from model names; each entry records the real tokenizer class.
- The cluster hosts may differ from the hosts behind other benchmark data.
- The fixed, unqueued overhead (`engine.extra_cpu_overhead`) is a separate collection; `json_ms` and `validate_ms` are stored, not fitted.

## Reproduce

`slurm/collect.sbatch` (one job per system), `collector.host_cpu.stage` (merge, gate, GPU time, ledger), see the
[`aisim-cpu-overhead-collect`](../../../.agents/skills/aisim-cpu-overhead-collect/references/tokenizer.md) skill. Per-job metadata
(`job.txt`, `entries.txt`, `sacct.txt`) is in [`jobs/`](jobs/).

## Where the evidence lives

The raw evidence (request-path audit, per-entry fits, cost ledger, Slurm job records, tokenizer manifest, cluster facts) and the
84 staged entries were moved out of this repository, parked in two Hugging Face pull requests until #932 lands:

- Public rows: https://huggingface.co/datasets/ai-dynamo/aisim-data/discussions/6
- Private evidence only (`evidence/data/host-cost/<framework>/<version>/`): https://huggingface.co/datasets/ai-dynamo/aisim-data-internal/discussions/2
