# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 on vLLM keeps its sliding-window layers as one fixed state per request."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from aisim import capacity, compiler
from aisim.compiler import prediction_to_replay_spec
from aisim.config import CorePredictionConfig
from aisim.config.engine import KvCachePredictionConfig
from aisim.runner import EngineReplayRunnerFactory

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

GEMMA4 = "google/gemma-4-26B-A4B"


class RecordingRuntime:
    execution_spec = None

    def run_replay_json(self, payload):
        self.execution_spec = json.loads(payload)
        return json.dumps({"duration_ms": 1.0, "completed_requests": 1})


def _payload(*, model=GEMMA4, backend="vllm", tensor=1, timing=None, **cache):
    kv_cache = {"capacity": {"type": "fixed", "blocks": 400}, **cache}
    return {
        "engine": {
            "mode": "aggregated",
            "backend": backend,
            "model": model,
            "hardware": "rtx_pro_6000_server",
            "context_length": 4096,
            "workers": {
                "aggregated": {
                    "parallelism": {"tensor": tensor, "moe_tensor": tensor},
                    "kv_cache": kv_cache,
                    "timing": timing or {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
                }
            },
        },
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 1},
            "load": {"type": "concurrency", "concurrency": 1},
            "stop": {"requests": 1},
        },
    }


def _args(payload):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    return spec, spec.backend_deployment.agg_engine_args


# Gemma-4-26B-A4B: 25 sliding layers (8 KV heads x 256) and 5 global layers
# (2 KV heads x 512, K stored in both slots). vLLM grows the smaller page's
# 64-token block to match the larger page, then makes six groups of five layers.
# One block id spans a page on each of a group's five layers, and each sliding
# group keeps cdiv(1024, 64) + 1 = 17 blocks: 85 window blocks per request.
@pytest.mark.parametrize(
    "controls,global_block,page,token_bytes",
    [
        # TP1 BF16: sliding 8192 B/token/layer, global 4096 -> global block 128.
        ({}, 128, 64 * 8192, 20480),
        # TP2 BF16: 4 and 1 KV heads per rank: 4096 and 2048 B/token/layer.
        ({"tp_size": 2}, 128, 64 * 4096, 10240),
        # TP4 BF16: global heads replicate (max(1, 2 // 4)), both pages 2048 B/token -> no growth.
        ({"tp_size": 4}, 64, 64 * 2048, 10240),
        # TP2 FP8: half the bytes of TP2 BF16, same blocks.
        ({"tp_size": 2, "kvcache_quant_mode": "fp8"}, 128, 64 * 2048, 5120),
    ],
)
def test_gemma4_state_estimator_follows_vllm_cache_groups(controls, global_block, page, token_bytes):
    from aisim_core.sdk import estimate_state_cache

    result = estimate_state_cache(GEMMA4, **controls)
    block_bytes = 5 * page
    assert result["block_size"] == global_block
    assert result["bytes_per_block"] == block_bytes
    assert result["kv_bytes_per_token"] == token_bytes == block_bytes // global_block
    assert result["fixed_blocks"] == 85
    assert result["bytes_per_request"] == 85 * block_bytes
    assert result["sliding_window_lookahead_tokens"] == 0
    groups = [(group["block_size"], group["sliding_window"], group["layers"]) for group in result["cache_groups"]]
    assert groups == [(global_block, None, 5)] + [(64, 1024, 5)] * 5
    json.dumps(result)


def test_gemma4_state_padding_counts_the_window_rounding():
    from aisim_core.sdk import estimate_state_cache

    result = estimate_state_cache(GEMMA4)
    # 25 sliding layers x 1024 tokens x 8192 B, against 85 blocks of 2,621,440 B.
    assert result["raw_bytes_per_request"] == 25 * 1024 * 8192
    assert result["padding_bytes_per_request"] == 85 * 2621440 - 25 * 1024 * 8192


@pytest.mark.parametrize(
    "controls,error",
    [
        ({"backend": "sglang"}, "requires vllm and PP=1"),
        ({"kv_bytes_per_token": 20480}, "leave bytes_per_token as auto"),
        ({"num_speculative_tokens": 1}, "draft tokens"),
        ({"kvcache_quant_mode": "nvfp4"}, "does not support kvcache_quant_mode"),
        ({"mamba_cache_dtype": "float32"}, "Kimi K3 only"),
    ],
)
def test_gemma4_state_estimator_rejects_unsupported_controls(controls, error):
    from aisim_core.sdk import estimate_state_cache

    with pytest.raises(ValueError, match=error):
        estimate_state_cache(GEMMA4, **controls)


def test_vllm_gemma4_gets_a_window_state_without_configuring_one():
    spec, args = _args(_payload(tensor=2))
    assert args["block_size"] == 128
    assert args["kv_cache_bytes_per_token"] == 10240
    assert args["state_cache"] == {
        "bytes_per_request": 85 * 1310720,
        "sliding_window": {"lookahead_tokens": 0},
    }
    assert args["num_gpu_blocks"] == 400
    info = spec.backend_deployment.performance_model_metadata["aggregated"]["state_cache"]
    assert (info["source"], info["state_blocks"]) == ("inferred", 85)
    runtime = RecordingRuntime()
    EngineReplayRunnerFactory(runtime=runtime).create(0).run(spec)
    rank = runtime.execution_spec["spec"]["engine"]["rank"]
    for field in (
        "block_size",
        "kv_cache_bytes_per_token",
        "num_gpu_blocks",
        "state_cache",
    ):
        assert rank[field] == args[field]


def test_gemma4_window_state_survives_public_export_and_reload():
    config = CorePredictionConfig.model_validate(_payload(tensor=2))
    reloaded = CorePredictionConfig.model_validate_json(config.model_dump_json())
    # Resolution happens at compile time, so the exported config stays as authored.
    assert reloaded.engine.workers.aggregated.kv_cache.state_cache is None
    first = prediction_to_replay_spec(config).backend_deployment.agg_engine_args
    again = prediction_to_replay_spec(reloaded).backend_deployment.agg_engine_args
    assert first == again


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_payload(backend="sglang"), id="sglang"),
        pytest.param(_payload(model="Qwen/Qwen3-32B"), id="other-model"),
    ],
)
def test_window_state_stays_off_elsewhere(payload):
    _, args = _args(payload)
    assert "state_cache" not in args


def test_an_unresolvable_model_is_not_sized_as_gemma4(monkeypatch):
    from aisim_core.sdk import utils

    def unavailable(model_path):
        raise utils.HuggingFaceDownloadError("no config")

    monkeypatch.setattr(utils, "_load_model_config_from_model_path", unavailable)
    compiler._is_gemma4.cache_clear()
    engine = SimpleNamespace(backend="vllm", model="not-a/real-model")
    assert compiler._vllm_sliding_window_model(engine) is False
    compiler._is_gemma4.cache_clear()


def test_a_local_model_directory_without_config_is_not_sized_as_gemma4(tmp_path):
    assert compiler._is_gemma4(str(tmp_path)) is False


_ENGINE = SimpleNamespace(speculation=None, nextn=0)


@pytest.mark.parametrize(
    "updates,engine,error",
    [
        (
            {"g3_offload": object()},
            _ENGINE,
            "kv_cache.g3_offload: replay models G3 only behind vLLM 0.24",
        ),
        (
            {"kvcr": object()},
            _ENGINE,
            "kv_cache.kvcr: a KVCR peer fetch carries no window snapshot",
        ),
        (
            {"host_offload": SimpleNamespace(connector="simple_cpu_v0_32")},
            _ENGINE,
            "simple_cpu_v0_32: replay does not model",
        ),
        ({}, SimpleNamespace(speculation=None, nextn=1), "speculative decoding"),
        ({"bytes_per_token": 64}, _ENGINE, "explicit kv_cache.bytes_per_token"),
    ],
)
def test_gemma4_window_state_rejects_features_that_lose_the_window(updates, engine, error):
    cache = KvCachePredictionConfig().model_copy(update=updates)
    with pytest.raises(ValueError, match=error):
        compiler._with_gemma4_window_state(engine, cache)


def test_gemma4_window_state_allows_the_v0_32_offloading_connector():
    cache = KvCachePredictionConfig().model_copy(update={"host_offload": SimpleNamespace(connector="offloading_v0_32")})
    assert compiler._with_gemma4_window_state(_ENGINE, cache).state_cache is not None


def test_analytical_state_pool_divides_the_uniform_budget(monkeypatch):
    probes = []

    def budget(raw, *, memory_diagnostics=None):
        probes.append(raw)
        memory_diagnostics["total_kv_size_bytes"] = 1000 * 1310720 + 5
        return raw

    monkeypatch.setattr(capacity, "materialize_aic_num_gpu_blocks", budget)
    payload = _payload(
        tensor=2,
        timing={"type": "polynomial"},
        capacity={"type": "default", "memory_fraction": 0.92},
    )
    _, args = _args(payload)
    assert args["num_gpu_blocks"] == 1000
    assert args["num_gpu_blocks_is_explicit"] is False
    assert "gpu_memory_utilization" not in args
    (probe,) = probes
    # The budget probe is the uniform worker's memory request, without the state pool.
    assert probe["gpu_memory_utilization"] == 0.92
    assert not {"state_cache", "kv_cache_bytes_per_token", "num_gpu_blocks"} & probe.keys()


def test_analytical_state_pool_keeps_the_uniform_byte_budget():
    payload = _payload(tensor=2, timing={"type": "default"}, capacity={"type": "default"})
    payload["engine"]["backend_version"] = "0.32.0"
    _, args = _args(payload)
    diagnostics = {}
    capacity.estimate_num_gpu_blocks(
        backend_name="vllm",
        system="rtx_pro_6000_server",
        model_path=GEMMA4,
        tp_size=2,
        moe_tp_size=2,
        block_size=64,
        max_num_batched_tokens=args["max_num_batched_tokens"],
        max_num_sequences=args["max_num_seqs"],
        gpu_memory_utilization=0.9,
        backend_version="0.32.0",
        diagnostics=diagnostics,
    )
    block_bytes = args["block_size"] * args["kv_cache_bytes_per_token"]
    assert args["num_gpu_blocks"] == diagnostics["total_kv_size_bytes"] // block_bytes


def test_fp8_kv_checkpoint_sizes_the_state_in_fp8():
    from aisim_core.sdk import estimate_state_cache

    # The NVFP4 checkpoint declares FP8 KV, which vLLM's --kv-cache-dtype auto follows.
    result = estimate_state_cache("nvidia/Gemma-4-26B-A4B-NVFP4", tp_size=2)
    assert (result["kv_bytes_per_token"], result["bytes_per_request"]) == (
        5120,
        85 * 655360,
    )


def _grouped_profile(layout, kv_cache_dtype="bfloat16"):
    groups = [
        {
            "name": "full",
            "kind": "attention",
            "num_layers": 5,
            "block_size_tokens": 128,
            "page_size_bytes": 524288,
        },
        {
            "name": "swa",
            "kind": "attention",
            "num_layers": 25,
            "block_size_tokens": 64,
            "page_size_bytes": 524288,
            "sliding_window": 1024,
        },
    ]
    resources = {
        "cache_layout": layout,
        "max_num_tokens": 8192,
        "max_batch_size": 256,
        "provenance": "test",
    }
    if layout == "grouped":
        resources["cache_groups"] = groups
    else:
        # The global group's rate at TP1 FP8: one 1,310,720 B block per 128 tokens.
        resources["kv_bytes_per_token"] = 10240
    resources["runtime_memory"] = {
        "kv_cache_bytes": 60 * 10**9,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "provenance": "test",
    }
    deployment = {
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.32.0",
        "tp": 1,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "cp": 1,
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
        "kv_cache_dtype": kv_cache_dtype,
        "worker_type": "aggregated",
        "resources": resources,
    }
    return {
        "schema_version": 1,
        "model": GEMMA4,
        "model_revision": "1" * 40,
        "architecture": "Gemma4ForConditionalGeneration",
        "context_length": 8192,
        "num_experts": 128,
        "provenance": "test",
        "deployments": [deployment],
    }


def test_a_grouped_fpm_profile_keeps_its_own_window_model():
    payload = _payload(timing={"type": "default"}, capacity={"type": "default"}, prefix_caching=False)
    engine = payload["engine"]
    engine.update(hardware="h200_sxm", backend_version="0.32.0", context_length=8192)
    engine["fpm_profile"] = _grouped_profile("grouped")
    engine["workers"]["aggregated"].pop("parallelism")
    _, args = _args(payload)
    assert "state_cache" not in args


def _linear_profile_payload(**engine):
    payload = _payload(timing={"type": "default"}, capacity={"type": "default"}, prefix_caching=False)
    payload["engine"].update(hardware="h200_sxm", backend_version="0.32.0", context_length=8192, **engine)
    payload["engine"]["fpm_profile"] = _grouped_profile("linear", kv_cache_dtype="fp8")
    payload["engine"]["workers"]["aggregated"].pop("parallelism")
    return payload


def test_a_linear_fpm_profile_sizes_the_window_at_its_kv_dtype():
    # The checkpoint is BF16, but the profile was measured with FP8 KV. TP1 FP8 halves
    # the TP1 BF16 page: 64 x 4096 B per layer, five layers per 1,310,720 B block.
    _, args = _args(_linear_profile_payload())
    assert (args["block_size"], args["kv_cache_bytes_per_token"]) == (128, 10240)
    assert args["state_cache"]["bytes_per_request"] == 85 * 1310720
    # The recorded 60 GB pool, in FP8 blocks.
    assert args["num_gpu_blocks"] == 60 * 10**9 // 1310720


def test_a_kv_dtype_override_that_contradicts_the_profile_is_rejected():
    with pytest.raises(ValueError, match="FPM profile identity conflict"):
        _args(_linear_profile_payload(kvcache_quant_mode="bfloat16"))


def _sweeper_sample(**extra):
    """A TP1 aggregated vLLM Gemma 4 candidate, as the sweeper unrolls it."""
    shape = dict.fromkeys(("tp", "pp", "attention_dp", "moe_tp", "moe_ep"), 1)
    return {"backend": "vllm", "model_name": GEMMA4, "hardware_sku": "h200_sxm", **shape, **extra}


def test_the_sweeper_sizes_the_window_at_the_profile_kv_dtype():
    from aisim.sweeper.deploy import gemma4_window_geometry

    sample = _sweeper_sample()
    profile = _grouped_profile("linear", kv_cache_dtype="fp8")
    state = gemma4_window_geometry(sample, "agg", 64, backend_version="0.32.0", fpm_profile=profile)
    assert (state["kv_bytes_per_token"], state["bytes_per_block"]) == (10240, 1310720)
    assert state["runtime_memory"].kv_cache_bytes == 60 * 10**9


def test_the_sweeper_rejects_a_kv_dtype_that_contradicts_the_profile():
    from aisim.sweeper.deploy import gemma4_window_geometry

    sample = _sweeper_sample(kvcache_quant_mode="bfloat16")
    profile = _grouped_profile("linear", kv_cache_dtype="fp8")
    with pytest.raises(ValueError, match="FPM profile identity conflict"):
        gemma4_window_geometry(sample, "agg", 64, backend_version="0.32.0", fpm_profile=profile)


def _gemma4_snapshot(backend="vllm", tp=1, kv_block_size=64, pp=1):
    from aisim.sweeper.state_snapshot import SnapshotRole

    identity = {
        "model": GEMMA4,
        "backend": backend,
        "backend_version": "0.32.0",
        "system": "h200_sxm",
        "kv_block_size": kv_block_size,
        "tp": tp,
        "pp": pp,
    }
    summary = {"identity": identity, "selected_estimation_mode": "op_level", "kind": "native"}
    return SnapshotRole(role="agg", path="/snapshots/gemma4", summary=summary, config={})


def _derive_snapshot_role(monkeypatch, snapshot, **space):
    """Run recommend's snapshot intake on an aggregated Gemma 4 role timed by ``snapshot``."""
    from aisim.sweeper import state_snapshot

    monkeypatch.setattr(state_snapshot, "load_snapshot_role", lambda role, reference: snapshot)
    value = {
        "model_name": GEMMA4,
        "hardware_sku": "h200_sxm",
        "deployment_mode": ["agg"],
        "role_estimator_controls": {"agg": {"estimator_config": {"state_snapshot": {"path": snapshot.path}}}},
        **space,
    }
    return state_snapshot.derive_snapshot_roles(value)


@pytest.mark.parametrize(
    "tp,kv_block_size",
    [
        # A deployment's 64-token block gives the window-state pool 128-token global blocks.
        (1, 64),
        # An AISim-compiled TP2 model saves the pool block (128), which a deployment launched with
        # --block-size 128 (pool block 256) would also record; the value cannot tell them apart.
        (2, 128),
    ],
)
def test_recommend_rejects_a_gemma4_vllm_snapshot_whose_block_changes(monkeypatch, tp, kv_block_size):
    snapshot = _gemma4_snapshot(tp=tp, kv_block_size=kv_block_size)
    with pytest.raises(ValueError, match=f"at tp={tp} .* kv_block_size={kv_block_size} does not record which"):
        _derive_snapshot_role(monkeypatch, snapshot)


def test_a_gemma4_tp4_snapshot_with_an_unknown_pipeline_takes_the_explicit_one(monkeypatch):
    from aisim.sweeper.state_snapshot import UNKNOWN

    # At TP4 both page sizes match, so the pool keeps the deployed 64-token block. The unknown saved
    # pipeline width is resolved from the role's single explicit value before the block check.
    snapshot = _gemma4_snapshot(tp=4, pp=UNKNOWN)
    value = _derive_snapshot_role(monkeypatch, snapshot, parallel_independent_by_mode={"agg": {"pp": [1]}})
    assert value["agg_block_size"] == 64
    assert value["parallel_independent_by_mode"]["agg"]["tp"] == [4]


def test_a_gemma4_snapshot_on_another_backend_passes_the_window_state_guard():
    from aisim.sweeper.state_snapshot import _reject_window_state_snapshot

    _reject_window_state_snapshot(_gemma4_snapshot(backend="sglang"), tp=1)


def _pd_profile(prefill_runtime_memory):
    """A linear FP8-KV profile with prefill and decode deployments; only decode records its pool unless asked."""
    profile = _grouped_profile("linear", kv_cache_dtype="fp8")
    decode = profile["deployments"][0]
    decode["worker_type"] = "decode"
    prefill = deepcopy(decode)
    prefill["worker_type"] = "prefill"
    if not prefill_runtime_memory:
        # Declared non-KV memory instead of a recorded pool: AISim cannot divide a prefill pool.
        del prefill["resources"]["runtime_memory"]
        prefill["resources"].update(
            weights_bytes=50 * 10**9, activations_bytes=10**9, runtime_overhead_bytes=10**9, comm_overhead_bytes=0
        )
    profile["deployments"] = [prefill, decode]
    return profile


def _pd_payload(profile):
    payload = _linear_profile_payload()
    engine = payload["engine"]
    engine["mode"] = "disaggregated"
    worker = engine["workers"].pop("aggregated")
    # The scheduler limits the profile recorded its runtime memory at.
    worker["scheduler"] = {"max_sequences": 256, "max_batched_tokens": 8192}
    engine["workers"] = {"prefill": deepcopy(worker), "decode": deepcopy(worker)}
    engine["fpm_profile"] = profile
    return payload


@pytest.mark.parametrize("prefill_runtime_memory", [True, False])
def test_disaggregated_roles_take_the_window_state_together(prefill_runtime_memory):
    spec = prediction_to_replay_spec(
        CorePredictionConfig.model_validate(_pd_payload(_pd_profile(prefill_runtime_memory)))
    )
    roles = (spec.backend_deployment.prefill_engine_args, spec.backend_deployment.decode_engine_args)
    # Decode resumes from the transferred state, so a prefill pool AISim cannot size keeps both roles uniform.
    assert ["state_cache" in args for args in roles] == [prefill_runtime_memory] * 2


@pytest.mark.parametrize("prefill_runtime_memory", [True, False])
def test_the_sweeper_gives_disaggregated_roles_the_window_state_together(prefill_runtime_memory):
    from aisim.sweeper.deploy import gemma4_window_geometry

    shape = {
        f"{role}_{key}": 1 for role in ("prefill", "decode") for key in ("tp", "pp", "attention_dp", "moe_tp", "moe_ep")
    }
    sample = {**_sweeper_sample(), **shape}
    profile = _pd_profile(prefill_runtime_memory)
    states = [
        gemma4_window_geometry(sample, role, 64, backend_version="0.32.0", fpm_profile=profile)
        for role in ("prefill", "decode")
    ]
    assert [state is not None for state in states] == [prefill_runtime_memory] * 2


@pytest.mark.parametrize("mode", ["aggregated", "disaggregated"])
def test_the_rust_engine_accepts_the_compiled_gemma4_window_state(mode):
    # Through the native runtime, not a recording stub: Rust validates the sliding-window state,
    # the 128-token block and the global-only token bytes, and allocates from that pool.
    payload = _payload(tensor=2, prefix_caching=True)
    payload["traffic"]["source"] = {"type": "synthetic", "input_tokens": 1500, "output_tokens": 4}
    payload["traffic"]["stop"] = {"requests": 2}
    if mode == "disaggregated":
        engine = payload["engine"]
        worker = engine["workers"].pop("aggregated")
        engine.update(mode="disaggregated", workers={"prefill": worker, "decode": deepcopy(worker)})
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
    deployment = spec.backend_deployment
    roles = (
        [deployment.agg_engine_args]
        if mode == "aggregated"
        else [
            deployment.prefill_engine_args,
            deployment.decode_engine_args,
        ]
    )
    for args in roles:
        assert (args["block_size"], args["kv_cache_bytes_per_token"]) == (128, 10240)
        assert args["state_cache"]["sliding_window"] == {"lookahead_tokens": 0}
    metrics = EngineReplayRunnerFactory().create(0).run(spec).metrics
    assert metrics["completed_requests"] == 2


def test_gemma4_window_state_runs_with_host_offload_in_the_rust_engine():
    # A rank-local G2 pool keeps the window state too. An unset connector defaults to
    # offloading_v0_24, which a rank-local window-state pool runs as vLLM 0.32's OffloadingConnector.
    for connector in (None, "offloading_v0_24", "offloading_v0_32"):
        # The host pool must exceed one 85-block state checkpoint.
        host_offload = {"num_host_blocks": 256, **({"connector": connector} if connector else {})}
        payload = _payload(tensor=2, prefix_caching=True, host_offload=host_offload)
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(payload))
        args = spec.backend_deployment.agg_engine_args
        assert args["native_host_offload"]["num_host_blocks"] == 256
        assert "sliding_window" in args["state_cache"]
        assert EngineReplayRunnerFactory().create(0).run(spec).metrics["completed_requests"] == 1


def test_a_gemma4_prefix_hit_resumes_at_a_window_snapshot_boundary():
    # Replay keeps each boundary's window as a full 85-block state copy; with 400 blocks only the
    # newest three fit and the 896-token one is evicted, where vLLM shares window blocks instead.
    payload = _payload(tensor=2, prefix_caching=True, capacity={"type": "fixed", "blocks": 4000})
    # The second request shares 1,000 prompt tokens with the first. Window snapshots exist only at
    # 128-token block boundaries, so it resumes after 7 blocks: 896 tokens.
    payload["traffic"]["source"] = {
        "type": "synthetic",
        "input_tokens": 1500,
        "output_tokens": 2,
        "cached_prefix_tokens": 1000,
    }
    payload["traffic"]["stop"] = {"requests": 2}
    metrics = (
        EngineReplayRunnerFactory()
        .create(0)
        .run(prediction_to_replay_spec(CorePredictionConfig.model_validate(payload)))
        .metrics
    )
    assert metrics["completed_requests"] == 2
    assert metrics["committed_prefill_tokens"] == 2 * 1500 - 896
    assert metrics["prefix_cache_reused_ratio"] == pytest.approx(896 / 3000)
