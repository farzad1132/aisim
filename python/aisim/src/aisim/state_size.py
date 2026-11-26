# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve CLI state-cache policy around the shared memory estimation API."""

from typing import Any

from .capacity import estimate_state_cache
from .config.engine import (
    _PD_CONNECTOR_ERROR,
    EnginePredictionConfig,
    StateCacheConfig,
    WorkerPredictionConfig,
    _require_block_drop_off,
    _validate_hybrid_host_offload,
    fpm_profile_deployment,
    recorded_state_pool,
)


def resolve_state_size(engine: EnginePredictionConfig, worker: WorkerPredictionConfig, role: str) -> dict[str, Any]:
    cache = worker.kv_cache
    sizing = cache.state_cache
    if sizing is None:
        return {"source": "disabled", "bytes_per_request": None}
    if sizing.bytes_per_request is not None:
        result = {
            "source": "overridden",
            "bytes_per_request": sizing.bytes_per_request,
            "block_size": cache.block_size,
            "kv_bytes_per_token": cache.bytes_per_token,
        }
        if sizing.sliding_window_lookahead_tokens is not None:
            # A resolved DeepSeek V4 state keeps the window it was resolved with.
            result["sliding_window_lookahead_tokens"] = sizing.sliding_window_lookahead_tokens
    else:
        if sizing.sliding_window_lookahead_tokens is not None:
            raise ValueError("state_cache.sliding_window_lookahead_tokens requires bytes_per_request")
        result = estimate_state_cache(
            engine.model,
            backend=engine.backend,
            tp_size=worker.parallelism.tensor,
            pp_size=worker.parallelism.pipeline,
            block_size=cache.block_size,
            kv_bytes_per_token=None if cache.bytes_per_token == "auto" else cache.bytes_per_token,
            kvcache_quant_mode=_kv_quant_mode(engine, worker, role),
            mamba_cache_dtype=sizing.mamba_cache_dtype,
            indexer_cache_dtype=sizing.indexer_cache_dtype,
            num_speculative_tokens=engine.speculation.num_speculative_tokens if engine.speculation else engine.nextn,
        )
    if sizing.retention_interval is not None:
        if "sliding_window_lookahead_tokens" not in result:
            raise ValueError("state_cache.retention_interval requires an inferred sliding-window geometry")
        if sizing.retention_interval % result["block_size"]:
            raise ValueError("state_cache.retention_interval must be a multiple of the resolved block_size")
    if "sliding_window_lookahead_tokens" in result and cache.prefix_match_unit is not None:
        raise ValueError("a sliding-window state resumes at block boundaries; leave prefix_match_unit unset")
    block_size = result["block_size"]
    bytes_per_token = result["kv_bytes_per_token"]
    if cache.prefix_match_unit is not None and block_size % cache.prefix_match_unit:
        raise ValueError("prefix_match_unit must be a positive divisor of resolved block_size")
    if sizing.prefill_checkpoint and cache.prefix_match_unit != block_size:
        raise ValueError("state_cache.prefill_checkpoint requires prefix_match_unit equal to block_size")
    if cache.host_offload is not None:
        # The state's family is known here, also for an inferred geometry.
        window = "sliding_window_lookahead_tokens" in result
        _validate_hybrid_host_offload(
            cache.host_offload.connector,
            cache.prefix_match_unit,
            block_size,
            window=window,
            kvcr=cache.kvcr is not None,
        )
        if engine.speculation is not None and not window:
            _require_block_drop_off(engine.speculation)
        if not window and role != "aggregated" and cache.host_offload.connector != "offloading_v0_24":
            raise ValueError(_PD_CONNECTOR_ERROR)
    size = StateCacheConfig(bytes_per_request=result["bytes_per_request"])
    blocks = size.state_blocks(block_size, bytes_per_token)
    block_bytes = block_size * bytes_per_token
    if cache.capacity.type != "fixed":
        # Without a profile the compiler sizes the pool from the analytical budget,
        # and the engine checks that it fits one token block and one request state.
        capacity = runtime_memory_blocks(engine, worker, role, block_bytes) if engine.fpm_profile is not None else None
    elif (capacity := cache.capacity.blocks) is None:
        capacity = cache.capacity.bytes // block_bytes
    if capacity is not None and not 0 < capacity <= (1 << 64) - 1:
        raise ValueError("fixed KV capacity must fit at least one block within u64")
    if capacity is not None and capacity < blocks + 1:
        raise ValueError("state_cache capacity must fit one token block and one request state")
    return {**result, "state_blocks": blocks, "allocated_bytes_per_request": blocks * block_bytes}


def _kv_quant_mode(engine: EnginePredictionConfig, worker: WorkerPredictionConfig, role: str) -> str | None:
    """KV precision for state sizing: an explicit override, then the FPM profile's, then ``None`` (the checkpoint's)."""
    explicit = worker.timing.kvcache_quant_mode or engine.kvcache_quant_mode
    if engine.fpm_profile is None:
        return explicit
    # The profile's timing and recorded pool were measured at this precision.
    deployment = fpm_profile_deployment(engine, worker, role)
    deployment.validate_overrides(kvcache_quant_mode=explicit)
    return deployment.kv_cache_dtype


def runtime_memory_blocks(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig, role: str, block_bytes: int
) -> int:
    """Whole blocks of the cache pool the FPM profile recorded for this worker (``resources.runtime_memory``)."""
    return recorded_state_pool(engine, worker, role).kv_cache_bytes // block_bytes


# Engine arguments that describe the state pool, not the memory the budget comes from.
_STATE_POOL_ARGS = ("state_cache", "kv_cache_bytes_per_token", "num_gpu_blocks", "num_gpu_blocks_is_explicit")
_MEMORY_FRACTION_ARG = {
    "vllm": "gpu_memory_utilization",
    "sglang": "mem_fraction_static",
    "trtllm": "free_gpu_memory_fraction",
}


def analytical_memory_blocks(
    payload: dict[str, Any],
    *,
    backend: str,
    memory_fraction: float,
    cuda_graph_reserved_bytes: int,
    block_bytes: int,
) -> int:
    """Whole state-pool blocks of the analytical KV byte budget, for a worker without an FPM profile.

    The budget is the one the uniform path uses (weights, activations and
    overheads subtracted from the memory fraction); only its division changes,
    by the state pool's block bytes instead of the model's all-layer token bytes.
    """
    from .capacity import materialize_aic_num_gpu_blocks

    probe = {key: value for key, value in payload.items() if key not in _STATE_POOL_ARGS}
    probe[_MEMORY_FRACTION_ARG[backend]] = memory_fraction
    probe["cuda_graph_reserved_bytes"] = cuda_graph_reserved_bytes
    diagnostics: dict[str, Any] = {}
    materialize_aic_num_gpu_blocks(probe, memory_diagnostics=diagnostics)
    adjusted = diagnostics.get("tolerance_adjusted")
    budget = (adjusted or diagnostics).get("total_kv_size_bytes")
    if budget is None:
        raise ValueError("state_cache without an FPM profile requires the analytical KV-cache budget")
    return int(budget) // block_bytes
