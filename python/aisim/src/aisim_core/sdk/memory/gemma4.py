# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified: CPU-only sizing of Gemma 4 cache groups for AISim's state cache.
# Adapted from https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be
# Original paths: vllm/v1/core/kv_cache_utils.py, vllm/v1/kv_cache_interface.py,
# vllm/model_executor/models/gemma4.py.

"""Resolve Gemma 4 cache geometry as vLLM allocates it, without GPU weights.

vLLM unifies every layer's page size (``unify_kv_cache_spec_page_size``) by
growing the block size of layers with smaller pages, then splits the layers
into equal-size groups (``_get_kv_cache_groups_uniform_page_size``): one
global group and several sliding-window groups. A block id spans one page per
layer of a group, so every group's block costs the same bytes. A request holds
one global block per ``global_block_size`` tokens, plus the blocks its sliding
groups keep for the window. AISim's state cache represents the first part as
token bytes and the second as one fixed per-request state.
"""

from __future__ import annotations

from typing import Any

VLLM_REVISION = "5df63a7203cdb9328d824a27b682e77c7b9928be"
# AISim's vLLM default (``capacity._DEFAULT_BLOCK_SIZES``), vLLM's --block-size.
DEFAULT_BLOCK_SIZE = 64


def _cdiv(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _rank_kv_heads(total: int, tp_size: int) -> int:
    # vLLM's ModelConfig.get_num_kv_heads replicates heads when TP exceeds them.
    return max(1, total // tp_size)


def gemma4_state_cache(
    geometry: Any,
    *,
    tp_size: int,
    kv_bytes_per_element: int,
    block_size: int | None = None,
    num_speculative_tokens: int = 0,
) -> dict[str, Any]:
    """Per-rank token bytes, fixed window state bytes and pool block size.

    ``geometry`` is the model's ``Gemma4MixConfig``. ``block_size`` is vLLM's
    requested block size; the pool block is the global group's block after
    page unification. ``attention_k_eq_v`` does not shrink the cache: vLLM
    loads K into both the K and V slots.
    """
    if block_size is None:
        block_size = DEFAULT_BLOCK_SIZE
    for name, value, minimum in (
        ("tp_size", tp_size, 1),
        ("kv_bytes_per_element", kv_bytes_per_element, 1),
        ("block_size", block_size, 2),
        ("num_speculative_tokens", num_speculative_tokens, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    window = int(geometry.sliding_window_size)
    layer_types = list(geometry.layer_types)
    sliding_layers = layer_types.count("sliding_attention")
    global_layers = layer_types.count("full_attention")
    if window <= 0 or not sliding_layers or not global_layers or sliding_layers + global_layers != len(layer_types):
        raise ValueError("Gemma 4 sizing expects sliding_attention and full_attention layers with a positive window")

    # Page bytes per layer at the requested block size: K and V, per rank.
    sliding_token_bytes = 2 * _rank_kv_heads(geometry.swa_num_kv_heads, tp_size) * geometry.swa_head_dim
    global_token_bytes = 2 * _rank_kv_heads(geometry.global_num_kv_heads, tp_size) * geometry.global_head_dim
    sliding_token_bytes *= kv_bytes_per_element
    global_token_bytes *= kv_bytes_per_element
    page = max(sliding_token_bytes, global_token_bytes) * block_size

    def unified_block_size(token_bytes: int) -> int:
        layer_page = token_bytes * block_size
        # A smaller page that divides the largest grows its block; otherwise
        # vLLM pads the page and keeps the block size.
        return block_size * (page // layer_page) if page % layer_page == 0 else block_size

    sliding_block = unified_block_size(sliding_token_bytes)
    global_block = unified_block_size(global_token_bytes)

    # Equal-size groups: the smaller layer count, unless the counts are close.
    group_size = min(sliding_layers, global_layers)
    if max(sliding_layers, global_layers) < group_size * 1.5:
        group_size = max(sliding_layers, global_layers)
    sliding_groups = _cdiv(sliding_layers, group_size)
    global_groups = _cdiv(global_layers, group_size)
    block_bytes = page * group_size

    # W window tokens plus k draft tokens can span cdiv(W + k, B) + 1 blocks of B tokens.
    window_blocks = _cdiv(window + num_speculative_tokens, sliding_block) + 1
    fixed_blocks = sliding_groups * window_blocks
    state = fixed_blocks * block_bytes
    raw_state = sliding_layers * window * sliding_token_bytes
    groups = [
        {"block_size": global_block, "sliding_window": None, "layers": group_size, "bytes_per_block": block_bytes}
        for _ in range(global_groups)
    ] + [
        {"block_size": sliding_block, "sliding_window": window, "layers": group_size, "bytes_per_block": block_bytes}
        for _ in range(sliding_groups)
    ]
    return {
        "source": "inferred",
        "block_size": global_block,
        "kv_bytes_per_token": _cdiv(global_groups * block_bytes, global_block),
        "bytes_per_request": state,
        "raw_bytes_per_request": raw_state,
        "padding_bytes_per_request": state - raw_state,
        "bytes_per_block": block_bytes,
        "fixed_blocks": fixed_blocks,
        "cache_groups": groups,
        "num_speculative_tokens": num_speculative_tokens,
        "sliding_window_lookahead_tokens": 0,
        "backend_revision": VLLM_REVISION,
    }
