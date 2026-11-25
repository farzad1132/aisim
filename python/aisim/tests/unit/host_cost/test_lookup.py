# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisim_core.host_cost.fit import Fit
from aisim_core.host_cost.lookup import Entry, lookup

pytestmark = pytest.mark.unit

CPU = "Intel(R) Xeon(R) Platinum 8559C"


def _entry(version, *, cpu=CPU, tokenizer="qwen2", per_token=1.5, base=-0.4, workers=1, framework="vllm"):
    fit = Fit(base, per_token, 0.99, 60, 0, 2000, 180000, 4.0, 8.0, (per_token - 0.05, per_token + 0.05))
    return Entry(framework, version, tokenizer, cpu, "b300_sxm", workers, fit)


def test_exact_match_has_no_reuse_and_clamps_base_for_replay():
    result = lookup([_entry("0.31.0")], framework="vllm", version="0.31.0", tokenizer="qwen2", cpu_model=CPU)
    assert result.reused_version is None
    assert result.engine_tokenization() == {"workers": 1, "base_ms": 0.0, "per_input_token_us": 1.5}


def test_uncollected_version_reuses_the_closest_at_or_below():
    entries = [_entry("0.29.0", per_token=1.1), _entry("0.31.0", per_token=1.5), _entry("0.35.0", per_token=1.9)]
    result = lookup(entries, framework="vllm", version="0.32.0", tokenizer="qwen2", cpu_model=CPU)
    assert result.reused_version == "0.31.0"
    assert result.entry.fit.per_input_token_us == 1.5
    assert result.requested_version == "0.32.0"


def test_version_older_than_everything_collected_uses_the_nearest_above():
    entries = [_entry("0.31.0", per_token=1.5), _entry("0.35.0", per_token=1.9)]
    assert (
        lookup(entries, framework="vllm", version="0.20.0", tokenizer="qwen2", cpu_model=CPU).reused_version == "0.31.0"
    )


def test_release_candidates_order_correctly():
    entries = [_entry("1.3.0rc29", framework="trtllm")]
    result = lookup(entries, framework="trtllm", version="1.3.0", tokenizer="qwen2", cpu_model=CPU)
    assert result.reused_version == "1.3.0rc29"


def test_never_crosses_cpu_tokenizer_or_framework_and_reuse_can_be_refused():
    entries = [_entry("0.31.0")]
    assert lookup(entries, framework="vllm", version="0.31.0", tokenizer="qwen2", cpu_model="AMD EPYC 7R13") is None
    assert lookup(entries, framework="vllm", version="0.31.0", tokenizer="llama3", cpu_model=CPU) is None
    assert lookup(entries, framework="sglang", version="0.31.0", tokenizer="qwen2", cpu_model=CPU) is None
    assert (
        lookup(entries, framework="vllm", version="0.32.0", tokenizer="qwen2", cpu_model=CPU, allow_version_reuse=False)
        is None
    )


def test_unknown_worker_count_falls_back_to_one():
    result = lookup(
        [_entry("1.3.0rc29", workers=None, framework="trtllm")],
        framework="trtllm",
        version="1.3.0rc29",
        tokenizer="qwen2",
        cpu_model=CPU,
    )
    assert result.engine_tokenization()["workers"] == 1
