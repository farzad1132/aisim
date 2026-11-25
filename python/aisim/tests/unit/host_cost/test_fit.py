# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from aisim_core.host_cost import fit_linear, fit_rows

pytestmark = pytest.mark.unit


def _points(base_ms, per_token_us, seed=1, noise=0.01, n=60):
    rng = random.Random(seed)
    tokens = [rng.randrange(2_000, 180_000) for _ in range(n)]
    totals = [(base_ms + per_token_us * t / 1000.0) * (1 + rng.uniform(-noise, noise)) for t in tokens]
    return tokens, totals


def test_recovers_a_known_line():
    tokens, totals = _points(2.0, 1.5)
    fit = fit_linear(tokens, totals)
    assert fit.per_input_token_us == pytest.approx(1.5, rel=0.02)
    assert fit.base_ms == pytest.approx(2.0, abs=3.0)
    assert fit.r > 0.999
    assert fit.n == 60 and fit.n_dropped == 0
    low, high = fit.per_input_token_us_ci95
    assert low <= 1.5 <= high
    assert fit.predict_ms(100_000) == pytest.approx(152.0, rel=0.03)


def test_drops_requests_slower_than_twice_the_median_cost_per_token():
    tokens, totals = _points(0.0, 1.5)
    totals[0] *= 5
    fit = fit_linear(tokens, totals)
    assert fit.n_dropped == 1
    assert fit.per_input_token_us == pytest.approx(1.5, rel=0.02)


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="at least 4"):
        fit_linear([1, 2, 3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        fit_linear([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="distinct"):
        fit_linear([5, 5, 5, 5], [1.0, 1.1, 0.9, 1.0])


def test_fit_rows_sums_the_queued_components_and_ignores_absent_ones():
    tokens, totals = _points(0.0, 2.0)
    rows = [
        {"tokens": t, "template_ms": 0.0, "encode_ms": total * 0.75, "hash_ms": total * 0.25}
        for t, total in zip(tokens, totals, strict=True)
    ]
    assert fit_rows(rows).per_input_token_us == pytest.approx(2.0, rel=0.02)
    no_hash = [{"tokens": t, "template_ms": 0.0, "encode_ms": total} for t, total in zip(tokens, totals, strict=True)]
    assert fit_rows(no_hash).per_input_token_us == pytest.approx(2.0, rel=0.02)
    with pytest.raises(ValueError, match="none of the components"):
        fit_rows([{"tokens": 10, "json_ms": 1.0}])
