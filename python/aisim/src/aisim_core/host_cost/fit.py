# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression of per-request host time against input tokens: ``base_ms + per_input_token_us * tokens``.

The fit is ordinary least squares after dropping requests whose cost per token exceeds
``outlier_factor`` times the median (shared-node noise). Nothing is extrapolated: a fit describes the
(framework, version, tokenizer, host CPU) it was measured on and the token range it covers.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

# Components whose cost queues on the tokenizer in replay (``engine.tokenization``).
QUEUED_COMPONENTS: tuple[str, ...] = ("template", "encode", "hash")


@dataclass(frozen=True)
class Fit:
    """One linear fit. Times are milliseconds; ``per_input_token_us`` is microseconds per token."""

    base_ms: float
    per_input_token_us: float
    r: float
    n: int
    n_dropped: int
    tokens_min: int
    tokens_max: int
    residual_p50_pct: float
    residual_p90_pct: float
    per_input_token_us_ci95: tuple[float, float]

    def predict_ms(self, tokens: int) -> float:
        return self.base_ms + self.per_input_token_us * tokens / 1000.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def fit_linear(
    tokens: Sequence[int],
    totals_ms: Sequence[float],
    *,
    outlier_factor: float = 2.0,
    bootstrap: int = 400,
    seed: int = 0,
) -> Fit:
    """Fit ``total_ms = base_ms + per_input_token_us * tokens / 1000`` after dropping noisy requests."""
    if len(tokens) != len(totals_ms) or len(tokens) < 4:
        raise ValueError("need at least 4 paired points to fit")
    if any(t <= 0 for t in tokens):
        raise ValueError("input token counts must be positive")
    points = list(zip(tokens, totals_ms, strict=True))
    median = statistics.median(total / count for count, total in points)
    kept = [(count, total) for count, total in points if total / count <= outlier_factor * median]
    if len(kept) < 4 or len({count for count, _ in kept}) < 2:
        raise ValueError("too few distinct token counts left after dropping outliers")
    x = [count for count, _ in kept]
    y = [total for _, total in kept]
    slope, intercept = statistics.linear_regression(x, y)
    residuals = sorted(abs(total - (intercept + slope * count)) / total * 100.0 for count, total in kept)
    rng = random.Random(seed)
    slopes = []
    for _ in range(bootstrap):
        sample = [kept[rng.randrange(len(kept))] for _ in kept]
        if len({count for count, _ in sample}) < 2:
            continue
        slopes.append(statistics.linear_regression([c for c, _ in sample], [t for _, t in sample])[0])
    slopes.sort()
    low = slopes[int(0.025 * (len(slopes) - 1))] if slopes else math.nan
    high = slopes[int(0.975 * (len(slopes) - 1))] if slopes else math.nan
    return Fit(
        base_ms=intercept,
        per_input_token_us=slope * 1000.0,
        r=statistics.correlation(x, y),
        n=len(kept),
        n_dropped=len(points) - len(kept),
        tokens_min=min(x),
        tokens_max=max(x),
        residual_p50_pct=residuals[len(residuals) // 2],
        residual_p90_pct=residuals[int(0.9 * (len(residuals) - 1))],
        per_input_token_us_ci95=(low * 1000.0, high * 1000.0),
    )


def fit_rows(rows: Iterable[Mapping[str, Any]], components: Sequence[str] = QUEUED_COMPONENTS) -> Fit:
    """Fit the sum of ``<component>_ms`` over ``components`` that a row has (a missing one counts as absent)."""
    tokens, totals = [], []
    for row in rows:
        present = [row[f"{name}_ms"] for name in components if row.get(f"{name}_ms") is not None]
        if not present:
            raise ValueError(f"row has none of the components {list(components)}")
        tokens.append(int(row["tokens"]))
        totals.append(float(sum(present)))
    return fit_linear(tokens, totals)
