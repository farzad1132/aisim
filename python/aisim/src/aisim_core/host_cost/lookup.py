# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Look up a measured tokenization cost and shape it as ``engine.tokenization``.

A result is exact for (framework, framework version, tokenizer, host CPU). Nothing is extrapolated across host
CPUs or tokenizers. Framework versions are recollected only every one or two releases, so an uncollected
version reuses the closest collected one (at or below it, else the nearest above) and the result says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from aisim_core.data import Manifest
from aisim_core.data.host_cost import CATEGORY, MANIFEST_NAME, read_fit
from aisim_core.host_cost.fit import Fit


@dataclass(frozen=True)
class Entry:
    framework: str
    version: str
    tokenizer: str
    cpu_model: str
    system: str
    tokenizer_workers: int | None
    fit: Fit


@dataclass(frozen=True)
class LookupResult:
    entry: Entry
    requested_version: str
    reused_version: str | None  # the collected version that stood in, or None for an exact match

    def engine_tokenization(self) -> dict[str, Any]:
        """The ``engine.tokenization`` block for this entry.

        ``base_ms`` is clamped at zero (a fit can have a small negative intercept, which replay rejects). ``workers``
        is the tokenizer worker count the entry was measured with, or 1 when the framework's count is unverified.
        """
        return {
            "workers": self.entry.tokenizer_workers or 1,
            "base_ms": max(self.entry.fit.base_ms, 0.0),
            "per_input_token_us": self.entry.fit.per_input_token_us,
        }


def _version(text: str) -> Version | None:
    try:
        return Version(text)
    except InvalidVersion:
        return None


def load_entries(root: str | Path) -> list[Entry]:
    """Every ``host-cost`` entry under a dataset root."""
    base = Path(root) / "data" / CATEGORY
    entries = []
    for path in sorted(base.rglob(MANIFEST_NAME)) if base.is_dir() else []:
        manifest = Manifest.model_validate_json(path.read_bytes())
        settings = manifest.identity.effective_settings
        workers = settings.get("tokenizer_workers")
        entries.append(
            Entry(
                framework=manifest.identity.framework,
                version=manifest.identity.framework_version,
                tokenizer=str(settings["tokenizer_key"]),
                cpu_model=str(settings["cpu_model"]),
                system=manifest.identity.hardware,
                tokenizer_workers=int(workers) if isinstance(workers, int) else None,
                fit=read_fit(manifest),
            )
        )
    return entries


def lookup(
    entries: list[Entry],
    *,
    framework: str,
    version: str,
    tokenizer: str,
    cpu_model: str,
    allow_version_reuse: bool = True,
) -> LookupResult | None:
    """The entry for this deployment, or ``None``. Host CPU and tokenizer must match exactly."""
    candidates = [e for e in entries if (e.framework, e.tokenizer, e.cpu_model) == (framework, tokenizer, cpu_model)]
    for entry in candidates:
        if entry.version == version:
            return LookupResult(entry, version, None)
    wanted = _version(version)
    if not allow_version_reuse or wanted is None:
        return None
    parsed = [(v, e) for e in candidates if (v := _version(e.version)) is not None]
    if not parsed:
        return None
    below = [(v, e) for v, e in parsed if v <= wanted]
    pool = below if below else parsed
    # Closest at or below; if every collected version is newer, the nearest one above.
    chosen = max(pool, key=lambda item: item[0]) if below else min(pool, key=lambda item: item[0])
    return LookupResult(chosen[1], version, chosen[1].version)
