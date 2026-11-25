# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``host-cost`` category: measured serving-path host time per request, and the fit over it.

One configuration is one (framework, framework version, system, tokenizer) measurement on that system's
host CPU. It stores one Parquet row per timed request (``aisim-host-cost/v1``) plus an envelope manifest
that carries the host CPU and software provenance, the regression, and the GPU time of the allocation.

The manifest is public. It names the tokenizer by its Hugging Face repository and keeps only the host facts
in :data:`PUBLIC_HOST_FACTS`; host names, scheduler job IDs, node load and local paths go to the internal
record that :func:`internal_record` returns, which belongs under ``evidence/`` in ``aisim-data-internal``.

Layout: ``data/host-cost/<framework>/<version>/<system>/<tokenizer>/{requests.parquet,aisim-manifest.json}``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

from aisim_core.data import Artifact, Identity, Manifest, _verify, configuration_id
from aisim_core.data.gpu_time import GpuTime
from aisim_core.data.parquet import read_parquet, write_parquet
from aisim_core.host_cost.fit import QUEUED_COMPONENTS, Fit, fit_rows

HOST_COST_SCHEMA = "aisim-host-cost/v1"
CATEGORY = "host-cost"
PURPOSE = "host-cost-profile"
MANIFEST_NAME = "aisim-manifest.json"
# Host facts a fit depends on and that identify no machine or job.
PUBLIC_HOST_FACTS = ("cpu_model", "cpu_sockets", "cpu_threads", "affinity_cpus", "python", "tokenizers_parallelism")
_HF_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class HostCostRow(BaseModel):
    """One timed request. Times are milliseconds, the median over the repeats of that request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    request: int = Field(ge=0)
    tokens: int = Field(gt=0)
    body_kb: float = Field(ge=0)
    json_ms: float = Field(ge=0)
    validate_ms: float = Field(ge=0)
    template_ms: float = Field(ge=0)
    encode_ms: float = Field(ge=0)
    hash_ms: float | None = Field(default=None, ge=0)  # only where the framework hashes the prompt before scheduling


_ARROW = pa.schema(
    [
        ("request", pa.int64()),
        ("tokens", pa.int64()),
        ("body_kb", pa.float64()),
        ("json_ms", pa.float64()),
        ("validate_ms", pa.float64()),
        ("template_ms", pa.float64()),
        ("encode_ms", pa.float64()),
        ("hash_ms", pa.float64()),
    ]
)


def entry_directory(framework: str, version: str, system: str, tokenizer: str, *, prefix: str = "data") -> str:
    return "/".join(part for part in (prefix.strip("/"), CATEGORY, framework, version, system, tokenizer) if part)


def _identity(environment: Mapping[str, Any], system: str, model: str) -> Identity:
    host = environment["host"]
    settings = {
        "cpu_model": host["cpu_model"],
        "tokenizer_key": environment["tokenizer_key"],
        "tokenizer_class": environment["tokenizer_class"],
        "tokenizer_is_fast": environment["tokenizer_is_fast"],
        "tokenizer_workers": environment["tokenizer_workers"],
        "hash_algorithm": environment.get("hash_algorithm"),
        "block_size": environment.get("block_size"),
        "endpoint": "chat",
    }
    return Identity(
        hardware=system,
        framework=environment["framework"],
        framework_version=environment["framework_version"],
        operation_family=None,
        model=model,
        effective_settings=settings,
        unresolved_identity=None,
    )


def write_entry(
    root: str | Path,
    run_dir: str | Path,
    *,
    system: str,
    model: str,
    gpu_time: GpuTime | None,
    prefix: str = "data",
    extra_provenance: Mapping[str, Any] | None = None,
) -> str:
    """Write one harness run directory (rows.jsonl + environment.json) as a configuration; return the manifest path.

    ``model`` is the Hugging Face repository the tokenizer files came from; the run's own ``model`` may be a local
    path and is not published. ``gpu_time`` must not carry scheduler job IDs (see :func:`internal_record`).
    """
    if not _HF_REPO.fullmatch(model):
        raise ValueError(f"model must be a Hugging Face repository ID such as 'org/name', not {model!r}")
    if gpu_time is not None and "job_ids" in gpu_time.source:
        raise ValueError("gpu_time.source carries job IDs; keep them in the internal record")
    root, run_dir = Path(root), Path(run_dir)
    environment = json.loads((run_dir / "environment.json").read_text())
    raw_rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines() if line.strip()]
    rows = [HostCostRow(**{k: v for k, v in row.items() if k != "hash_ms" or v is not None}) for row in raw_rows]
    fit = fit_rows([row.model_dump() for row in rows])
    identity = _identity(environment, system, model)
    directory = entry_directory(
        environment["framework"], environment["framework_version"], system, environment["tokenizer_key"], prefix=prefix
    )
    relative = f"{directory}/requests.parquet"
    table = pa.Table.from_pylist([row.model_dump() for row in rows], schema=_ARROW)
    write_parquet(table, root / relative, HOST_COST_SCHEMA)
    data = (root / relative).read_bytes()
    import hashlib

    artifact = Artifact(
        artifact_id="requests",
        path=relative,
        role="host-cost-requests",
        format="parquet",
        schema=HOST_COST_SCHEMA,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
        rows=len(rows),
    )
    provenance: dict[str, Any] = {
        "host_cost": {
            "environment": {
                **{k: v for k, v in environment.items() if k not in {"host", "schema", "model"}},
                "model": model,
            },
            "host": {k: environment["host"].get(k) for k in PUBLIC_HOST_FACTS},
            "fit": {"components": list(QUEUED_COMPONENTS), **fit.to_json()},
            "gpu_time": gpu_time.model_dump(mode="json", by_alias=True) if gpu_time is not None else None,
            **dict(extra_provenance or {}),
        }
    }
    manifest = Manifest(
        schema_version="aisim-artifact-manifest/v1",
        category=CATEGORY,
        purpose=PURPOSE,
        configuration_id=configuration_id(CATEGORY, PURPOSE, identity),
        identity=identity,
        requested_settings=None,
        artifacts=[artifact],
        provenance=json.loads(json.dumps(provenance)),
        evidence_gaps=[],
    )
    path = f"{directory}/{MANIFEST_NAME}"
    (root / path).write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def internal_record(run_dir: str | Path, *, job_ids: list[str]) -> dict[str, Any]:
    """The cluster facts of one entry that the public manifest leaves out: host, jobs, load and tokenizer path."""
    environment = json.loads((Path(run_dir) / "environment.json").read_text())
    return {
        "schema": "aisim-host-cost-internal/v1",
        "tokenizer_path": environment["model"],
        "host": environment["host"],
        "job_ids": list(job_ids),
    }


def read_fit(manifest: Manifest) -> Fit:
    """The fit stored in a manifest."""
    stored = dict(manifest.provenance["host_cost"]["fit"])
    stored.pop("components", None)
    stored["per_input_token_us_ci95"] = tuple(stored["per_input_token_us_ci95"])
    return Fit(**stored)


def check_entry(root: str | Path, path: str) -> Manifest:
    """Verify the envelope, the Parquet rows, and that the stored fit is the fit of those rows."""
    root = Path(root)
    manifest = Manifest.model_validate_json((root / path).read_bytes())
    if manifest.category != CATEGORY or manifest.purpose != PURPOSE:
        raise ValueError("not a host-cost manifest")
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    (artifact,) = manifest.artifacts
    table = read_parquet(_verify(root, artifact), HOST_COST_SCHEMA)
    rows = [HostCostRow(**row) for row in table.to_pylist()]
    recomputed = fit_rows([row.model_dump() for row in rows])
    stored = read_fit(manifest)
    if abs(recomputed.per_input_token_us - stored.per_input_token_us) > 1e-9 * max(1.0, abs(stored.per_input_token_us)):
        raise ValueError("stored fit differs from the fit of the rows")
    return manifest


def register_entries(root: str | Path, manifest_paths: list[str]) -> str:
    """Register entries in the ``host-cost`` category index and ``catalog/dataset.json``; return the index path.

    ``catalog/dataset.json`` must exist (a fresh dataset has one); other categories are left untouched.
    """
    import hashlib

    from aisim_core.data import CategoryIndex, CategoryReference, DatasetIndex, Entry

    root = Path(root)

    def ref(relative: str, schema: str, kind: str) -> Artifact:
        data = (root / relative).read_bytes()
        return Artifact(
            artifact_id=kind,
            path=relative,
            role=kind,
            format="json",
            schema=schema,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )

    def dump(model: BaseModel) -> bytes:
        text = json.dumps(model.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True, allow_nan=False)
        return (text + "\n").encode()

    index_path = f"catalog/{CATEGORY}/index.json"
    existing: dict[str, Entry] = {}
    if (root / index_path).is_file():
        existing = {
            e.configuration_id: e for e in CategoryIndex.model_validate_json((root / index_path).read_bytes()).entries
        }
    for path in manifest_paths:
        manifest = Manifest.model_validate_json((root / path).read_bytes())
        existing[manifest.configuration_id] = Entry(
            configuration_id=manifest.configuration_id,
            purpose=manifest.purpose,
            identity=manifest.identity,
            manifest=ref(path, "aisim-artifact-manifest/v1", "manifest"),
        )
    index = CategoryIndex(
        schema_version="aisim-category-index/v1",
        category=CATEGORY,
        entries=[existing[key] for key in sorted(existing)],
    )
    (root / index_path).parent.mkdir(parents=True, exist_ok=True)
    (root / index_path).write_bytes(dump(index))
    dataset = DatasetIndex.model_validate_json((root / "catalog/dataset.json").read_bytes())
    categories = dict(dataset.categories)
    categories[CATEGORY] = CategoryReference(
        root=f"data/{CATEGORY}", layout="category", index=ref(index_path, "aisim-category-index/v1", "index")
    )
    (root / "catalog/dataset.json").write_bytes(dump(dataset.model_copy(update={"categories": categories})))
    return index_path
