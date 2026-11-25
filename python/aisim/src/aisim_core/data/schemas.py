# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated JSON Schema snapshots for dataset documents, with a drift check.

The library is the single source of schemas. Snapshots are generated, never
hand-edited, and published under ``catalog/schemas/`` in the dataset for
readers without AISim. A committed copy ships with the package so CI can fail
when the library and the snapshots diverge.

Regenerate: ``python -m aisim_core.data.schemas --write``
Check:      ``python -m aisim_core.data.schemas --check``
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from aisim_core.data import CategoryIndex, DatasetIndex, Manifest
from aisim_core.data.catalog import CatalogCoverage, CatalogRow, CatalogRowV1, SystemRow
from aisim_core.data.e2e import (
    E2eInputSnapshot,
    E2eManifestBody,
    MetricRow,
    ObservationRow,
    RequestRow,
    SeriesRow,
    SnapshotAcquisition,
    SnapshotMember,
)
from aisim_core.data.eval import EvalRecord
from aisim_core.data.fpm_library import FpmLibraryMetadata, FpmLibraryRow, FpmMeasurementManifest
from aisim_core.data.gpu_time import GpuTime
from aisim_core.data.host_cost import HostCostRow
from aisim_core.data.op_schema import (
    OpDeclaredReuseRow,
    OpLatencyRow,
    OpQueryVersionsRow,
    OpReuseManifestRow,
    OpSourceMapRow,
)

# Schema ID (as stored in documents) -> model. Add new versions as new entries;
# never edit an existing entry's meaning.
SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "aisim-artifact-manifest/v1": Manifest,
    "aisim-catalog-coverage/v1": CatalogCoverage,
    "aisim-catalog-systems/v1": SystemRow,
    "aisim-catalog/v1": CatalogRowV1,
    "aisim-catalog/v2": CatalogRow,
    "aisim-category-index/v1": CategoryIndex,
    "aisim-dataset-index/v1": DatasetIndex,
    "aisim-e2e-input-snapshot/v1": E2eInputSnapshot,
    "aisim-e2e-manifest/v1": E2eManifestBody,
    "aisim-e2e-metrics/v1": MetricRow,
    "aisim-e2e-observations/v1": ObservationRow,
    "aisim-e2e-requests/v1": RequestRow,
    "aisim-e2e-series/v1": SeriesRow,
    "aisim-e2e-snapshot-acquisition/v1": SnapshotAcquisition,
    "aisim-e2e-snapshot-members/v1": SnapshotMember,
    "aisim-eval/v1": EvalRecord,
    "aisim-gpu-time/v1": GpuTime,
    # FPM contract. The data schema id ``aic_fpm_forward_perf`` stays as published; the snapshots below describe its
    # v6 table notes and sidecar (with the optional ``resources`` block), and the measurement manifest v5.
    "aic_fpm_forward_perf/v6": FpmLibraryRow,
    "aic_fpm_forward_perf.metadata/v6": FpmLibraryMetadata,
    "measurement-manifest/v5": FpmMeasurementManifest,
    "aisim-host-cost/v1": HostCostRow,
    "aisim-op-declared-reuse/v1": OpDeclaredReuseRow,
    "aisim-op-latency/v1": OpLatencyRow,
    "aisim-op-query-versions/v1": OpQueryVersionsRow,
    "aisim-op-reuse-manifest/v1": OpReuseManifestRow,
    "aisim-op-source-map/v1": OpSourceMapRow,
}

PACKAGED_SNAPSHOT_DIR = Path(__file__).parent / "schema_snapshots"


def snapshot_filename(schema_id: str) -> str:
    """``aisim-dataset-index/v1`` -> ``aisim-dataset-index.v1.json``."""
    name, _, version = schema_id.rpartition("/")
    if not name or not version:
        raise ValueError(f"schema ID must look like name/vN: {schema_id!r}")
    return f"{name}.{version}.json"


def render_snapshot(schema_id: str) -> str:
    """Deterministic JSON Schema text for one registered schema ID."""
    schema = SCHEMA_MODELS[schema_id].model_json_schema(by_alias=True)
    schema["$id"] = schema_id
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def render_all() -> dict[str, str]:
    return {snapshot_filename(sid): render_snapshot(sid) for sid in SCHEMA_MODELS}


def write_snapshots(directory: str | Path = PACKAGED_SNAPSHOT_DIR) -> list[Path]:
    """Write every registered snapshot without deleting unrecognized files, and return the paths.

    A caller-supplied directory can contain unrelated JSON. Obsolete snapshots remain visible to
    ``check_snapshots`` so their owner can explicitly remove them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render_all()
    paths = []
    for name, text in rendered.items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def check_snapshots(directory: str | Path = PACKAGED_SNAPSHOT_DIR) -> list[str]:
    """Return drift problems (empty when the directory matches the library)."""
    directory = Path(directory)
    problems = []
    rendered = render_all()
    existing = {p.name for p in directory.glob("*.json")} if directory.is_dir() else set()
    for name, text in rendered.items():
        path = directory / name
        if name not in existing:
            problems.append(f"missing snapshot: {name}")
        elif path.read_text(encoding="utf-8") != text:
            problems.append(f"snapshot differs from library: {name}")
    problems.extend(f"unexpected snapshot: {name}" for name in sorted(existing - set(rendered)))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate snapshots")
    mode.add_argument("--check", action="store_true", help="fail when snapshots drift from the library")
    parser.add_argument("--dir", type=Path, default=PACKAGED_SNAPSHOT_DIR)
    args = parser.parse_args(argv)
    if args.write:
        for path in write_snapshots(args.dir):
            print(path)
        return 0
    problems = check_snapshots(args.dir)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print("Run: python -m aisim_core.data.schemas --write", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
