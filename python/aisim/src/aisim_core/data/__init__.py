# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared public/private dataset contracts and offline profile preparation.

This module never authenticates, downloads, publishes, or runs stored scripts.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from aisim_core.data.fpm_layout import OTHER_CATEGORY_PREFIXES, PHYSICAL_PREFIX, storage_path

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Category = Literal["op-latency", "fpm", "e2e", "eval", "systems", "host-cost"]
Purpose = Literal[
    "operation-tables",
    "execution-profile",
    "e2e-configuration",
    "evaluation-record",
    "system-spec",
    "host-cost-profile",
]
# Version 1 hashed only the original Identity fields. Version 2 adds the
# first-class deployment fields below. Version 3 adds the deployment facts that
# distinguish otherwise identical E2E configurations (attention data parallelism,
# multinode topology, KV offload mode). A version is chosen from the identity
# (see configuration_id) so existing IDs never change and a field of a later
# version can never be silently dropped from a hash.
IDENTITY_HASH_VERSIONS = (1, 2, 3)
_V2_FIELDS = ("parallelism", "parallelism_complete", "precision", "deployment")
_V3_FIELDS = ("dp_attention", "multinode", "offload_mode")
_FIELDS_ADDED_AFTER = {1: _V2_FIELDS + _V3_FIELDS, 2: _V3_FIELDS, 3: ()}
_DIMENSION = re.compile(r"^(x-)?[a-z][a-z0-9_]*$")


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("expected a canonical repository-relative path")
    return value


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class Artifact(_Record):
    artifact_id: str = Field(min_length=1)
    path: str
    role: str = Field(min_length=1)
    format: str = Field(min_length=1)
    artifact_schema: str = Field(alias="schema", min_length=1)
    sha256: Digest
    bytes: int = Field(ge=0)
    rows: int | None = Field(default=None, ge=0)
    # Only runtime inputs have a destination in the materialized systems root.
    runtime_path: str | None = None

    @field_validator("path", "runtime_path")
    @classmethod
    def relative_paths(cls, value):
        return None if value is None else _relative(value)


class Identity(_Record):
    hardware: str
    framework: str
    framework_version: str
    operation_family: str | None
    model: str | None
    effective_settings: dict[str, JsonValue]
    unresolved_identity: dict[str, JsonValue] | None
    # Identity hash v2 fields. None means unknown, never a default.
    # Parallelism is an open {worker_role: {dimension: degree}} map, e.g.
    # {"aggregated": {"tp": 4, "ep": 8}}. Degree 1 is normalized away, so a new
    # dimension at degree 1 leaves existing IDs unchanged. Unregistered
    # dimensions need an "x-" prefix (registry checks live in the rule set).
    parallelism: dict[str, dict[str, int]] | None = None
    # True: absent dimensions are 1. Otherwise absent dimensions are unknown.
    parallelism_complete: bool | None = None
    precision: dict[str, str] | None = None  # e.g. {"weights": "fp8", "kv_cache": "fp8"}
    deployment: Literal["aggregated", "disaggregated"] | None = None
    # Identity hash v3 fields. None means not reported, never a default.
    # ``dp_attention`` is the source's per-role boolean (never a degree; degrees
    # live in ``parallelism``), ``multinode`` is the source's multinode flag and
    # ``offload_mode`` its KV offload setting (e.g. "off", "on").
    dp_attention: dict[str, bool] | None = None
    multinode: bool | None = None
    offload_mode: str | None = None

    @field_validator("dp_attention")
    @classmethod
    def normalize_dp_attention(cls, value):
        if value is None:
            return None
        if any(not role for role in value):
            raise ValueError("worker role must be non-empty")
        return dict(sorted(value.items()))

    @field_validator("parallelism")
    @classmethod
    def normalize_parallelism(cls, value):
        if value is None:
            return None
        out = {}
        for role, dims in value.items():
            if not role:
                raise ValueError("worker role must be non-empty")
            kept = {}
            for name, degree in dims.items():
                if not _DIMENSION.match(name):
                    raise ValueError(f"invalid parallelism dimension: {name!r}")
                if type(degree) is not int or degree < 1:
                    raise ValueError(f"parallelism degree must be a positive integer: {name}={degree!r}")
                if degree != 1:
                    kept[name] = degree
            out[role] = dict(sorted(kept.items()))
        return dict(sorted(out.items()))

    def uses_v2_fields(self) -> bool:
        return any(getattr(self, f) is not None for f in _V2_FIELDS)

    def uses_v3_fields(self) -> bool:
        return any(getattr(self, f) is not None for f in _V3_FIELDS)


class Manifest(_Record):
    schema_version: Literal["aisim-artifact-manifest/v1"]
    category: Category
    purpose: Purpose
    configuration_id: str = Field(pattern=r"^cfg-[0-9a-f]{64}$")
    identity: Identity
    requested_settings: dict[str, JsonValue] | None
    artifacts: list[Artifact] = Field(min_length=1)
    provenance: dict[str, JsonValue]
    evidence_gaps: list[dict[str, JsonValue]]


class Entry(_Record):
    configuration_id: str = Field(pattern=r"^cfg-[0-9a-f]{64}$")
    purpose: Purpose
    identity: Identity
    manifest: Artifact


class CategoryIndex(_Record):
    schema_version: Literal["aisim-category-index/v1"]
    category: Category
    entries: list[Entry]
    # Files every selection of the category needs but no entry owns (op-latency: the two lookup tables). Listed
    # here they are digest-pinned by this index, fetched with any selection and verified before use.
    support: list[Artifact] = Field(default_factory=list)


class CategoryReference(_Record):
    root: str
    index: Artifact
    layout: Literal["category"]

    _path = field_validator("root")(_relative)


class DatasetIndex(_Record):
    schema_version: Literal["aisim-dataset-index/v1"]
    dataset_id: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    categories: dict[Category, CategoryReference]


def configuration_id(category: str, purpose: str, identity: Identity, version: int | None = None) -> str:
    """Hash effective identity, including explicit nulls and unresolved source facts.

    version=None picks the lowest version that can represent every set field: 1
    when no v2 or v3 field is set (keeping every existing ID stable), 2 when only
    v2 fields are set, 3 otherwise. An older version cannot hash an identity that
    sets a later field.
    """
    if version is None:
        version = 3 if identity.uses_v3_fields() else 2 if identity.uses_v2_fields() else 1
    if version not in IDENTITY_HASH_VERSIONS:
        raise ValueError(f"unsupported identity hash version: {version}")
    exclude = set(_FIELDS_ADDED_AFTER[version])
    if version == 1 and identity.uses_v2_fields():
        raise ValueError("identity hash v1 cannot represent parallelism, precision or deployment")
    if version < 3 and identity.uses_v3_fields():
        raise ValueError(f"identity hash v{version} cannot represent dp_attention, multinode or offload_mode")
    payload = {
        "version": version,
        "category": category,
        "purpose": purpose,
        "identity": identity.model_dump(exclude=exclude),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "cfg-" + hashlib.sha256(canonical.encode()).hexdigest()


def path_component(value: str) -> str:
    """``value`` as one directory or file stem; refuses what could leave its parent directory."""
    if value in ("", ".", "..") or any(c in value for c in "/\\\0"):
        raise ValueError(f"not a safe path component: {value!r}")
    return value


def _plain_file(root: Path, relative: str) -> Path:
    path = root
    for part in PurePosixPath(_relative(relative)).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in dataset path: {relative}")
    if not path.is_file():
        raise ValueError(f"missing dataset artifact: {relative}")
    return path


def _file(root: Path, relative: str) -> Path:
    # Moved FPM manifests retain logical legacy keys. Resolve only storage locations; manifest bytes and
    # configuration/case/observation identities stay untouched. Canonical category paths need no catalog read.
    _relative(relative)
    if (
        relative.startswith("data/")
        and not relative.startswith((PHYSICAL_PREFIX, *OTHER_CATEGORY_PREFIXES))
        and (root / "catalog/dataset.json").exists()
    ):
        index = DatasetIndex.model_validate_json(_plain_file(root, "catalog/dataset.json").read_bytes())
        relative = storage_path(relative, {name: ref.root for name, ref in index.categories.items()})
    return _plain_file(root, relative)


def _verify(root: Path, artifact: Artifact) -> Path:
    path = _file(root, artifact.path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != artifact.bytes or digest != artifact.sha256:
        raise ValueError(f"artifact integrity mismatch: {artifact.path}")
    if artifact.role == "fpm-library":
        # Native libraries are identified by the manifest declaration alone, so validate it before any branch on
        # the declared format.
        from aisim_core.data.fpm_library import LIBRARY_SCHEMA_ID

        if artifact.format != "parquet" or artifact.rows is None:
            raise ValueError(f"an fpm-library artifact must declare format parquet and rows: {artifact.path}")
        if artifact.artifact_schema != LIBRARY_SCHEMA_ID:
            raise ValueError(f"unsupported FPM library schema ID {artifact.artifact_schema!r}: {artifact.path}")
    if artifact.format == "parquet" and (
        artifact.rows is None or pq.ParquetFile(path).metadata.num_rows != artifact.rows
    ):
        raise ValueError(f"Parquet row count mismatch: {artifact.path}")
    return path


def _verified_bytes(root: Path, artifact: Artifact) -> bytes:
    """The bytes of an artifact after the same size, digest and Parquet row-count checks as :func:`_verify`.

    Parse these bytes, not the path again: a file replaced after the check would otherwise be read unverified.
    """
    data = _file(root, artifact.path).read_bytes()
    if len(data) != artifact.bytes or hashlib.sha256(data).hexdigest() != artifact.sha256:
        raise ValueError(f"artifact integrity mismatch: {artifact.path}")
    if artifact.format == "parquet" and (
        artifact.rows is None or pq.ParquetFile(pa.BufferReader(data)).metadata.num_rows != artifact.rows
    ):
        raise ValueError(f"Parquet row count mismatch: {artifact.path}")
    return data


def check_manifest(root: Path, path: str) -> Manifest:
    """Check one manifest and all declared inputs without network access."""
    root = Path(root)
    manifest = Manifest.model_validate_json(_file(root, path).read_bytes())
    return _check_manifest(root, manifest)


def _check_manifest(root: Path, manifest: Manifest) -> Manifest:
    """Validate an already parsed manifest and all its declared inputs."""
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    # fpm manifests are execution-profile bundles (the shared-envelope FPM layout);
    # e2e/eval/systems manifests have no offline checker here yet.
    if manifest.category not in ("op-latency", "fpm"):
        raise ValueError("use the existing category protocol for E2E manifests")
    if manifest.category == "fpm" and manifest.purpose != "execution-profile":
        raise ValueError("fpm manifests must have purpose execution-profile")
    if manifest.purpose == "operation-tables" and manifest.identity.model is not None:
        raise ValueError("operation table storage must remain model-independent")
    for field in ("artifact_id", "path", "runtime_path"):
        values = [getattr(a, field) for a in manifest.artifacts if getattr(a, field) is not None]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate artifact {field}")
    for artifact in manifest.artifacts:
        _verify(root, artifact)
    return manifest


def check_envelope(root: Path, path: str) -> Manifest:
    """Check a manifest of a category with no offline checker here: its identity hash and every declared artifact.

    The category's own validators check the content; this keeps a missing or altered artifact from passing.
    """
    root = Path(root)
    return _check_envelope(root, Manifest.model_validate_json(_file(root, path).read_bytes()))


def _check_envelope(root: Path, manifest: Manifest) -> Manifest:
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    for artifact in manifest.artifacts:
        _verify(root, artifact)
    return manifest


def check_dataset(root: Path, *, unchecked_categories: tuple[str, ...] = ()) -> DatasetIndex:
    """Verify shared discovery indexes and the manifests they reference.

    Manifests of ``unchecked_categories`` (categories with no offline checker here) get the envelope checks of
    :func:`check_envelope` and the catalog identity comparison, but no category-specific checks.
    """
    root = Path(root)
    index = DatasetIndex.model_validate_json(_file(root, "catalog/dataset.json").read_bytes())
    for category, reference in index.categories.items():
        entries = CategoryIndex.model_validate_json(_verified_bytes(root, reference.index))
        if entries.category != category:
            raise ValueError("category index identity mismatch")
        for artifact in entries.support:
            _verify(root, artifact)
        seen = set()
        for entry in entries.entries:
            if entry.configuration_id in seen:
                raise ValueError("duplicate configuration ID")
            seen.add(entry.configuration_id)
            # Parse the bytes that passed the integrity check, then run the category's checks on that manifest.
            manifest = Manifest.model_validate_json(_verified_bytes(root, entry.manifest))
            if category in unchecked_categories:
                manifest = _check_envelope(root, manifest)
            elif category == "e2e":
                # Import lazily: the E2E protocol builds on this shared envelope module.
                from aisim_core.data.e2e import load_e2e

                loaded = load_e2e(root, entry.manifest.path).manifest
                if loaded.model_dump_json() != manifest.model_dump_json():
                    raise ValueError("manifest changed during dataset validation")
            elif category == "host-cost":
                from aisim_core.data.host_cost import check_entry

                if check_entry(root, entry.manifest.path).model_dump_json() != manifest.model_dump_json():
                    raise ValueError("manifest changed during dataset validation")
            else:
                manifest = _check_manifest(root, manifest)
            if (manifest.category, manifest.configuration_id, manifest.identity, manifest.purpose) != (
                category,
                entry.configuration_id,
                entry.identity,
                entry.purpose,
            ):
                raise ValueError("catalog and manifest identity mismatch")
    return index


def prepare_execution_profile(root: Path, manifest_path: str, output_dir: Path) -> dict:
    """Materialize verified local inputs and return canonical best_available config.

    output_dir must not exist. Canonical construction validates the compiled
    basis and retained runtime inputs without changing profile bytes or pins.
    Model metadata must be bundled or in an existing local model directory.
    Callers keep both the output and any local model directory for later reloads.
    """
    from aisim_core.sdk import RustForwardPassPerfModel
    from aisim_core.sdk.common import DefaultHFModels

    root, output_dir = Path(root), Path(output_dir).absolute()
    manifest = check_manifest(root, manifest_path)
    if manifest.purpose != "execution-profile":
        raise ValueError("an execution-profile manifest is required")
    profiles = [a for a in manifest.artifacts if a.role == "execution-profile"]
    configs = [a for a in manifest.artifacts if a.role == "predictor-config"]
    if len(profiles) != 1 or len(configs) != 1:
        raise ValueError("exactly one profile and predictor config are required")
    profile, config_artifact = profiles[0], configs[0]
    profile_bytes = _verify(root, profile).read_bytes()
    config_bytes = _verify(root, config_artifact).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != config_artifact.sha256:
        raise ValueError("configuration changed during preparation")
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("predictor config must be a JSON object")
    model = config.get("model")
    if isinstance(model, str) and model not in DefaultHFModels and not Path(model).is_dir():
        raise ValueError(
            "offline profile preparation requires bundled model metadata or an existing local model directory"
        )
    selection = config
    for key in ("estimator_config", "op_level", "execution_profile"):
        selection = selection.get(key) if isinstance(selection, dict) else None
    if not isinstance(selection, dict):
        raise ValueError("predictor config lacks estimator_config.op_level.execution_profile")
    if selection.get("sha256") != profile.sha256:
        raise ValueError("configuration profile pin mismatch")
    declared_identity = (manifest.identity.hardware, manifest.identity.framework, manifest.identity.framework_version)
    if declared_identity != (config.get("system"), config.get("backend"), config.get("backend_version")):
        raise ValueError("manifest identity differs from predictor configuration")
    declared_model, configured_model = manifest.identity.model, config.get("model")
    if (
        declared_model is not None
        and configured_model is not None
        and declared_model != configured_model
        # A local model directory stands in for the published ID, so only a named model can disagree.
        and not Path(str(configured_model)).is_dir()
    ):
        # Rust admission sees only the configuration: a bundle for model A must not come back as one for model B.
        raise ValueError("manifest model differs from predictor configuration")
    inputs = {a.runtime_path: a for a in manifest.artifacts if a.runtime_path is not None}
    profile_doc = json.loads(profile_bytes)
    retained = profile_doc.get("retained_files") if isinstance(profile_doc, dict) else None
    if not isinstance(retained, list) or not all(
        isinstance(f, dict)
        and isinstance(f.get("path"), str)
        and isinstance(f.get("sha256"), str)
        and type(f.get("size_bytes")) is int
        for f in retained
    ):
        raise ValueError("profile retained_files is malformed")
    declared = {f["path"]: (f["sha256"], f["size_bytes"]) for f in retained}
    actual = {p: (a.sha256, a.bytes) for p, a in inputs.items()}
    if len(declared) != len(retained) or declared != actual:
        raise ValueError("runtime inputs differ from the pinned profile inventory")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        for relative, artifact in inputs.items():
            destination = output_dir / "systems" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_verify(root, artifact), destination)
            _verify(output_dir / "systems", artifact.model_copy(update={"path": relative}))
        destination = output_dir / "profile.json"
        destination.write_bytes(profile_bytes)
        _verify(output_dir, profile.model_copy(update={"path": "profile.json"}))
        config["systems_paths"] = [str(output_dir / "systems")]
        selection["path"] = str(destination)
        # Use the same Rust-owned admission/readiness checks as prediction;
        # manifest/profile agreement alone cannot establish runtime completeness.
        # The saved configuration is what gets checked: an AISIM_DATASET/AISIM_DATA_LOCAL pin in the environment
        # must neither add a dataset root to it nor fetch, so the check runs with the pin suppressed.
        from aisim_core.data.ensure import _guard

        with _guard():
            RustForwardPassPerfModel.best_available(config)
        (output_dir / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return config


# The FPM library contract (staging, sidecar schema, resource precedence). Defined in a leaf module that does not
# import this package, so the re-export is safe here.
from aisim_core.data.fpm_library import (  # noqa: F401
    FpmContractError,
    FpmLibraryMetadata,
    FpmResourcesPendingError,
    StagedFpmLibrary,
    complete_fpm_profile,
    resolve_resources,
    stage_fpm_library,
)
