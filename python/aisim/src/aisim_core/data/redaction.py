# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redaction rules and the publication content check (DEP #435, "Accidental disclosure").

What may move from ``aisim-data-internal`` to the public dataset is decided by a
:class:`RedactionPolicy`. The defaults are the most conservative reading of the
disclosure decision:

* Visibility is a label per path prefix in ``mirror/visibility.json``
  (:class:`VisibilityLedger`); an unlabeled path is ``internal``.
* ``producer.run_url`` (``producer_id``), image names and digests, patch
  references, logs and raw tool output are withheld. They must be absent or the
  literal ``[REDACTED]``; their original hash is kept as lineage in the private
  promotion ledger, never as a public field.
* Data derived from InferenceX is published as aggregates only. Per-point truth,
  per-request rows, raw output and logs are never public, so an InferenceX-derived
  configuration or evaluation record is not promotable; a page may publish rows that
  pass :func:`check_disclosure`.
* Promotion is a path-identical copy, so redaction happens before a directory is
  labeled ``public``. This module checks the result; it never rewrites bytes.

Every check is pure and offline. ``check_publishable`` is also what the ``public``
profile of :mod:`aisim_core.data.mirror` runs, so the public repository CI
re-checks anything that was promoted.
"""

from __future__ import annotations

import base64
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aisim_core.data import Artifact, CategoryIndex, DatasetIndex, Manifest, _verify
from aisim_core.data.catalog import CATALOG_DIR, COVERAGE_FILE, catalog_files, read_catalog_coverage
from aisim_core.data.e2e import (
    OBSERVATIONS_SCHEMA,
    REQUESTS_SCHEMA,
    E2eManifestBody,
    body_from_manifest,
    load_e2e,
    read_snapshot,
)
from aisim_core.data.eval import load_record, record_paths
from aisim_core.data.hygiene import (
    DuplicateJsonKeyError,
    Finding,
    _unique_json_keys,
    bounded_json_bytes,
    checked_path,
    json_sidecar_value,
    yaml_sidecar_value,
)
from aisim_core.data.parquet import read_parquet

Label = Literal["public", "internal", "restricted"]
LEDGER_PATH = "mirror/visibility.json"
LEDGER_SCHEMA = "aisim-visibility/v1"
REDACTED = "[REDACTED]"
INFX_MARKERS = ("inferencex", "infx")

_RUN_URL = re.compile(r"https?://[^\s\"']+/(?:actions/runs|-/pipelines|-/jobs|pipelines|builds)/\d+", re.I)
_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://", re.I)
_DIGEST = re.compile(r"@sha256:[0-9a-f]{16,}", re.I)
_REGISTRY = re.compile(r"(?:nvcr\.io|gitlab-master|ghcr\.io|docker\.io|gcr\.io|\.dkr\.ecr\.|urm\.|artifactory)", re.I)
_INTERNAL_HOST = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:nvidia\.com|nvidia\.net)\b|gitlab-master", re.I)
_PATCH_VALUE = re.compile(r"(?m)^(?:diff --git |--- a/|\+\+\+ b/)|\.(?:patch|diff)\b", re.I)
_SECRET = re.compile(
    r"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,}|bearer\s+[A-Za-z0-9._\-]{16,})", re.I
)
# A URL whose userinfo carries a password (``scheme://user:password@host``), whatever the host or token format.
_URL_PASSWORD = re.compile(r"[a-z][a-z0-9+.\-]*://[^/\s?#@:]*:[^/\s?#@]+@", re.I)
_CREDENTIAL_KEY = re.compile(
    r"^(password|passwd|secret|client_secret|api_key|access_token|auth_token|refresh_token|id_token|"
    r"private_key|authorization|credential|credentials|"
    r"aws_secret_access_key|aws_session_token|docker_auth_config)$",
    re.I,
)
_IMAGE_KEY = re.compile(r"(^|_)(image|images|container|docker_image)(_|$)", re.I)
_PATCH_KEY = re.compile(r"(^|_)patch(es)?($|_)", re.I)
_RUN_KEY = re.compile(r"(^|_)run_url$|^producer_id$", re.I)
# Cluster facts that identify a machine or a scheduler job: they stay in ``aisim-data-internal``.
_CLUSTER_KEY = re.compile(
    r"^(hostnames?|host_names?|node_?names?|node_?list|job_ids?|slurm_(job_ids?|job_name|nodelist|node_?names?)"
    r"|load_average(_[a-z0-9]+)?)$",
    re.I,
)
_CLUSTER_HOST = re.compile(r"\b[\w-]+-slurm-[\w-]+\b", re.I)
_LOCAL_PATH = re.compile(r"(?:^|[\s\"'=:(])/(?:workspace|lustre|gpfs|scratch|raid|home|mnt|data)/", re.I)


@dataclass(frozen=True)
class RedactionPolicy:
    """Conservative defaults; widening any field is a recorded maintainer decision."""

    allow_provenance_artifacts: bool = False  # logs and raw tool output (tier 3)
    allow_request_rows: bool = False  # per-request tables, for every configuration
    # Per-request tables of self-collected configurations (body ``origin == "self-collected"``).
    # InferenceX-derived and official (upstream-licensed) configurations never use this.
    allow_self_collected_request_rows: bool = True
    allowed_run_url_prefixes: tuple[str, ...] = ()
    allowed_image_prefixes: tuple[str, ...] = ()
    # "aggregates-only": InferenceX-derived configurations and records are not promotable.
    # "point-level": they may be promoted once every other rule passes.
    inferencex_disclosure: Literal["aggregates-only", "point-level"] = "aggregates-only"
    min_scored: int = 5  # smallest cell an aggregate row may describe


DEFAULT_POLICY = RedactionPolicy()

# Fields a published aggregate row may carry. Anything else (case ids, per-point
# truth, request rows, URLs, images) is refused rather than filtered.
AGGREGATE_FIELDS = frozenset(
    {
        "hardware",
        "model",
        "framework",
        "framework_version",
        "layer",
        "n_total",
        "n_scored",
        "n_configs",
        "mape_pct",
        "residual_quantiles_pct",
        "coverage",
        "status",
        "created_at",
        "snapshot_id",
        "dataset_sha",
    }
)


# --------------------------------------------------------------------------- visibility


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


_LABEL_RANK = ("public", "restricted", "internal")  # least to most restrictive; unknown labels fail validation


class VisibilityEntry(_Model):
    path: str = Field(min_length=1)  # a file or a directory prefix, repository-relative
    label: Label
    origin: str | None = None  # e.g. "inferencex"; marks derived data for the disclosure rule
    # path -> SHA-256 of the unredacted original, when the labeled bytes are a redaction of it.
    original_sha256: dict[str, str] = Field(default_factory=dict)


class VisibilityLedger(_Model):
    schema_version: Literal["aisim-visibility/v1"] = LEDGER_SCHEMA
    entries: list[VisibilityEntry] = Field(default_factory=list)

    def entries_for(self, path: str) -> list[VisibilityEntry]:
        """All covering entries contribute origin, even when a child overrides visibility."""
        return [e for e in self.entries if path == e.path.rstrip("/") or path.startswith(e.path.rstrip("/") + "/")]

    def entry_for(self, path: str) -> VisibilityEntry | None:
        """Longest matching prefix wins, so a file can be narrower than its directory."""
        hits = self.entries_for(path)
        # Longest normalized prefix wins; equal prefixes resolve to the most restrictive label.
        return max(hits, key=lambda e: (len(e.path.rstrip("/")), _LABEL_RANK.index(e.label)), default=None)

    def label_for(self, path: str) -> Label:
        entry = self.entry_for(path)
        return entry.label if entry else "internal"  # unlabeled fails closed


def load_ledger(root: str | Path) -> VisibilityLedger:
    path = checked_path(Path(root), LEDGER_PATH)
    if not path.exists():
        return VisibilityLedger()
    return VisibilityLedger.model_validate(json_sidecar_value(bounded_json_bytes(path)))


def write_ledger(root: str | Path, ledger: VisibilityLedger) -> Path:
    path = Path(root) / LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(ledger.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- text scanning


def _walk(value: Any, key: str = "", seen: set[tuple[int, str]] | None = None) -> Iterable[tuple[str, str]]:
    """Yield (key, string) for every string in a JSON-like value, with the nearest key name."""
    if seen is None:
        seen = set()
    if isinstance(value, (Mapping, list, tuple)):
        marker = (id(value), key)
        if marker in seen:
            return  # YAML aliases are scanned once per field context, without exponential expansion
        seen.add(marker)
    if isinstance(value, Mapping):
        for k, v in value.items():
            # Arrow binary map keys arrive as bytes: decode them strictly (invalid UTF-8 raises, as for binary
            # values) so the field-name rules see "password", not "b'password'". Entries stay separate.
            name = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            yield "", name  # content rules apply to keys; field-name rules apply only to their values
            inherited = (
                _IMAGE_KEY.search(key) or _PATCH_KEY.search(key) or _RUN_KEY.search(key) or _CREDENTIAL_KEY.search(key)
            )
            # A value under a ruled key keeps that key's rule. A nested credential field also keeps its own, so a
            # credential inside an image configuration is judged as a credential (a number is a value too).
            contexts = [key] if inherited else [name]
            if inherited and _CREDENTIAL_KEY.search(name):
                contexts.append(name)
            for context in contexts:
                yield from _walk(v, context, seen)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk(v, key, seen)
    elif isinstance(value, str):
        yield key, value
    elif (
        value is not None and not isinstance(value, bytes) and (_CREDENTIAL_KEY.search(key) or _CLUSTER_KEY.search(key))
    ):
        yield key, str(value)
    elif isinstance(value, bytes):
        # Binary extension columns and Arrow metadata must not bypass disclosure
        # checks. Invalid UTF-8 is refused rather than decoded lossily.
        yield key, value.decode("utf-8")


def _json_text(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_unique_json_keys)
    except DuplicateJsonKeyError:
        raise
    except (ValueError, RecursionError):
        return text


def scan_value(value: Any, policy: RedactionPolicy, path: str, where: str, *, _key: str = "") -> list[Finding]:
    """Findings for run URLs, images, digests, patch references, secrets, internal hosts, cluster facts, local paths."""
    out: list[Finding] = []

    def flag(code: str, detail: str) -> None:
        out.append(Finding(code, f"{where}: {detail}", path))

    scanned: set[tuple[str, str]] = set()
    for key, text in _walk(value, _key):
        if text == REDACTED or not text:
            continue
        if (key, text) in scanned:
            continue  # an aliased or repeated scalar in the same field context has the same findings
        scanned.add((key, text))
        if _CREDENTIAL_KEY.search(key):
            flag("secret", "credential field must be removed or [REDACTED]")
        try:
            nested = _json_text(text) if text.lstrip(" \t\r\n")[:1] in ("{", "[") else None
        except DuplicateJsonKeyError:
            flag("duplicate-json-key", "JSON-encoded metadata repeats an object key")
            continue
        if isinstance(nested, (dict, list)):
            out.extend(scan_value(nested, policy, path, where, _key=key))
            continue
        if (_RUN_KEY.search(key) or _RUN_URL.search(text)) and not any(
            text.startswith(p) for p in policy.allowed_run_url_prefixes
        ):
            flag("run-url", "producer run URL must be removed or [REDACTED]")
        if _DIGEST.search(text):
            flag("image-digest", "image digest must be removed")
        elif (_IMAGE_KEY.search(key) or _REGISTRY.search(text)) and not any(
            text.startswith(p) for p in policy.allowed_image_prefixes
        ):
            flag("image-name", "image name must be removed or [REDACTED]")
        if _PATCH_KEY.search(key) or _PATCH_VALUE.search(text):
            flag("patch-reference", "patch reference or diff text must be removed")
        if _SECRET.search(text) or _URL_PASSWORD.search(text):
            flag("secret", "credential-like text")
        if _INTERNAL_HOST.search(text) or _CLUSTER_HOST.search(text):
            flag("internal-host", "internal host name")
        if _CLUSTER_KEY.search(key):
            flag("cluster-detail", "host names, scheduler job IDs and node load stay internal")
        if _LOCAL_PATH.search(text):
            flag("local-path", "a path on the collection host must be replaced by a public identifier")
    return out


def scan_json_file(path: Path, policy: RedactionPolicy, rel: str, where: str) -> list[Finding]:
    """Scan a JSON file with the shared bounded, strict parser; ambiguous JSON raises ``ValueError``."""
    return scan_value(json_sidecar_value(bounded_json_bytes(path)), policy, rel, where)


def scan_yaml(data: bytes, policy: RedactionPolicy, path: str, where: str) -> list[Finding]:
    """Scan original mapping values, including overridden merge defaults, and source text."""
    yaml_sidecar_value(data)  # also require valid effective YAML merge semantics
    value = yaml_sidecar_value(data, preserve_merges=True)
    return [*scan_value(value, policy, path, where), *scan_value(data.decode("utf-8"), policy, path, f"{where} source")]


# --------------------------------------------------------------------------- content check


def _is_infx(source_kind: str | None) -> bool:
    return bool(source_kind) and any(m in source_kind.lower() for m in INFX_MARKERS)


def _request_rows_allowed(body: E2eManifestBody | None, policy: RedactionPolicy) -> bool:
    if policy.allow_request_rows:
        return True
    return (
        policy.allow_self_collected_request_rows
        and body is not None
        and body.origin == "self-collected"
        and not _is_infx(body.source_kind)
    )


def _check_manifest(
    root: Path, rel: str, manifest: Manifest, policy: RedactionPolicy, infx_ids: set[str]
) -> list[Finding]:
    out: list[Finding] = []
    body: E2eManifestBody | None = None
    if manifest.category == "e2e":
        body = body_from_manifest(manifest)
        if manifest.configuration_id in infx_ids and policy.inferencex_disclosure == "aggregates-only":
            out.append(
                Finding(
                    "infx-aggregates-only",
                    "InferenceX-derived configurations are not published; publish aggregates (check_disclosure)",
                    rel,
                )
            )
        for info in body.provenance_artifacts:
            if not policy.allow_provenance_artifacts:
                out.append(
                    Finding("provenance-withheld", f"{info.kind} artifacts are withheld ({info.artifact_id})", rel)
                )
            elif not info.redacted:
                out.append(Finding("unredacted-provenance", f"{info.artifact_id} is not redacted", rel))
    for artifact in manifest.artifacts:
        suffixes = {"parquet": ".parquet", "json": ".json", "yaml": ".yaml", "gzip": ".gz"}
        if artifact.format not in suffixes or Path(artifact.path).suffix.lower() != suffixes[artifact.format]:
            out.append(
                Finding(
                    "artifact-format", f"unsupported or mismatched artifact format: {artifact.format}", artifact.path
                )
            )
            continue
        if artifact.format == "gzip" and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", "logs and raw output are withheld", artifact.path))
        if artifact.artifact_schema == REQUESTS_SCHEMA and not _request_rows_allowed(body, policy):
            out.append(Finding("request-rows", "per-request rows are not published", artifact.path))
        if artifact.artifact_schema == OBSERVATIONS_SCHEMA:
            out.extend(_check_observations(root / artifact.path, artifact.path, policy))
        elif artifact.artifact_schema == REQUESTS_SCHEMA and _request_rows_allowed(body, policy):
            out.extend(_check_requests(root / artifact.path, artifact.path, policy))
        elif artifact.format == "parquet":
            # Metrics, series, op tables, snapshot members/acquisition and eval
            # truth all have free-text columns. Scan every column, including
            # extension schemas, rather than exempting unmatched schema IDs.
            out.extend(
                _check_table_text(
                    read_parquet(root / artifact.path, artifact.artifact_schema),
                    artifact.path,
                    policy,
                    "column",
                    path=root / artifact.path,
                )
            )
        elif artifact.format == "json":
            out.extend(scan_json_file(checked_path(root, artifact.path), policy, artifact.path, "JSON sidecar"))
        elif artifact.format == "yaml":
            data = checked_path(root, artifact.path).read_bytes()
            out.extend(scan_yaml(data, policy, artifact.path, "YAML sidecar"))
    out.extend(scan_value(manifest.model_dump(mode="json", by_alias=True), policy, rel, "manifest"))
    return out


def _check_support(root: Path, artifact: Artifact, policy: RedactionPolicy) -> list[Finding]:
    """Scan a category-level support file (op-latency lookup tables) like a manifest's declared artifact."""
    suffixes = {"parquet": ".parquet", "json": ".json", "yaml": ".yaml"}
    if artifact.format not in suffixes or Path(artifact.path).suffix.lower() != suffixes[artifact.format]:
        return [
            Finding("artifact-format", f"unsupported or mismatched support format: {artifact.format}", artifact.path)
        ]
    if artifact.format == "parquet":
        path = root / artifact.path
        return _check_table_text(
            read_parquet(path, artifact.artifact_schema), artifact.path, policy, "support column", path=path
        )
    if artifact.format == "json":
        return scan_json_file(checked_path(root, artifact.path), policy, artifact.path, "support JSON")
    return scan_yaml(checked_path(root, artifact.path).read_bytes(), policy, artifact.path, "support YAML")


def _check_observations(path: Path, rel: str, policy: RedactionPolicy) -> list[Finding]:
    table: pa.Table = read_parquet(path, OBSERVATIONS_SCHEMA)
    out: list[Finding] = []
    for name in table.column_names:
        values = [v for v in table.column(name).to_pylist() if v not in (None, "", REDACTED)]
        if not values:
            continue
        if name in ("raw_artifact_id", "log_artifact_id") and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", f"{name} links withheld evidence", rel))
        elif name == "evidence_refs" and any(values) and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", "evidence_refs link withheld evidence", rel))
    out.extend(_check_table_text(table, rel, policy, "column", path=path))
    return out


def _check_requests(path: Path, rel: str, policy: RedactionPolicy) -> list[Finding]:
    """Scan every string column of an accepted request table; one finding per column stops the copy."""
    return _check_table_text(read_parquet(path, REQUESTS_SCHEMA), rel, policy, "request column", path=path)


def _is_list_view(dtype: pa.DataType) -> bool:
    """List-view types only exist in newer PyArrow releases; the declared minimum (15.0.0) has no predicates."""
    return any(
        getattr(pa.types, name)(dtype) for name in ("is_list_view", "is_large_list_view") if hasattr(pa.types, name)
    )


# What every data type has; an extension type's own parameters are the attributes beyond these.
_TYPE_ATTRIBUTES = frozenset(dir(pa.DataType)) | {"extension_name", "storage_type"}


def _is_extension(dtype: pa.DataType) -> bool:
    return isinstance(dtype, getattr(pa, "BaseExtensionType", pa.ExtensionType))


def _extension_parameters(dtype: pa.DataType) -> tuple[dict[str, Any], list[pa.DataType], bool]:
    """Text, types and inspectability of an extension type's parameters.

    The reader restores a built-in extension (opaque, tensor, ...) into type attributes and leaves
    ``field.metadata`` empty, so its name and parameters are visible only here. Parameters are collected by probing
    the type, not from a fixed list, so one a newer PyArrow adds is scanned too. Returns the scannable text by name,
    the types the parameters refer to, and ``False`` when a serialization exists that cannot be read as text.
    """
    text: dict[str, Any] = {"extension_name": dtype.extension_name}
    nested: list[pa.DataType] = []
    inspectable = True
    for name in dir(dtype):
        if name.startswith("_") or name in _TYPE_ATTRIBUTES:
            continue
        try:
            value = getattr(dtype, name)
        except Exception:  # an attribute that does not apply to this type (a width of a variable-size one)
            continue
        if callable(value):
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        nested.extend(v for v in values if isinstance(v, pa.DataType))
        strings = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for v in values]
        strings = [v for v in strings if isinstance(v, str)]
        if strings:
            text[name] = strings if isinstance(value, (list, tuple)) else strings[0]
    serialize = getattr(dtype, "__arrow_ext_serialize__", None)
    if serialize is not None:
        try:
            serialized = serialize()
            text["serialized"] = serialized.decode("utf-8") if serialized else ""
        except Exception:
            inspectable = False
    return text, nested, inspectable


def _stored_dictionaries(array: pa.Array | pa.ChunkedArray) -> Iterator[pa.Array]:
    """Every dictionary ``array`` stores, in all chunks and nested in lists, structs, maps and other dictionaries.

    A dictionary keeps the values that no row references after filtering or slicing, and the Parquet writer
    stores the whole dictionary, so reading the visible rows (``to_pylist``) does not cover everything a file holds.
    """
    for _, dictionary in _stored_dictionaries_at(array):
        yield dictionary


def _stored_dictionaries_at(
    array: pa.Array | pa.ChunkedArray, fields: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], pa.Array]]:
    """:func:`_stored_dictionaries` with each dictionary's struct field names, from the column down.

    The field names are the key context the credential rules need: an unused value stored for a ``password``
    field must be judged as that field's value, not as a value of the top-level column.
    """
    if isinstance(array, pa.ChunkedArray):
        for chunk in array.chunks:
            yield from _stored_dictionaries_at(chunk, fields)
        return
    dtype = array.type
    if pa.types.is_dictionary(dtype):
        yield fields, array.dictionary
        yield from _stored_dictionaries_at(array.dictionary, fields)
    elif pa.types.is_struct(dtype):
        for index in range(dtype.num_fields):
            yield from _stored_dictionaries_at(array.field(index), (*fields, dtype.field(index).name))
    elif pa.types.is_map(dtype):
        yield from _stored_dictionaries_at(array.keys, fields)
        # A dictionary of item values keeps no record of the key each value was stored for, so judge every stored
        # value as it would be for each key whose name carries a rule, as well as under the column alone.
        for key in _map_key_contexts(array.keys):
            yield from _stored_dictionaries_at(array.items, (*fields, key))
        yield from _stored_dictionaries_at(array.items, fields)
    elif (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
        or _is_list_view(dtype)
    ):
        yield from _stored_dictionaries_at(array.values, fields)
    elif isinstance(array, pa.ExtensionArray):
        yield from _stored_dictionaries_at(array.storage, fields)


def _map_key_contexts(keys: pa.Array | pa.ChunkedArray) -> list[str]:
    """Map key names, visible or only stored in a dictionary, that a field-name rule applies to."""
    names: dict[str, None] = {}
    candidates = [keys.to_pylist()]
    candidates.extend(dictionary.to_pylist() for dictionary in _stored_dictionaries(keys))
    for values in candidates:
        for value in values:
            name = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            if isinstance(name, str) and name not in names and _is_context_key(name):
                names[name] = None
    return list(names)


def _is_context_key(name: str) -> bool:
    return any(rule.search(name) for rule in (_CREDENTIAL_KEY, _IMAGE_KEY, _PATCH_KEY, _RUN_KEY))


def _file_dictionaries(path: Path, footer: pq.FileMetaData) -> list[tuple[str, tuple[str, ...], pa.Array]]:
    """Dictionaries stored in the file, as ``(top-level column, struct fields, dictionary)``, whatever its schema says.

    A reader returns a plain string column unless the embedded Arrow schema names a dictionary, yet the dictionary
    page itself keeps every value its writer put there. Reading the dictionary-encoded byte-array columns with
    ``read_dictionary`` returns those pages as written.
    """
    encoded: dict[str, None] = {}
    for group in range(footer.num_row_groups):
        for index in range(footer.num_columns):
            chunk = footer.row_group(group).column(index)
            if chunk.has_dictionary_page and chunk.physical_type == "BYTE_ARRAY":
                encoded[chunk.path_in_schema] = None
    if not encoded:
        return []
    try:
        table = pq.ParquetFile(path, read_dictionary=list(encoded)).read()
    except pa.ArrowException as exc:  # unreadable pages cannot be vouched for; callers report a ValueError
        raise ValueError(f"cannot inspect the stored dictionaries of {path.name}: {exc}") from exc
    return [
        (name, fields, dictionary)
        for name in table.column_names
        for fields, dictionary in _stored_dictionaries_at(table[name])
    ]


_TEXT_PHYSICAL_TYPES = ("BYTE_ARRAY", "FIXED_LEN_BYTE_ARRAY")


def _footer_statistics(footer: pq.FileMetaData) -> tuple[dict[str, dict[str, None]], list[str], list[str]]:
    """Text held by the footer's column-chunk statistics, by column path, plus what cannot be read as text.

    A column chunk's min and max are stored as raw bytes in the footer, whatever its visible cells hold, and
    promotion copies them verbatim. Returns the decoded values per column path, the paths whose statistics are not
    valid UTF-8 (refused, as undecodable cells are), and the text columns that carry a page index (per-page min and
    max values this reader cannot inspect). Attributes are probed so older PyArrow releases keep working.
    """
    texts: dict[str, dict[str, None]] = {}
    undecodable: dict[str, None] = {}
    indexed: dict[str, None] = {}
    for group in range(footer.num_row_groups):
        for index in range(footer.num_columns):
            chunk = footer.row_group(group).column(index)
            if chunk.physical_type not in _TEXT_PHYSICAL_TYPES:
                continue  # a number holds no text
            column = chunk.path_in_schema
            if getattr(chunk, "has_column_index", False):
                indexed[column] = None
            statistics = chunk.statistics if getattr(chunk, "is_stats_set", True) else None
            if statistics is None or not statistics.has_min_max:
                continue
            for bound in ("min", "max"):
                try:
                    value = getattr(statistics, f"{bound}_raw", None)  # the stored bytes, when this release offers them
                    value = getattr(statistics, bound) if value is None else value
                    text = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
                except (UnicodeDecodeError, ValueError, pa.ArrowException):
                    undecodable[column] = None
                    continue
                if isinstance(text, str):
                    texts.setdefault(column, {})[text] = None
    return texts, list(undecodable), list(indexed)


def _check_table_text(table: pa.Table, rel: str, policy: RedactionPolicy, where: str, *, path: Path) -> list[Finding]:
    """Scan metadata and text, including UTF-8 binary cells; undecodable bytes are invalid.

    Stored dictionary values are text too: they are scanned like the visible cells of their column.
    """
    footer = pq.ParquetFile(path).metadata
    out = scan_value(footer.created_by, policy, rel, "Parquet producer")
    # The footer's column-chunk ``file_path`` is readable text that the Arrow schema metadata does not carry.
    chunk_paths = {
        footer.row_group(group).column(column).file_path
        for group in range(footer.num_row_groups)
        for column in range(footer.num_columns)
    }
    out.extend(scan_value(sorted(p for p in chunk_paths if p), policy, rel, "Parquet column chunk file path"))
    metadata = [table.schema.metadata or {}]
    field_names: dict[str, None] = {}

    extensions: list[pa.DataType] = []

    def field_metadata(field: pa.Field) -> None:
        field_names[field.name] = None
        metadata.append(field.metadata or {})
        type_metadata(field.type)

    def type_metadata(dtype: pa.DataType) -> None:
        if pa.types.is_struct(dtype):
            for child in dtype:
                field_metadata(child)
        elif (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
            or _is_list_view(dtype)
        ):
            field_metadata(dtype.value_field)
        elif pa.types.is_map(dtype):
            field_metadata(dtype.key_field)
            field_metadata(dtype.item_field)
        elif pa.types.is_dictionary(dtype):
            type_metadata(dtype.value_type)
        elif _is_extension(dtype) and not any(dtype.equals(seen) for seen in extensions):
            # (the stored and the embedded schema repeat the same types, so each is inspected once)
            extensions.append(dtype)
            type_metadata(dtype.storage_type)

    for field in table.schema:
        field_metadata(field)
    # The reader merges the footer's key/value metadata over the Arrow schema it embeds (``ARROW:schema``), so a
    # redacted override can hide the original: scan the stored key/values and the decoded embedded schema as well.
    stored = footer.metadata or {}
    metadata.append({key: value for key, value in stored.items() if key != b"ARROW:schema"})
    if (embedded := stored.get(b"ARROW:schema")) is not None:
        schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(embedded)))
        metadata.append(schema.metadata or {})
        for field in schema:
            field_metadata(field)
    # Extension parameters (a built-in's name, vendor or dimension names) are type attributes, not field metadata.
    for extension in extensions:
        text, nested, inspectable = _extension_parameters(extension)
        for dtype in nested:
            type_metadata(dtype)
        out.extend(scan_value(text, policy, rel, "Arrow extension type"))
        if not inspectable:
            out.append(
                Finding(
                    "extension-type",
                    f"extension type {extension.extension_name} has parameters that cannot be inspected",
                    rel,
                )
            )
    # Field names are disclosure text even when the table has no rows or every cell is null.
    out.extend(scan_value(field_names, policy, rel, f"{where} name"))
    for entry in metadata:
        decoded = {key.decode("utf-8"): value.decode("utf-8") for key, value in entry.items()}
        out.extend(scan_value(decoded, policy, rel, "Parquet metadata"))

    def disclosure_value(value, dtype):
        if value is None:
            return None
        if pa.types.is_map(dtype):
            # Keep duplicate map entries and their disclosure keys; converting
            # straight to dict would discard all but the last repeated key.
            return [{key: disclosure_value(item, dtype.item_type)} for key, item in value]
        if pa.types.is_struct(dtype):
            return {field.name: disclosure_value(value[field.name], field.type) for field in dtype}
        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
            return [disclosure_value(item, dtype.value_type) for item in value]
        return value

    # Dictionaries kept by the table itself and by the stored file, by top-level column.
    stored: dict[str, list[tuple[tuple[str, ...], pa.Array]]] = {}
    for name in table.column_names:
        stored.setdefault(name, []).extend(_stored_dictionaries_at(table[name]))
    for name, fields, dictionary in _file_dictionaries(path, footer):
        stored.setdefault(name, []).append((fields, dictionary))
    for name in table.column_names:
        column = table.column(name)
        values = [disclosure_value(value, column.type) for value in column.to_pylist()]
        found = scan_value({name: values}, policy, rel, f"{where} {name}")[:1] if values else []
        if not found:
            # Not a visible cell: a value left in a dictionary that filtering or slicing stopped referencing.
            for fields, dictionary in stored.get(name, ()):
                dictionary_values = [disclosure_value(value, dictionary.type) for value in dictionary.to_pylist()]
                if not dictionary_values:
                    continue
                # Keep the struct field names as key context, so a stored ``password`` value is judged as one.
                nested: Any = dictionary_values
                for field in reversed(fields):
                    nested = {field: nested}
                found = scan_value({name: nested}, policy, rel, f"{where} {name} (stored dictionary)")[:1]
                if found:
                    break
        out.extend(found)
    statistics, undecodable, indexed = _footer_statistics(footer)
    for column, values in statistics.items():
        # The column path's names are the key context, as for the visible cells of that column.
        nested: Any = list(values)
        for part in reversed(column.split(".")):
            nested = {part: nested}
        out.extend(scan_value(nested, policy, rel, f"{where} {column} (Parquet statistics)")[:1])
    for column in undecodable:
        out.append(Finding("parquet-statistics", f"column {column} has statistics that are not valid UTF-8 text", rel))
    for column in indexed:
        out.append(
            Finding(
                "parquet-page-index", f"column {column} has a page index whose min/max values cannot be inspected", rel
            )
        )
    return out


def _check_snapshot(root: Path, rel: str, policy: RedactionPolicy, payload: set[str]) -> list[Finding]:
    bounded_json_bytes(checked_path(root, rel))
    snapshot, _ = read_snapshot(root, rel, payload=payload)
    out = scan_value(snapshot.model_dump(mode="json", by_alias=True), policy, rel, "snapshot metadata")
    for artifact in (snapshot.members, snapshot.acquisition):
        if artifact is not None:
            path = _verify(root, artifact)
            out.extend(
                _check_table_text(
                    read_parquet(path, artifact.artifact_schema), artifact.path, policy, "snapshot column", path=path
                )
            )
    return out


def _check_eval_record(
    root: Path,
    rel: str,
    policy: RedactionPolicy,
    infx_ids: set[str],
    classified_ids: set[str],
    snapshot_configurations: Callable[[str], set[str] | None],
) -> list[Finding]:
    record = load_record(root / rel)
    out = scan_value(record.model_dump(mode="json", by_alias=True), policy, rel, "eval record")
    if missing := set(record.truth.configuration_ids) - classified_ids:
        out.append(
            Finding(
                "eval-configuration-unresolved", f"cannot classify evaluated configurations: {sorted(missing)}", rel
            )
        )
    # A named input snapshot is truth provenance too: classify its configurations as well as the listed ones.
    snapshot_id = record.truth.snapshot_id
    members = None if snapshot_id is None else snapshot_configurations(snapshot_id)
    if snapshot_id is not None:
        if members is None:
            out.append(Finding("eval-configuration-unresolved", f"cannot resolve snapshot {snapshot_id}", rel))
        elif unclassified := members - classified_ids:
            out.append(
                Finding(
                    "eval-configuration-unresolved",
                    f"cannot classify snapshot configurations: {sorted(unclassified)}",
                    rel,
                )
            )
    for artifact in record.truth.artifacts:
        path = _verify((root / rel).parent, artifact)
        artifact_rel = path.relative_to(root).as_posix()
        if artifact.format == "gzip" and not policy.allow_provenance_artifacts:
            out.append(Finding("provenance-withheld", "logs and raw output are withheld", artifact.path))
        if artifact.format == "parquet":
            out.extend(
                _check_table_text(
                    read_parquet(path, artifact.artifact_schema), artifact_rel, policy, "eval truth column", path=path
                )
            )
        elif artifact.format == "json":
            out.extend(scan_json_file(path, policy, artifact_rel, "eval truth JSON"))
        elif artifact.format != "gzip":
            out.append(Finding("artifact-format", f"unsupported eval truth format: {artifact.format}", artifact_rel))
    if record.findings and not policy.allow_provenance_artifacts:
        out.append(Finding("provenance-withheld", "linked finding artifacts may be logs", rel))
    for artifact in record.findings:
        path = _verify((root / rel).parent, artifact)
        if policy.allow_provenance_artifacts and artifact.format == "json":
            out.extend(scan_json_file(path, policy, path.relative_to(root).as_posix(), "eval findings JSON"))
    if policy.inferencex_disclosure == "aggregates-only" and infx_ids & (
        set(record.truth.configuration_ids) | (members or set())
    ):
        out.append(Finding("infx-aggregates-only", "record scores InferenceX-derived truth; publish aggregates", rel))
    return out


def check_publishable(
    root: str | Path,
    paths: Iterable[str],
    *,
    manifests: Mapping[str, Manifest] | None = None,
    policy: RedactionPolicy = DEFAULT_POLICY,
) -> list[Finding]:
    """Content-check ``paths`` (repository-relative) for publication. Empty list means publishable.

    ``manifests`` maps manifest path to a manifest already loaded by the caller; manifests
    among ``paths`` that are missing from it are read here.
    """
    root = Path(root)
    chosen = sorted(set(paths))
    loaded = dict(manifests or {})
    findings: list[Finding] = []
    from aisim_core.data.mirror import DATA_ROOTS, list_tree, snapshot_documents

    tree = list_tree(root)
    payload_paths = {f.path for f in tree if f.path.startswith(DATA_ROOTS)}
    for f in tree:
        if f.is_symlink:
            findings.append(Finding("symlink", "symbolic links are not published", f.path))
    try:
        for rel in chosen:
            checked_path(root, rel)
    except ValueError as exc:
        findings.append(Finding("path-unsafe", str(exc)))
    if findings:
        return findings
    eval_records = {p.relative_to(root).as_posix() for p in record_paths(root / "eval", recursive=True)}
    snapshot_members: dict[str, set[str]] | None = None

    def snapshot_configurations(snapshot_id: str) -> set[str] | None:
        """Configurations of the tree's snapshot with this ID (read once, on first use); None when absent."""
        nonlocal snapshot_members
        if snapshot_members is None:
            snapshot_members = {}
            for document in sorted(snapshot_documents(root, (f.path for f in tree))):
                try:
                    snapshot, members = read_snapshot(root, document)
                except (OSError, ValidationError, ValueError, RecursionError):
                    continue  # unreadable documents resolve nothing; the validators report them
                snapshot_members.setdefault(snapshot.snapshot_id, set()).update(m.configuration_id for m in members)
        return snapshot_members.get(snapshot_id)

    for rel in chosen:
        if rel.endswith("/manifest.json") and rel not in loaded:
            try:
                loaded[rel] = Manifest.model_validate_json((root / rel).read_bytes())
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
    infx_ids = set()
    invalid = set()
    for rel, manifest in loaded.items():
        try:
            for artifact in manifest.artifacts:
                checked_path(root, artifact.path)
            if manifest.category == "e2e":
                configuration = load_e2e(root, rel)
                if _is_infx(configuration.body.source_kind) or any(
                    _is_infx(observation.source_kind) for observation in configuration.observations
                ):
                    infx_ids.add(manifest.configuration_id)
        except (ValueError, OSError) as exc:
            findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
            invalid.add(rel)
    chosen_set = set(chosen)
    index_rel = "catalog/dataset.json"
    if (root / index_rel).is_file():
        try:
            dataset = DatasetIndex.model_validate_json(bounded_json_bytes(checked_path(root, index_rel)))
            indexes = {index_rel: dataset}
            for ref in dataset.categories.values():
                if ref.index.path not in payload_paths:
                    raise ValueError("category index must be a listed dataset payload file")
                checked_path(root, ref.index.path)
                indexes[ref.index.path] = CategoryIndex.model_validate_json(
                    bounded_json_bytes(_verify(root, ref.index))
                )
            for rel, index in indexes.items():
                if rel in chosen_set:
                    findings.extend(scan_value(index.model_dump(mode="json", by_alias=True), policy, rel, "index"))
                for artifact in getattr(index, "support", ()):  # lookup tables no manifest declares
                    if artifact.path in chosen_set:
                        findings.extend(_check_support(root, artifact, policy))
        except (OSError, ValueError, RecursionError) as exc:
            findings.append(Finding("document-invalid", str(exc).splitlines()[0], index_rel))
    catalog_shards = {p.relative_to(root).as_posix() for p in catalog_files(root)}
    snapshots = snapshot_documents(root, chosen)
    for rel, manifest in sorted(loaded.items()):
        if (rel in chosen_set or any(a.path in chosen_set for a in manifest.artifacts)) and rel not in invalid:
            try:
                findings.extend(_check_manifest(root, rel, manifest, policy, infx_ids))
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
    for rel in chosen:
        if rel == f"{CATALOG_DIR}/{COVERAGE_FILE}":
            try:
                findings.extend(scan_value(read_catalog_coverage(root).model_dump(), policy, rel, "catalog coverage"))
            except (OSError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        if rel in catalog_shards:
            try:
                parquet = pq.ParquetFile(checked_path(root, rel))
                findings.extend(
                    _check_table_text(parquet.read(), rel, policy, "catalog column", path=checked_path(root, rel))
                )
            except (OSError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        if rel.startswith("systems/") and rel.endswith(".yaml"):
            try:
                data = checked_path(root, rel).read_bytes()
                findings.extend(scan_yaml(data, policy, rel, "system spec"))
            except (OSError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        # Match the snapshot documents declared by mirror._check_snapshots.
        if rel in snapshots:
            try:
                findings.extend(_check_snapshot(root, rel, policy, payload_paths))
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
        if rel in eval_records:
            try:
                findings.extend(
                    _check_eval_record(
                        root,
                        rel,
                        policy,
                        infx_ids,
                        {m.configuration_id for m in loaded.values()},
                        snapshot_configurations,
                    )
                )
            except (OSError, ValidationError, ValueError, RecursionError) as exc:
                findings.append(Finding("document-invalid", str(exc).splitlines()[0], rel))
    return findings


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_disclosure(rows: Iterable[Mapping[str, Any]], policy: RedactionPolicy = DEFAULT_POLICY) -> list[Finding]:
    """Check aggregate rows for a public page or catalog (InferenceX-derived or not).

    Only :data:`AGGREGATE_FIELDS` may appear; each row must cover at least ``policy.min_scored``
    scored points; no value may carry a URL, image, patch reference or credential.
    """
    out: list[Finding] = []
    for i, row in enumerate(rows):
        where = f"row {i}"
        for key in sorted(set(row) - AGGREGATE_FIELDS):
            out.append(Finding("not-aggregate", f"field {key!r} is not an allowed aggregate field", where))
        scored = row.get("n_scored")
        if type(scored) is not int or scored < policy.min_scored:
            out.append(Finding("small-cell", f"n_scored must be an integer >= {policy.min_scored}", where))
        total = row.get("n_total")
        if type(scored) is int and type(total) is int and scored >= 0 and total >= 0 and scored > total:
            out.append(Finding("not-aggregate", "n_scored exceeds n_total", where))
        for key, value in row.items():
            if key not in AGGREGATE_FIELDS:
                continue
            if key in {"n_total", "n_scored", "n_configs"}:
                valid = type(value) is int and value >= 0
            elif key in {"coverage", "mape_pct"}:
                valid = value is None or (_finite_number(value) and value >= 0)
                if key == "coverage" and valid:
                    valid = value is None or value <= 1
            elif key == "residual_quantiles_pct":
                valid = value is None or (
                    isinstance(value, Mapping)
                    and set(value) <= {"p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100"}
                    and all(_finite_number(v) for v in value.values())
                )
            else:
                # Identifiers and status labels are text, never a channel for
                # serialized per-point measurements (including malformed containers).
                valid = isinstance(value, str) and not value.lstrip().startswith(("[", "{"))
            if not valid:
                out.append(Finding("not-aggregate", f"invalid aggregate shape for {key!r}", where))
        for key, value in row.items():
            if key in AGGREGATE_FIELDS and isinstance(value, str) and _URL.search(value):
                out.append(Finding("not-aggregate", f"aggregate text field {key!r} must not contain a URL", where))
        out.extend(scan_value({k: v for k, v in row.items() if k in AGGREGATE_FIELDS}, policy, where, where))
    return out
