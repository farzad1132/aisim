# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One dataset-tree validator for the public dataset and its private mirror (DEP #435, Phase 7).

``ai-dynamo/aisim-data-internal`` has the same layout and schemas as the public
dataset, so its CI runs exactly this function, from a pinned checkout of this
repository. There is no separate "internal" rule set: a tree the public CI would
reject fails the mirror CI for the same reason.

Two profiles exist and differ only in what the *public* repository additionally
forbids:

* ``mirror``  structure, identity, integrity, schema and catalog checks. Sensitive
  content (unredacted logs, run URLs, image digests) is allowed.
* ``public``  everything in ``mirror`` plus the publication rules in
  :mod:`aisim_core.data.redaction`, and no ``mirror/`` control directory.

Both profiles separate the installed ``.github/workflows/validate.yml`` from
dataset payloads. This exact repository infrastructure path is reviewed as code,
never loaded as data, and still must not be a symlink. Root ``README.md`` (the
dataset card) and ``.gitattributes`` are also repository metadata, rather than
manifest artifacts. They must be UTF-8 text of at most 4 MiB, pass content-signature
checks and, in the public profile, pass disclosure scanning. Candidate hygiene
does not permit contributors to change these repository files.

``aisim-infx-cache`` is a separate private cache that is never promoted
(:data:`NEVER_PROMOTED`). It uses the same E2E layout, so it can be validated with
``repo=INFX_CACHE_REPO``.

Nothing here touches the network. The report carries a validator fingerprint (the
hash of every generated schema snapshot plus :data:`VALIDATOR_VERSION`), so a CI
log shows which rule set judged the tree.

CLI: ``python -m aisim_core.data.mirror validate --root DIR [--repo ID]
[--profile mirror|public]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from pydantic import ValidationError

from aisim_core.data import CategoryIndex, DatasetIndex, Manifest, _verify, check_manifest, configuration_id
from aisim_core.data.catalog import CATALOG_DIR, COVERAGE_FILE, _write, build_catalog, catalog_files, known_system_specs
from aisim_core.data.e2e import SNAPSHOT_SCHEMA, load_e2e, read_snapshot, verify_snapshot_members
from aisim_core.data.eval import TIER_RANK, load_record, record_dir, record_paths
from aisim_core.data.evidence import EVIDENCE_ROOT, check_evidence
from aisim_core.data.hygiene import (
    CandidateFile,
    CandidatePolicy,
    Finding,
    bounded_json_bytes,
    check_content,
    check_listing,
    checked_path,
)
from aisim_core.data.loaders import LoaderError
from aisim_core.data.schemas import check_snapshots, render_all

VALIDATOR_VERSION = "9"
INTERNAL_REPO = "ai-dynamo/aisim-data-internal"
INFX_CACHE_REPO = "ai-dynamo/aisim-infx-cache"
# Never a promotion source: private caches whose terms forbid public export.
NEVER_PROMOTED = frozenset({INFX_CACHE_REPO})
MIRROR_CONTROL_DIR = "mirror"  # private-only control files; absent from the public layout
DATA_ROOTS = ("catalog/", "eval/", "data/", "systems/")
ROOT_METADATA_FILES = frozenset({"README.md", ".gitattributes"})
REPOSITORY_FILES = ROOT_METADATA_FILES | {".github/workflows/validate.yml"}
MAX_METADATA_BYTES = 4 * 1024 * 1024
OP_YAML_SIDECARS = {
    "collection_meta.yaml": ("collection-meta", "aisim-collection-meta/v"),
    "reuse.yaml": ("reuse-declaration", "aisim-reuse-declaration/v"),
}
Profile = Literal["mirror", "public"]


def validator_fingerprint() -> str:
    """Hash of the rule set: validator version plus every generated schema snapshot."""
    digest = hashlib.sha256(VALIDATOR_VERSION.encode())
    for name, text in sorted(render_all().items()):
        digest.update(name.encode())
        digest.update(text.encode())
    return digest.hexdigest()


@dataclass
class TreeReport:
    repo: str | None
    profile: str
    fingerprint: str = field(default_factory=validator_fingerprint)
    files: int = 0
    configurations: int = 0
    eval_records: int = 0
    findings: list[Finding] = field(default_factory=list)
    manifests: dict[str, Manifest] = field(default_factory=dict, repr=False)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, code: str, message: str, path: str | None = None, severity: str = "error") -> None:
        self.findings.append(Finding(code, message, path, severity))  # type: ignore[arg-type]

    def to_json(self) -> dict:
        return {
            "repo": self.repo,
            "profile": self.profile,
            "validator_fingerprint": self.fingerprint,
            "ok": self.ok,
            "files": self.files,
            "configurations": self.configurations,
            "eval_records": self.eval_records,
            "findings": [vars(f) for f in self.findings],
        }


def _tree_policy(extra_roots: tuple[str, ...]) -> CandidatePolicy:
    # Whole-tree limits are the host's business; candidate size limits stay in hygiene.
    return CandidatePolicy(
        max_files=10**9,
        max_total_bytes=1 << 62,
        max_file_bytes=1 << 40,
        allowed_roots=DATA_ROOTS + extra_roots,
        allowed_suffixes=frozenset({".parquet", ".json", ".gz", ".yaml"}),
    )


_CARD_OPENING = re.compile(r"\s*---[ \t]*(?:\r\n|\n|\r|\Z)")
# The Hub's closing delimiter: an unindented ``---`` alone on its line (an indented one is YAML block-scalar text).
_CARD_CLOSING = re.compile(r"(?:^|(?<=[\r\n]))---(?=\r\n|\n|\Z)")


def _card_front_matter(text: str) -> str | None:
    """The YAML front matter of a dataset card, with the Hub parser's delimiter rules; None without any.

    Leading whitespace may precede the opening ``---``. Raises ``ValueError`` for an opening delimiter with no
    closing one, so a card never hides metadata from the structured scan.
    """
    opening = _CARD_OPENING.match(text)
    if opening is None:
        return None
    rest = text[opening.end() :]
    closing = _CARD_CLOSING.search(rest)
    if closing is None:
        raise ValueError("dataset card front matter has no closing delimiter")
    return rest[: closing.start()]


def _check_root_metadata(root: Path, tree: list[CandidateFile], profile: Profile, report: TreeReport) -> None:
    for file in tree:
        if file.path not in ROOT_METADATA_FILES:
            continue
        if file.size > MAX_METADATA_BYTES:
            report.add("metadata-too-large", f"repository metadata exceeds {MAX_METADATA_BYTES} bytes", file.path)
            continue
        try:
            data = (root / file.path).read_bytes()
        except OSError as exc:
            report.add("metadata-unreadable", str(exc).splitlines()[0], file.path)
            continue
        report.findings.extend(check_content(_tree_policy(()), file, data))
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            report.add("metadata-invalid", "repository metadata must be UTF-8 text", file.path)
            continue
        if profile == "public":
            from aisim_core.data.redaction import DEFAULT_POLICY, scan_value, scan_yaml

            report.findings.extend(scan_value(text, DEFAULT_POLICY, file.path, "repository metadata"))
            try:
                front_matter = _card_front_matter(text) if file.path == "README.md" else None
                if front_matter is not None:
                    report.findings.extend(
                        scan_yaml(front_matter.encode(), DEFAULT_POLICY, file.path, "dataset card front matter")
                    )
            except ValueError as exc:
                report.add("metadata-invalid", str(exc).splitlines()[0], file.path)


def list_tree(root: Path) -> list[CandidateFile]:
    """Every file under ``root`` (symlinks are listed, never followed), skipping ``.git``."""
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel == ".git" or rel.startswith(".git/") or (path.is_dir() and not path.is_symlink()):
            continue
        out.append(CandidateFile(rel, 0 if path.is_symlink() else path.stat().st_size, is_symlink=path.is_symlink()))
    return out


def _read_json(root: Path, rel: str, model, report: TreeReport):
    try:
        return model.model_validate_json(bounded_json_bytes(root / rel))
    except (OSError, ValidationError, ValueError) as exc:
        report.add("document-invalid", f"{model.__name__}: {str(exc).splitlines()[0]}", rel)
        return None


def _check_manifest_any(root: Path, rel: str, category: str):
    """Category-specific loader: the same code the evaluators use."""
    if category == "e2e":
        return load_e2e(root, rel).manifest
    if category in ("op-latency", "fpm"):
        return check_manifest(root, rel)
    if category == "host-cost":
        from aisim_core.data.host_cost import check_entry

        return check_entry(root, rel)
    manifest = Manifest.model_validate_json((root / rel).read_bytes())
    if manifest.configuration_id != configuration_id(manifest.category, manifest.purpose, manifest.identity):
        raise ValueError("configuration identity hash mismatch")
    for artifact in manifest.artifacts:
        _verify(root, artifact)
    return manifest


def _check_indexes(root: Path, report: TreeReport, declared: set[str], eligible: set[str]) -> dict[str, Manifest]:
    index_rel = f"{CATALOG_DIR}/dataset.json"
    if not (root / index_rel).is_file():
        report.add("dataset-index-missing", "catalog/dataset.json is missing", index_rel)
        return {}
    dataset = _read_json(root, index_rel, DatasetIndex, report)
    if dataset is None:
        return {}
    if report.repo is not None and dataset.dataset_id != report.repo:
        report.add("dataset-id-mismatch", f"dataset_id {dataset.dataset_id!r} is not {report.repo!r}", index_rel)
    declared.add(index_rel)
    manifests: dict[str, Manifest] = {}
    for category, ref in dataset.categories.items():
        try:
            if ref.index.path not in eligible:
                raise ValueError("category index must be a listed dataset payload file")
            _verify(root, ref.index)
        except (ValueError, OSError) as exc:
            report.add("index-integrity", str(exc), ref.index.path)
            continue
        declared.add(ref.index.path)
        entries = _read_json(root, ref.index.path, CategoryIndex, report)
        if entries is None:
            continue
        if entries.category != category:
            report.add("index-category", "category index identity mismatch", ref.index.path)
        for artifact in entries.support:  # files every selection of the category needs, pinned by this index
            try:
                if artifact.path not in eligible:
                    raise ValueError(f"support artifact {artifact.path} must be a listed dataset payload file")
                _verify(root, artifact)
            except (ValueError, OSError) as exc:
                report.add("index-integrity", str(exc).splitlines()[0], artifact.path)
                continue
            declared.add(artifact.path)
        seen: set[str] = set()
        for entry in entries.entries:
            path = entry.manifest.path
            if entry.configuration_id in seen:
                report.add("duplicate-configuration", entry.configuration_id, path)
            seen.add(entry.configuration_id)
            try:
                if path not in eligible:
                    raise ValueError("manifest must be a listed dataset payload file")
                _verify(root, entry.manifest)
                manifest = _check_manifest_any(root, path, category)
                if (manifest.category, manifest.configuration_id, manifest.identity, manifest.purpose) != (
                    category,
                    entry.configuration_id,
                    entry.identity,
                    entry.purpose,
                ):
                    raise ValueError("catalog and manifest identity mismatch")
                for artifact in manifest.artifacts:
                    if artifact.path not in eligible:
                        raise ValueError(f"artifact {artifact.path} must be a listed dataset payload file")
            except (ValueError, LoaderError, ValidationError, OSError) as exc:
                report.add("manifest-invalid", str(exc).splitlines()[0], path)
                continue
            manifests[path] = manifest
            report.configurations += 1
            declared.add(path)
            declared.update(a.path for a in manifest.artifacts)
    return manifests


def _check_eval(root: Path, report: TreeReport, declared: set[str]) -> None:
    eval_root = root / "eval"
    if not eval_root.is_dir():
        return
    verified = {}
    for path in record_paths(eval_root, recursive=True):
        rel = path.relative_to(root).as_posix()
        try:
            record = load_record(path)
            expected = record_dir(root, record.truth.identity)
            if path.parent != expected:
                raise ValueError(f"record is not at its identity path {expected.relative_to(root).as_posix()}")
            for artifact in record.truth.artifacts:
                _verify(path.parent, artifact)
                declared.add((path.parent / artifact.path).relative_to(root).as_posix())
            for artifact in record.findings:
                _verify(path.parent, artifact)
                declared.add((path.parent / artifact.path).relative_to(root).as_posix())
        except (ValueError, ValidationError, OSError) as exc:
            report.add("eval-record-invalid", str(exc).splitlines()[0], rel)
            continue
        verified[(path.parent, record.record_id)] = (rel, record)
        declared.add(rel)
        report.eval_records += 1
    successors: dict[tuple[Path, str], list[tuple[str, str]]] = {}
    for (directory, _), (rel, record) in verified.items():
        if record.supersedes is not None and (directory, record.supersedes) not in verified:
            report.add("eval-predecessor-missing", "superseded record must be verified in the same directory", rel)
        elif record.supersedes is not None:
            # The successor hides its predecessor from the current view, so it must not rank below it (attest
            # enforces the same order). A same-tier retry and an upgrade stay valid.
            _, predecessor = verified[(directory, record.supersedes)]
            if TIER_RANK[record.tier] < TIER_RANK[predecessor.tier]:
                report.add(
                    "eval-tier-downgrade",
                    f"a {record.tier} record cannot supersede the {predecessor.tier} record {predecessor.record_id}",
                    rel,
                )
        if record.supersedes is not None:
            successors.setdefault((directory, record.supersedes), []).append((rel, record.record_id))
    # Two branches can each publish a valid successor of one record and then merge: both files stay, because their
    # names differ, and the catalog would count the same evaluation twice. A linear history has one per record.
    for (_, predecessor), found in successors.items():
        if len(found) > 1:
            ids = ", ".join(sorted(record_id for _, record_id in found))
            for rel, _ in sorted(found):
                report.add("eval-competing-successors", f"record {predecessor} has competing successors: {ids}", rel)


def snapshot_documents(root: Path, paths: Iterable[str]) -> set[str]:
    """Find snapshot schemas, preserving custom filenames without misclassifying JSON sidecars."""
    found = set()
    for rel in paths:
        if not rel.startswith("data/") or Path(rel).suffix.lower() != ".json":
            continue
        if Path(rel).name.lower() == "snapshot.json":
            found.add(rel)  # a malformed document at the conventional path must fail validation too
            continue
        try:
            limit = CandidatePolicy().max_json_bytes
            with checked_path(root, rel).open("rb") as handle:
                data = handle.read(limit + 1)
            if len(data) > limit:
                continue  # leave oversized sidecars to the bounded content check
            document = json.loads(data)
        except (OSError, ValueError, RecursionError):
            continue  # the content and integrity checks report unreadable or invalid JSON
        if isinstance(document, dict) and document.get("schema_version") == SNAPSHOT_SCHEMA:
            found.add(rel)
    return found


def _check_snapshots(
    root: Path, report: TreeReport, declared: set[str], manifests: dict[str, Manifest], eligible: set[str]
) -> None:
    candidates = (p.relative_to(root).as_posix() for p in (root / "data").rglob("*") if p.suffix.lower() == ".json")
    for rel in sorted(snapshot_documents(root, candidates)):
        try:
            bounded_json_bytes(checked_path(root, rel))
            # A fresh checkout only has the dataset payload: tables outside it (e.g. under .git) cannot be loaded.
            snapshot, members = read_snapshot(root, rel, payload=eligible)
            for member in members:
                checked_path(root, member.manifest_path)
                manifest = manifests.get(member.manifest_path)
                if manifest is None or manifest.configuration_id != member.configuration_id:
                    raise ValueError(f"snapshot member is not a verified configuration: {member.manifest_path}")
            verify_snapshot_members(root, members)
        except (ValueError, ValidationError, OSError) as exc:
            report.add("snapshot-invalid", str(exc).splitlines()[0], rel)
            continue
        declared.update({rel, snapshot.members.path} | ({snapshot.acquisition.path} if snapshot.acquisition else set()))


def _check_schema_snapshots(root: Path, report: TreeReport, declared: set[str]) -> None:
    directory = root / CATALOG_DIR / "schemas"
    if not directory.is_dir():
        return
    try:
        for problem in check_snapshots(directory):
            report.add("schema-drift", problem, f"{CATALOG_DIR}/schemas")
    except (OSError, UnicodeError) as exc:
        report.add("schema-unreadable", str(exc).splitlines()[0], f"{CATALOG_DIR}/schemas")
    declared.update(p.relative_to(root).as_posix() for p in directory.glob("*.json"))


def _check_catalog_drift(root: Path, report: TreeReport, declared: set[str]) -> None:
    coverage_rel = f"{CATALOG_DIR}/{COVERAGE_FILE}"
    if (root / coverage_rel).is_file():
        declared.add(coverage_rel)
    existing = catalog_files(root)
    declared.update(p.relative_to(root).as_posix() for p in existing)
    if not existing:
        report.add("catalog-missing", "no catalog shards; regenerate with build_catalog", f"{CATALOG_DIR}/")
        return
    try:
        rows = build_catalog(root, write=False)
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp)
            _write(fresh, rows, known_system_specs())
            want = {p.relative_to(fresh).as_posix(): pq.read_table(p) for p in catalog_files(fresh)}
    except (ValueError, ValidationError, OSError) as exc:
        report.add("catalog-unbuildable", str(exc).splitlines()[0], f"{CATALOG_DIR}/")
        return
    have = {}
    for path in existing:
        name = path.relative_to(root).as_posix()
        try:
            have[name] = pq.read_table(path)
        except ValueError as exc:
            report.add("catalog-drift", f"invalid catalog shard: {str(exc).splitlines()[0]}", name)
        except (OSError, NotImplementedError) as exc:  # NotImplementedError: unsupported Parquet codec
            report.add("catalog-unreadable", str(exc).splitlines()[0], name)
    existing_names = {p.relative_to(root).as_posix() for p in existing}
    for name in sorted(set(want) | existing_names):
        if name not in existing_names:
            report.add("catalog-drift", "catalog shard is missing; regenerate the catalog", name)
        elif name not in want:
            report.add("catalog-drift", "stale catalog shard; regenerate the catalog", name)
        elif name in have and not want[name].equals(have[name], check_metadata=True):
            report.add("catalog-drift", "catalog shard differs from the data; regenerate the catalog", name)


def validate_tree(
    root: str | Path,
    *,
    repo: str | None = None,
    profile: Profile = "mirror",
    allow_empty: bool = False,
) -> TreeReport:
    """Validate a dataset checkout. ``repo`` (when given) must equal ``dataset.json``'s ``dataset_id``.

    ``allow_empty`` accepts a tree with no dataset payload files, for a freshly created mirror.
    """
    if profile not in ("mirror", "public"):
        raise ValueError(f"unsupported validation profile: {profile!r}")
    root = Path(root)
    report = TreeReport(repo=repo, profile=profile)
    if not root.is_dir():
        report.add("root-missing", f"{root} is not a directory")
        return report
    tree = list_tree(root)
    # Reject the complete unsafe tree before indexes, references or redaction can read it.
    for f in tree:
        if f.is_symlink:
            report.add("symlink", "symbolic links are not allowed", f.path)
    if root.is_symlink():
        report.add("symlink", "dataset root must not be a symbolic link")
    if report.errors:
        return report
    _check_root_metadata(root, tree, profile, report)
    for f in tree:
        if f.path in REPOSITORY_FILES and f.is_symlink:
            report.add("symlink", "symbolic links are not allowed", f.path)
    listing = [
        f
        for f in tree
        if f.path not in REPOSITORY_FILES and not (profile == "mirror" and f.path.startswith(f"{MIRROR_CONTROL_DIR}/"))
    ]
    evidence: list[CandidateFile] = []
    if profile == "mirror":  # internal-only root: hash-checked blobs, never loaded; the public profile rejects it
        evidence = [f for f in listing if f.path.startswith(EVIDENCE_ROOT)]
        listing = [f for f in listing if not f.path.startswith(EVIDENCE_ROOT)]
        for f in evidence:
            if f.is_symlink:
                report.add("symlink", "symbolic links are not allowed", f.path)
    if profile == "mirror":
        control = [f for f in tree if f.path.startswith(f"{MIRROR_CONTROL_DIR}/")]
        for f in control:
            if f.is_symlink or not f.path.endswith(".json"):
                report.add("control-file", "mirror/ may hold plain .json files only", f.path)
        from aisim_core.data.redaction import LEDGER_PATH, load_ledger

        try:
            load_ledger(root)
        except (OSError, ValueError, RecursionError) as exc:
            report.add("document-invalid", f"VisibilityLedger: {str(exc).splitlines()[0]}", LEDGER_PATH)
    report.files = len(listing) + len(evidence)
    if not listing and not evidence and allow_empty:
        return report
    # Preflight before index/system loaders and catalog rebuilding consume structured data.
    # Evidence blobs are exempt, but their manifests use bounded_json_bytes.
    oversized = [
        file
        for file in listing
        if Path(file.path).suffix.lower() in (".json", ".yaml") and file.size > CandidatePolicy().max_json_bytes
    ]
    for file in oversized:
        kind = Path(file.path).suffix.lower()[1:]
        report.add(f"{kind}-too-large", f"{kind.upper()} exceeds the structured-content size limit", file.path)
    if oversized:
        return report

    declared: set[str] = set()
    eligible = {f.path for f in listing if not check_listing(_tree_policy(()), [f])}
    manifests = _check_indexes(root, report, declared, eligible)
    report.manifests = manifests
    system_files = _check_system_bundles(root, report)
    declared.update(system_files)
    yaml_sidecars = {
        artifact.path
        for manifest in manifests.values()
        if manifest.category == "op-latency"
        for artifact in manifest.artifacts
        for name, (role, schema_prefix) in OP_YAML_SIDECARS.items()
        if Path(artifact.path).name == name
        and artifact.format == "yaml"
        and artifact.role == role
        and artifact.artifact_schema.startswith(schema_prefix)
    }
    for f in listing:
        if Path(f.path).suffix.lower() == ".yaml" and f.path not in yaml_sidecars | system_files:
            report.add("yaml-sidecar", "YAML must be a declared operation collection/reuse sidecar", f.path)
    _check_eval(root, report, declared)
    _check_snapshots(root, report, declared, manifests, eligible)
    _check_schema_snapshots(root, report, declared)
    if not any(f.code == "index-integrity" for f in report.errors):
        _check_catalog_drift(root, report, declared)

    evidence_findings, _ = check_evidence(root, [f.path for f in evidence if not f.is_symlink], manifests=manifests)
    report.findings.extend(evidence_findings)

    gz_declared = {p for p in declared if p.lower().endswith(".gz")}
    for finding in check_listing(_tree_policy(()), listing):
        report.findings.append(finding)
    skip = report_paths_with_errors(report, "path-", "symlink")
    for f in listing:
        if f.is_symlink or f.path in skip:
            continue
        try:
            data = (root / f.path).read_bytes() if f.size <= 4 * 1024 * 1024 else None
            head = data if data is not None else (root / f.path).open("rb").read(16)
            tail = data[-4:] if data is not None else b""
            if data is None and f.path.lower().endswith(".parquet"):
                with (root / f.path).open("rb") as handle:
                    handle.seek(-4, 2)
                    tail = handle.read(4)
        except OSError as exc:
            report.add("content-unreadable", str(exc).splitlines()[0], f.path)
            continue
        report.findings.extend(check_content(_tree_policy(()), f, head, tail))
    for f in listing:
        if f.path.lower().endswith(".gz") and f.path not in gz_declared:
            report.add("undeclared-archive", "gzip file is not a declared provenance artifact", f.path)
        elif f.path not in declared and not f.path.lower().endswith(".gz"):
            # Unreferenced files escape the content checks, so the public profile rejects them.
            severity = "error" if profile == "public" else "warning"
            report.add("orphan-file", "file is not referenced by any manifest, index or record", f.path, severity)

    if profile == "public":
        from aisim_core.data.redaction import check_publishable

        for f in listing:
            if f.path.startswith(f"{MIRROR_CONTROL_DIR}/"):
                report.add("control-file", "mirror/ control files must not exist in the public dataset", f.path)
        report.findings.extend(check_publishable(root, [f.path for f in listing], manifests=manifests))
    return report


def _check_system_bundles(root: Path, report: TreeReport) -> set[str]:
    from aisim_core.data.hygiene import yaml_sidecar_value
    from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR, check_published_system, copy_filename, read_manifest

    declared = set()
    for path in sorted((root / "systems").glob("*/manifest.yaml")):
        rel = path.relative_to(root).as_posix()
        system = path.parent.name
        try:
            yaml_sidecar_value(bounded_json_bytes(checked_path(root, rel)))
            manifest = read_manifest(path)
            for entry in manifest.history:
                copy_rel = f"systems/{system}/{copy_filename(system, entry.spec_sha256)}"
                yaml_sidecar_value(bounded_json_bytes(checked_path(root, copy_rel)))
            problems = check_published_system(system, root, PACKAGED_SYSTEMS_DIR / f"{system}.yaml")
            if problems:
                raise ValueError("; ".join(problems))
        except (ValueError, OSError) as exc:
            report.add("system-spec-invalid", str(exc).splitlines()[0], rel)
            continue
        declared.add(rel)
        declared.update(f"systems/{system}/{copy_filename(system, h.spec_sha256)}" for h in manifest.history)
    return declared


def report_paths_with_errors(report: TreeReport, *prefixes: str) -> set[str]:
    return {f.path for f in report.errors if f.path and f.code.startswith(prefixes)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="validate a dataset checkout")
    val.add_argument("--root", type=Path, required=True)
    val.add_argument("--repo", default=INTERNAL_REPO, help="expected dataset_id (default: the internal mirror)")
    val.add_argument("--profile", choices=("mirror", "public"), default="mirror")
    val.add_argument(
        "--allow-empty", action="store_true", help="accept a tree with no dataset payload files (new mirror)"
    )
    val.add_argument("--output", type=Path, help="write the JSON report here")
    sub.add_parser("fingerprint", help="print the validator fingerprint")
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        print(validator_fingerprint())
        return 0
    report = validate_tree(args.root, repo=args.repo, profile=args.profile, allow_empty=args.allow_empty)
    if args.output:
        args.output.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n")
    print(f"validator {report.fingerprint[:12]} profile={report.profile} repo={report.repo}")
    for f in report.findings:
        print(f"{f.severity.upper()} {f.code} {f.path or ''}: {f.message}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
