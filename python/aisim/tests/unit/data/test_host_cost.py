# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random

import pytest

from aisim_core.data.gpu_time import GpuTime
from aisim_core.data.host_cost import check_entry, entry_directory, internal_record, read_fit, write_entry
from aisim_core.data.redaction import DEFAULT_POLICY, scan_value

pytestmark = pytest.mark.unit

MODEL = "Qwen/Qwen3.5-397B-A17B"


def _run_dir(tmp_path, *, per_token_us=1.5, with_hash=True):
    rng = random.Random(3)
    rows = []
    for index in range(60):
        tokens = rng.randrange(2_000, 180_000)
        total = per_token_us * tokens / 1000.0 * (1 + rng.uniform(-0.01, 0.01))
        row = {
            "request": index,
            "tokens": tokens,
            "body_kb": tokens * 4 / 1024,
            "json_ms": 1.0,
            "validate_ms": 0.5,
            "template_ms": 0.0,
            "encode_ms": total * 0.8,
        }
        if with_hash:
            row["hash_ms"] = total * 0.2
        rows.append(row)
    run = tmp_path / "run"
    run.mkdir()
    (run / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    environment = {
        "schema": "aisim-host-cost-run/v1",
        "framework": "vllm",
        "framework_version": "0.31.0",
        "tokenizer_key": "qwen2",
        "model": "/workspace/tokenizers/qwen2",
        "tokenizer_class": "Qwen2TokenizerFast",
        "tokenizer_is_fast": True,
        "tokenizer_workers": 1,
        "hash_algorithm": "sha256_cbor",
        "block_size": 16,
        "requests": 60,
        "host": {
            "cpu_model": "Intel(R) Xeon(R) Platinum 8559C",
            "cpu_threads": 192,
            "hostname": "cluster-slurm-1-gpu-50",
            "slurm_job_id": "1",
            "load_average_1m": 12.5,
        },
    }
    (run / "environment.json").write_text(json.dumps(environment))
    return run


def test_write_then_check_round_trips_and_stores_the_fit(tmp_path):
    root = tmp_path / "dataset"
    gpu_time = GpuTime(authority="declared", gpu_type="B300", gpu_count=1, wall_seconds=600.0, gpu_seconds=600.0)
    path = write_entry(root, _run_dir(tmp_path), system="b300_sxm", model=MODEL, gpu_time=gpu_time)
    assert path == entry_directory("vllm", "0.31.0", "b300_sxm", "qwen2") + "/aisim-manifest.json"
    manifest = check_entry(root, path)
    assert manifest.identity.hardware == "b300_sxm"
    assert manifest.identity.effective_settings["cpu_model"] == "Intel(R) Xeon(R) Platinum 8559C"
    assert read_fit(manifest).per_input_token_us == pytest.approx(1.5, rel=0.03)
    assert manifest.provenance["host_cost"]["gpu_time"]["gpu_seconds"] == 600.0


def test_rows_without_a_hash_component_are_supported(tmp_path):
    root = tmp_path / "dataset"
    path = write_entry(root, _run_dir(tmp_path, with_hash=False), system="h100_sxm", model=MODEL, gpu_time=None)
    assert check_entry(root, path).provenance["host_cost"]["gpu_time"] is None


def test_a_tampered_table_or_fit_is_refused(tmp_path):
    root = tmp_path / "dataset"
    path = write_entry(root, _run_dir(tmp_path), system="b300_sxm", model=MODEL, gpu_time=None)
    manifest_file = root / path
    document = json.loads(manifest_file.read_text())
    document["provenance"]["host_cost"]["fit"]["per_input_token_us"] += 1.0
    manifest_file.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="stored fit differs"):
        check_entry(root, path)
    table = root / entry_directory("vllm", "0.31.0", "b300_sxm", "qwen2") / "requests.parquet"
    table.write_bytes(table.read_bytes()[:-8] + b"corrupt!")
    with pytest.raises(ValueError):
        check_entry(root, path)


def test_the_public_manifest_carries_no_cluster_facts_and_the_internal_record_does(tmp_path):
    root = tmp_path / "dataset"
    run = _run_dir(tmp_path)
    path = write_entry(root, run, system="b300_sxm", model=MODEL, gpu_time=None)
    manifest = check_entry(root, path)
    assert manifest.identity.model == MODEL
    assert manifest.provenance["host_cost"]["environment"]["model"] == MODEL
    host = manifest.provenance["host_cost"]["host"]
    assert host["cpu_threads"] == 192 and not {"hostname", "slurm_job_id", "load_average_1m"} & set(host)
    assert not scan_value(json.loads((root / path).read_text()), DEFAULT_POLICY, path, "manifest")
    record = internal_record(run, job_ids=["1", "2"])
    assert record["host"]["hostname"] == "cluster-slurm-1-gpu-50"
    assert record["tokenizer_path"] == "/workspace/tokenizers/qwen2" and record["job_ids"] == ["1", "2"]


@pytest.mark.parametrize("model", ["/workspace/tokenizers/qwen2", "qwen2", "./Qwen/Qwen3"])
def test_a_local_tokenizer_path_is_not_a_public_model(tmp_path, model):
    with pytest.raises(ValueError, match="Hugging Face repository"):
        write_entry(tmp_path / "dataset", _run_dir(tmp_path), system="b300_sxm", model=model, gpu_time=None)


def test_job_ids_in_gpu_time_are_refused(tmp_path):
    gpu_time = GpuTime(authority="measured", gpu_type="B300", gpu_count=1, wall_seconds=60.0, source={"job_ids": "1,2"})
    with pytest.raises(ValueError, match="job IDs"):
        write_entry(tmp_path / "dataset", _run_dir(tmp_path), system="b300_sxm", model=MODEL, gpu_time=gpu_time)
