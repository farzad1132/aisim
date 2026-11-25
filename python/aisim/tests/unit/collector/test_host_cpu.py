# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from collector.host_cpu import adapters
from collector.host_cpu.ledger import job_cost, read_entries, render_markdown
from collector.host_cpu.merge import merge_runs
from collector.host_cpu.run import time_entry
from collector.host_cpu.workload import load_payloads, normalize_messages, workload_digest

pytestmark = pytest.mark.unit


def test_normalize_messages_flattens_parts_folds_tool_calls_and_merges_roles():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "rules"}, "more"]},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": None},
    ]
    assert normalize_messages(messages) == [
        {"role": "system", "content": "rules\nmore"},
        {"role": "user", "content": "a\nb"},
        {"role": "assistant", "content": '\n{"arguments": "{}", "name": "f"}'},
        {"role": "user", "content": "result"},
    ]


def test_load_payloads_takes_every_kth_profiling_success(tmp_path):
    lines = []
    for index in range(20):
        phase = "warmup" if index < 2 else "profiling"
        lines.append(
            {"metadata": {"benchmark_phase": phase}, "status": 200 if index != 5 else 500, "payload": {"i": index}}
        )
    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    taken = [payload["i"] for payload in load_payloads(path, 5)]
    # 17 profiling successes (index 5 failed), so every 3rd of them, the first 5.
    assert taken == [2, 6, 9, 12, 15]
    assert workload_digest([{"a": 1}]) == workload_digest([{"a": 1}]) != workload_digest([{"a": 2}])


class _Fake:
    framework = "fake"

    def __init__(self):
        self.calls = 0

    def environment(self):
        return {}

    def time_request(self, body):
        self.calls += 1
        return {"tokens": body["n"], "body_kb": 1.0, "json": 1.0, "encode": float(body["n"]) * 0.001 + self.calls % 3}


def test_time_entry_warms_up_and_takes_the_median_of_repeats():
    adapter = _Fake()
    rows = time_entry(adapter, [{"n": 1000}, {"n": 2000}], warmup=1, repeats=3)
    assert [row["tokens"] for row in rows] == [1000, 2000]
    assert adapter.calls == 1 + 6
    assert set(rows[0]) >= {"request", "tokens", "body_kb", "json_ms", "encode_ms"}


def test_time_entry_rejects_unstable_tokenization():
    class Unstable(_Fake):
        def time_request(self, body):
            row = super().time_request(body)
            row["tokens"] += self.calls
            return row

    with pytest.raises(RuntimeError, match="different lengths"):
        time_entry(Unstable(), [{"n": 10}], warmup=0, repeats=2)


def _write_run(directory, scale, job):
    directory.mkdir(parents=True)
    rows = [
        {"request": i, "tokens": 1000 * (i + 1), "body_kb": 1.0, "encode_ms": scale * (i + 1), "json_ms": 0.1}
        for i in range(4)
    ]
    (directory / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    env = {
        "framework": "vllm",
        "framework_version": "0.31.0",
        "tokenizer_key": "qwen2",
        "model": "m",
        "tokenizer_class": "T",
        "workload_sha256": "w",
        "requests": 4,
        "host": {"cpu_model": "cpu", "load_average_1m": 1.0, "slurm_job_id": job, "hostname": f"node-{job}"},
    }
    (directory / "environment.json").write_text(json.dumps(env))


def test_merge_runs_takes_the_per_request_median_and_refuses_mismatches(tmp_path):
    for name, scale, job in (("a", 1.0, "1"), ("b", 2.0, "2"), ("c", 3.0, "3")):
        _write_run(tmp_path / name, scale, job)
    environment = merge_runs([tmp_path / "a", tmp_path / "b", tmp_path / "c"], tmp_path / "out")
    rows = [json.loads(line) for line in (tmp_path / "out" / "rows.jsonl").read_text().splitlines()]
    assert [row["encode_ms"] for row in rows] == [2.0, 4.0, 6.0, 8.0]
    assert environment["merged_runs"] == 3 and environment["host"]["slurm_job_ids"] == ["1", "2", "3"]
    assert environment["host"]["hostnames"] == ["node-1", "node-2", "node-3"]
    other = json.loads((tmp_path / "c" / "environment.json").read_text())
    other["workload_sha256"] = "different"
    (tmp_path / "c" / "environment.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="workload_sha256"):
        merge_runs([tmp_path / "a", tmp_path / "c"], tmp_path / "out2")


def test_ledger_entry_costs_plus_remainder_equal_the_allocation(tmp_path):
    (tmp_path / "entries.txt").write_text("vllm qwen2 0 100 130\nvllm llama3 0 130 170\nvllm kimi 1 170 175\n")
    sacct = "JobID|State|ElapsedRaw|AllocTRES|NNodes\n77|COMPLETED|100|billing=1,cpu=16,gres/gpu=1|1\n"
    job = job_cost("b300_sxm", "77", sacct, tmp_path / "entries.txt", gpu_type="B300")
    assert [entry.wall_seconds for entry in job.entries] == [30.0, 40.0, 5.0]
    assert job.gpu_time.gpu_seconds == 100.0
    assert job.unattributed_gpu_seconds == pytest.approx(25.0)
    assert read_entries(tmp_path / "entries.txt", system="s", gpu_count=8)[0].gpu_seconds == 240.0
    markdown = render_markdown([job])
    assert "**Total**" in markdown and "0.028" in markdown


def test_every_campaign_framework_has_an_adapter():
    assert set(adapters.ADAPTERS) == {"vllm", "sglang", "trtllm"}


def test_internal_evidence_lists_the_host_record_and_job_records(tmp_path):
    from collector.host_cpu.stage import _write_internal

    from aisim_core.data.evidence import check_evidence

    job = tmp_path / "campaign" / "vllm-1"
    job.mkdir(parents=True)
    for name in ("job.txt", "entries.txt", "sacct.txt"):
        (job / name).write_text(f"{name} of job 7\n")
    internal = tmp_path / "internal"
    config_id = "cfg-" + "a" * 64
    manifest = "data/host-cost/vllm/0.31.0/b300_sxm/qwen2/aisim-manifest.json"
    _write_internal(internal, manifest, config_id, {"host": {"hostname": "node-7"}, "job_ids": ["7"]}, [job])
    directory = "evidence/data/host-cost/vllm/0.31.0/b300_sxm/qwen2"
    paths = [p.relative_to(internal).as_posix() for p in internal.rglob("*") if p.is_file()]
    assert sorted(p.removeprefix(directory + "/") for p in paths) == [
        "evidence-manifest.json",
        "host.json",
        "jobs/vllm-1/entries.txt",
        "jobs/vllm-1/job.txt",
        "jobs/vllm-1/sacct.txt",
    ]
    findings, _ = check_evidence(internal, paths)
    assert not findings
