# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the pinned Dynamo callback, unmodified, against the selected AIS core/SDK."""

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_simulation_dynamo import locked_pyo3_version

from scripts.release.package_identity import core_package

# Dynamo's callback at DYNAMO_REVISION, reviewed and compiled without changes.
CALLBACK_PATH = "lib/bindings/python/rust/llm/ais_callback.rs"
CALLBACK_SHA256 = "4e0f53f55f77ed5e5c5899702da0880846405c977110f3665aa2e5b64d641791"

# Minimal interface scaffolding adapted from NVIDIA Dynamo, Apache-2.0:
# https://github.com/ai-dynamo/dynamo/blob/df666fbc6284e540ea0f5cac613da7f992b1fc7b/lib/mocker/src/common/perf_model.rs
# https://github.com/ai-dynamo/dynamo/blob/df666fbc6284e540ea0f5cac613da7f992b1fc7b/lib/router/scheduler/src/prefill_load.rs
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modified: retain only the callback traits; no Dynamo scheduling implementation.
SCAFFOLD = """// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted interfaces from NVIDIA Dynamo, modified to omit scheduling implementation:
// https://github.com/ai-dynamo/dynamo/blob/df666fbc6284e540ea0f5cac613da7f992b1fc7b/lib/mocker/src/common/perf_model.rs
// https://github.com/ai-dynamo/dynamo/blob/df666fbc6284e540ea0f5cac613da7f992b1fc7b/lib/router/scheduler/src/prefill_load.rs
#![allow(dead_code)]
extern crate self as dynamo_mocker;
extern crate self as dynamo_scheduler;
pub mod host {
    pub trait PrefillLoadEstimator: Send + Sync {
        fn predict_prefill_duration(&self, batch_size: usize, effective_isl: usize,
            prefix: usize) -> anyhow::Result<std::time::Duration>;
    }
}
pub mod common { pub mod perf_model {
    pub trait AisCallback: Send + Sync {
        fn predict_prefill(&self, batch_size: usize, effective_isl: usize,
            prefix: usize) -> anyhow::Result<f64>;
        fn predict_decode(&self, batch_size: usize, isl: usize,
            osl: usize) -> anyhow::Result<f64>;
    }
}}
mod callback;
"""


def check(source: Path, dynamo: Path) -> None:
    source, dynamo = source.resolve(), dynamo.resolve()
    callback = dynamo / CALLBACK_PATH
    callback_source = callback.read_bytes()
    if hashlib.sha256(callback_source).hexdigest() != CALLBACK_SHA256:
        raise ValueError("callback does not match the reviewed pinned Dynamo source")
    core = source / "crates/core"
    shape = re.search(
        r"pub (struct|enum) ForwardPassSpeculationConfig\b", (core / "src/perfmodel/fpm/config.rs").read_text()
    )
    # The pinned callback reads `kind` and `params` from the generic configuration.
    if shape is None or shape[1] != "struct":
        raise ValueError("the pinned Dynamo callback needs the generic speculation configuration API")
    # Validate every Python source file, so a different checkout's installed SDK
    # cannot silently supply the Python half of the selected Rust core.
    python_paths = site.getsitepackages()
    for package in ("aisim", "aisim_core"):
        installed = Path(importlib.import_module(package).__file__).resolve().parent
        expected = source / "python/aisim/src" / package
        for path in expected.rglob("*.py"):
            if (installed / path.relative_to(expected)).read_bytes() != path.read_bytes():
                raise ValueError(f"installed SDK differs from selected source: {path}")
        python_paths.append(str(installed.parent))
        print(f"SDK {package}: {installed}", flush=True)
    print(f"Callback: {callback}; sha256={hashlib.sha256(callback_source).hexdigest()}", flush=True)
    print(f"AIS core: {core}; speculation API={shape[1]}", flush=True)
    env = os.environ | {
        "PYO3_PYTHON": sys.executable,
        "PYTHONHOME": sys.base_prefix,
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_paths)),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with tempfile.TemporaryDirectory(prefix="dynamo-callback-contract-") as directory:
        root = Path(directory)
        (root / "Cargo.toml").write_text(
            '[package]\nname = "dynamo-callback-contract"\nversion = "0.0.0"\nedition = "2024"\n'
            '[lib]\npath = "lib.rs"\n[features]\ndefault = ["ais-forward-pass"]\nais-forward-pass = []\n'
            "[dependencies]\n"
            f"aisim-core = {{ package = {json.dumps(core_package(core))}, "
            f'path = {json.dumps(str(core))}, features = ["embed-python"] }}\n'
            f'pyo3 = {{ version = "={locked_pyo3_version(source / "Cargo.lock")}", '
            'features = ["auto-initialize", "abi3-py311"] }\n'
            'serde_json = "1"\nanyhow = "1"\n[profile.dev]\ndebug = 0\n[profile.test]\ndebug = 0\n'
        )
        shutil.copyfile(source / "Cargo.lock", root / "Cargo.lock")
        (root / "lib.rs").write_text(SCAFFOLD)
        contract = Path(__file__).resolve().with_name("dynamo_callback_contract.rs")
        (root / "callback.rs").write_bytes(
            callback_source
            + f"\n#[cfg(test)]\nmod consumer_contract {{ include!({json.dumps(str(contract))}); }}\n".encode()
        )
        manifest = ["--manifest-path", str(root / "Cargo.toml")]
        subprocess.run(
            ["cargo", "metadata", *manifest, "--offline", "--format-version", "1"],
            env=env,
            stdout=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            ["cargo", "test", *manifest, "--offline", "--locked", "--lib", "--", "--test-threads=1"],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dynamo", type=Path)
    args = parser.parse_args()
    check(args.source, args.dynamo)
