# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download only the tokenizer files (never weights) of each campaign model into a directory.

Run where Hugging Face is reachable, then copy the directory to the cluster. Writes a manifest with the
file hashes so a run records exactly which tokenizer files it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import TOKENIZER_MODELS

TOKENIZER_PATTERNS = (
    "tokenizer*",
    "*.model",
    "vocab*",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template*",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "tiktoken*",
    "*.tiktoken",
    "*.py",
)


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="tokenizer keys to fetch (default: all)")
    args = parser.parse_args(argv)
    manifest = {}
    for key, model in TOKENIZER_MODELS.items():
        if args.only and key not in args.only:
            continue
        target = args.output / key
        try:
            snapshot_download(model, local_dir=target, allow_patterns=list(TOKENIZER_PATTERNS))
        except Exception as exc:  # recorded, never papered over
            manifest[key] = {"model": model, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
            print(f"{key}: FAILED {manifest[key]['error']}")
            continue
        files = {
            path.relative_to(target).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
        }
        manifest[key] = {"model": model, "files": files}
        print(f"{key}: {len(files)} files")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
