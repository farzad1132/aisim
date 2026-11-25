# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print where a framework's installed source handles the request path (read-only, no GPU).

Usage inside the framework image::

    python3 audit.py PACKAGE REGEX [REGEX ...] [--files GLOB ...] [--max N]

Each match prints ``relative/path.py:line: text`` so a request-path map can cite exact lines of the
installed version instead of memory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import re
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--files", nargs="*", default=["**/*.py"])
    parser.add_argument("--max", type=int, default=12, help="matches printed per pattern")
    args = parser.parse_args(argv)
    # Locate the package without importing it: some frameworks need a GPU driver just to import.
    spec = importlib.util.find_spec(args.package)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"package {args.package} not found")
    root = Path(next(iter(spec.submodule_search_locations)))
    try:
        version = importlib.metadata.version(args.package)
    except importlib.metadata.PackageNotFoundError:
        version = "?"
    print(f"package {args.package} version {version} at {root}")
    files = sorted({path for pattern in args.files for path in root.glob(pattern) if path.is_file()})
    for pattern in args.patterns:
        regex = re.compile(pattern)
        shown = 0
        print(f"\n## {pattern}")
        for path in files:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if regex.search(line):
                    print(f"{path.relative_to(root)}:{number}: {line.strip()[:150]}")
                    shown += 1
                    if shown >= args.max:
                        break
            if shown >= args.max:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
