#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run `aisim predict` with every output file in a fresh, dated results folder.
#
# Usage:
#   ./aisim-run.sh [-m "comment"] [predict] <aisim predict args...>
#
# Example:
#   ./aisim-run.sh -m "npc trace baseline" --stack engine --config prediction.yaml
#   -> results/26-11-24-14:05:37_npc-trace-baseline/{prediction.json,...,console.log}
#
# The folder name is yy-mm-dd-hh:mm:ss, plus "_<comment>" when -m is given.
# Characters outside [A-Za-z0-9._-] in the comment become "-".
# console.log holds the command line and everything aisim printed.
# Do not pass --output-dir: the wrapper sets it.

set -euo pipefail

REPO_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
RESULTS_DIR="$REPO_ROOT/results"

die() {
    echo "aisim-run: $*" >&2
    exit 1
}

# Leading "-m comment" and "predict" are accepted in either order.
comment=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m)
            [[ $# -ge 2 ]] || die "-m needs a comment"
            comment="$(printf '%s' "$2" | tr -c 'A-Za-z0-9._-' '-')"
            shift 2
            ;;
        predict) shift ;;
        *) break ;;
    esac
done

for arg in "$@"; do
    [[ "$arg" == --output* ]] && die "do not pass $arg; the wrapper sets --output-dir"
done

if [[ -x "$REPO_ROOT/python/aisim/.venv/bin/aisim" ]]; then
    AISIM="$REPO_ROOT/python/aisim/.venv/bin/aisim"
elif command -v aisim >/dev/null 2>&1; then
    AISIM="aisim"
else
    die "aisim not found on PATH or in python/aisim/.venv"
fi

RUN_DIR="$RESULTS_DIR/$(date +%y-%m-%d-%H:%M:%S)${comment:+_$comment}"
mkdir -p "$RESULTS_DIR"
# mkdir without -p reserves the name atomically; aisim accepts an empty folder.
mkdir "$RUN_DIR" 2>/dev/null ||
    die "cannot create $RUN_DIR (already exists?); wait a second or use a different -m comment"

# aisim refuses a folder holding files it did not write, so log beside it
# and move the log in on exit, including after Ctrl-C or a failed run.
LOG="$(mktemp "$RESULTS_DIR/.console-XXXXXX.log")"
finish() {
    chmod 644 "$LOG"
    mv "$LOG" "$RUN_DIR/console.log"
    echo "aisim-run: results in $RUN_DIR" >&2
}
trap finish EXIT

cmd=("$AISIM" predict "$@" --output-dir "$RUN_DIR")
{ printf '#'; printf ' %q' "${cmd[@]}"; echo; } >"$LOG"

# tee ignores Ctrl-C so aisim's cancellation output still reaches the log.
set +e
PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | (trap '' INT; exec tee -a "$LOG")
status=${PIPESTATUS[0]}
set -e
exit "$status"
