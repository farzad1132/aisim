#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs inside one framework image: time every requested tokenizer for that framework.
# Usage: run_entries.sh FRAMEWORK OUT_DIR CODE_DIR TOKENIZER_DIR PAYLOADS COUNT REPEATS KEY...
set -uo pipefail
framework=$1 out=$2 code=$3 tok=$4 payloads=$5 count=$6 repeats=$7
shift 7
export PYTHONPATH="$code/python/aisim" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
for key in "$@"; do
    start=$(date +%s)
    python3 -m collector.host_cpu.run --framework "$framework" --tokenizer "$key" --model "$tok/$key" \
        --payloads "$payloads" --count "$count" --repeats "$repeats" --output "$out/$framework/$key" \
        > "$out/$framework-$key.log" 2>&1
    status=$?
    # One line per entry: wall seconds in the allocation, for the GPU-hour ledger.
    echo "$framework $key $status $start $(date +%s)" >> "$out/entries.txt"
done
