#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One job per system: all requested frameworks run in order on the same node, each in its pinned image.
# Required environment: SYSTEM, OUT_DIR, CODE_DIR, TOKENIZER_DIR, PAYLOADS, STORAGE_ROOT,
#   IMAGE_VLLM / IMAGE_SGLANG / IMAGE_TRTLLM (sqsh files for the frameworks in FRAMEWORKS)
# Optional: FRAMEWORKS (default "vllm sglang trtllm"), KEYS (default all 7), COUNT (60), REPEATS (3)
# Submit with the account, partition and a --time up to the partition limit; the node must be the
# system's own GPU node because its host CPU is what is measured.
#SBATCH --job-name=tokenization-cost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
set -uo pipefail
: "${SYSTEM:?}" "${OUT_DIR:?}" "${CODE_DIR:?}" "${TOKENIZER_DIR:?}" "${PAYLOADS:?}" "${STORAGE_ROOT:?}"
frameworks=${FRAMEWORKS:-"vllm sglang trtllm"}
keys=${KEYS:-"qwen2 llama3 deepseek kimi glm minimax gpt-oss"}
mkdir -p "$OUT_DIR"
{
    echo "system=$SYSTEM job=$SLURM_JOB_ID node=$SLURMD_NODENAME start=$(date +%s)"
    lscpu | grep -E 'Model name|Socket|Thread|^CPU\(s\)'
    sha256sum "$PAYLOADS"
} > "$OUT_DIR/job.txt"
export ENROOT_CACHE_PATH="$STORAGE_ROOT/.cache/enroot" ENROOT_DATA_PATH="$OUT_DIR/enroot" \
    ENROOT_CONFIG_PATH="$STORAGE_ROOT/.config/enroot"
mkdir -p "$ENROOT_DATA_PATH"
in_container() { echo "${1/#$STORAGE_ROOT//workspace}"; }
for framework in $frameworks; do
    var="IMAGE_${framework^^}"
    image=${!var:?"$var is not set"}
    sha256sum "$image" >> "$OUT_DIR/job.txt"
    srun --ntasks=1 --container-image="$image" --container-mounts="$STORAGE_ROOT:/workspace" \
        bash "$(in_container "$CODE_DIR")/python/aisim/collector/host_cpu/slurm/run_entries.sh" \
        "$framework" "$(in_container "$OUT_DIR")" "$(in_container "$CODE_DIR")" \
        "$(in_container "$TOKENIZER_DIR")" "$(in_container "$PAYLOADS")" "${COUNT:-60}" "${REPEATS:-3}" $keys
done
echo "end=$(date +%s)" >> "$OUT_DIR/job.txt"
