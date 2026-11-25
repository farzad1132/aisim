# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope of the tokenization-cost campaign: frameworks, versions, tokenizers and systems."""

from __future__ import annotations

# Framework versions that match Dynamo 1.6.0.
FRAMEWORK_VERSIONS: dict[str, str] = {"vllm": "0.31.0", "sglang": "0.5.21", "trtllm": "1.3.0rc29"}

# One representative model per tokenizer family. Only tokenizer files are read, never weights.
# The family names are inferred from the model family; the harness records the real tokenizer class.
TOKENIZER_MODELS: dict[str, str] = {
    "qwen2": "Qwen/Qwen3.5-397B-A17B",
    # Ungated mirror of the Llama 3.1 tokenizer and chat template (the meta-llama repos are gated).
    "llama3": "NousResearch/Meta-Llama-3.1-70B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "kimi": "moonshotai/Kimi-K2.5",
    "glm": "zai-org/GLM-5",
    "minimax": "MiniMaxAI/MiniMax-M2.5",
    "gpt-oss": "openai/gpt-oss-120b",
}

# Systems whose host CPU is measured (x86 hosts of the Blackwell and Hopper systems).
SYSTEMS: tuple[str, ...] = ("b200_sxm", "b300_sxm", "h100_sxm", "h200_sxm")

# Request components timed per request. ``hash`` exists only where the framework hashes the prompt
# for prefix caching before scheduling (vLLM).
COMPONENTS: tuple[str, ...] = ("json", "validate", "template", "encode", "hash")

# Components whose cost queues on the tokenizer in replay (engine.tokenization).
QUEUED_COMPONENTS: tuple[str, ...] = ("template", "encode", "hash")
