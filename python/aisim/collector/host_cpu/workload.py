# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real chat request bodies shared by every tokenizer, so tokenizers see the same text."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_ROLES = {"system", "user", "assistant"}


def _flatten(content: Any) -> str:
    """Text of a message ``content`` that is a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def normalize_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Plain role/text messages every chat template accepts.

    Tool calls and tool results are folded into plain text so templates that reject them (or render
    them differently) do not change what is compared across tokenizers. Consecutive messages of the
    same role are merged because several templates require alternating roles.
    """
    merged: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        role = role if role in _ROLES else "user"
        text = _flatten(message.get("content"))
        for call in message.get("tool_calls") or []:
            text += "\n" + json.dumps(call.get("function", call), sort_keys=True)
        if not text:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + text
        else:
            merged.append({"role": role, "content": text})
    return merged


def load_payloads(raw_jsonl: str | Path, count: int) -> Iterator[dict[str, Any]]:
    """Every k-th profiling-phase, HTTP 200 request payload of an AgentX raw stream (``count`` of them)."""
    payloads = []
    with Path(raw_jsonl).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("metadata", {}).get("benchmark_phase") == "profiling" and record.get("status") == 200:
                payloads.append(record["payload"])
    step = max(1, len(payloads) // count)
    yield from payloads[::step][:count]


def workload_digest(bodies: Iterable[dict[str, Any]]) -> str:
    """SHA-256 over the canonical bodies, identifying the exact request set that was timed."""
    digest = hashlib.sha256()
    for body in bodies:
        digest.update(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()
