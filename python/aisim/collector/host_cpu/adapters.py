# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-framework request-path adapters.

Each adapter calls the framework's own code (imported from the serving image) and returns the time
of every step in milliseconds for one request body. An adapter never estimates a step it cannot run:
it raises, and the entry is recorded as failed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from .workload import normalize_messages


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = fn()
    return result, (time.perf_counter_ns() - start) / 1e6


class Adapter(Protocol):
    """One framework's request path for one tokenizer."""

    framework: str

    def environment(self) -> dict[str, Any]:
        """Versions and settings that identify what was timed."""

    def time_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """``{"tokens": int, "json": ms, "validate": ms, "template": ms, "encode": ms, ["hash": ms]}``."""


class VllmAdapter:
    """vLLM 0.31 chat path: JSON, request validation, chat template, encode and prefix-cache block hashing."""

    framework = "vllm"
    block_size = 16

    def __init__(self, model: str):
        import vllm
        from transformers import AutoTokenizer

        self._vllm = vllm
        # Load the tokenizer the way the vLLM server does (``vllm.tokenizers.get_tokenizer``), so custom
        # tokenizers get the same wrapper class and call path as in a served deployment.
        self._loader = "vllm.tokenizers.get_tokenizer"
        try:
            from vllm.tokenizers import get_tokenizer

            self._tokenizer = get_tokenizer(model, trust_remote_code=True)
        except Exception:
            self._loader = "transformers.AutoTokenizer"
            self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._request_type = None
        for module in (
            "vllm.entrypoints.openai.chat_completion.protocol",
            "vllm.entrypoints.openai.protocol",
        ):
            try:
                self._request_type = __import__(module, fromlist=["ChatCompletionRequest"]).ChatCompletionRequest
                break
            except (ImportError, AttributeError):
                continue
        if self._request_type is None:
            raise RuntimeError("vLLM ChatCompletionRequest is not importable; the request model moved")
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

        init_none_hash(sha256_cbor)
        self._hash_fn = sha256_cbor
        self._hash_block = hash_block_tokens

    def environment(self) -> dict[str, Any]:
        import transformers

        return {
            "framework_version": self._vllm.__version__,
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(self._tokenizer, "is_fast", False)),
            "tokenizer_loader": self._loader,
            "hash_algorithm": "sha256_cbor",
            "block_size": self.block_size,
            "tokenizer_workers": 1,
        }

    def time_request(self, body: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        raw = json.dumps(body).encode()
        obj, row["json"] = _timed(lambda: json.loads(raw))
        request = {**obj, "messages": normalize_messages(obj["messages"])}
        _, row["validate"] = _timed(lambda: self._request_type.model_validate(request))
        text, row["template"] = _timed(
            lambda: self._tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
        )
        ids, row["encode"] = _timed(lambda: self._tokenizer(text, add_special_tokens=False)["input_ids"])

        def block_hashes() -> None:
            parent = None
            for start in range(0, len(ids) - len(ids) % self.block_size, self.block_size):
                block = self._hash_block(
                    self._hash_fn, getattr(parent, "block_hash", parent), ids[start : start + self.block_size]
                )
                parent = block

        _, row["hash"] = _timed(block_hashes)
        row["tokens"] = len(ids)
        row["body_kb"] = len(raw) / 1024
        return row


class SglangAdapter:
    """SGLang 0.5.21 chat path (``OpenAIServingChat``): request validation, chat-template render, encode.

    Audited in the installed image: ``handle_request`` stamps ``received_time`` after FastAPI has parsed and
    validated the body, ``_process_messages`` renders the template (``apply_chat_template(tokenize=False)``)
    and ``tokenizer.encode`` turns it into ids (``_render_and_encode_chat_template``), and the tokenizer manager
    then skips tokenizing because ``input_ids`` are present. SGLang hashes no prompt blocks before scheduling:
    the radix-cache match runs in the scheduler. The template cache is off for text-only models
    (``use_cache=is_multimodal``), so this is the cold path every request takes.
    """

    framework = "sglang"

    def __init__(self, model: str):
        import sglang
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        self._sglang = sglang
        self._request_type = ChatCompletionRequest
        self._loader = "transformers.AutoTokenizer"
        try:
            try:
                from sglang.srt.utils.hf_transformers_utils import get_tokenizer
            except ImportError:
                from sglang.srt.hf_transformers_utils import get_tokenizer
            self._tokenizer = get_tokenizer(model, trust_remote_code=True)
            self._loader = "sglang get_tokenizer"
        except Exception:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # SGLang adds ``add_special_tokens=False`` to the encode only when the tokenizer auto-adds specials.
        try:
            self._auto_adds_specials = len(self._tokenizer.encode("")) > 0
        except Exception:
            self._auto_adds_specials = True

    def environment(self) -> dict[str, Any]:
        import transformers

        return {
            "framework_version": self._sglang.__version__,
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(self._tokenizer, "is_fast", False)),
            "tokenizer_loader": self._loader,
            "encode_adds_special_tokens": not self._auto_adds_specials,
            "hash_algorithm": None,
            "block_size": None,
            "tokenizer_workers": 1,  # --tokenizer-worker-num default
        }

    def time_request(self, body: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        raw = json.dumps(body).encode()
        obj, row["json"] = _timed(lambda: json.loads(raw))
        request = {**obj, "messages": normalize_messages(obj["messages"])}
        _, row["validate"] = _timed(lambda: self._request_type.model_validate(request))
        text, row["template"] = _timed(
            lambda: self._tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True, return_dict=False
            )
        )
        encode_kwargs = {"add_special_tokens": False} if self._auto_adds_specials else {}
        ids, row["encode"] = _timed(lambda: self._tokenizer.encode(text, **encode_kwargs))
        row["tokens"] = len(ids)
        row["body_kb"] = len(raw) / 1024
        return row


class TrtllmAdapter:
    """TRT-LLM 1.3.0rc29 chat path (``trtllm-serve``): request validation, chat-template render, encode.

    Audited in the installed image: ``openai_chat`` renders the template with ``async_apply_chat_template``
    (``inputs.utils.apply_chat_template`` on a worker thread) and ``DefaultInputProcessor.__call__`` then
    encodes the prompt with ``tokenizer.encode(prompt, add_special_tokens=sampling_params.add_special_tokens)``.
    No prompt block hashing happens before the request reaches the executor. The thread hop of the template
    call and the tokenizer worker count are not measured or verified here (``tokenizer_workers`` is null).
    """

    framework = "trtllm"

    def __init__(self, model: str):
        import tensorrt_llm
        from tensorrt_llm.inputs.utils import apply_chat_template
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
        from tensorrt_llm.tokenizer.tokenizer import tokenizer_factory

        self._trtllm = tensorrt_llm
        self._apply_chat_template = apply_chat_template
        self._request_type = ChatCompletionRequest
        self._tokenizer = tokenizer_factory(model, trust_remote_code=True)
        with open(f"{model}/config.json") as stream:
            self._model_type = json.load(stream).get("model_type", "")
        self._add_special_tokens = bool(self._request_type.model_fields["add_special_tokens"].default)
        # trtllm-serve rejects request fields it does not define (extra="forbid"); the client's extra inputs
        # (such as return_token_ids) are dropped, as the Dynamo capture runs dropped them.
        self._fields = set(self._request_type.model_fields)
        self._dropped: set[str] = set()

    def environment(self) -> dict[str, Any]:
        import transformers

        return {
            "framework_version": self._trtllm.__version__,
            "transformers_version": transformers.__version__,
            "tokenizer_class": type(self._tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(getattr(self._tokenizer, "tokenizer", None), "is_fast", False)),
            "model_type": self._model_type,
            "encode_adds_special_tokens": self._add_special_tokens,
            "hash_algorithm": None,
            "block_size": None,
            "tokenizer_workers": None,
            "dropped_request_fields": sorted(self._dropped),
        }

    def time_request(self, body: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        raw = json.dumps(body).encode()
        obj, row["json"] = _timed(lambda: json.loads(raw))
        self._dropped |= set(obj) - self._fields
        request = {k: v for k, v in obj.items() if k in self._fields}
        request["messages"] = normalize_messages(obj["messages"])
        _, row["validate"] = _timed(lambda: self._request_type.model_validate(request))
        messages = request["messages"]
        text, row["template"] = _timed(
            lambda: self._apply_chat_template(
                model_type=self._model_type,
                tokenizer=self._tokenizer,
                processor=None,
                conversation=[dict(message) for message in messages],
                add_generation_prompt=True,
                mm_placeholder_counts=[{} for _ in messages],
                enable_tokenize=False,
            )
        )
        ids, row["encode"] = _timed(lambda: self._tokenizer.encode(text, add_special_tokens=self._add_special_tokens))
        row["tokens"] = len(ids)
        row["body_kb"] = len(raw) / 1024
        return row


ADAPTERS: dict[str, Callable[[str], Adapter]] = {
    "vllm": VllmAdapter,
    "sglang": SglangAdapter,
    "trtllm": TrtllmAdapter,
}
