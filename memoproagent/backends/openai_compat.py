# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""OpenAI-compatible chat and embedding backends.

Covers anything that speaks the OpenAI REST shape: the OpenAI API itself, a local
vLLM / SGLang / llama.cpp server, Azure OpenAI, OpenRouter, and most internal
gateways. This is the default path for every model in the paper:

    Qwen3.8-27B            vLLM  -> http://127.0.0.1:8000/v1
    GPT / Claude / Gemini  their own or a gateway endpoint

No provider SDK is required; only `requests`.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import requests

from .base import ChatResult, EmbeddingBackend, GenerationConfig, Message, batched, l2_normalize

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class BackendError(RuntimeError):
    pass


def _restore_stop(content: str, choice: dict[str, Any], stops: list[str]) -> str:
    """Re-append a server-stripped stop delimiter, or return content unchanged.

    Restores ONLY when the choice proves the stop fired (``finish_reason`` is
    ``"stop"`` and ``stop_reason`` exactly equals one requested stop). Any
    other finish reason (``length``, ``eos``, missing, ...) leaves content
    untouched: blindly closing a truncated span would manufacture a well-formed
    trajectory the model never produced. An already-closed string is returned
    as-is, never duplicated.
    """
    if not isinstance(content, str) or not stops:
        return content
    stop_reason = choice.get("stop_reason")
    if not isinstance(stop_reason, str) or stop_reason not in stops:
        return content
    if choice.get("finish_reason") != "stop":
        return content
    if content.endswith(stop_reason):
        return content
    return content + stop_reason


def _normalize_finish_reason(value: Any) -> str | None:
    """Preserve server finish_reason; unknown/missing stays None (never forced)."""
    return value if isinstance(value, str) else None


def _normalize_stop_reason(value: Any) -> str | int | None:
    """Preserve string/int stop_reason; bools and other types stay None."""
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (str, int)) else None


def _normalize_usage_field(usage: Any, key: str) -> int | None:
    """Normalize one usage count; missing/invalid stays None without masking chat errors."""
    if usage is None:
        return None
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


class _RateLimiter:
    """Process-local request pacing. `rpm<=0` disables it."""

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            time.sleep(wait)


@dataclass
class OpenAICompatConfig:
    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    timeout: float = 180.0
    max_retries: int = 4
    rpm: int = 0
    max_images_per_request: int = 32
    image_detail: str = "auto"
    # Some servers reject unknown fields; drop them here rather than per call site.
    send_seed: bool = True
    send_stop: bool = True
    extra_body: dict[str, Any] | None = None

    def resolve_key(self) -> str:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env, "") or "EMPTY"


class OpenAICompatChat:
    """Chat backend. Satisfies both ChatBackend and VisionChatBackend."""

    def __init__(self, config: OpenAICompatConfig) -> None:
        self.config = config
        self.name = f"{config.model}@{config.base_url}"
        self.max_images_per_request = config.max_images_per_request
        self._limiter = _RateLimiter(config.rpm)
        self._session = requests.Session()

    def _build_payload(
        self, messages: Sequence[Message], config: GenerationConfig
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_openai(self.config.image_detail) for m in messages],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if config.include_stop_str_in_output and not self.config.send_stop:
            raise BackendError("include_stop_str_in_output requires send_stop=True")
        if config.include_stop_str_in_output and not config.stop:
            raise BackendError("include_stop_str_in_output requires stop to be set")
        if config.stop and self.config.send_stop:
            payload["stop"] = list(config.stop)
        if config.include_stop_str_in_output:
            # vLLM per-request flag: server returns the matched stop string
            # instead of stripping it. Only sent when opted in, so generic
            # servers see byte-identical payloads by default.
            payload["include_stop_str_in_output"] = True
        if config.seed is not None and self.config.send_seed:
            payload["seed"] = config.seed
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        if config.include_stop_str_in_output:
            # extra_body must not silently drop the canonical stop request.
            if payload.get("stop") != list(config.stop):
                raise BackendError("extra_body must not overwrite the requested stop")
            if payload.get("include_stop_str_in_output") is not True:
                raise BackendError("extra_body must not overwrite include_stop_str_in_output")
        return payload

    def chat_with_metadata(
        self, messages: Sequence[Message], config: GenerationConfig
    ) -> ChatResult:
        """Single-request structured chat preserving termination metadata.

        Optional capability (not part of the ChatBackend/VisionChatBackend
        protocols). Unknown finish/stop/usage metadata stays None; stop
        restoration reuses the existing _restore_stop gate.
        """
        payload = self._build_payload(messages, config)
        data = self._post("/chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise BackendError(f"unexpected chat response: {json.dumps(data)[:400]}") from error
        if content is None:
            # A reasoning model that spent its whole budget on hidden tokens returns
            # null content. Treat it as a hard failure: scoring it as silence would
            # quietly turn an infrastructure problem into a model result.
            raise BackendError(
                f"empty content (finish_reason={choice.get('finish_reason')!r}); "
                "raise max_tokens or disable reasoning for this model"
            )
        if config.include_stop_str_in_output and config.stop:
            content = _restore_stop(content, choice, list(config.stop))
        return ChatResult(
            text=content,
            finish_reason=_normalize_finish_reason(choice.get("finish_reason")),
            stop_reason=_normalize_stop_reason(choice.get("stop_reason")),
            prompt_tokens=_normalize_usage_field(data.get("usage"), "prompt_tokens"),
            completion_tokens=_normalize_usage_field(data.get("usage"), "completion_tokens"),
            total_tokens=_normalize_usage_field(data.get("usage"), "total_tokens"),
        )

    def chat(self, messages: Sequence[Message], config: GenerationConfig) -> str:
        return self.chat_with_metadata(messages, config).text

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self.config.resolve_key()}",
            "Content-Type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._limiter.acquire()
            try:
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=self.config.timeout
                )
            except requests.RequestException as error:
                last = error
            else:
                if response.status_code == 200:
                    return response.json()
                last = BackendError(
                    f"HTTP {response.status_code} from {url}: {response.text[:400]}"
                )
                if response.status_code not in RETRYABLE_STATUS:
                    raise last
            if attempt < self.config.max_retries:
                time.sleep(min(2.0**attempt, 30.0) * (0.5 + random.random()))
        raise BackendError(f"request failed after {self.config.max_retries + 1} attempts: {last}")


class OpenAICompatEmbedding(EmbeddingBackend):
    """Embeddings over an OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, config: OpenAICompatConfig, dim: int, batch_size: int = 64) -> None:
        self.config = config
        self.name = config.model
        self.dim = dim
        self.batch_size = batch_size
        self._chat = OpenAICompatChat(config)

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for chunk in batched(list(texts), self.batch_size):
            payload = {"model": self.config.model, "input": list(chunk)}
            data = self._chat._post("/embeddings", payload)
            try:
                items = sorted(data["data"], key=lambda item: item["index"])
            except (KeyError, TypeError) as error:
                raise BackendError(
                    f"unexpected embeddings response: {json.dumps(data)[:400]}"
                ) from error
            vectors.extend(item["embedding"] for item in items)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape[1] != self.dim:
            raise BackendError(
                f"{self.name} returned dim {matrix.shape[1]}, config declares {self.dim}"
            )
        return l2_normalize(matrix)


class SentenceTransformerEmbedding(EmbeddingBackend):
    """Local embedding backend. Default model matches the one used in the paper.

    `model_name` may be a local directory, which is how this runs on a machine
    with no access to the model hub. In that case pass `signature_name` as well:
    the signature is what the memory manifest is checked against, and a path is
    machine-specific, so an index built on one machine would look foreign on
    another purely because the weights live somewhere else.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        signature_name: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.name = signature_name or model_name
        # The accessor was renamed in sentence-transformers 5; support both so a
        # dimension mismatch is never masked by an AttributeError.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        self.batch_size = batch_size

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return l2_normalize(matrix)
