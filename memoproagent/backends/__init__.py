# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Backend construction from configuration."""
from __future__ import annotations

from typing import Any

from .base import (
    ChatBackend,
    ChatResult,
    EmbeddingBackend,
    GenerationConfig,
    Message,
    VisionChatBackend,
    encode_jpeg,
)
from .openai_compat import (
    BackendError,
    OpenAICompatChat,
    OpenAICompatConfig,
    OpenAICompatEmbedding,
    SentenceTransformerEmbedding,
)

__all__ = [
    "BackendError",
    "ChatBackend",
    "ChatResult",
    "EmbeddingBackend",
    "GenerationConfig",
    "Message",
    "OpenAICompatChat",
    "OpenAICompatConfig",
    "OpenAICompatEmbedding",
    "SentenceTransformerEmbedding",
    "VisionChatBackend",
    "build_chat_backend",
    "build_embedding_backend",
    "encode_jpeg",
]


def build_chat_backend(spec: dict[str, Any]) -> OpenAICompatChat:
    """Build a chat backend from a config mapping.

    ``provider: openai_compat`` is currently the only provider; it covers the OpenAI
    API, a local vLLM/SGLang server, and most gateways. Unknown keys are rejected so
    a typo in a config file fails loudly instead of silently taking a default.
    """
    spec = dict(spec)
    provider = spec.pop("provider", "openai_compat")
    if provider != "openai_compat":
        raise ValueError(f"unknown chat provider: {provider!r}")
    known = set(OpenAICompatConfig.__dataclass_fields__)
    unknown = set(spec) - known
    if unknown:
        raise ValueError(f"unknown chat backend options: {sorted(unknown)}")
    return OpenAICompatChat(OpenAICompatConfig(**spec))


def build_embedding_backend(spec: dict[str, Any]) -> EmbeddingBackend:
    """Build an embedding backend.

    ``provider: sentence_transformers`` runs locally (default, reproducible).
    ``provider: openai_compat`` calls a served ``/embeddings`` endpoint and requires
    an explicit ``dim`` so a silent dimension change cannot corrupt an index.
    """
    spec = dict(spec)
    provider = spec.pop("provider", "sentence_transformers")
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedding(**spec)
    if provider == "openai_compat":
        dim = spec.pop("dim", None)
        if dim is None:
            raise ValueError("openai_compat embedding backend requires an explicit `dim`")
        batch_size = spec.pop("batch_size", 64)
        known = set(OpenAICompatConfig.__dataclass_fields__)
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"unknown embedding backend options: {sorted(unknown)}")
        return OpenAICompatEmbedding(OpenAICompatConfig(**spec), dim=int(dim), batch_size=batch_size)
    raise ValueError(f"unknown embedding provider: {provider!r}")
