"""Backend interfaces.

ProPerAgent needs exactly three capabilities from a model provider:

    ChatBackend       text-only chat            (the gate, the judge)
    VisionChatBackend chat with image inputs    (the generator / policy)
    EmbeddingBackend  dense text embeddings     (memory construction + retrieval)

Everything else in the package is written against these three protocols, so a new
provider is added by implementing them rather than by touching the harness.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np

Role = Literal["system", "user", "assistant", "tool"]

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _image_mime(raw: bytes) -> str:
    """Sniff PNG from the bytes header, otherwise assume JPEG.

    Judge frames are PNG-encoded; policy frames are JPEG-encoded. Sniffing
    keeps old JPEG payloads byte-identical while emitting the correct MIME
    for PNG without a format flag threading through every call site.
    """
    if isinstance(raw, (bytes, bytearray)) and bytes(raw[:8]) == _PNG_MAGIC:
        return "image/png"
    return "image/jpeg"


@dataclass
class Message:
    """One chat message.

    Images are attached in order before `text`. `image_labels`, when given, emits
    a short text part before each image -- which is how a judge is told which
    frame is the current one and which are context.

    `leading_text`, when given, is emitted once before all images; `text` stays
    trailing. The default (empty `leading_text`) preserves the existing order.
    """

    role: Role
    text: str = ""
    images: list[bytes] = field(default_factory=list)
    image_labels: list[str] = field(default_factory=list)
    leading_text: str = ""

    def __post_init__(self) -> None:
        if self.image_labels and len(self.image_labels) != len(self.images):
            raise ValueError(
                f"image_labels has {len(self.image_labels)} entries "
                f"for {len(self.images)} images"
            )

    def to_openai(self, image_detail: str = "auto") -> dict[str, Any]:
        if not self.images:
            if self.leading_text and self.text:
                return {"role": self.role, "content": f"{self.leading_text}\n\n{self.text}"}
            if self.leading_text:
                return {"role": self.role, "content": self.leading_text}
            return {"role": self.role, "content": self.text}
        parts: list[dict[str, Any]] = []
        if self.leading_text:
            parts.append({"type": "text", "text": self.leading_text})
        for index, raw in enumerate(self.images):
            if self.image_labels:
                parts.append({"type": "text", "text": self.image_labels[index]})
            mime = _image_mime(raw)
            url = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
        if self.text:
            parts.append({"type": "text", "text": self.text})
        return {"role": self.role, "content": parts}


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stop: Sequence[str] | None = None
    seed: int | None = None
    include_stop_str_in_output: bool = False
    """Opt-in vLLM per-request ``include_stop_str_in_output``. When True the
    backend sends the flag and restores a stop delimiter the server stripped.
    False (default) keeps payloads byte-identical for generic servers."""


@runtime_checkable
class ChatBackend(Protocol):
    """Text-only chat completion."""

    name: str

    def chat(self, messages: Sequence[Message], config: GenerationConfig) -> str:
        """Return the assistant message content. Must raise on a failed request
        rather than returning a sentinel, so callers never score a silent error."""


@runtime_checkable
class VisionChatBackend(ChatBackend, Protocol):
    """Chat completion that accepts image parts in user messages."""

    max_images_per_request: int


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Dense text embeddings.

    The same backend MUST be used to build the memory index and to embed queries at
    inference time; a mismatch silently destroys retrieval. `signature` is recorded
    in the memory manifest and verified on load.
    """

    name: str
    dim: int

    @property
    def signature(self) -> str:
        """Stable identifier of the embedding space, e.g. ``bge-large-en-v1.5:1024``."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalized ``(len(texts), dim)`` float32 array."""


def encode_jpeg(image: Any, quality: int = 85) -> bytes:
    """Encode a PIL image or an HxWx3 uint8 array (RGB) to JPEG bytes."""
    from PIL import Image

    if isinstance(image, bytes):
        return image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
