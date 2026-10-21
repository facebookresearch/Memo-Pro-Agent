# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Focused backend metadata tests (fake HTTP only, CPU-only)."""
from __future__ import annotations

import pytest

from memoproagent.backends import ChatBackend, VisionChatBackend
from memoproagent.backends.base import ChatResult, GenerationConfig, Message
from memoproagent.backends.openai_compat import (
    BackendError,
    OpenAICompatChat,
    OpenAICompatConfig,
)

CLOSER = "</observe>"


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = "" if status == 200 else "boom"

    def json(self):
        return self._payload


def _make_backend(monkeypatch, payload=None, status=200, calls=None):
    backend = OpenAICompatChat(
        OpenAICompatConfig(model="test-model", base_url="http://127.0.0.1:8000/v1", rpm=0, max_retries=0)
    )
    if calls is None:
        calls = []
    if payload is None:
        payload = {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json})
        return _FakeResponse(payload, status=status)

    monkeypatch.setattr(backend._session, "post", fake_post)
    return backend, calls


def _cfg(**over):
    base = {"max_tokens": 16, "temperature": 0.0}
    base.update(over)
    return GenerationConfig(**base)


def test_metadata_preserved(monkeypatch):
    payload = {
        "choices": [
            {"message": {"content": "hi"}, "finish_reason": "length", "stop_reason": "len"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    backend, _ = _make_backend(monkeypatch, payload=payload)
    result = backend.chat_with_metadata([Message(role="user", text="q")], _cfg())
    assert isinstance(result, ChatResult)
    assert result.text == "hi"
    assert result.finish_reason == "length"
    assert result.stop_reason == "len"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (10, 4, 14)


def test_single_http_call_and_string_compat(monkeypatch):
    backend, calls = _make_backend(monkeypatch)
    msgs = [Message(role="user", text="q")]
    text = backend.chat(msgs, _cfg())
    assert text == "hello"
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/chat/completions")

    backend2, calls2 = _make_backend(monkeypatch)
    result = backend2.chat_with_metadata(msgs, _cfg())
    assert result.text == "hello"
    assert len(calls2) == 1


def test_unknown_metadata_stays_none(monkeypatch):
    for payload in [
        {"choices": [{"message": {"content": "x"}}]},
        {
            "choices": [{"message": {"content": "x"}, "finish_reason": 5, "stop_reason": True}],
            "usage": {"prompt_tokens": "3", "completion_tokens": True},
        },
        {
            "choices": [{"message": {"content": "x"}, "finish_reason": None}],
            "usage": "malformed",
        },
    ]:
        backend, _ = _make_backend(monkeypatch, payload=payload)
        result = backend.chat_with_metadata([Message(role="user", text="q")], _cfg())
        assert result.finish_reason is None
        assert result.stop_reason is None
        assert result.prompt_tokens is None
        assert result.completion_tokens is None
        assert result.total_tokens is None


def test_restore_stop_requires_matching_stop_metadata(monkeypatch):
    cfg = _cfg(stop=[CLOSER], include_stop_str_in_output=True)
    msgs = [Message(role="user", text="q")]
    # Restored: finish stop + matching stop_reason.
    payload = {
        "choices": [
            {
                "message": {"content": "<observe>abc"},
                "finish_reason": "stop",
                "stop_reason": CLOSER,
            }
        ]
    }
    backend, _ = _make_backend(monkeypatch, payload=payload)
    assert backend.chat_with_metadata(msgs, cfg).text == "<observe>abc" + CLOSER

    # No restore on length even with matching stop_reason.
    payload_len = {
        "choices": [
            {
                "message": {"content": "<observe>abc"},
                "finish_reason": "length",
                "stop_reason": CLOSER,
            }
        ]
    }
    backend, _ = _make_backend(monkeypatch, payload=payload_len)
    assert backend.chat_with_metadata(msgs, cfg).text == "<observe>abc"

    # No restore on eos.
    payload_eos = {
        "choices": [
            {
                "message": {"content": "<observe>abc"},
                "finish_reason": "eos",
                "stop_reason": CLOSER,
            }
        ]
    }
    backend, _ = _make_backend(monkeypatch, payload=payload_eos)
    assert backend.chat_with_metadata(msgs, cfg).text == "<observe>abc"

    # No restore on unknown finish.
    payload_unknown = {
        "choices": [{"message": {"content": "<observe>abc"}, "stop_reason": CLOSER}]
    }
    backend, _ = _make_backend(monkeypatch, payload=payload_unknown)
    assert backend.chat_with_metadata(msgs, cfg).text == "<observe>abc"

    # No duplication when already closed.
    payload_closed = {
        "choices": [
            {
                "message": {"content": "<observe>abc" + CLOSER},
                "finish_reason": "stop",
                "stop_reason": CLOSER,
            }
        ]
    }
    backend, _ = _make_backend(monkeypatch, payload=payload_closed)
    assert backend.chat_with_metadata(msgs, cfg).text == "<observe>abc" + CLOSER


def test_error_preservation(monkeypatch):
    msgs = [Message(role="user", text="q")]
    # Null content raises.
    null_payload = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    backend, _ = _make_backend(monkeypatch, payload=null_payload)
    with pytest.raises(BackendError):
        backend.chat_with_metadata(msgs, _cfg())
    backend, _ = _make_backend(monkeypatch, payload=null_payload)
    with pytest.raises(BackendError):
        backend.chat(msgs, _cfg())
    # Malformed response raises.
    backend, _ = _make_backend(monkeypatch, payload={"oops": 1})
    with pytest.raises(BackendError):
        backend.chat_with_metadata(msgs, _cfg())
    # Non-retryable HTTP error raises.
    backend, calls = _make_backend(monkeypatch, status=400)
    with pytest.raises(BackendError):
        backend.chat_with_metadata(msgs, _cfg())
    assert len(calls) == 1


def test_chat_only_backend_still_satisfies_protocols():
    class ChatOnly:
        name = "scripted"
        max_images_per_request = 1

        def chat(self, messages, config):
            return "ok"

    assert isinstance(ChatOnly(), ChatBackend)
    assert isinstance(ChatOnly(), VisionChatBackend)
    assert not hasattr(ChatBackend, "chat_with_metadata")
