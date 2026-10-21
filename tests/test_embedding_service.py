# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The shared embedding service uses the public OpenAI embedding contract."""
from __future__ import annotations

import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests

from memoproagent.backends.openai_compat import OpenAICompatConfig, OpenAICompatEmbedding


spec = importlib.util.spec_from_file_location(
    "embedding_service", Path(__file__).resolve().parents[1] / "tools" / "serve_embeddings.py"
)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class FakeEmbedding:
    name = "test-model"
    dim = 3
    signature = "test-model:3"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.tile(np.array([0.0, 0.6, 0.8], dtype=np.float32), (len(texts), 1))


@pytest.fixture
def endpoint():
    backend = FakeEmbedding()
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.handler_for(backend))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", backend
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_health_and_models(endpoint):
    root, _ = endpoint
    assert requests.get(root + "/health", timeout=2).json()["signature"] == "test-model:3"
    assert requests.get(root + "/v1/models", timeout=2).json()["data"][0]["id"] == "test-model"


def test_public_embedding_client_preserves_order_and_signature(endpoint):
    root, backend = endpoint
    client = OpenAICompatEmbedding(OpenAICompatConfig(model="test-model", base_url=root + "/v1"), dim=3)
    actual = client.embed(["cup", "bowl"])
    assert client.signature == backend.signature
    np.testing.assert_allclose(actual, [[0, 0.6, 0.8], [0, 0.6, 0.8]])
    assert backend.calls == [["cup", "bowl"]]


@pytest.mark.parametrize("body", [
    {"model": "wrong", "input": ["cup"]},
    {"model": "test-model", "input": []},
    {"model": "test-model", "input": [123]},
    {"model": "test-model", "input": None},
])
def test_bad_requests_do_not_call_model(endpoint, body):
    root, backend = endpoint
    assert requests.post(root + "/v1/embeddings", json=body, timeout=2).status_code == 400
    assert not backend.calls


def test_string_input_and_error_paths(endpoint):
    root, backend = endpoint
    response = requests.post(root + "/v1/embeddings", json={"model": "test-model", "input": "cup"}, timeout=2)
    assert response.status_code == 200
    assert backend.calls == [["cup"]]
    assert requests.get(root + "/unknown", timeout=2).status_code == 404
    assert requests.post(root + "/unknown", json={}, timeout=2).status_code == 404


def test_model_failure_is_not_a_successful_embedding(endpoint):
    root, backend = endpoint
    backend.embed = lambda texts: np.zeros((len(texts), 2), dtype=np.float32)
    response = requests.post(root + "/v1/embeddings", json={"model": "test-model", "input": ["cup"]}, timeout=2)
    assert response.status_code == 500
