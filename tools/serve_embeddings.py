# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Job-local OpenAI-compatible embedding service backed by one CPU model."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memoproagent.backends.openai_compat import SentenceTransformerEmbedding


def handler_for(backend):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ok", "signature": backend.signature})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": backend.name, "object": "model"}]})
            else:
                self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/embeddings":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or request.get("model") != backend.name:
                    raise ValueError("model must match the served embedding model")
                texts = request.get("input")
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
                    raise ValueError("input must be a string or a nonempty string list")
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})
                return
            try:
                # Model execution is serialized; HTTP clients can wait independently.
                with lock:
                    vectors = backend.embed(texts)
                if vectors.shape != (len(texts), backend.dim) or not np.isfinite(vectors).all():
                    raise ValueError("invalid embedding shape or values")
                self.reply(200, {"object": "list", "model": backend.name, "data": [
                    {"object": "embedding", "index": i, "embedding": vector.tolist()}
                    for i, vector in enumerate(vectors)
                ]})
            except Exception as error:
                self.log_error("embedding request failed: %s", error)
                self.reply(500, {"error": "embedding model execution failed"})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.threads < 1:
        parser.error("port must be valid and threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    backend = SentenceTransformerEmbedding(args.model_path, signature_name=args.model_name, device="cpu")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(backend))
    print(f"embedding service ready: {backend.signature} on port {args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
