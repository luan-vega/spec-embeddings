import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


class FakeOllama:
    def __init__(self, dim=4, models=("nomic-embed-text:latest",), embed_status=200, embeddings=None, generate_response=None):
        self.dim, self.models, self.embed_status, self.embeddings = dim, models, embed_status, embeddings
        self.generate_response = generate_response
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.requests.append(("GET", self.path, None))
                self._send(200, {"models": [{"name": m, "digest": "abc123"} for m in outer.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(("POST", self.path, body))
                if self.path == "/api/show":
                    return self._send(200, {"model_info": {"bert.context_length": 2048}})
                if self.path == "/api/generate":
                    if outer.generate_response is None:
                        return self._send(500, {"error": "x"})
                    return self._send(200, {"response": outer.generate_response})
                if outer.embed_status != 200:
                    return self._send(outer.embed_status, {"error": "x"})
                vecs = outer.embeddings or [outer.vector(t) for t in body["input"]]
                self._send(200, {"embeddings": vecs})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def vector(self, text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        return [float(x) for x in np.random.default_rng(seed).standard_normal(self.dim)]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
