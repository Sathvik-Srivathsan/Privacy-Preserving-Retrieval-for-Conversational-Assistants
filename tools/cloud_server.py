#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cloud server (Phase-2 P2-D / DD-1): stdlib HTTP, cipher-side only.

Bind ``127.0.0.1:0`` (ephemeral port), print port to stdout so the test
fixture can capture it.  JSON-line protocol (every request/response body
is ``application/json``).

Endpoints
---------
PUT    /doc      body: {doc_id, ct:[int], tag, meta}      -> {"ok":true}
GET    /docs                                          -> {"doc_ids":[...]}
GET    /doc/<id>                                      -> {doc_id, ct, tag, meta}
DELETE /doc/<id>                                      -> {"ok":true}
POST   /setup   body: {p, q, g, mpk, vec_len, quant_bits}
                                                        -> {"ok":true}
POST   /query   body: {sk_q:int, q_ints:[int], attrs:[str]}
                                                        -> {"results":[{doc_id,score}...]}  (authorised docs only, desc)
POST   /shutdown                                      -> {"ok":true} + exit(0)

Server-side compute mirrors Phase-1 ``RetrievalEngine.rank`` semantics but
CIPHER-SIDE (``IPFEScheme(msk=None)``); it never sees ``msk``.  Authorisation
is SERVER-side (paper Algorithm 4): the caller's ``attrs`` cross the wire in
the query body so the server derives ``Dauth`` and only scores/returns docs
whose policy tag the attrs satisfy.

Wire-assertion (tested in ``tests/test_cloud.py``): captured body bytes
contain NONE of the corpus content words, no float q coefficients, no ``msk``,
no ``k_int``.  ``sk_q`` IS on the wire (it is the functional key, disclosed
by design — DD-2(b)).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.ipfe import IPFEScheme                 # noqa: E402
from src.store import StoredChunk               # noqa: E402
from src.attributes import build_tree, Attributes  # noqa: E402

# ---------------------------------------------------------------------------
# Shared mutable state (single-threaded server, one request at a time)
# ---------------------------------------------------------------------------

_scheme: IPFEScheme | None = None               # set by POST /setup
_store: dict[str, StoredChunk] = {}             # doc_id -> StoredChunk
_lock = threading.Lock()                        # guard _store for safety


class _Handler(BaseHTTPRequestHandler):
    """JSON-line request handler."""

    # Suppress default access-log noise.
    def log_message(self, fmt, *args):           # noqa: D401
        pass

    # -- helpers ----------------------------------------------------------- #

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # -- routing ----------------------------------------------------------- #

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._respond(200, {"ok": True})
        elif path == "/docs":
            with _lock:
                ids = sorted(_store.keys())
            self._respond(200, {"doc_ids": ids})
        elif path.startswith("/doc/"):
            doc_id = path[len("/doc/"):]
            with _lock:
                chunk = _store.get(doc_id)
            if chunk is None:
                self._respond(404, {"error": "not found"})
            else:
                self._respond(200, {
                    "doc_id": chunk.chunk_id,
                    "ct": json.loads(chunk.blob.decode("utf-8")),
                    "tag": chunk.tag,
                    "meta": chunk.meta,
                })
        else:
            self._respond(404, {"error": "unknown path"})

    def do_PUT(self):
        path = urlparse(self.path).path
        if path != "/doc":
            self._respond(404, {"error": "unknown path"})
            return
        data = self._read_json()
        doc_id = data["doc_id"]
        ct = data["ct"]             # list[int]
        tag = data.get("tag", "")
        meta = data.get("meta", {})
        blob = json.dumps(ct, separators=(",", ":")).encode("utf-8")
        chunk = StoredChunk(doc_id, blob, tag, meta)
        with _lock:
            _store[doc_id] = chunk
        self._respond(200, {"ok": True})

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/doc/"):
            self._respond(404, {"error": "unknown path"})
            return
        doc_id = path[len("/doc/"):]
        with _lock:
            gone = _store.pop(doc_id, None) is not None
        if not gone:
            self._respond(404, {"error": "not found"})
        else:
            self._respond(200, {"ok": True})

    def do_POST(self):
        global _scheme
        path = urlparse(self.path).path

        if path == "/setup":
            data = self._read_json()
            mpk = [int(x) for x in data["mpk"]]
            _scheme = IPFEScheme(
                p=int(data["p"]),
                q=int(data["q"]),
                g=int(data["g"]),
                mpk=mpk,
                msk=None,                       # cipher-side only
                vec_len=int(data["vec_len"]),
                quant_bits=int(data.get("quant_bits", 2)),
            )
            self._respond(200, {"ok": True})

        elif path == "/query":
            if _scheme is None:
                self._respond(400, {"error": "call /setup first"})
                return
            data = self._read_json()
            sk_q = int(data["sk_q"])
            q_ints = [int(x) for x in data["q_ints"]]
            attrs = Attributes(data.get("attrs", []) or [])
            with _lock:
                docs = list(_store.values())
            # Server-side authorisation gate (paper Algorithm 4: Dauth computed
            # from Au and each doc's policy tree BEFORE any scoring).  Docs the
            # caller is not authorised for are never scored, so their similarity
            # is never disclosed to anyone -- matches RetrievalEngine.rank.
            authorized = []
            for chunk in docs:
                policy = build_tree(chunk.tag)
                if not policy.satisfies(attrs):
                    continue
                authorized.append(chunk)
            results = []
            for chunk in authorized:
                ct = json.loads(chunk.blob.decode("utf-8"))
                score = _scheme.inner_product_q(ct, sk_q, q_ints)
                results.append({"doc_id": chunk.chunk_id, "score": score})
            results.sort(key=lambda r: r["score"], reverse=True)
            self._respond(200, {"results": results})

        elif path == "/shutdown":
            self._respond(200, {"ok": True})
            threading.Thread(target=lambda: os._exit(0)).start()

        else:
            self._respond(404, {"error": "unknown path"})


def main():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    # Print port to stdout so the subprocess fixture can capture it.
    sys.stdout.write(f"port={port}\n")
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
