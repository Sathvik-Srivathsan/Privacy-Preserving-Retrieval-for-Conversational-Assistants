# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Cloud client (Phase-2 P2-D): HTTP wrappers for the cloud server.

``CloudStore`` provides ``put / get / list_ids / delete`` (matching the
``LocalStore`` interface) over HTTP; ``CloudQuery`` wraps the query path
(derives ``sk_q`` + ``q_ints`` client-side, sends them to the server, receives
scored results).

Both use ``urllib.request`` (stdlib) — no new deps.  Each captured request body
is appended (raw bytes) to ``.captured`` so tests can assert wire properties.

The server never sees ``msk`` or ``k_int``.  ``sk_q`` (the functional key) and
``q_ints`` (the quantised query) are the only query-path disclosures; they
cross the wire by design (DD-2(b)).
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from .ipfe import IPFEScheme
from .store import StoredChunk


def _no_floats(value: object) -> bool:
    """True iff ``value`` contains no float anywhere (wire has ints only)."""
    if isinstance(value, float):
        return False
    if isinstance(value, dict):
        return all(_no_floats(k) and _no_floats(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_no_floats(x) for x in value)
    return True


class CloudStore:
    """HTTP-backed store (mirrors ``LocalStore.put/get/list_ids``)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.captured: list[bytes] = []          # raw request bodies (wire)

    # -- LocalStore-compatible interface ----------------------------------- #

    def put(self, chunk: StoredChunk) -> None:
        body = {
            "doc_id": chunk.chunk_id,
            "ct": json.loads(chunk.blob.decode("utf-8")),
            "tag": chunk.tag,
            "meta": chunk.meta,
        }
        self._request("PUT", "/doc", body)

    def get(self, chunk_id: str) -> Optional[StoredChunk]:
        try:
            data = self._request("GET", f"/doc/{chunk_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        blob = json.dumps(data["ct"], separators=(",", ":")).encode("utf-8")
        return StoredChunk(data["doc_id"], blob, data["tag"], data["meta"])

    def list_ids(self) -> list[str]:
        data = self._request("GET", "/docs")
        return data["doc_ids"]

    def delete(self, chunk_id: str) -> bool:
        self._request("DELETE", f"/doc/{chunk_id}")
        return True

    # -- internal ---------------------------------------------------------- #

    def _request(self, method: str, path: str,
                 body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        payload = (json.dumps(body, separators=(",", ":")).encode("utf-8")
                   if body is not None else None)
        self.captured.append(payload or b"")
        req = urllib.request.Request(
            url, data=payload, method=method,
            headers={"Content-Type": "application/json"} if payload else {},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))


class CloudQuery:
    """Query client: derives sk_q + q_ints locally, sends to cloud server."""

    def __init__(self, base_url: str, scheme: IPFEScheme):
        self.base_url = base_url.rstrip("/")
        self.scheme = scheme
        self.captured: list[bytes] = []          # raw request bodies (wire)

    def setup_server(self) -> None:
        """Send the public IPFE group params to the server via POST /setup."""
        body = {
            "p": self.scheme.p,
            "q": self.scheme.q,
            "g": self.scheme.g,
            "mpk": self.scheme.mpk,
            "vec_len": self.scheme.vec_len,
            "quant_bits": self.scheme.quant_bits,
        }
        self._request("/setup", body)

    def query(self, query_vec: list[float],
              attrs: list[str] | None = None) -> list[dict]:
        """Derive sk_q + q_ints from the float query, POST to server, return
        the authorised top-k as ``[{doc_id, score}, ...]``.

        Server-side authorisation (paper Algorithm 4 / DD-4(4)): the caller's
        own attributes are sent (``attrs``) so the SERVER derives ``Dauth`` --
        the set of docs whose policy tag the attrs satisfy -- and only ever
        scores/returns those.  Docs the user is not authorised for are never
        scored, so the cloud never computes *or* discloses a similarity for
        them.  This mirrors Phase-1 ``RetrievalEngine.rank`` exactly (the same
        ``if not authorized(...): continue`` before scoring), moved cipher-side.

        ``attrs`` is a disclosed-by-design query-path field, the same
        disclosure class as ``sk_q`` (Algorithm 4 requires Au to compute
        Dauth); the roles are the caller's functional access set, not corpus
        content.
        """
        attrs = list(attrs or [])
        sk_q = self.scheme.key_derive(query_vec)
        q_ints = self.scheme.quant(query_vec)
        data = self._request(
            "/query", {"sk_q": sk_q, "q_ints": q_ints, "attrs": attrs})
        return data["results"]

    def shutdown(self) -> None:
        url = f"{self.base_url}/shutdown"
        payload = b"{}"
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass  # server may exit before sending a response

    # -- internal ---------------------------------------------------------- #

    def _request(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.captured.append(payload)
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))