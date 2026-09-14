# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Corpus integrity (Phase-2 P2-B / DD-3): per-record HMAC-SHA256 MACs.

TERMINOLOGY (naming nit, resolved here): ``tag`` means TWO different things in
this codebase and is never conflated in this module — (1) the ACCESS-POLICY
tag: ``StoredChunk.tag`` / ``IntegrityRecord.tag``, the DSL string from
``src/attributes.py``; (2) the INTEGRITY tag = the MAC = ``tag_d`` = the
``_mac`` meta field: the per-record HMAC-SHA256 value. Docstrings say "policy
tag" for (1) and "MAC"/"integrity tag" for (2). The function names
``compute_tag`` / ``verify_tag`` mean the MAC (context: "HMAC tag").

Primary defence against corpus tampering / poisoning (v2 §3.2, milestone (D)):
every record ``(doc_id, ct, policy tag, meta, version)`` carries a per-doc MAC

    tag_d = HMAC_SHA256(k_int, canonical_json({doc_id, ct, policy tag, meta, version}))

where ``canonical_json`` is the DD-3 BYTE-STABLE serialisation
(``json.dumps(sort_keys=True, separators=(",", ":"))``): identical structured
content MUST produce identical bytes on every machine, so the write-side MAC
and the read-side recomputation always agree. This is the classic cross-process
bug this module exists to prevent; it is tested by its own dedicated test
(``test_canonical_serialization_byte_stable_across_processes``), not just
folded into the tamper-injection battery.

CORPUS-INTEGRITY CLAIMS — feed into P2-I's security analysis VERBATIM:
  * Content SUBSTITUTION / MUTATION is detected loudly: any changed field (ct,
    policy tag, meta value, version) fails the recomputation → EXCLUDE + flag.
  * DELETION is detected ONLY against the client-side EXPECTED-ID MANIFEST:
    ``verify_corpus()`` flags a manifest id that is absent from ``list_ids()``
    as a ``TamperReport(reason="doc_missing")``. The manifest is grown on
    ``put()`` and persisted beside ``integrity.key``; a doc registered but
    never actually written, or a manifest that is itself deleted, is out of
    scope (same trust level as ``store.key`` / the MAC index).
  * AVAILABILITY / censorship by deleting a doc NEVER registered in the
    manifest is indistinguishable from that doc never having existed (silent
    ``None`` at ``get()``). This is the documented, asymmetric residual risk —
    a real gap called out in review and now made explicit, not silent.

Threat-surface honesty: ``k_int`` lives ONLY on the assistant side — it is
never serialised, never written into a record, and never crosses a wire
(P2-D asserts this). The MAC index and the manifest are likewise client-side
roots of trust; a tampered MAC index or manifest is out of scope (same trust
level as ``store.key``).

``CorpusIntegrity`` wraps any store exposing ``put(StoredChunk)`` / ``get`` /
``list_ids`` (LocalStore today, CloudStore in P2-D) so verified reads are
EXCLUDE-AND-FLAG, never silent fallback: a MAC mismatch drops the doc and
appends a ``TamperReport`` to an audit log (JSONL) + in-memory list.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from src.store import StoredChunk

# Reserved meta keys that ride inside a stored chunk's ``meta`` dict but are
# stripped before the user sees them and are never themselves MAC'd.
MAC_META_KEY = "_mac"
VERSION_META_KEY = "_v"

MAC_KEY_BYTES = 32  # HMAC key length (256-bit)
_SHA = "sha256"
MANIFEST_FILENAME = "integrity.manifest.json"


# --------------------------------------------------------------------------- #
# Canonical (byte-stable) serialisation — DD-3
# --------------------------------------------------------------------------- #

def canonical_json(payload: dict) -> str:
    """json.dumps(sort_keys=True, separators=(",", ":")) — byte-stable.

    Nested dict keys are sorted recursively (sort_keys), separators are the
    no-space form, and ``ensure_ascii=True`` keeps the bytes pure-ASCII so the
    same document produces the same BYTES on every machine/locale.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _ct_to_text(ct: Any) -> str:
    """Byte-faithful, unambiguous text form of a ciphertext value.

    Accepts bytes/bytearray (the LocalStore blob form) or an iterable of ints
    (the P2-D cloud record form). The type prefix keeps the two encodings from
    ever colliding into identical MAC inputs.
    """
    if isinstance(ct, (bytes, bytearray)):
        return "B" + bytes(ct).decode("latin-1")
    try:
        body = canonical_json({"ct": [int(x) for x in ct]})
    except (TypeError, ValueError):
        raise TypeError("ct must be bytes or an iterable of ints") from None
    return "L" + body


def mac_input(*, doc_id: str, ct: Any, tag: str, meta: dict,
              version: int) -> bytes:
    """The exact byte string that is MAC'd (DD-3 concatenation).

    One canonical JSON object frames every field, so field boundaries are
    unambiguous (no ``"a1"‖"b"`` vs ``"a"‖"1b"`` class of collision) and
    identical structured content yields identical bytes by construction.
    """
    payload = {
        "doc_id": doc_id,
        "ct": _ct_to_text(ct),
        "tag": tag,
        "meta": dict(meta),
        "version": int(version),
    }
    return canonical_json(payload).encode("utf-8")


def compute_tag(k_int: bytes, *, doc_id: str, ct: Any, tag: str, meta: dict,
                version: int) -> str:
    """``tag_d`` = the integrity tag = HMAC-SHA256(k_int, mac_input(...)), hex."""
    if len(k_int) != MAC_KEY_BYTES:
        raise ValueError(f"k_int must be {MAC_KEY_BYTES} bytes, got {len(k_int)}")
    return hmac.new(k_int, mac_input(doc_id=doc_id, ct=ct, tag=tag,
                                     meta=meta, version=version),
                    hashlib.sha256).hexdigest()


def verify_tag(k_int: bytes, *, doc_id: str, ct: Any, tag: str, meta: dict,
               version: int, mac: str) -> bool:
    """Constant-time compare of a freshly computed MAC against ``mac``."""
    expected = compute_tag(k_int, doc_id=doc_id, ct=ct, tag=tag, meta=meta,
                           version=version)
    return hmac.compare_digest(expected, mac)


# --------------------------------------------------------------------------- #
# Key management (mirrors LocalStore's store.key pattern)
# --------------------------------------------------------------------------- #

class IntegrityKey:
    """File-backed 256-bit HMAC key (``integrity.key`` beside ``store.key``).

    Generated on first open and reused thereafter; callers may pass an explicit
    ``key`` (their responsibility to keep it — if ``key`` is given, the file is
    left untouched/not written).
    """

    FILENAME = "integrity.key"

    def __init__(self, root: str, key: Optional[bytes] = None,
                 key_file: Optional[str] = None):
        self.key_file = key_file or os.path.join(root, self.FILENAME)
        if key is not None:
            if len(key) != MAC_KEY_BYTES:
                raise ValueError(
                    f"explicit k_int must be {MAC_KEY_BYTES} bytes, got {len(key)}")
            self.key = key
            self._persisted = False
        elif os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                self.key = bytes(f.read())
            if len(self.key) != MAC_KEY_BYTES:
                raise ValueError(
                    f"{self.key_file} holds {len(self.key)} bytes, expected "
                    f"{MAC_KEY_BYTES}")
            self._persisted = True
        else:
            self.key = os.urandom(MAC_KEY_BYTES)
            self._persisted = True
            self.save()

    def save(self, path: Optional[str] = None) -> None:
        with open(path or self.key_file, "wb") as f:
            f.write(self.key)

    @property
    def persisted(self) -> bool:
        return self._persisted


# --------------------------------------------------------------------------- #
# Records + tamper report
# --------------------------------------------------------------------------- #

@dataclass
class IntegrityRecord:
    """A verified (or to-be-verified) corpus record, MAC index stripped."""

    doc_id: str
    ct: bytes
    tag: str
    meta: dict
    version: int


@dataclass
class TamperReport:
    """One detected integrity failure (EXCLUDE + flag; never silent)."""

    doc_id: str
    reason: str      # "mac_mismatch" | "missing_mac" | "store_unreadable" |
                     # "doc_missing" (manifest id absent from store)
    ts: float
    note: str = ""


# --------------------------------------------------------------------------- #
# CorpusIntegrity wrapper
# --------------------------------------------------------------------------- #

class CorpusIntegrity:
    """Adds MAC-on-put / verify-on-get around a ``put/get/list_ids`` store.

    The wrapped store sees only ``StoredChunk`` values (blob = ct, policy tag,
    meta with the two reserved MAC keys attached). Verified reads strip the
    reserved keys and hand back an ``IntegrityRecord``; any failure returns
    ``None``, emits a ``TamperReport`` (in-memory + JSONL audit log), and never
    falls back to serving the unverified doc.

    DELETION DETECTION: ``put()`` records the id in a persisted EXPECTED-ID
    manifest; ``verify_corpus()`` flags manifest ids that ``list_ids()`` no
    longer shows as ``TamperReport(reason="doc_missing")``. This is what makes
    deletion loud — for ids that were put() through this wrapper. An id never
    registered stays silently absent (documented residual risk).
    """

    def __init__(self, store: Any, *, root: Optional[str] = None,
                 key: Optional[bytes] = None,
                 log_path: Optional[str] = None,
                 manifest_path: Optional[str] = None):
        self.store = store
        self.root = root or getattr(store, "root", ".")
        self._key = IntegrityKey(self.root, key=key)
        self.log_path = log_path or os.path.join(self.root,
                                                 "integrity_incidents.jsonl")
        self.manifest_path = manifest_path or os.path.join(
            self.root, MANIFEST_FILENAME)
        self.incidents: list[TamperReport] = []
        self._manifest: set[str] = self._load_manifest()

    # -- expected-id manifest (deletion detection) -------------------- #
    def _load_manifest(self) -> set[str]:
        if not os.path.exists(self.manifest_path):
            return set()
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "doc_ids" not in data:
            raise ValueError(f"{self.manifest_path}: malformed manifest")
        return set(data["doc_ids"])

    def _save_manifest(self) -> None:
        payload = canonical_json({"doc_ids": sorted(self._manifest)}) + "\n"
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(payload)

    def manifest_ids(self) -> list[str]:
        return sorted(self._manifest)

    # -- key surface -------------------------------------------------- #
    @property
    def k_int(self) -> bytes:
        return self._key.key

    def key_persisted(self) -> bool:
        return self._key.persisted

    # -- write path --------------------------------------------------- #
    def put(self, record: IntegrityRecord) -> None:
        mac = compute_tag(self.k_int, doc_id=record.doc_id, ct=record.ct,
                          tag=record.tag, meta=record.meta, version=record.version)
        stored_meta = dict(record.meta)
        stored_meta[MAC_META_KEY] = mac
        stored_meta[VERSION_META_KEY] = int(record.version)
        self.store.put(StoredChunk(record.doc_id, record.ct, record.tag,
                                   stored_meta))
        self._manifest.add(record.doc_id)
        self._save_manifest()

    # -- read path (verify or EXCLUDE) -------------------------------- #
    def _flag(self, doc_id: str, reason: str, note: str = "") -> None:
        rep = TamperReport(doc_id=doc_id, reason=reason, ts=time.time(),
                           note=note)
        self.incidents.append(rep)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(canonical_json({
                "doc_id": doc_id, "reason": reason, "ts": rep.ts, "note": note,
            }) + "\n")

    def get(self, doc_id: str) -> IntegrityRecord | None:
        try:
            chunk = self.store.get(doc_id)
        except Exception as exc:  # noqa: BLE001  (e.g. ciphertext invalid)
            self._flag(doc_id, "store_unreadable", f"{type(exc).__name__}: {exc}")
            return None
        if chunk is None:
            return None
        meta = dict(chunk.meta)
        mac = meta.pop(MAC_META_KEY, None)
        version = meta.pop(VERSION_META_KEY, None)
        if mac is None or version is None:
            self._flag(doc_id, "missing_mac", "record carries no MAC/version")
            return None
        version = int(version)
        if not verify_tag(
                self.k_int, doc_id=doc_id, ct=chunk.blob, tag=chunk.tag,
                meta=meta, version=version, mac=mac):
            self._flag(doc_id, "mac_mismatch",
                       "recomputed HMAC != stored MAC (tampered record)")
            return None
        return IntegrityRecord(doc_id=doc_id, ct=chunk.blob, tag=chunk.tag,
                               meta=meta, version=version)

    def verify(self, doc_id: str) -> bool:
        return self.get(doc_id) is not None

    def list_ids(self) -> list[str]:
        return self.store.list_ids()

    def verified_ids(self) -> list[str]:
        """All ids that verify; failures are EXCLUDED and flagged."""
        return [i for i in self.store.list_ids() if self.verify(i)]

    def verify_corpus(self) -> tuple[list[str], list[TamperReport]]:
        """Verify every present id AND flag registered-but-missing ids.

        The ``doc_missing`` check runs first so the returned "fresh" incidents
        include deletions alongside any MAC failures from verification.
        """
        before = len(self.incidents)
        present = set(self.store.list_ids())
        for doc_id in sorted(self._manifest - present):
            self._flag(doc_id, "doc_missing",
                       "in expected-id manifest but absent from store "
                       "(deletion / censorship)")
        ok = self.verified_ids()
        return ok, self.incidents[before:]