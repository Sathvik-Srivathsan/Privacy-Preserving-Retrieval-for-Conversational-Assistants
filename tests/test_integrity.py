# -*- coding: utf-8 -*-
"""
Integrity / HMAC tagging tests (Phase-2 P2-B, DD-3).

Acceptance criteria (per plan P2-B + green-light addendum):
  * canonical (byte-stable) serialisation is the load-bearing property and gets
    its OWN dedicated test — identical structured content must produce
    identical bytes AND identical MACs across separate processes
    (json.dumps sort_keys=True, separators=(",",":"); the subprocess check
    also proves dict-key insertion order on one side is irrelevant to the
    other). Cross-process assertion: in-process compute_tag == subprocess.
  * tamper harness: N=20 docs, five tamper classes (ct byte / tag / meta value
    / version / doc_id-swap) + an unreadable-ciphertext case -> 0 FALSE
    NEGATIVES on the tampered set and 0 FALSE POSITIVES on a clean set.
  * EXCLUDE-and-flag, no silent fallback: a tampered id is dropped from
    verified_ids() and a TamperReport is recorded (in-memory + JSONL audit).
  * DELETION detection (review-closed gap): put() grows a persisted
    EXPECTED-ID manifest; verify_corpus() flags manifest ids absent from the
    store as reason="doc_missing" — loud for registered-but-deleted docs while
    an id never registered stays silently absent (the documented boundary).
  * terminology: "policy tag" = StoredChunk.tag DSL; "MAC/integrity tag" =
    the per-record HMAC (tag_d, _mac) — never conflated.
  * k_int management mirrors store.key (integrity.key): generated on first
    open, reused thereafter, explicit override without a side-file.
  * k_int never appears inside the MAC'd bytes; verify_tag is a constant-time
    compare and rejects a mismatch on ANY single field including version.

Run:  python tests\\test_integrity.py      (from repo root)
Exit 0 iff ALL_OK. Also pytest-compatible.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.integrity import (                 # noqa: E402
    CorpusIntegrity, IntegrityKey, IntegrityRecord, MAC_META_KEY,
    VERSION_META_KEY, canonical_json, compute_tag, mac_input, verify_tag,
)
from src.store import LocalStore, StoredChunk  # noqa: E402

KEY = b"\x41" * 32          # fixed 256-bit test key
BLOB = bytes(range(16))     # deterministic ct for reproducibility


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _tmp_store():
    tmp = tempfile.mkdtemp(prefix="saferag_integrity_")
    return LocalStore(tmp), tmp


def _doc_id(i, n_width=3):
    return f"d{i:0{n_width}d}"


def _record(doc_id, version=1, seed=None):
    """Deterministic record so the harness is reproducible."""
    base = (seed if seed is not None else int(doc_id[1:]) * 17) % 240
    ct = bytes([base + i for i in range(16)])
    return IntegrityRecord(doc_id=doc_id, ct=ct, tag="role:Nurse",
                           meta={"dim": 8, "seed": base}, version=version)


def _build_corpus(ci, n):
    for i in range(n):
        ci.put(_record(_doc_id(i)))


def _raw_chunk(store, doc_id):
    return store.get(doc_id)


def _tamper(store, doc_id, kind, version=1):
    """Re-put a doc's record (server/wire-side attack) with one field mutated.

    The stored MAC / version meta are the STALE attacker-visible ones, exactly
    as they would be on the wire/cloud after the client's upload.
    """
    c = _raw_chunk(store, doc_id)
    if kind == "ct":
        mut = bytearray(c.blob)
        mut[0] ^= 0xFF
        store.put(StoredChunk(doc_id, bytes(mut), c.tag, c.meta))
    elif kind == "tag":
        store.put(StoredChunk(doc_id, c.blob, c.tag + "-tampered", c.meta))
    elif kind == "meta":
        m = dict(c.meta)
        m["dim"] = int(m["dim"]) + 100
        store.put(StoredChunk(doc_id, c.blob, c.tag, m))
    elif kind == "version":
        m = dict(c.meta)
        m[VERSION_META_KEY] = int(m[VERSION_META_KEY]) + 1
        store.put(StoredChunk(doc_id, c.blob, c.tag, m))
    else:
        raise ValueError(f"unknown tamper kind: {kind}")


def _tamper_swap(store, a, b):
    ca, cb = _raw_chunk(store, a), _raw_chunk(store, b)
    store.put(StoredChunk(a, cb.blob, cb.tag, cb.meta))
    store.put(StoredChunk(b, ca.blob, ca.tag, ca.meta))


def _corrupt_blob_file(root, doc_id):
    p = os.path.join(root, f"{doc_id}.blob")
    data = bytearray(Path(p).read_bytes())
    data[3] ^= 0xFF                      # corrupt the Fernet token
    Path(p).write_bytes(bytes(data))


# --------------------------------------------------------------------------- #
# Canonical (byte-stable) serialization — DD-3 dedicated tests
# --------------------------------------------------------------------------- #

def test_canonical_json_is_compact_and_nested_sorted():
    assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'
    assert canonical_json({"x": {"b": 1, "a": 2}, "y": [1, 2]}) == \
        '{"x":{"a":2,"b":1},"y":[1,2]}'


def test_canonical_hash_is_key_order_invariant():
    meta1 = {"dim": 768, "z": 1, "a": 2.5}
    meta2 = {"a": 2.5, "z": 1, "dim": 768}
    assert canonical_json({"meta": meta1}) == canonical_json({"meta": meta2})
    t1 = compute_tag(KEY, doc_id="d001", ct=BLOB, tag="role:Nurse",
                     meta=meta1, version=8)
    t2 = compute_tag(KEY, doc_id="d001", ct=BLOB, tag="role:Nurse",
                     meta=meta2, version=8)
    assert t1 == t2


def test_canonical_serialization_byte_stable_across_processes():
    """THE DD-3 test: same structured record -> same bytes -> same MAC,
    computed here and in a fresh subprocess (different process, different
    locale-default JSON formatting path, and a deliberately different dict
    insertion order than ours). Any serialisation divergence breaks it."""
    key = b"\x5a" * 32
    doc_id = "d_\u03b1ff"                    # non-ASCII exercised too
    ct = b"\x00\xff\x10\x7f"
    tag = "role:Doctor"
    meta = {"a": 2.5, "dim": 768, "z": 1}    # insertion order A
    version = 7

    in_proc = compute_tag(key, doc_id=doc_id, ct=ct, tag=tag, meta=meta,
                          version=version)

    code = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "from src.integrity import compute_tag;"
        "print(compute_tag(%r, doc_id=%r, ct=%r, tag=%r, meta=%r, version=%r))"
        % (str(REPO), key, doc_id, ct, tag,
           {"dim": 768, "z": 1, "a": 2.5},  # insertion order B in subprocess
           version)
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    subpid_hex = proc.stdout.strip()
    assert len(subpid_hex) == 64 and all(c in "0123456789abcdef"
                                         for c in subpid_hex)
    assert in_proc == subpid_hex, "DD-3: cross-process byte stability violated"


def test_mac_binds_every_field_and_version():
    base = dict(doc_id="d042", ct=BLOB, tag="role:Nurse", meta={"dim": 8},
                version=3)
    original = compute_tag(KEY, **base)
    variants = []
    for name, kw in [
        ("doc_id", {"doc_id": "d043"}),
        ("ct", {"ct": b"\xff\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e"}),
        ("tag", {"tag": "role:Engineer"}),
        ("meta", {"meta": {"dim": 9}}),
        ("version", {"version": 4}),
    ]:
        base2 = dict(base)
        base2.update(kw)
        variants.append((name, compute_tag(KEY, **base2)))
    seen = set()
    for name, t in variants:
        assert t != original, f"{name} change did not alter the MAC"
        assert t not in seen
        seen.add(t)


def test_reupload_version_bump_yields_new_mac():
    t1 = compute_tag(KEY, doc_id="d001", ct=BLOB, tag="role:Nurse",
                     meta={"dim": 8}, version=1)
    t2 = compute_tag(KEY, doc_id="d001", ct=BLOB, tag="role:Nurse",
                     meta={"dim": 8}, version=2)
    assert t1 != t2


def test_k_int_never_inside_mac_bytes():
    blob_str = BLOB.decode("latin-1")
    assert KEY not in mac_input(doc_id="d001", ct=BLOB, tag="role:Nurse",
                                meta={"dim": 8}, version=1)
    assert blob_str.encode("latin-1") == BLOB    # sanity for review
    assert str(KEY) not in repr(mac_input(doc_id="d001", ct=BLOB,
                                          tag="role:Nurse", meta={"dim": 8},
                                          version=1))


# --------------------------------------------------------------------------- #
# compute_tag / verify_tag unit behaviour
# --------------------------------------------------------------------------- #

def test_compute_verify_roundtrip():
    mac = compute_tag(KEY, doc_id="d007", ct=BLOB, tag="role:Nurse",
                      meta={"dim": 8}, version=1)
    assert len(mac) == 64
    assert verify_tag(KEY, doc_id="d007", ct=BLOB, tag="role:Nurse",
                      meta={"dim": 8}, version=1, mac=mac)


def test_verify_rejects_wrong_mac_or_any_field():
    mac = compute_tag(KEY, doc_id="d007", ct=BLOB, tag="role:Nurse",
                      meta={"dim": 8}, version=1)
    assert not verify_tag(b"\x42" * 32, doc_id="d007", ct=BLOB,
                          tag="role:Nurse", meta={"dim": 8}, version=1,
                          mac=mac)
    assert not verify_tag(KEY, doc_id="d007", ct=BLOB, tag="role:Nurse",
                          meta={"dim": 8}, version=1,
                          mac="0" * 64)


def test_compute_tag_rejects_wrong_key_length():
    try:
        compute_tag(b"\x00" * 16, doc_id="d", ct=BLOB, tag="t",
                    meta={}, version=1)
    except ValueError:
        pass
    else:
        assert False, "16-byte key must be rejected (k_int = 256-bit)"


# --------------------------------------------------------------------------- #
# IntegrityKey management (mirrors store.key)
# --------------------------------------------------------------------------- #

def test_integrity_key_generated_persisted_reused():
    root = tempfile.mkdtemp(prefix="saferag_key_")
    k1 = IntegrityKey(root)
    kf = os.path.join(root, IntegrityKey.FILENAME)
    assert os.path.exists(kf)
    assert k1.persisted
    assert len(k1.key) == 32
    k2 = IntegrityKey(root)          # fresh instance, same dir, no key arg
    assert k2.key == k1.key           # reused, not regenerated


def test_integrity_key_explicit_override_no_side_file():
    root = tempfile.mkdtemp(prefix="saferag_key2_")
    explicit = bytes(range(32))
    k = IntegrityKey(root, key=explicit)
    assert not k.persisted
    assert k.key == explicit
    assert not os.path.exists(os.path.join(root, IntegrityKey.FILENAME))


def test_integrity_key_wrong_length_rejected():
    root = tempfile.mkdtemp(prefix="saferag_key3_")
    try:
        IntegrityKey(root, key=b"\x00" * 16)
    except ValueError:
        pass
    else:
        assert False, "short explicit key must be rejected"


# --------------------------------------------------------------------------- #
# CorpusIntegrity wrapper
# --------------------------------------------------------------------------- #

def test_corpus_integrity_roundtrip_strips_reserved_meta():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    rec = _record("d001")
    ci.put(rec)
    got = ci.get("d001")
    assert got is not None
    assert got.doc_id == rec.doc_id
    assert got.ct == rec.ct
    assert got.tag == rec.tag
    assert got.meta == rec.meta                      # reserved keys stripped
    assert got.version == rec.version
    stored = s.get("d001")                           # raw layer sees reserved
    assert MAC_META_KEY in stored.meta and VERSION_META_KEY in stored.meta
    assert MAC_META_KEY not in got.meta


def test_corpus_integrity_missing_mac_flagged():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    s.put(StoredChunk("d009", BLOB, "role:Nurse", {"dim": 8}))  # no MAC
    assert ci.get("d009") is None
    assert ci.verify("d009") is False
    assert ci.incidents and ci.incidents[0].reason == "missing_mac"


def test_zfc_tamper_harness_20_docs_zero_false_negatives():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 20)
    kinds = ["ct", "tag", "meta", "version"]         # five each across 20
    for i in range(20):
        _tamper(s, _doc_id(i), kinds[i % 4])
    for i in range(20):
        assert ci.get(_doc_id(i)) is None, f"{_doc_id(i)} not detected"
    assert len(ci.incidents) == 20
    reasons = {r.reason for r in ci.incidents}
    assert reasons == {"mac_mismatch"}
    assert [r.doc_id for r in ci.incidents] == \
        [_doc_id(i) for i in range(20)]


def test_zfp_tamper_harness_20_docs_zero_false_positives():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 20)
    for i in range(20):
        got = ci.get(_doc_id(i))
        assert got is not None, f"{_doc_id(i)} flagged though clean"
        assert got.version == 1
    assert ci.incidents == []
    assert ci.verified_ids() == [_doc_id(i) for i in range(20)]


def test_docid_swap_detected():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 2)
    _tamper_swap(s, "d000", "d001")                  # contents exchanged
    assert ci.get("d000") is None
    assert ci.get("d001") is None
    assert {r.doc_id for r in ci.incidents} == {"d000", "d001"}


def test_corrupt_blob_file_flagged_unreadable():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    ci.put(_record("d020"))
    _corrupt_blob_file(root, "d020")                 # Fernet token damaged
    assert ci.get("d020") is None
    assert ci.incidents and ci.incidents[0].reason == "store_unreadable"


# --------------------------------------------------------------------------- #
# Deletion detection (review-closed gap): expected-id manifest -> doc_missing
# --------------------------------------------------------------------------- #

def test_deletion_detected_as_doc_missing():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 3)
    os.remove(os.path.join(root, "d001.blob"))       # attacker deletes a doc
    ok, fresh = ci.verify_corpus()
    assert set(ok) == {"d000", "d002"}
    assert [(r.doc_id, r.reason) for r in fresh] == [("d001", "doc_missing")]
    assert ci.manifest_ids() == ["d000", "d001", "d002"]


def test_deletion_vs_never_existed_now_distinguishable():
    """Review wording: absence must no longer silently read as "never had it"."""
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 2)
    os.remove(os.path.join(root, "d000.blob"))       # registered, then deleted
    assert ci.get("d000") is None                    # silent at get() level
    assert ci.get("d999") is None                    # d999 was never registered
    assert ci.incidents == []                        # absence alone never flags
    ok, fresh = ci.verify_corpus()
    assert set(ok) == {"d001"}                       # d999 not an expected id
    assert [(r.doc_id, r.reason) for r in fresh] == [("d000", "doc_missing")]
    assert "d999" not in ci.manifest_ids()


def test_manifest_persisted_across_instances():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 2)
    os.remove(os.path.join(root, "d001.blob"))
    reopen = CorpusIntegrity(s, root=root, key=KEY)  # fresh instance, same dir
    assert reopen.manifest_ids() == ["d000", "d001"]
    ok, fresh = reopen.verify_corpus()
    assert [r.doc_id for r in fresh if r.reason == "doc_missing"] == ["d001"]


def test_unregistered_raw_doc_deletion_silent_boundary():
    """Manifest only protects ids put() through this wrapper (documented)."""
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    s.put(StoredChunk("raw1", BLOB, "role:Nurse", {"dim": 8}))  # bypasses put()
    os.remove(os.path.join(root, "raw1.blob"))
    assert ci.get("raw1") is None
    ok, fresh = ci.verify_corpus()
    assert "raw1" not in ci.manifest_ids()
    assert all(r.reason != "doc_missing" for r in fresh)  # out-of-scope, explicit


def test_injected_unregistered_doc_is_excluded_loudly():
    """Bypass-ARRIVAL is loud: a doc that appears without a valid MAC is not
    served. This is the mechanism side of the P2-D ingestion contract — the
    routing side (docs MUST reach the store via put(), never raw store.put)
    is enforced at P2-D build; see plan P2-D CorpusIngress contract."""
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY)
    _build_corpus(ci, 2)
    anon1 = StoredChunk("anon1", BLOB, "role:Nurse", {"dim": 8})  # no MAC
    anon2 = StoredChunk("anon2", BLOB, "role:Nurse", {"dim": 8})
    anon2.meta[MAC_META_KEY] = "0" * 64                 # guessed MAC, no k_int
    anon2.meta[VERSION_META_KEY] = 1
    s.put(anon1)
    s.put(anon2)
    ok, fresh = ci.verify_corpus()
    assert set(ok) == {"d000", "d001"}                  # injected ids not served
    assert ci.manifest_ids() == ["d000", "d001"]        # never registered
    assert {r.doc_id for r in fresh} == {"anon1", "anon2"}
    assert {r.reason for r in fresh} == {"missing_mac", "mac_mismatch"}


def test_verified_ids_excludes_and_flags_no_silent_fallback():
    s, root = _tmp_store()
    ci = CorpusIntegrity(s, root=root, key=KEY, log_path=os.path.join(
        root, "incidents.jsonl"))
    _build_corpus(ci, 5)
    tampered = ["d000", "d002", "d004"]
    _tamper(s, "d000", "ct", version=1)
    _tamper(s, "d002", "tag", version=1)
    _tamper(s, "d004", "version", version=1)
    ok, fresh = ci.verify_corpus()
    assert set(ok) == {"d001", "d003"}               # tampered ids EXCLUDED
    assert {r.doc_id for r in fresh} == set(tampered)
    assert len(ci.incidents) == 3


def test_incident_log_jsonl_audit_trail():
    s, root = _tmp_store()
    log = os.path.join(root, "audit.jsonl")
    ci = CorpusIntegrity(s, root=root, key=KEY, log_path=log)
    _build_corpus(ci, 3)
    _tamper(s, "d000", "ct", version=1)
    _tamper(s, "d001", "version", version=1)
    assert ci.verify("d000") is False and ci.verify("d001") is False
    assert os.path.exists(log)
    lines = [json.loads(l) for l in Path(log).read_text().splitlines()
             if l.strip()]
    assert len(lines) == 2
    assert {line["doc_id"] for line in lines} == {"d000", "d001"}
    assert all(line["reason"] == "mac_mismatch" for line in lines)
    assert all(isinstance(line["ts"], float) for line in lines)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    fails = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fails.append((name, repr(exc)))
    report = {
        "module": "integrity",
        "tests": len(tests),
        "passed": len(tests) - len(fails),
        "failed": len(fails),
        "failures": [f[0] for f in fails],
        "all_ok": not fails,
    }
    print(json.dumps(report, indent=2))
    print(f"ALL_OK: {not fails}   tests: {len(tests)}   passed: {len(tests) - len(fails)}")
    if fails:
        for name, exc in fails:
            print(f"FAIL {name}: {exc}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())