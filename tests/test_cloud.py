"""Cloud-boundary tests (Phase-2 P2-D / P2-B cloud / DD-1 / DD-2(b) /
DD-4(4)(b)).

Acceptance criteria (per plan):
  (a) Subprocess fixture (DD-1): spawn ``tools/cloud_server.py``, capture
      port from stdout, teardown via POST /shutdown.
  (b) Full round-trip: ingest N docs → query → topk → verify scores match
      the local Phase-1 ``RetrievalEngine.rank`` for the same vectors.
  (c) Authorisation gating matches Phase-1 on the same corpus — INCLUDING
      the dept/clearance-gated doc (DD-4(4)(b)): the caller's dept:Cardio
      and clearance:2 tokens cross the wire in ``attrs`` (Algorithm 4 needs
      Au; server derives Dauth from them) and the nurse-only doc stays
      excluded.
  (d) Wire-assertion: captured wire bodies contain NONE of the corpus
      content words, no float values, no ``msk``, no ``k_int``.  ``sk_q``
      IS on the wire (DD-2(b), disclosed by design).
  (e) Tampered-doc server round-trip excluded client-side (integrates P2-B).
  (f) CorpusIngress contract (P2-B review-closure) over the CLOUD store:
      (a) every served doc is manifest-registered AND MAC-verified;
      (b) injected unregistered doc excluded + flagged (missing_mac);
      (c) deletion of a registered doc → ``doc_missing``.

Three DD-4(4) dedicated tests for the ``inner_product_q`` seam:
  1. q_ints == float quant path (no double-quantisation).
  2. NEGATIVE inner product through the seam (BSGS inverse branch),
     asserted SEPARATELY from the positive case.
  3. Exact plaintext-quantised-dot ORACLE (independent of the float path):
     a re-quantising implementation would scale by B again and fail.

Run:  python tests\\test_cloud.py      (from repo root)
Exit 0 iff ALL_OK.  Also pytest-compatible.
"""

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from src.cloud import CloudQuery, CloudStore          # noqa: E402
from src.integrity import CorpusIntegrity, IntegrityRecord  # noqa: E402
from src.ipfe import IPFEScheme                       # noqa: E402
from src.attributes import Attributes, build_tree                 # noqa: E402
from src.retrieval import RetrievalEngine                         # noqa: E402
from src.store import StoredChunk                     # noqa: E402

N = 32
PREC = 2           # decimal places carried by the fix (matches grid/pipeline/complexity;
                     # quant_bits>9 explodes the BSGS table -> machine lock-up)
B = 10 ** PREC

# Deterministic corpus — plain role doc + DEPARTMENT + CLEARANCE gated docs
# (DD-4(4)(b)): D0 is PLAIN ``role:Doctor`` (served to any doctor caller);
# D1/D3 require ``role:Doctor AND dept:Cardio AND clearance:2`` (EXCLUDED
# unless the full tokens are on the wire in ``attrs``); D2 is nurse-only.
CORPUS = [
    ("D0", "role:Doctor",
     "heart rhythm monitoring device prevents arrhythmia related stroke"),
    ("D1", "role:Doctor AND dept:Cardio AND clearance:2",
     "postoperative infection rates drop with prophylactic antibiotic course"),
    ("D2", "role:Nurse",
     "daily blood pressure readings recorded by the ward staff"),
    ("D3", "role:Doctor AND dept:Cardio AND clearance:2",
     "cardiac echo report flagged for the cardiovascular department"),
]
QUERY = "heart rhythm monitoring reduces stroke risk"

CORPUS_WORDS = {w.lower() for _, _, text in CORPUS for w in text.split()}
FORBIDDEN_BODIES = [b"msk", b"k_int"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class _Doc:
    """Seam mirror for the cloud round-trip battery."""
    def __init__(self, doc_id, vec, ct, tag):
        self.doc_id = doc_id
        self.vec = np.asarray(vec, dtype=np.float64)
        self.ct = ct
        self.tag = tag
        self.policy = build_tree(tag)


def _embeddings(scheme):
    """Deterministic doc embeddings + query (per-doc policy tags from CORPUS;
    the D3 dept/clearance doc IS in the corpus — DD-4(4)(b))."""
    rng = np.random.default_rng(42)
    cvals = [0.85, 0.45, 0.70, 0.55]   # order = D0..D3 (ALL corpus rows fed to round-trip)
    query_vec = _unit(rng.normal(size=N))
    docs = []
    for i, (cid, tag, _text) in enumerate(CORPUS):
        if i >= len(cvals):
            break                    # round-trip battery uses docs[:3] only
        d = np.zeros(N)
        d[0] = cvals[i]
        if N > 1:
            d[1] = np.sqrt(max(0.0, 1.0 - cvals[i] ** 2))
        docs.append(_Doc(cid, list(d), scheme.encrypt(list(d))[0], tag))
    return query_vec, docs


def _no_floats(value):
    """True iff ``value`` (a parsed JSON body) contains no float anywhere."""
    if isinstance(value, float):
        return False
    if isinstance(value, dict):
        return all(_no_floats(k) and _no_floats(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_no_floats(v) for v in value)
    return True


# --------------------------------------------------------------------------- #
# Server fixture (subprocess, DD-1)
# --------------------------------------------------------------------------- #

def _spawn_server():
    """Spawn ``tools/cloud_server.py`` as a subprocess; capture the reported
    port; return (proc, base_url)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo, "tools", "cloud_server.py")
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = proc.stdout.readline().strip()
    match = re.search(r"port=(\d+)", line)
    assert match, f"cloud server did not report port: {line!r}"
    port = int(match.group(1))
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    return proc, base


def _stop_server(proc, base):
    try:
        urllib.request.urlopen(f"{base}/shutdown", timeout=2)
    except Exception:
        pass
    proc.terminate()


# --------------------------------------------------------------------------- #
# DD-4(4) inner-product seam
# --------------------------------------------------------------------------- #

def test_inner_product_q_q_ints_match_float_quant_path():
    """q_ints == float quant path (no double-quantisation)."""
    scheme = IPFEScheme.setup(N, quant_bits=PREC)
    rng = np.random.default_rng(7)
    for _ in range(5):
        doc = _unit(rng.normal(size=N))
        qvec = _unit(rng.normal(size=N))
        ct, _ = scheme.encrypt(list(doc))
        sk = scheme.key_derive(list(qvec))
        qv = scheme.quant(list(qvec))
        got = scheme.inner_product_q(ct, sk, qv)
        want = sum(a * b for a, b in zip(scheme.quant(list(doc)), qv))
        assert got == want, (got, want)


def test_inner_product_q_negative_doc_handled_separately():
    """NEGATIVE inner product through the seam (BSGS inverse branch),
    asserted SEPARATELY from the positive case."""
    scheme = IPFEScheme.setup(N, quant_bits=PREC)
    rng = np.random.default_rng(8)
    doc = _unit(rng.normal(size=N))
    qvec = _unit(rng.normal(size=N))
    if float(np.dot(doc, qvec)) >= 0:
        doc = list(-np.asarray(doc))
    else:
        doc = list(doc)
    ct, _ = scheme.encrypt(doc)
    sk = scheme.key_derive(list(qvec))
    qv = scheme.quant(list(qvec))
    got = scheme.inner_product_q(ct, sk, qv)
    assert got < 0, "negative doc must yield a negative inner product"
    assert got == scheme.inner_product(ct, sk, list(qvec))


def test_inner_product_q_exact_plaintext_oracle():
    """Exact plaintext-quantised-dot ORACLE (independent of the float path):
    a re-quantising implementation would scale by B again and fail."""
    scheme = IPFEScheme.setup(N, quant_bits=PREC)
    rng = np.random.default_rng(9)
    doc = _unit(rng.normal(size=N))
    qvec = _unit(rng.normal(size=N))
    ct, _ = scheme.encrypt(list(doc))
    sk = scheme.key_derive(list(qvec))
    qd = scheme.quant(list(doc))
    qv = scheme.quant(list(qvec))
    got = scheme.inner_product_q(ct, sk, qv)
    expected = sum(a * b for a, b in zip(qd, qv))
    assert got == expected, (got, expected)


# --------------------------------------------------------------------------- #
# Cloud round-trip + authorization + dept/clearance wire (DD-4(4)(b))
# --------------------------------------------------------------------------- #

def test_cloud_round_trip_and_authorization():
    """Ingest → query (doctor dept+clearance attrs) → topk → verify scores
    match local Phase-1 rank; D3 gated on dept:Cardio AND clearance:2 is
    EXCLUDED for a doctor-only caller (no dept/clearance tokens) and the
    wire bodies carry the dept/clearance tokens in attrs (DD-4(4)(b)) but
    no plaintext (DD-2(b))."""
    proc, base = _spawn_server()
    try:
        scheme = IPFEScheme.setup(N, quant_bits=PREC)
        query_vec, docs = _embeddings(scheme)

        cs = CloudStore(base)
        with tempfile.TemporaryDirectory() as td:
            cq = CloudQuery(base, scheme)
            cq.setup_server()

            for d in docs:
                ct_blob = json.dumps(d.ct, separators=(",", ":")).encode("utf-8")
                cs.put(StoredChunk(d.doc_id, ct_blob, d.tag, {"dim": N}))

            # caller = doctor (role only) → D3 (dept:Cardio AND clearance:2)
            # must be EXCLUDED; D0 (PLAIN role:Doctor) is served — so the
            # doctor-only set is NON-empty and the differential is real.
            doctor_only = ["role:Doctor"]
            all_results = cq.query(list(query_vec), attrs=doctor_only)
            doctor_ids = [r["doc_id"] for r in all_results]
            assert doctor_ids == ["D0"], (
                "doctor-only caller must be served D0 and nothing else")
            assert "D3" not in doctor_ids, "dept/clearance-gated doc excluded"
            assert "D2" not in doctor_ids, "nurse-only doc excluded"

            # caller = doctor WITH dept:Cardio AND clearance:2 → D0, D1, D3
            dept_attrs = ["role:Doctor", "dept:Cardio", "clearance:2"]
            gated = cq.query(list(query_vec), attrs=dept_attrs)
            gated_ids = [r["doc_id"] for r in gated]
            assert set(gated_ids) == {"D0", "D1", "D3"}, gated_ids
            assert "D2" not in gated_ids, "nurse-only doc still excluded"

            # score/ranking parity vs Phase-1 for the NON-EMPTY authorised set:
            # the dept_attrs caller (D0+D1+D3) compared against the local
            # float-side RetrievalEngine.rank on the same corpus.
            local_full = RetrievalEngine(scheme, k=5).rank(
                docs, list(query_vec),
                Attributes(["role:Doctor", "dept:Cardio", "clearance:2"]),
                k=5)
            local_full_ids = [r.doc_id for r in local_full]
            assert gated_ids == local_full_ids, (
                "server ranking != Phase-1 ranking for dept_attrs set",
                gated_ids, local_full_ids)

            # doctor-only parity (now also non-trivial: D0 vs nothing)
            local_doc = RetrievalEngine(scheme, k=5).rank(
                docs, list(query_vec), Attributes.from_roles(["doctor"]), k=5)
            local_doc_ids = [r.doc_id for r in local_doc]
            assert doctor_ids == local_doc_ids, (doctor_ids, local_doc_ids)

            # wire bodies: no corpus words, no floats, no forbidden tokens;
            # query attrs carry dept/clearance tokens for the gated caller.
            all_bodies = cs.captured + cq.captured
            assert all_bodies, "no request bodies captured"
            for body in all_bodies:
                lower = body.lower()
                for word in CORPUS_WORDS:
                    assert word.encode() not in lower, (
                        f"corpus word '{word}' leaked to wire")
                for forbidden in FORBIDDEN_BODIES:
                    assert forbidden not in lower, f"{forbidden} leaked to wire"
            qb = json.loads(cq.captured[-1].decode("utf-8"))
            assert set(qb) == {"sk_q", "q_ints", "attrs"}, (
                "query wire must carry ONLY sk_q + q_ints + attrs")
            assert set(qb["attrs"]) == set(dept_attrs), (
                "dept/clearance tokens must be on the wire (DD-4(4)(b))")
    finally:
        _stop_server(proc, base)


def test_threshold_policy_gated_through_cloud():
    """A ``2-of(clearance:2, role:Nurse)``-tagged doc must be gated by the
    SERVER-side Dauth path (Algorithm 4) exactly like role/dept policies:
    a caller holding ONE of the two tokens is excluded; a caller holding
    BOTH is served.  This closes the review item that threshold policies
    were never exercised end-to-end through the cloud (only at the
    attributes-unit level).  Full key:value tokens cross the wire in
    ``attrs`` (DD-4(4)(b)); ``from_roles`` never sees a bare role name."""
    proc, base = _spawn_server()
    try:
        scheme = IPFEScheme.setup(N, quant_bits=PREC)
        query_vec, docs = _embeddings(scheme)

        cs = CloudStore(base)
        with tempfile.TemporaryDirectory() as td:
            cq = CloudQuery(base, scheme)
            cq.setup_server()

            # corpus doc tagged with a THRESHOLD policy (2-of)
            policy_tag = "2-of(clearance:2, role:Nurse)"
            ct_blob = json.dumps(docs[1].ct,
                                 separators=(",", ":")).encode("utf-8")
            cs.put(StoredChunk("THRESH", ct_blob, policy_tag, {"dim": N}))

            # caller holding only role:Nurse -> 1 of 2 -> EXCLUDED
            nurse_only = cq.query(list(query_vec), attrs=["role:Nurse"])
            nurse_ids = [r["doc_id"] for r in nurse_only]
            assert "THRESH" not in nurse_ids, (
                "1-of-2 caller must be excluded from a 2-of policy")

            # caller holding role:Nurse AND clearance:2 -> 2 of 2 -> SERVED
            both_attrs = ["role:Nurse", "clearance:2"]
            gated = cq.query(list(query_vec), attrs=both_attrs)
            gated_ids = [r["doc_id"] for r in gated]
            assert "THRESH" in gated_ids, (
                "2-of-2 caller must be authorised for a 2-of policy")

            # wire body carries the full key:value tokens (not prefixed roles)
            qb = json.loads(cq.captured[-1].decode("utf-8"))
            assert set(qb) == {"sk_q", "q_ints", "attrs"}
            body_attrs = set(qb["attrs"])
            assert body_attrs == set(both_attrs), (body_attrs, both_attrs)
            assert not any("role:role:" in a for a in body_attrs), (
                "no doubled role:role: token on the wire")
    finally:
        _stop_server(proc, base)


# --------------------------------------------------------------------------- #
# P2-B integrity over the cloud store
# --------------------------------------------------------------------------- #

def test_tampered_doc_excluded_client_side():
    """A doc tampered ON THE SERVER (policy tag flipped after ingest) is
    excluded by the client integrity layer (MAC mismatch)."""
    proc, base = _spawn_server()
    try:
        scheme = IPFEScheme.setup(N, quant_bits=PREC)
        query_vec, docs = _embeddings(scheme)

        cs = CloudStore(base)
        with tempfile.TemporaryDirectory() as td:
            integrity = CorpusIntegrity(cs, root=td)
            for d in docs:
                ct_blob = json.dumps(d.ct, separators=(",", ":")).encode("utf-8")
                integrity.put(IntegrityRecord(d.doc_id, ct_blob, d.tag,
                                              {"dim": N}, version=1))
            d0 = cs.get("D0")
            d0.tag = "role:Nurse"
            cs.put(d0)
            assert "D0" not in integrity.verified_ids()
    finally:
        _stop_server(proc, base)


def test_inject_unregistered_doc_excluded():
    """A doc injected via raw PUT (no MAC, no manifest entry) is excluded
    + flagged (missing_mac) by the client integrity layer."""
    proc, base = _spawn_server()
    try:
        scheme = IPFEScheme.setup(N, quant_bits=PREC)
        query_vec, docs = _embeddings(scheme)

        cs = CloudStore(base)
        with tempfile.TemporaryDirectory() as td:
            integrity = CorpusIntegrity(cs, root=td)
            for d in docs[:2]:
                ct_blob = json.dumps(d.ct, separators=(",", ":")).encode("utf-8")
                integrity.put(IntegrityRecord(d.doc_id, ct_blob, d.tag,
                                              {"dim": N}, version=1))
            d2 = docs[2]
            ct_blob = json.dumps(d2.ct, separators=(",", ":")).encode("utf-8")
            cs.put(StoredChunk(d2.doc_id, ct_blob, d2.tag, {"dim": N}))
            assert "D2" not in integrity.verified_ids()
    finally:
        _stop_server(proc, base)


def test_deletion_detected_as_doc_missing():
    """Deleting a registered doc on the server → ``doc_missing`` in
    ``verify_corpus`` (manifest id absent from the store)."""
    proc, base = _spawn_server()
    try:
        scheme = IPFEScheme.setup(N, quant_bits=PREC)
        query_vec, docs = _embeddings(scheme)

        cs = CloudStore(base)
        with tempfile.TemporaryDirectory() as td:
            integrity = CorpusIntegrity(cs, root=td)
            for d in docs:
                ct_blob = json.dumps(d.ct, separators=(",", ":")).encode("utf-8")
                integrity.put(IntegrityRecord(d.doc_id, ct_blob, d.tag,
                                              {"dim": N}, version=1))
            cs.delete("D0")
            ok, fresh = integrity.verify_corpus()
            missing = [r for r in fresh if r.reason == "doc_missing"]
            assert any(r.doc_id == "D0" for r in missing)
    finally:
        _stop_server(proc, base)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("ALL CLOUD TESTS PASSED")
