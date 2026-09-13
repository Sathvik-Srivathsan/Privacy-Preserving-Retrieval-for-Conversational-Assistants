# -*- coding: utf-8 -*-
"""
Phase-1 END-TO-END pipeline smoke (T7) —— one green chain:

    qwen (offline deterministic embeddings)
      -> store (LocalStore Fernet AND GoogleDriveStore via in-memory fake)
      -> access tree (explicit-role policies, Attributes.from_roles)
      -> encrypted IPFE top-k (real ciphertext inner products)
      -> grounding (stopword-aware overlap vs retrieved chunks)

Anchor numbers: ALL stores list the full corpus; only authorised docs reach
top-k; the best-matching authorised doc ranks first; a hallucinated claim is
flagged ungrounded; plaintext never hits the disk (Fernet) — the blob file
must not contain the chunk text or its policy tag.

Run:  python tests\test_pipeline.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from src.store import GoogleDriveStore, LocalStore, StoredChunk     # noqa: E402
from src.attributes import Attributes, build_tree                   # noqa: E402
from src.ipfe import IPFEScheme                                     # noqa: E402
from src.retrieval import RetrievalEngine                            # noqa: E402
from src.grounding import verify_grounded                           # noqa: E402
from src.llm import OllamaAdapter                                   # noqa: E402
from test_llm import _FakeSession                                    # noqa: E402
from test_store import FakeDrive                                    # noqa: E402


N = 32                       # embedding dim == IPFE vector length (fixed at Setup)
PREC = 2

CORPUS = [
    ("D0", "role:Doctor",
     "heart rhythm monitoring device prevents arrhythmia related stroke"),
    ("D1", "role:Doctor",
     "postoperative infection rates drop with prophylactic antibiotic course"),
    ("D2", "role:Nurse",
     "daily blood pressure readings recorded by the ward staff"),
]
QUERY = "heart rhythm monitoring reduces stroke risk"

ANSWER_GROUNDED = "Heart rhythm monitoring prevents stroke risk."
ANSWER_HALLUCINATED = ["Approve the moon landing fund immediately.",
                       "Heart rhythm monitoring prevents stroke risk."]


class _Doc:
    """Bridge a stored chunk into the RetrievalEngine doc interface."""

    def __init__(self, chunk: StoredChunk, ct, text: str):
        self.doc_id = chunk.chunk_id
        self.ct = ct
        self.policy = build_tree(chunk.tag)
        self.text = text
        self.group_bits = 0

    def policy_cached(self):
        return self.policy


def _offline_adapter():
    # qwen stage: deterministic hash-BoW fallback (server down), dim = N
    a = OllamaAdapter(host="http://127.0.0.1:9999")
    a._session = _FakeSession(down=True)
    return a


def _ct_bytes(scheme, vec):
    Ct, _ = scheme.encrypt(list(vec))
    return json.dumps(Ct).encode("utf-8")


def _ct_from_bytes(blob):
    return json.loads(blob.decode("utf-8"))


def _ingest(store, scheme, adapter) -> dict:
    """Write the corpus. Chunk TEXT never enters chunk.meta: for the GDrive
    path meta would ride in the Drive `description` (plaintext leak!); the
    honest-but-curious contract is ciphertext + tag only. Text stays in a
    client-side map, decrypted/known only on the local side."""
    texts = {cid: text for cid, _, text in CORPUS}
    for cid, tag, text in CORPUS:
        vec = adapter.embed(text, dim=N)          # qwen (offline) -> unit vec
        store.put(StoredChunk(cid, _ct_bytes(scheme, vec), tag, {"dim": N}))
    return texts


def _load_docs(store, texts):
    docs = []
    for cid in sorted(store.list_ids()):
        ch = store.get(cid)
        docs.append(_Doc(ch, _ct_from_bytes(ch.blob), texts[cid]))
    return docs


def _engine(scheme):
    return RetrievalEngine(scheme, k=5)


def _query_vec(adapter):
    return adapter.embed(QUERY, dim=N)


# --------------------------------------------------------------------------- #
# LocalStore pipeline
# --------------------------------------------------------------------------- #

def _local_pipeline():
    scheme = IPFEScheme.setup(N, quant_bits=PREC)
    adapter = _offline_adapter()
    store = LocalStore(tempfile.mkdtemp(prefix="saferag_pipe_"))
    texts = _ingest(store, scheme, adapter)
    docs = _load_docs(store, texts)
    return scheme, adapter, store, docs


def test_pipeline_local_roundtrip_full_corpus():
    scheme, adapter, store, docs = _local_pipeline()
    assert sorted(store.list_ids()) == ["D0", "D1", "D2"]
    assert {d.text for d in docs} == {t for _, _, t in CORPUS}


def test_pipeline_local_encrypted_at_rest():
    _, _, store, _ = _local_pipeline()
    root = store.root
    raw = (Path(root) / "D0.blob").read_bytes()
    text = CORPUS[0][2]
    assert text.encode() not in raw, "chunk plaintext must never hit disk"
    assert b"role:Doctor" not in raw, "policy tag must be encrypted too"


def test_pipeline_topk_doctor_auth_order():
    scheme, adapter, _, docs = _local_pipeline()
    engine = _engine(scheme)
    res = engine.rank(docs, _query_vec(adapter),
                      Attributes.from_roles(["doctor"]), k=5)
    ids = [r.doc_id for r in res]
    # only the two role:Doctor docs are authorised for a doctor caller
    assert sorted(ids) == ["D0", "D1"], ids
    # D0 shares content tokens with the query -> must rank first
    assert ids[0] == "D0", ids
    assert res[0].score > res[1].score
    # magnitude sane: |cos| cannot exceed 1 for unit vectors
    assert all(abs(r.score) <= 1.05 for r in res)


def test_pipeline_topk_nurse_only_sees_nurse():
    scheme, adapter, _, docs = _local_pipeline()
    engine = _engine(scheme)
    res = engine.rank(docs, _query_vec(adapter),
                      Attributes.from_roles(["nurse"]), k=5)
    assert [r.doc_id for r in res] == ["D2"]


def test_pipeline_grounding_flags_hallucinated_claim():
    _, _, _, docs = _local_pipeline()
    chunk_texts = [d.text for d in docs]

    gone = verify_grounded(ANSWER_GROUNDED, chunk_texts)
    assert gone.fraction_grounded == 1.0, gone

    mixed = verify_grounded(" ".join(ANSWER_HALLUCINATED), chunk_texts)
    assert 0.0 < mixed.fraction_grounded < 1.0, mixed
    # the ungrounded span is exactly the hallucinated sentence
    bad = [sp[2] for sp in mixed.ungrounded_spans if "moon landing" in sp[2]]
    assert bad == [ANSWER_HALLUCINATED[0]], bad


def test_pipeline_scores_match_plaintext_cosine():
    scheme, adapter, _, docs = _local_pipeline()
    engine = _engine(scheme)
    v = _query_vec(adapter)
    res = engine.rank(docs, v, Attributes.from_roles(["doctor"]), k=5)
    plain = {d.doc_id: sum(a * b for a, b in zip(v, adapter.embed(d.text, dim=N)))
             for d in docs}
    for r in res:
        assert abs(r.score - plain[r.doc_id]) <= N / (10 ** PREC), (r.doc_id, r.score, plain[r.doc_id])


# --------------------------------------------------------------------------- #
# GoogleDriveStore (offline fake) pipeline — same chain through the cloud path
# --------------------------------------------------------------------------- #

def test_pipeline_gdrive_offline_chain():
    scheme = IPFEScheme.setup(N, quant_bits=PREC)
    adapter = _offline_adapter()
    fake = FakeDrive()
    st = GoogleDriveStore(creds_path="gdrive/service_account.json")
    st._svc = fake
    texts = _ingest(st, scheme, adapter)      # Drive sees only ciphertext+tag+dim

    docs = _load_docs(st, texts)
    assert sorted(st.list_ids()) == ["D0", "D1", "D2"]
    assert {d.text for d in docs} == {t for _, _, t in CORPUS}

    engine = _engine(scheme)
    res = engine.rank(docs, _query_vec(adapter), Attributes.from_roles(["doctor"]), k=5)
    ids = [r.doc_id for r in res]
    assert sorted(ids) == ["D0", "D1"]
    assert ids[0] == "D0"

    # honest-but-curious guarantee: fake Drive holds NO plaintext text —
    # descriptions carry only tag + dim, blobs carry only serialised ciphertext.
    all_descs = " | ".join(rec["description"] for rec in fake._files.values())
    for morsel in ("heart", "rhythm", "arrhythmia", "stroke"):
        assert morsel not in all_descs, morsel
    assert "role:Doctor" in all_descs          # tag travels as Drive description
    for blob in fake._blobs.values():
        for morsel in ("heart", "arrhythmia", "stroke"):
            assert morsel not in blob.decode("latin-1"), morsel


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
        "module": "pipeline",
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