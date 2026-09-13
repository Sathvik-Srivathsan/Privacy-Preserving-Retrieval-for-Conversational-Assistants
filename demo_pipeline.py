"""Standalone real-docs demo for the Phase-1 SafeRAG pipeline (T10).

NOT part of the test suite (tests stay deterministic/offline). Reads
``docs/`` + ``docs/manifest.json``, builds an IPFE-encrypted ``LocalStore``
index with live ``nomic-embed-text`` embeddings when the Ollama server
answers (offline ``_hash_embed`` fallback otherwise — reuse ``OllamaAdapter``
as-is), then answers a query under a caller attribute set via the encrypted
RetrievalEngine, with readable output (rank, score, excerpt), optional
grounding, and a live-vs-hash comparison.

Two critical decisions (from T10 plan review):
1. EXPLICIT ``quant_bits=2`` on ``IPFEScheme.setup`` - never the default
   (``quant_bits=12`` -> B=10^12 -> BSGS range ~2*768*10^24 ~ 10^27, baby
   table ~4*10^13 entries ~ machine lock-up).
2. Scheme PERSISTENCE: ``(p, q, g, msk)`` is serialized to
   ``demo_index/ipfe_scheme.json`` on first build and loaded on later runs,
   so ingest-in-one-process / query-in-a-separate-process use the SAME
   master secret (mirrors the LocalStore Fernet key-persistence fix, T2).
   ``demo_index/`` is gitignored (msk lives there uncommitted).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from src.attributes import Attribute, Attributes, build_tree              # noqa: E402
from src.grounding import verify_grounded                  # noqa: E402
from src.ipfe import IPFEScheme                            # noqa: E402
from src.llm import OllamaAdapter, _hash_embed, _l2normalize  # noqa: E402
from src.retrieval import RetrievalEngine                  # noqa: E402
from src.store import LocalStore, StoredChunk              # noqa: E402

N_DEFAULT = 768
GROUP_BITS = 128
QUANT_BITS = 2  # explicit; never rely on the setup() default


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def excerpt(text: str, n: int = 120) -> str:
    s = norm_ws(text)
    return s[:n] + ("..." if len(s) > n else "")


def parse_caller(spec: str) -> Attributes:
    """Parse a caller spec like 'role:Doctor, dept: Cardio' into Attributes.

    Each ``key:value`` token is routed through ``Attribute.parse`` so key and
    value are stripped/lowercased SEPARATELY (``Attributes.__init__`` only
    strips/lowercases the whole token — 'dept: Cardio' would otherwise stay
    'dept: cardio' vs the canonical 'dept:cardio' and silently fail every
    gate: T10 review fix). Bare tokens pass through unchanged.
    """
    raw = [t.strip() for t in spec.split(",") if t.strip()]
    if not raw:
        raise SystemExit("--caller must be a comma list like 'role:Doctor,dept:Cardio'")
    norm = []
    for t in raw:
        if ":" in t:
            a = Attribute.parse(t)
            norm.append(str(a))
        else:
            norm.append(t.strip().lower())
    return Attributes(norm)


class RankableDoc:
    """Minimal doc wrapper giving RetrievalEngine the fields it reads
    (``_policy_of`` reads ``.policy``/``.policy_cached``; rank reads
    ``.doc_id``, ``.ct``, ``.group_bits``)."""

    __slots__ = ("doc_id", "ct", "policy", "group_bits")

    def __init__(self, doc_id: str, ct: list[int], policy, group_bits: int = 0):
        self.doc_id = doc_id
        self.ct = ct
        self.policy = policy
        self.group_bits = group_bits


def detect_embed_mode(adapter: OllamaAdapter, dim: int) -> str:
    """Return 'live (model)' if the adapter's embedding is NOT the
    deterministic hash-BoW fallback, else 'fallback (hash-BoW)'."""
    canary = "canary seizure screening protocol reference alpha 7741"
    vec = adapter.embed(canary, dim=dim)
    hb = _l2normalize(_hash_embed(canary, dim), dim)
    diff = max(abs(a - b) for a, b in zip(vec[:64], hb[:64]))
    use = adapter.embed_model if diff > 1e-9 else "hash-BoW"
    return f"live ({use})" if diff > 1e-9 else f"fallback (hash-BoW)"


def load_or_create_scheme(index_dir: str, n: int, rebuild: bool):
    path = os.path.join(index_dir, "ipfe_scheme.json")
    if not rebuild and os.path.exists(path):
        d = json.loads(open(path, encoding="utf-8").read())
        if len(d["msk"]) == n:
            p, q, g, msk, qb = d["p"], d["q"], d["g"], d["msk"], d["quant_bits"]
            mpk = [pow(g, si, p) for si in msk]
            return IPFEScheme(p, q, g, mpk, msk, n, qb), False
    # fresh setup with EXPLICIT quant_bits
    scheme = IPFEScheme.setup(n, group_bits=GROUP_BITS, quant_bits=QUANT_BITS)
    return scheme, True


def save_scheme(index_dir: str, scheme: IPFEScheme) -> None:
    path = os.path.join(index_dir, "ipfe_scheme.json")
    data = {
        "p": scheme.p, "q": scheme.q, "g": scheme.g,
        "msk": list(scheme.msk), "n": scheme.vec_len,
        "quant_bits": scheme.quant_bits,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def read_manifest(manifest_path: str):
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def ingest(adapter, scheme, doc_dir, manifest, store, index_dir, n, texts_out,
           cache_policies) -> dict:
    """(Re)build the encrypted index from docs/ + manifest. Returns {id: text}."""
    texts = {}
    policies = {}
    for fname, pol in manifest.items():
        path = os.path.join(doc_dir, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing document from manifest: {path}")
        text = open(path, encoding="utf-8").read()
        vec = adapter.embed(text, dim=n)
        if len(vec) != n:
            raise SystemExit(f"embedding dim {len(vec)} != {n} for {fname}")
        C, qq = scheme.encrypt(vec)
        store.put(StoredChunk(fname, json.dumps(C).encode("utf-8"),
                              pol, {"dim": n}))
        texts[fname] = text
        policies[fname] = build_tree(pol)
    with open(texts_out, "w", encoding="utf-8") as f:
        json.dump(texts, f)
    cache_policies.update(policies)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real-docs encrypted-IPFE retrieval demo (T10).")
    ap.add_argument("query", nargs="?", default=None,
                    help="query string (also accepted via --query)")
    ap.add_argument("--query", dest="query_opt", default=None)
    ap.add_argument("--caller", required=True,
                    help="caller attribute set, e.g. 'role:Doctor,dept:Cardio'")
    ap.add_argument("-k", type=int, default=5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--answer", default=None,
                   help="manually-supplied candidate answer for the grounding check")
    g.add_argument("--generate", action="store_true",
                   help="LLM-generate an answer from the top-k excerpts via "
                        "OllamaAdapter.complete(), then run the grounding check "
                        "on it (mutually exclusive with --answer)")
    ap.add_argument("--compare-hashb", action="store_true",
                    help="also rank the query via hash-BoW and show top-1 both ways")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe index + scheme and rebuild from scratch")
    ap.add_argument("--dim", type=int, default=N_DEFAULT)
    ap.add_argument("--doc-dir", default="docs")
    ap.add_argument("--manifest", default="docs/manifest.json")
    ap.add_argument("--index", default="demo_index")
    ap.add_argument("--artifact", default="artifacts/demo_run.json")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    args = ap.parse_args()

    query = args.query or args.query_opt
    if not query:
        ap.error("a query string is required (positional or --query)")

    n = args.dim
    os.makedirs(args.index, exist_ok=True)
    os.makedirs(os.path.dirname(args.artifact), exist_ok=True)
    texts_path = os.path.join(args.index, "texts.json")

    print("=" * 72)
    print("Real-docs encrypted-IPFE retrieval demo (T10)")
    print("=" * 72)

    t0 = time.perf_counter()
    adapter = OllamaAdapter(model=args.model, embed_model=args.embed_model)
    mode = detect_embed_mode(adapter, n)
    print(f"embed mode  : {mode}")
    print(f"dim n       : {n}   group_bits: {GROUP_BITS}   quant_bits: {QUANT_BITS}")

    manifest = read_manifest(args.manifest)
    attrs = parse_caller(args.caller)

    store = LocalStore(args.index)
    stored = set(store.list_ids())
    expected = set(manifest)

    # scheme: load persisted if present (not rebuilding); else fresh + save.
    scheme, created_scheme = load_or_create_scheme(args.index, n, args.rebuild)

    need_rebuild = args.rebuild or created_scheme or (expected != stored)
    if need_rebuild:
        if created_scheme and (expected != stored) and stored:
            print(f"  note: no persisted scheme but {len(stored)} blobs exist; a "
                  f"previous index cannot be queried without its msk -> rebuilding.")
        # drop stale blobs (not in manifest)
        stale = [f for f in stored if f not in expected]
        for f in stale:
            p = os.path.join(args.index, f + ".blob")
            if os.path.exists(p):
                os.remove(p)
        t1 = time.perf_counter()
        texts = ingest(adapter, scheme, args.doc_dir, manifest, store,
                       args.index, n, texts_path, {})
        t_ingest = time.perf_counter() - t1
        print(f"ingest      : {len(texts)} chunks  ({t_ingest:.2f}s)  "
              f"mode={mode}")
        # pin the round-trip: every stored blob must deserialize to N+1 ints
        for fname in texts:
            chunk = store.get(fname)
            ct = json.loads(chunk.blob.decode("utf-8"))
            if not (isinstance(ct, list) and len(ct) == n + 1
                    and all(isinstance(x, int) for x in ct)):
                raise RuntimeError(
                    f"ingest->get round-trip failed for {fname}: bad ciphertext")
        if created_scheme:
            save_scheme(args.index, scheme)
    else:
        texts = json.loads(open(texts_path, encoding="utf-8").read())
        print(f"ingest      : reuse {len(texts)} chunks from {args.index}")

    scheme_src = (f"{args.index}/ipfe_scheme.json" if not created_scheme
                  else "freshly created + saved")
    print(f"scheme      : {scheme_src}  (n={scheme.vec_len})")

    # policies from manifest (client side)
    policies = {fname: build_tree(pol) for fname, pol in manifest.items()}

    docs = []
    for fname in expected:
        chunk = store.get(fname)
        if chunk is None:
            continue
        ct = json.loads(chunk.blob.decode("utf-8"))
        docs.append(RankableDoc(fname, ct, policies[fname]))
    print(f"corpus      : {len(docs)} docs (manifest), caller "
          f"authorized: {sum(p.satisfies(attrs) for p in policies.values())}")

    engine = RetrievalEngine(scheme, k=args.k)
    query_vec = adapter.embed(query, dim=n)

    t2 = time.perf_counter()
    results = engine.rank(docs, query_vec, attrs)
    t_rank = time.perf_counter() - t2

    print()
    print(f"top-{args.k} for {query!r}  (rank {t_rank*1000:.0f} ms):")
    print(f"  {'#':>2}  {'doc_id':<26} {'score':>7}   excerpt")
    print("  " + "-" * 110)
    for i, r in enumerate(results, 1):
        print(f"  {i:>2}  {r.doc_id:<26} {r.score:>7.4f}   {excerpt(texts[r.doc_id])}")

    # optional grounding check (--answer or --generate)
    answer = args.answer
    answer_source = "manual" if answer else None
    if args.generate and results:
        if not adapter.available():
            print("\n  [NOTE] Ollama generation endpoint unavailable; "
                  "--generate skipped (retrieval-only mode)")
        else:
            chunks_text = "\n---\n".join(texts[r.doc_id] for r in results)
            prompt = ("Using ONLY the following retrieved notes, answer the "
                      "question. Do not invent details not present in the notes.\n\n"
                      f"Notes:\n{chunks_text}\n\n"
                      f"Question: {query}\nAnswer:")
            print("\ngenerating answer from top-{0} excerpts ...".format(args.k),
                  end="", flush=True)
            t3 = time.perf_counter()
            answer = adapter.complete(prompt, temperature=0.2, max_tokens=256)
            t_gen = time.perf_counter() - t3
            print(f" ({t_gen:.2f}s)")
            print(f"generated    : {norm_ws(answer)}")
            answer_source = "generated"

    ground = None
    if answer:
        chunks = [texts[r.doc_id] for r in results]
        rep = verify_grounded(answer, chunks)
        ground = {"answer": answer, "source": answer_source,
                  "fraction_grounded": rep.fraction_grounded,
                  "ungrounded_spans": [list(s) for s in rep.ungrounded_spans]}
        print()
        print("grounding   :")
        print(f"  answer from : {answer_source}")
        print(f"  fraction grounded: {rep.fraction_grounded:.2f}")
        for start, end, span in rep.ungrounded_spans:
            print(f"  UNGROUNDED '{norm_ws(span)}' [{start}:{end}]")

    # optional live-vs-hash comparison
    comp = None
    if args.compare_hashb:
        hqv = _l2normalize(_hash_embed(query, n), n)
        r1 = engine.rank(docs, query_vec, attrs, k=1)[0].doc_id if results else None
        r2 = engine.rank(docs, hqv, attrs, k=1)[0].doc_id if docs else None
        comp = {"live_top1": r1, "hashb_top1": r2}
        print()
        print("live-vs-hash:")
        print(f"  live  top-1: {r1}")
        print(f"  hash  top-1: {r2}")

    record = {
        "embed_mode": mode, "n": n, "group_bits": GROUP_BITS,
        "quant_bits": QUANT_BITS, "scheme_source": scheme_src,
        "corpus": {"docs": len(docs), "manifest": sorted(manifest)},
        "caller": sorted(attrs.toks),
        "query": query,
        "timings_ms": {"ingest_s": round(t_ingest if need_rebuild else 0.0, 3),
                       "rank_ms": round(t_rank * 1000.0, 1),
                       "total_s": round(time.perf_counter() - t0, 3)},
        "top_k": [{"doc_id": r.doc_id, "score": round(r.score, 6)}
                  for r in results],
        "grounding": ground,
        "compare_hashb": comp,
    }
    with open(args.artifact, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print()
    print(f"wrote : {args.artifact}")
    print(f"done  : total {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, int):
            raise
        print(f"FATAL: {e}")
        sys.exit(1)