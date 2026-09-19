"""Standalone P2-F demo: the FULL Phase-2 stack over a cloud boundary.

SafeRAG-Improved Phase-2 walkthrough (checks phase2-checklist P2-F):

  * CLOUD boundary (P2-D): the encrypted index lives in a SEPARATE
    ``tools/cloud_server.py`` subprocess; the client speaks HTTP/JSON only.
  * INGEST via ``CorpusIntegrity.put`` wrapping ``CloudStore`` (P2-B
    review-closure): every doc is MAC-tagged and manifest-registered before it
    ever reaches the wire.  The old pre-integrity ``store.put(StoredChunk(..))``
    pattern is deliberately NOT used here.
  * ZKP gate (P2-E): the caller PROVES possession of each attribute token
    client-side via Schnorr Fiat-Shamir (``prove_attributes`` /
    ``verify_attributes``) — ONE round versus the N-round Bayesian dialogue —
    and then the query wire carries ``{sk_q, q_ints, attrs}`` ONLY, never a
    proof (verified against the captured wire body).
  * Embedding dim is pinned to the SAME ``n`` at corpus build, query, and
    Setup (review item: ``OllamaAdapter.embed`` pads/truncates live vectors to
    exactly ``dim``, so n must be fixed to match the IPFE vector length).
  * Grounding + REDACT seam (P2-F §2.5): ungrounded spans are resolved
    against the real response and masked length-preservingly via
    ``RedactSeam`` (once-per-message, caps, audit).  The redact pass emits NO
    store writes and no ``█`` byte ever reaches the wire.
  * Optional ``--dp EPS``: fixed-eps Laplace on a numeric token in the answer.

The orchestration is split so the Streamlit dashboard (``tools/dashboard.py``)
runs the SAME code path as this CLI: ``build_rig`` (one-time: adapter + cloud
server + IPFE scheme + MAC-tagged ingest) + ``access_matrix`` (per-caller
ALLOW/DENY policy view) + ``run_pipeline`` (ZKP -> encrypted query -> top-k ->
grounding/redact -> DP).  The dashboard caches the rig and calls
``run_pipeline`` per Run.

The offline route (``--offline``) forces the hash-BoW deterministic embedding
path and is the regression harness; the live Llama/Qwen path runs when Ollama
is up.  A JSON ``--artifact`` is written either way
(``--artifact`` disabled by the dashboard, which reuses the rig in-process).

Run:  python tools/demo.py <query> --caller role:Doctor,dept:Cardio [options]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# The redact seam emits U+2588 masking; force UTF-8 console output so the
# suite never dies on a cp1252 terminal mid-evidence.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402

from src.attributes import Attribute, Attributes, build_tree       # noqa: E402
from src.cloud import CloudQuery, CloudStore                       # noqa: E402
from src.dp import FixedEpsDP                                      # noqa: E402
from src.grounding import (resolve_spans, RedactSeam, verify_grounded)  # noqa: E402
from src.integrity import CorpusIntegrity, IntegrityRecord         # noqa: E402
from src.ipfe import IPFEScheme                                    # noqa: E402
from src.llm import OllamaAdapter, _hash_embed, _l2normalize       # noqa: E402
from src.zkp import prove_attributes, verify_attributes            # noqa: E402

N_DEFAULT = 64
GROUP_BITS = 128
QUANT_BITS = 2          # explicit; never rely on the IPFEScheme default
REDACT_MARK_UTF8 = "\u2588".encode("utf-8")

CORPUS = [
    ("D0", "role:Doctor",
     "Heart rhythm monitoring device prevents arrhythmia related stroke."),
    ("D1", "role:Nurse",
     "The ward staff record daily blood pressure readings each morning."),
    ("D2", "role:Doctor AND dept:Cardio",
     "Postoperative infection rates drop with a prophylactic antibiotic "
     "course."),
    ("D3", "role:Doctor AND dept:Cardio AND clearance:2",
     "The cardiac echo report was flagged for the cardiovascular department."),
    ("D4", "role:Doctor",
     "A pacing wire infection adds ten days to the hospital stay."),
]
DEFAULT_QUERY = "heart rhythm monitoring reduces stroke risk"
CALLER_TOKENS = ["role:doctor", "dept:cardio", "clearance:2", "role:nurse"]


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def excerpt(text: str, n: int = 120) -> str:
    s = norm_ws(text)
    return s[:n] + ("..." if len(s) > n else "")


def parse_caller(spec: str) -> list[str]:
    """'role:Doctor, dept: Cardio' -> ['role:doctor', 'dept:cardio']."""
    raw = [t.strip() for t in spec.split(",") if t.strip()]
    if not raw:
        raise SystemExit("--caller must be a comma list like 'role:Doctor,dept:Cardio'")
    toks = []
    for t in raw:
        if ":" in t:
            toks.append(str(Attribute.parse(t)))
        else:
            toks.append(t.strip().lower())
    return toks


def _no_floats(value):
    if isinstance(value, float):
        return False
    if isinstance(value, dict):
        return all(_no_floats(k) and _no_floats(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_no_floats(v) for v in value)
    return True


def detect_embed_mode(adapter: OllamaAdapter, dim: int) -> str:
    canary = "canary seizure screening protocol reference alpha 7741"
    vec = adapter.embed(canary, dim=dim)
    hb = _l2normalize(_hash_embed(canary, dim), dim)
    diff = max(abs(a - b) for a, b in zip(vec[:64], hb[:64]))
    return f"live ({adapter.embed_model})" if diff > 1e-9 else "fallback (hash-BoW)"


class _DownSession:
    """Offline-forcing session (mirrors the offline battery fake)."""

    def get(self, url, timeout=None):
        raise OSError("connection refused (offline --down session)")

    def post(self, url, json=None, timeout=None):
        raise OSError("connection refused (offline --down session)")


# --------------------------------------------------------------------------- #
# Cloud server fixture (P2-D: boundary is a separate process)
# --------------------------------------------------------------------------- #

def _spawn_cloud():
    """Spawn the cloud-server subprocess with stdout+stderr BOTH redirected
    to ``artifacts/cloud_server.log`` (no PIPE: a full 4KB Windows pipe would
    stall the child mid-request, and a swallowed pipe hides any crash).  The
    server's very first line is ``port=NNNN``; we poll the log file for it."""
    artifacts = os.path.join(REPO, "artifacts")
    os.makedirs(artifacts, exist_ok=True)
    log_path = os.path.join(artifacts, "cloud_server.log")
    logf = open(log_path, "w", encoding="utf-8")
    script = os.path.join(REPO, "tools", "cloud_server.py")
    try:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=logf, stderr=subprocess.STDOUT, close_fds=True)
    except Exception:
        logf.close()
        raise
    proc._cloud_log = logf            # keep the handle alive for the child
    port = None
    for _ in range(150):              # 15s max for the port banner
        with open(log_path, "r", encoding="utf-8") as pollf:
            m = re.search(r"port=(\d+)", pollf.read())
        if m:
            port = int(m.group(1))
            break
        time.sleep(0.1)
    if port is None:
        proc.terminate()
        raise RuntimeError(
            f"cloud server did not report a port (see {log_path})")
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    return proc, base, log_path


def _stop_cloud(proc, base):
    try:
        urllib.request.urlopen(f"{base}/shutdown", timeout=2)
    except Exception:
        pass
    proc.terminate()
    handle = getattr(proc, "_cloud_log", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:            # noqa: BLE001
            pass


def _health(base: str, timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen(f"{base}/health", timeout=timeout).close()
        return True
    except Exception:                # noqa: BLE001
        return False


def ensure_cloud(rig: DemoRig, *, timeout: float = 2.0) -> str:
    """Return the rig's cloud-server state: ``"up"`` if the cached child
    answers ``/health`` in time, ``"restarted"`` if it was gone and has just
    been respawned + re-ingested IN PLACE, or raise ``RuntimeError`` on a
    rebuild failure.

    The dashboard caches ONE long-lived rig across Streamlit reruns, but the
    cloud server is an OS child that nothing guarantees survives (a kill from
    outside the app, a crash, an idle watchdog).  ``ensure_cloud`` health-
    checks the child before any query; if it is gone it respawns and re-
    ingests from the cached ciphertext blobs (``rig.blobs``) using the SAME
    ``CorpusIntegrity.put`` path the build used — so the cached rig object
    (and the dashboard's reference to it) transparently points at the new
    server for all subsequent renders.
    """
    if _health(rig.base, timeout=timeout):
        return "up"
    print("cloud  : server gone; restarting + re-ingesting in place ...")
    old_proc, old_base = rig.proc, rig.base
    try:
        proc, base, log_path = _spawn_cloud()
        cs = CloudStore(base)
        integrity = CorpusIntegrity(cs, root="demo")
        for doc_id, tag, _text in CORPUS:
            integrity.put(IntegrityRecord(doc_id, rig.blobs[doc_id], tag,
                                          {"dim": rig.dim}, version=1))
        verified, incidents = integrity.verify_corpus()
        assert len(verified) == len(CORPUS), \
            f"integrity verify failed: {incidents}"
        rig.proc, rig.base, rig.cs, rig.integrity = proc, base, cs, integrity
        rig.log_path = log_path
        print(f"cloud  : restarted on {base}; 5/5 re-ingested + verified")
        return "restarted"
    except Exception as exc:          # noqa: BLE001
        _stop_cloud(old_proc, old_base)
        raise RuntimeError(f"cloud server rebuild failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# DemoRig — one-time state shared by CLI runs AND the cached dashboard
# --------------------------------------------------------------------------- #

@dataclass
class DemoRig:
    adapter: OllamaAdapter
    mode: str
    scheme: IPFEScheme
    proc: object
    base: str
    cs: CloudStore
    integrity: CorpusIntegrity
    texts: dict
    verified: list
    dim: int
    blobs: dict = None
    log_path: str = ""

    def close(self):
        try:
            _stop_cloud(self.proc, self.base)
        except Exception:  # noqa: BLE001
            pass


def build_rig(*, dim: int = N_DEFAULT, embed_model: str = "nomic-embed-text",
              model: str = "qwen2.5:1.5b", work: str = "demo",
              offline: bool = False) -> DemoRig:
    """Adapter + spawned cloud server + IPFE scheme + MAC-tagged ingest."""
    adapter = OllamaAdapter(model=model, embed_model=embed_model)
    if offline:
        adapter._session = _DownSession()
    mode = detect_embed_mode(adapter, dim)
    proc, base, log_path = _spawn_cloud()
    cs = CloudStore(base)
    try:
        scheme = IPFEScheme.setup(dim, group_bits=GROUP_BITS, quant_bits=QUANT_BITS)
        integrity = CorpusIntegrity(cs, root=work)
        texts, blobs = {}, {}
        for doc_id, tag, text in CORPUS:
            vec = adapter.embed(text, dim=dim)
            assert len(vec) == dim, f"embed dim {len(vec)} != {dim} (build)"
            ct, _ = scheme.encrypt(vec)
            blob = json.dumps(ct, separators=(",", ":")).encode("utf-8")
            integrity.put(IntegrityRecord(doc_id, blob, tag, {"dim": dim},
                                          version=1))
            texts[doc_id] = text
            blobs[doc_id] = blob
        verified, incidents = integrity.verify_corpus()
        print(f"ingest : {len(verified)}/{len(CORPUS)} docs verified via "
              f"Mac-tagged CloudStore  incidents={len(incidents)}")
        assert len(verified) == len(CORPUS), f"integrity verify failed: {incidents}"
        return DemoRig(adapter, mode, scheme, proc, base, cs, integrity,
                       texts, verified, dim, blobs=blobs, log_path=log_path)
    except Exception:
        _stop_cloud(proc, base)
        raise


def access_matrix(rig: DemoRig, caller_toks: list[str]) -> list[dict]:
    """Per-doc ALLOW/DENY + reason, from the SAME policy gate the pipeline
    uses (``build_tree(tag).satisfies`` — pinned equal to the server-side Dauth
    by the P2-D parity batteries)."""
    attrs = Attributes(caller_toks)
    held = set(caller_toks)
    rows = []
    for doc_id, tag, _text in CORPUS:
        allowed = build_tree(tag).satisfies(attrs)
        policy_toks = sorted({t.lower() for t in re.findall(r"[A-Za-z0-9_]+:[^() ]+", tag)})
        missing = [t for t in policy_toks if t not in held]
        rows.append({
            "doc_id": doc_id, "tag": tag, "allowed": bool(allowed),
            "missing": missing,
        })
    return rows


# --------------------------------------------------------------------------- #
# run_pipeline — one query through the full stack; shared CLI <-> dashboard
# --------------------------------------------------------------------------- #

def run_pipeline(rig: DemoRig, *, query: str, caller_toks: list[str], k: int = 5,
                 answer: str | None = None, generate: bool = False,
                 redact: bool = False, dp: float | None = None,
                 dp_seed: int | None = None, audit: str | None = None,
                 artifact: str | None = None) -> dict:
    """Run one NL query end-to-end: ZKP gate -> encrypted query -> top-k ->
    grounding (+ optional redact seam) (+ optional DP). Returns the artifact
    record (also carries the per-caller access matrix)."""
    # 1. CLOUD HEALTH — the cached rig's server child may not have survived
    # since the last run (dashboard sessions, external kills); self-heal
    # transparently BEFORE any query so a dead child can never stall a run.
    cloud_status = ensure_cloud(rig, timeout=2.0)
    t0 = time.perf_counter()
    scheme = rig.scheme
    cs = rig.cs
    texts = rig.texts

    # 3. ZKP gate — ONE round, proofs stay CLIENT-SIDE forever
    proofs = prove_attributes(caller_toks, scheme)
    ok = verify_attributes(proofs, required=caller_toks, group=scheme)
    assert ok, "ZKP verification FAILED for caller attributes"
    print(f"zkp    : {len(caller_toks)} tokens proven + verified in ONE round "
          f"(Schnorr Fiat-Shamir)")

    # 4. QUERY — same n as the corpus build (review item)
    query_vec = rig.adapter.embed(query, dim=rig.dim)
    assert len(query_vec) == rig.dim, \
        f"embed dim {len(query_vec)} != {rig.dim} (query)"
    cq = CloudQuery(rig.base, scheme)
    cq.setup_server()
    results = cq.query(query_vec, attrs=caller_toks)
    ids = [r["doc_id"] for r in results]
    t_query = time.perf_counter() - t0
    print(f"query  : top-{len(results)} for {query!r} "
          f"({t_query * 1000:.0f} ms)  ids={ids}")
    for i, r in enumerate(results, 1):
        print(f"   {i:>2}  {r['doc_id']:<6} score={r['score']:.6f}  "
              f"{excerpt(texts[r['doc_id']])}")

    # wire assertion (SS): body carries ONLY sk_q + q_ints + attrs, no
    # floats (P2-D parity).  The EXACT key-set equality is the honest
    # no-proofs guard: exactly three keys leaves no room for any proof
    # field to be smuggled onto the wire (P2-E ZKP stays client-side).
    wire = json.loads(cq.captured[-1].decode("utf-8"))
    assert set(wire) == {"sk_q", "q_ints", "attrs"}, f"wire shape: {set(wire)}"
    assert _no_floats(wire), "float leaked to wire"
    print("wire   : POST /query body == {sk_q, q_ints, attrs}  "
          "(no floats, exact key-set -> no proof can ride along)")

    # 5. ANSWER — manual / live-generated / derived from top-1 chunk
    answer_text = answer
    answer_source = "manual"
    gen_info = None
    if answer_text is None and generate:
        answer_source = "generated"
        gen_info = {"attempted": True, "elapsed_s": None, "fell_back": False,
                    "reason": None}
        if not rig.adapter.available():
            gen_info["fell_back"] = True
            gen_info["reason"] = "Ollama not reachable"
            print("  [NOTE] Ollama generation unavailable; "
                  "--generate skipped -> retrieval-derived answer")
            answer_source = "derived"
    if answer_text is None and answer_source == "generated":
        chunks_text = "\n---\n".join(texts[r["doc_id"]] for r in results)
        prompt = ("Using ONLY the following retrieved notes, answer the "
                  "question. Do not invent details not present in the notes."
                  f"\n\nNotes:\n{chunks_text}\n\nQuestion: {query}\nAnswer:")
        print(f"  generating answer from top-{len(results)} ...")
        t1 = time.perf_counter()
        answer_text = rig.adapter.complete(prompt, temperature=0.2,
                                           max_tokens=256)
        gen_info["elapsed_s"] = round(time.perf_counter() - t1, 2)
        print(f"  generated ({gen_info['elapsed_s']}s): "
              f"{norm_ws(answer_text)}")
    if answer_text is None:
        answer_text = norm_ws(texts[results[0]["doc_id"]])
        answer_source = "derived"
    answer_text = norm_ws(answer_text)
    print(f"answer : ({answer_source}) {answer_text}")

    # 6. GROUNDING + REDACT seam (§2.5)
    ground = {"answer": answer_text, "source": answer_source}
    if gen_info:
        ground["generation"] = gen_info
    rep = verify_grounded(answer_text, [texts[r["doc_id"]] for r in results])
    ground["fraction_grounded"] = rep.fraction_grounded
    ground["ungrounded_spans"] = [list(s) for s in rep.ungrounded_spans]
    redact_info = None
    if redact:
        spans = resolve_spans(answer_text, rep.ungrounded_spans)
        if not spans:
            # Fully-grounded answer: nothing to mask.  Do NOT hand the seam an
            # empty pass (its empty pass would count as "applied" with text
            # unchanged and pollute the once-per-message guard); declare the
            # no-op up-front instead.
            redact_info = {"applied": False, "refused": "no_ungrounded_spans",
                           "n_spans": 0, "masked_chars": 0,
                           "text": answer_text}
            print("redact : 0 ungrounded spans — fully grounded answer; "
                  "nothing to mask")
        else:
            seam = RedactSeam(audit_path=audit)
            n_ids_before = len(cs.list_ids())
            n_cap_before = len(cs.captured)  # snapshot right before the seam
            redacted, applied, why = seam.redact("demo", answer_text, spans)
            redact_info = {
                "applied": applied, "refused": why or None,
                "n_spans": len(spans),
                "masked_chars": sum(e - s for s, e in spans),
            }
            # guard checks FIRST, with no store ask between the snapshot and
            # this count (a list_ids() GET itself appends to .captured)
            assert len(cs.captured) == n_cap_before, "redaction emitted writes"
            assert len(cs.list_ids()) == n_ids_before, \
                "redaction touched the store"
            assert redacted != answer_text or not applied
            print(f"redact : {len(spans)} span(s) "
                  + ("applied" if applied else f"refused: {why}"))
            if applied:
                print(f"  -> {redacted}")
            redact_info["text"] = redacted
            # guard 2: no mask byte on the wire before or after the pass
            assert all(REDACT_MARK_UTF8 not in b
                       for b in cs.captured + cq.captured), \
                "redacted byte reached the vector path"
            print("guard2 : store untouched; no U+2588 byte on the wire")

    # 7. optional DP (fixed eps) on a numeric token in the answer
    dp_info = None
    if dp is not None:
        use = redact_info["text"] if redact_info and redact_info["applied"] \
            else answer_text
        m = re.search(r"\d+(\.\d+)?", use)
        if not m:
            dp_info = {"eps": dp, "privatised": None,
                       "note": "no numeric token in answer"}
            print("dp     : no numeric token in answer; skipped")
        else:
            raw = float(m.group(0))
            out_v = FixedEpsDP(dp).privatise(raw)
            dp_info = {"eps": dp, "input": raw, "output": round(out_v, 4)}
            print(f"dp     : eps={dp}  {raw} -> {out_v:.4f}")

    record = {
        "embed_mode": rig.mode, "n": rig.dim, "group_bits": GROUP_BITS,
        "quant_bits": QUANT_BITS,
        "corpus": {"docs": len(rig.verified), "ids": sorted(rig.verified)},
        "caller": sorted(set(caller_toks)),
        "query": query,
        "query_ms": round(t_query * 1000.0, 1),
        "access": access_matrix(rig, caller_toks),
        "cloud": {"base": rig.base, "healed": cloud_status == "restarted"},
        "top_k": [{"doc_id": r["doc_id"], "score": round(r["score"], 6)}
                  for r in results],
        "wire": {"shape": sorted(wire), "no_floats": True, "no_proofs": True},
        "wire_body": wire,
        "grounding": ground,
        "redact": redact_info,
        "dp": dp_info,
        "total_s": round(time.perf_counter() - t0, 3),
    }
    if artifact:
        with open(artifact, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
        print(f"wrote : {artifact}")
    return record


# --------------------------------------------------------------------------- #
# CLI entry (dashboards call build_rig / run_pipeline directly)
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="P2-F demo: cloud + integrity + ZKP + grounding/redact + DP.")
    ap.add_argument("query", nargs="?", default=None, help="query string")
    ap.add_argument("--query", dest="query_opt", default=None)
    ap.add_argument("--caller", default="role:Doctor,dept:Cardio")
    ap.add_argument("-k", type=int, default=5)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--answer", default=None,
                   help="manually-supplied candidate answer (grounding + redact)")
    g.add_argument("--generate", action="store_true",
                   help="LLM-generate an answer from top-k excerpts (live Ollama)")
    ap.add_argument("--redact", action="store_true",
                    help="run the RedactSeam on any ungrounded spans")
    ap.add_argument("--dp", type=float, default=None, metavar="EPS",
                    help="privatise a numeric token in the answer (fixed eps)")
    ap.add_argument("--offline", action="store_true",
                    help="force the deterministic hash-BoW path (no Ollama)")
    ap.add_argument("--dim", type=int, default=N_DEFAULT)
    ap.add_argument("--artifact", default="artifacts/demo_run.json")
    ap.add_argument("--audit", default="artifacts/redact_audit.jsonl")
    ap.add_argument("--work", default="demo")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--model", default="qwen2.5:1.5b")
    args = ap.parse_args()

    query = args.query or args.query_opt
    if not query:
        ap.error("a query string is required (positional or --query)")
    n = args.dim
    os.makedirs(os.path.dirname(args.artifact), exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    t0 = time.perf_counter()
    print("=" * 72)
    print("P2-F demo — cloud + integrity + ZKP + grounding/redact + DP")
    print("=" * 72)

    rig = build_rig(dim=n, embed_model=args.embed_model, model=args.model,
                    work=args.work, offline=args.offline)
    print(f"embed mode: {rig.mode}    dim n: {n}    group_bits: {GROUP_BITS}    "
          f"quant_bits: {QUANT_BITS}")
    try:
        record = run_pipeline(rig, query=query,
                              caller_toks=parse_caller(args.caller), k=args.k,
                              answer=args.answer, generate=args.generate,
                              redact=args.redact, dp=args.dp,
                              audit=args.audit, artifact=args.artifact)
        print(f"done  : total {time.perf_counter() - t0:.2f}s")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if isinstance(e.code, int):
            raise
        print(f"FATAL: {e}")
        sys.exit(1)