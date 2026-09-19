"""Interactive local dashboard for the P2-F demo (SafeRAG-Improved Phase 2).

Runs the SAME ``build_rig`` / ``run_pipeline`` / ``access_matrix`` code path
as ``tools/demo.py`` — nothing here re-implements the pipeline, so CLI and
dashboard cannot drift.

Faculty-visible content (see phase2-checklist P2-F dashboard sub-note):

  1. ENCRYPTED-RETRIEVAL STAGES — ZKP gate -> POST /query wire body ->
     top-3 ranking -> grounding + redaction seam -> optional DP, rendered
     after every Run.
  2. ACCESS MATRIX — every corpus doc with its policy tag and ALLOW/DENY for
     the CURRENT caller, plus the missing token(s) as the reason.  Computed
     from the SAME ``build_tree(tag).satisfies`` gate the pipeline uses
     (pinned equal to the server-side Dauth by the P2-D parity batteries).
  3. INLINE READER — full text of every AUTHORIZED doc is rendered as a card;
     DENIED docs stay locked (policy shown, text not rendered).  Credentials
     literally decide what you can read.

Live-embedding mode is the default (nomic-embed-text @ n=64); a hash-BoW
fallback banner shows if Ollama is down, and the sidebar can force offline.
The rig (adapter + spawned cloud server + IPFE scheme + MAC-tagged ingest)
is cached with ``@st.cache_resource`` so each Run is ~1-4 s.

Run:  python -m streamlit run tools/dashboard.py
       (binds 127.0.0.1; check "Enable LAN" in the sidebar to reach it from
       other machines on the demo network)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from tools.demo import (CALLER_TOKENS, CORPUS, GROUP_BITS, QUANT_BITS,   # noqa: E402
                        DemoRig, _DownSession, access_matrix, build_rig,
                        ensure_cloud, run_pipeline)


# --------------------------------------------------------------------------- #
# startup sweep: kill orphaned cloud servers from earlier dashboard sessions
# --------------------------------------------------------------------------- #

def _sweep_stale_cloud_servers():
    if os.name != "nt":
        return
    script = os.path.join(REPO, "tools", "cloud_server.py").lower()
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'",
             "get", "ProcessId,CommandLine", "/format:list"],
            capture_output=True, text=True, timeout=15)
    except Exception:  # noqa: BLE001
        return
    pids = set()
    for match in re.finditer(r"CommandLine=(.+?)\r?\nProcessId=(\d+)",
                             out.stdout, flags=re.DOTALL):
        cmd, pid = match.group(1), match.group(2)
        if script in cmd.lower() and pid != str(os.getpid()):
            pids.add(pid)
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/PID", pid, "/F"],
                           capture_output=True, text=True, timeout=10)
        except Exception:  # noqa: BLE001
            pass


_sweep_stale_cloud_servers()

# --------------------------------------------------------------------------- #
# cached rig
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="starting encrypted-retrieval rig ...")
def load_rig() -> DemoRig:
    return build_rig(dim=64, embed_model="nomic-embed-text",
                     model="qwen2.5:1.5b", work="demo")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Phase-2 Demo — Encrypted Conversational "
                             "Retrieval", layout="wide")
st.title("SafeRAG-Improved — Phase 2 demo")
st.caption("Cloud + integrity + 1-round ZKP + grounding/redact + DP. "
           "The rig (encrypted index + MAC-tagged ingest) is cached; "
           "\"Run query\" sends one request through the full stack.")

rig = load_rig()
cloud_state = ensure_cloud(rig, timeout=2.0)
if cloud_state == "restarted":
    st.info("Cloud server was down; **restarted + re-ingested automatically "
            "in place** (self-healing rig).")
else:
    st.caption(f"cloud server: up at {rig.base} — log: "
               f"`artifacts/cloud_server.log`")

with st.sidebar:
    st.header("Query")
    query = st.text_input("Question", value="heart rhythm monitoring reduces "
                                            "stroke risk")
    st.header("Caller credentials (ZKP tokens)")
    caller = st.multiselect("Attributes the caller proves",
                            options=CALLER_TOKENS,
                            default=["role:doctor", "dept:cardio"])
    if not caller:
        st.warning("Pick at least one credential.")
    k = st.slider("Top-k return", 1, 5, 3)
    st.header("Pipeline options")
    generate = st.checkbox("LLM-generate answer (live qwen2.5:1.5b)",
                           value=True)
    redact = st.checkbox(
        "Redact ungrounded spans (RedactSeam)", value=False)
    do_dp = st.checkbox("Privatise a numeric token (fixed-eps DP)", value=False)
    if do_dp:
        eps = st.slider("epsilon", 0.1, 10.0, 2.0, 0.1)
    else:
        eps = None
    offline = st.checkbox("Force offline (hash-BoW, no Ollama)", value=False)
    lan = st.checkbox("Enable LAN access (bind 0.0.0.0)", value=False)
    run = st.button("Run query", type="primary", width="stretch")

if offline:
    rig.adapter._session = _DownSession()

mode = rig.mode
if offline or ("fallback" in mode):
    st.warning("Embedding mode: **fallback (hash-BoW)** — deterministic "
               "no-Ollama route. Live mode restarts once Ollama answers "
               "and the offline toggle is off.")
else:
    st.success(f"Embedding mode: **live ({rig.adapter.embed_model})** — "
               f"dim n={rig.dim}, group {GROUP_BITS} bits, "
               f"quant {QUANT_BITS} bits")

if not run:
    st.info("Choose the caller credentials and press **Run query** to see the "
            "full encrypted-retrieval stages.")

# ---------------- access matrix + caller inventory ----------------
# (always visible; depends only on the sidebar caller selection)
st.header("What can THIS caller read?")
inv = ", ".join(sorted(set(caller))) if caller else "(none)"
st.caption(f"Credentials proved in one ZKP round: **{inv}**")

matrix = access_matrix(rig, sorted(set(caller))) if caller else [
    {"doc_id": d, "tag": t, "allowed": False,
     "missing": ["no credentials selected"]} for d, t, _x in CORPUS]

dfa = pd.DataFrame([
    {"doc": r["doc_id"], "policy": r["tag"], "access": "ALLOW" if r["allowed"]
     else "DENY", "reason": ("—" if r["allowed"]
                              else "needs " + " ^ ".join(r["missing"]))}
    for r in matrix
])

def _style(v):
    return ("background-color: #dfedd8; color: #1f5a24"
            if v == "ALLOW" else "background-color: #f4dddc; color: #8a2323")

st.dataframe(dfa.style.map(_style, subset=["access"]),
             width="stretch", hide_index=True)

# ---------------- inline reader ----------------
st.header("Reader")
cols = st.columns(len(CORPUS))
for col, (doc_id, _tag, text) in zip(cols, CORPUS):
    row = next(r for r in matrix if r["doc_id"] == doc_id)
    if row["allowed"]:
        col.success(f"**{doc_id}** — authorized")
        col.markdown(text)
    else:
        col.error(f"**{doc_id}** — locked")
        col.markdown(f"policy: `{row['tag']}`")
        if row["missing"]:
            col.caption("missing: " + ", ".join(row["missing"]))
        col.caption("*text not rendered for unauthorized callers*")

# ---------------- pipeline stages (require a run) ----------------
if not run:
    st.stop()

if not caller:
    st.error("No caller selected; nothing was run.")
    st.stop()

with st.spinner("running the encrypted-retrieval stack ..."):
    try:
        rec = run_pipeline(rig, query=query, caller_toks=sorted(set(caller)),
                           k=k, generate=generate, redact=redact, dp=eps,
                           artifact=None, audit=None)
        st.session_state["last"] = rec
    except Exception as exc:  # noqa: BLE001
        st.session_state.pop("last", None)
        st.error(f"pipeline failed: {exc}")
        st.stop()

rec = st.session_state["last"]
if rec.get("cloud", {}).get("healed"):
    st.info("This run **self-healed**: the cloud server had died and was "
            "restarted + re-ingested automatically before the query.")
st.header("How the answer was computed")

top = pd.DataFrame([
    {"doc": r["doc_id"], "score": r["score"], "rank": i + 1}
    for i, r in enumerate(rec["top_k"])
])
lt, rt = st.columns([1, 1])
with lt:
    st.subheader("Top-k retrieval (encrypted dot-score)")
    st.bar_chart(top, x="doc", y="score")
with rt:
    st.subheader("ZKP gate")
    st.markdown(
        f"{len(rec['caller'])} tokens proven + verified in **one round** "
        "(Schnorr Fiat-Shamir); proofs never leave the client.")
    st.subheader("Wire body (POST /query)")
    st.code(json.dumps(rec["wire_body"], indent=2), language="json")
    st.caption("exact key-set {sk_q, q_ints, attrs} — no floats, no proofs")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Grounding")
    g = rec["grounding"]
    st.metric("fraction grounded", f"{g['fraction_grounded']:.2f}")
    if g["ungrounded_spans"]:
        st.write("ungrounded spans", g["ungrounded_spans"])
    else:
        st.write("fully grounded against retrieved excerpts.")
    st.subheader("Answer")
    if g["source"] == "generated":
        gen = g.get("generation") or {}
        st.markdown("**LLM-generated · live `qwen2.5:1.5b`**" + (
            "" if not gen.get("fell_back")
            else " — *Ollama unreachable, fell back to top-1 excerpt*"))
        if gen.get("elapsed_s") is not None:
            st.caption(f"prompt from top-k notes → model call "
                       f"{gen['elapsed_s']}s")
    else:
        st.markdown(
            "**Retrieved (derived from top-1 excerpt)** — tick "
            "*LLM-generate answer* in the sidebar to see a live model "
            "response.")
    st.write(g["answer"])
with c2:
    st.subheader("Redaction pass")
    if rec["redact"]:
        rd = rec["redact"]
        if rd["applied"]:
            st.code(rd["text"], language=None)
        else:
            st.write("no-op:", rd["refused"] or "applied")
        st.caption(f"spans={rd['n_spans']}  masked_chars={rd['masked_chars']}")
    else:
        st.write("(redact off)")
    st.subheader("DP (numeric token)")
    if rec["dp"]:
        d = rec["dp"]
        if d["privatised"] is None and "note" in d:
            st.write(d["note"])
        else:
            st.write(f"eps={d['eps']}: {d['input']} -> {d['output']}")
    else:
        st.write("(DP off)")

st.caption(f"query took {rec['query_ms']:.0f} ms; total {rec['total_s']:.2f}s "
           f"· corpus {len(rec['corpus']['ids'])}/5 verified · n={rec['n']}")