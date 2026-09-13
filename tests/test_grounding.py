# -*- coding: utf-8 -*-
"""
Grounding tests (Phase-1 T5) —— verify_grounded overlap scorer.

Contract under test: every sentence in the response must have a lexical
overlap anchor (>=1 shared token) with at least one authorised chunk;
`GroundingReport` yields fraction_grounded + (start,end,text) spans for the
ungrounded sentences.

Run:  python tests\test_grounding.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO))

from src.grounding import GroundingReport, verify_grounded  # noqa: E402


def test_fully_grounded_single_sentence():
    rep = verify_grounded("The patient has heart failure.",
                          ["heart failure is common"])
    assert rep.fraction_grounded == 1.0
    assert rep.ungrounded_spans == []


def test_partially_grounded_fraction():
    rep = verify_grounded(
        "Fever was recorded. Purple elephants flew.",
        ["Fever is a symptom."])
    assert rep.fraction_grounded == 0.5
    assert len(rep.ungrounded_spans) == 1
    text = rep.ungrounded_spans[0][2]
    assert text == "Purple elephants flew."


def test_fully_ungrounded_no_chunks():
    rep = verify_grounded("Something entirely novel.", [])
    assert rep.fraction_grounded == 0.0
    assert rep.ungrounded_spans == [(0, len("Something entirely novel."),
                                     "Something entirely novel.")]


def test_empty_response_is_safe():
    rep = verify_grounded("", ["any chunk"])
    assert rep.fraction_grounded == 1.0
    assert rep.ungrounded_spans == []


def test_whitespace_only_response_is_safe():
    rep = verify_grounded("   \n\t  ", ["any chunk"])
    assert rep.fraction_grounded == 1.0
    assert rep.ungrounded_spans == []


def test_case_insensitive_overlap():
    rep = verify_grounded("HEART failing.",
                          ["heart failure is common"])
    assert rep.fraction_grounded == 1.0


def test_numeric_tokens_overlap():
    rep = verify_grounded("The dose is 10mg twice daily.",
                          ["prescribe 10mg on day one"])
    assert rep.fraction_grounded == 1.0


def test_any_chunk_may_anchor():
    rep = verify_grounded("take aspirin daily.",
                          ["nothing relevant here", "aspirin reduces risk"])
    assert rep.fraction_grounded == 1.0


def test_question_and_exclamation_split():
    rep = verify_grounded(
        "Is it serious? Stop worrying!",
        ["it is serious", "stop worrying"])
    assert rep.fraction_grounded == 1.0
    assert rep.ungrounded_spans == []


def test_apostrophes_treated_as_single_token():
    rep = verify_grounded("The patient's chart is available.",
                          ["patient's chart"])
    assert rep.fraction_grounded == 1.0


def test_span_positions_advance_across_sentences():
    rep = verify_grounded("Good first sentence. Bad second! And bad third.",
                          ["first sentence"])
    assert rep.fraction_grounded == 1.0 / 3.0
    spans = rep.ungrounded_spans
    assert len(spans) == 2
    # advance = len(sentence)+1 per processed sentence
    assert spans[0][0] == len("Good first sentence.") + 1
    assert spans[1][0] == spans[0][1] + 1
    assert spans[0][2] == "Bad second!"
    assert spans[1][2] == "And bad third."


def test_multi_chunk_no_match_is_ungrounded():
    rep = verify_grounded("Zebra quokka frolic.",
                          ["matching is missing", "still nothing similar"])
    assert rep.fraction_grounded == 0.0
    assert len(rep.ungrounded_spans) == 1


def test_stopword_only_overlap_is_NOT_grounded():
    # the failure mode stopword-stripping exists for: shares only
    # "the"/"is"/"a" with the chunk -> content is completely unrelated
    rep = verify_grounded("The dog is a cat.",
                          ["The stock market is a factor."])
    assert rep.fraction_grounded == 0.0
    assert [sp[2] for sp in rep.ungrounded_spans] == ["The dog is a cat."]


def test_content_word_overlap_is_grounded_despite_stopwords():
    # contract stays honest: one shared CONTENT word still anchors, even
    # though the only "the/is" tokens also coincide
    rep = verify_grounded("Patients present with fever.",
                          ["The fever is notable."])
    assert rep.fraction_grounded == 1.0
    assert rep.ungrounded_spans == []


def test_report_default_spans_list():
    rep = GroundingReport(0.5)
    assert rep.ungrounded_spans == []


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
        "module": "grounding",
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