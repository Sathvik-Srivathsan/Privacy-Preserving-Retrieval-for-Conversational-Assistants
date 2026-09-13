# -*- coding: utf-8 -*-
"""
Access-tree tests (Phase-1 T1) —— SafeRAG Section III-C.

Covers: Attribute / Attributes universe, Leaf / And / Or / Threshold(k-of-n)
nodes, operator composition (|, &), and the build_tree DSL
(leaf, AND, OR, parens, 2-of/k-of with comma args, nested, quoted, case).

Run:  python tests\test_attributes.py      (from repo root)
Exit 0 iff ALL_OK.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
REPO = SELF.parents[1]
sys.path.insert(0, str(REPO / "src"))

from attributes import (  # noqa: E402
    Attribute, Attributes, LeafNode, AndNode, OrNode,
    ThresholdNode, k_of_n, build_tree)

# --------------------------------------------------------------------------- #
# Attribute universe
# --------------------------------------------------------------------------- #


def test_attribute_parse_key_lower():
    a = Attribute.parse("Role:Doctor")
    assert a.key == "role" and a.value == "doctor"


def test_attribute_parse_value_lower():
    a = Attribute.parse("role:DOCTOR")
    assert a.value == "doctor"


def test_attribute_str_roundtrip():
    a = Attribute.parse("role:Doctor")
    assert str(a) == "role:doctor"


def test_attribute_parse_no_colon_raises():
    try:
        Attribute.parse("  Doctor ")
    except ValueError as exc:
        assert "role:Doctor" in str(exc)
        return
    raise AssertionError("colonless attribute should raise ValueError")


def test_leaf_explicit_role_matches_from_roles():
    # regression: an explicit role: leaf must match the SAME token namespace
    # that Attributes.from_roles() emits ("role:doctor")
    leaf = LeafNode("role:Doctor")
    assert leaf.tok == "role:doctor"
    assert leaf.satisfies(Attributes.from_roles(["doctor"]))
    assert not leaf.satisfies(Attributes.from_roles(["nurse"]))


def test_bare_word_requires_namespace():
    # no guessing: 'Doctor' and 'Cardio' are NOT distinguished implicitly;
    # both must carry an explicit key (role: vs dept:). Colonless words raise.
    for bad in ("Doctor", "Cardio", "Doctor AND (Cardio OR Neuro)"):
        try:
            build_tree(bad)
        except ValueError as exc:
            assert "role:Doctor" in str(exc)
        else:
            raise AssertionError(f"bare word should raise: {bad!r}")
    try:
        LeafNode("Cardio")
    except ValueError:
        pass
    else:
        raise AssertionError("bare LeafNode should raise")


def test_attribute_parse_key_value_whitespace():
    a = Attribute.parse(" role : Doctor ")
    assert a.key == "role" and a.value == "doctor"


# --------------------------------------------------------------------------- #
# Attributes (caller token set)
# --------------------------------------------------------------------------- #


def test_attributes_normalize_case():
    at = Attributes(["Role:Doctor", "Dept:Cardio"])
    assert at.toks == frozenset({"role:doctor", "dept:cardio"})


def test_attributes_contains_and_len():
    at = Attributes(["role:doctor", "clearance:2"])
    assert len(at) == 2
    assert "role:doctor" in at
    assert "role:nurse" not in at


def test_attributes_get():
    at = Attributes(["role:doctor", "clearance:2"])
    assert at.get("role") == "doctor"
    assert at.get("clearance") == "2"
    assert at.get("dept") is None


def test_attributes_has():
    at = Attributes(["role:doctor", "clearance:2"])
    assert at.has("role:")
    assert not at.has("dept:")


def test_attributes_role():
    at = Attributes(["role:doctor", "role:nurse", "clearance:2"])
    assert at.role() == {"doctor", "nurse"}


def test_attributes_from_roles():
    at = Attributes.from_roles(["doctor", "NURSE"])
    assert at.toks == frozenset({"role:doctor", "role:nurse"})


def test_attributes_empty():
    at = Attributes([])
    assert len(at) == 0


# --------------------------------------------------------------------------- #
# Leaf node
# --------------------------------------------------------------------------- #


def test_leaf_satisfies_positive():
    leaf = LeafNode("role:Doctor")
    assert leaf.satisfies(Attributes(["role:doctor"]))


def test_leaf_satisfies_negative():
    leaf = LeafNode("role:Doctor")
    assert not leaf.satisfies(Attributes(["role:nurse"]))


def test_leaf_min_attrs():
    assert LeafNode("role:Doctor").min_attrs() == 1


def test_leaf_from_attribute_obj():
    leaf = LeafNode(Attribute.parse("dept:Cardio"))
    assert leaf.tok == "dept:cardio"
    assert leaf.key == "dept" and leaf.value == "cardio"


# --------------------------------------------------------------------------- #
# And / Or
# --------------------------------------------------------------------------- #


def test_and_all_true():
    node = AndNode([LeafNode("a:1"), LeafNode("b:2")])
    assert node.satisfies(Attributes(["a:1", "b:2"]))


def test_and_one_false():
    node = AndNode([LeafNode("a:1"), LeafNode("b:2")])
    assert not node.satisfies(Attributes(["a:1"]))


def test_and_min_attrs_sum():
    node = AndNode([LeafNode("a:1"), LeafNode("b:2")])
    assert node.min_attrs() == 2


def test_or_any_true():
    node = OrNode([LeafNode("a:1"), LeafNode("b:2")])
    assert node.satisfies(Attributes(["a:1"]))


def test_or_none_true():
    node = OrNode([LeafNode("a:1"), LeafNode("b:2")])
    assert not node.satisfies(Attributes(["c:3"]))


def test_or_min_attrs_min():
    node = OrNode([LeafNode("a:1"), ThresholdNode(3, [LeafNode("b:2"), LeafNode("c:3"), LeafNode("d:4")])])
    assert node.min_attrs() == 1


# --------------------------------------------------------------------------- #
# Threshold (k-of-n)
# --------------------------------------------------------------------------- #


def test_kof_satisfies_at_k():
    node = k_of_n(2, LeafNode("a:1"), LeafNode("b:2"), LeafNode("c:3"))
    assert node.satisfies(Attributes(["a:1", "b:2"]))


def test_kof_more_than_k():
    node = k_of_n(2, LeafNode("a:1"), LeafNode("b:2"), LeafNode("c:3"))
    assert node.satisfies(Attributes(["a:1", "b:2", "c:3"]))


def test_kof_under_threshold():
    node = k_of_n(2, LeafNode("a:1"), LeafNode("b:2"), LeafNode("c:3"))
    assert not node.satisfies(Attributes(["a:1"]))


def test_kof_min_attrs():
    assert k_of_n(3, LeafNode("a:1"), LeafNode("b:2"), LeafNode("c:3"), LeafNode("d:4")).min_attrs() == 3


def test_threshold_children_exceed_count():
    node = ThresholdNode(2, [LeafNode("a:1"), LeafNode("b:2"), LeafNode("c:3")])
    assert node.satisfies(Attributes(["b:2", "c:3"]))


# --------------------------------------------------------------------------- #
# Operator composition (AccessNode | / &)
# --------------------------------------------------------------------------- #


def test_operator_and_or_compose():
    clue = LeafNode("role:doctor") & (LeafNode("dept:cardio") | LeafNode("dept:neuro"))
    assert isinstance(clue, AndNode)
    assert clue.satisfies(Attributes(["role:doctor", "dept:cardio"]))
    assert clue.satisfies(Attributes(["role:doctor", "dept:neuro"]))
    assert not clue.satisfies(Attributes(["role:doctor", "dept:onco"]))


# --------------------------------------------------------------------------- #
# build_tree DSL
# --------------------------------------------------------------------------- #


def test_dsl_single_leaf():
    node = build_tree("role:Doctor")
    assert isinstance(node, LeafNode)
    assert node.satisfies(Attributes(["role:doctor"]))


def test_dsl_and():
    node = build_tree("role:Doctor AND clearance:2")
    assert isinstance(node, AndNode)
    assert node.satisfies(Attributes(["role:doctor", "clearance:2"]))
    assert not node.satisfies(Attributes(["role:doctor"]))


def test_dsl_or():
    node = build_tree("dept:Cardio OR dept:Neuro")
    assert isinstance(node, OrNode)
    assert node.satisfies(Attributes(["dept:cardio"]))
    assert node.satisfies(Attributes(["dept:neuro"]))
    assert not node.satisfies(Attributes(["dept:onco"]))


def test_dsl_paren_grouping():
    node = build_tree("role:Doctor AND (dept:Cardio OR dept:Neuro)")
    assert isinstance(node, AndNode)
    assert len(node.children) == 2
    assert isinstance(node.children[1], OrNode)
    assert node.satisfies(Attributes(["role:doctor", "dept:cardio"]))
    assert node.satisfies(Attributes(["role:doctor", "dept:neuro"]))
    assert not node.satisfies(Attributes(["role:doctor", "dept:onco"]))


def test_dsl_2of_with_commas():
    node = build_tree("2-of(Clearance:2,role:Nurse)")
    assert isinstance(node, ThresholdNode)
    assert node.threshold == 2
    assert len(node.children) == 2
    assert node.satisfies(Attributes(["clearance:2", "role:nurse"]))
    assert not node.satisfies(Attributes(["clearance:2"]))


def test_dsl_3of_with_commas():
    node = build_tree("3-of(a:1,b:2,c:3)")
    assert isinstance(node, ThresholdNode)
    assert node.threshold == 3
    assert node.satisfies(Attributes(["a:1", "b:2", "c:3"]))
    assert not node.satisfies(Attributes(["a:1", "b:2"]))


def test_dsl_nested_full_example():
    # explicit namespaces throughout: role:Doctor is a role, dept:Cardio/Neuro
    # are departments, clearance:2 and role:Nurse explicit. No guessing anywhere.
    node = build_tree("role:Doctor AND (dept:Cardio OR dept:Neuro) AND 2-of(Clearance:2,role:Nurse)")
    assert isinstance(node, AndNode)
    assert len(node.children) == 3
    base = set(Attributes.from_roles(["doctor", "nurse"]).toks)
    ok = Attributes(base | {"dept:cardio", "clearance:2"})
    thresh = set(Attributes.from_roles(["doctor"]).toks)
    thresh_fail = Attributes(thresh | {"dept:cardio", "clearance:2"})       # 1/2 of threshold (no role:nurse)
    wrong_dept = set(Attributes.from_roles(["doctor", "nurse"]).toks)
    miss_cardio = Attributes(wrong_dept | {"dept:onco", "clearance:2"})     # onco != cardio/neuro
    assert node.satisfies(ok)
    assert not node.satisfies(thresh_fail)
    assert not node.satisfies(miss_cardio)


def test_dsl_role_and_dept_disambiguated():
    # the reviewer's exact question: 'Doctor' vs 'Cardio' are told apart ONLY
    # by their explicit keys (role: vs dept:) - never by a keyword list.
    node = build_tree("role:Doctor AND (dept:Cardio OR dept:Neuro)")
    assert node.satisfies(Attributes(set(Attributes.from_roles(["doctor"]).toks) | {"dept:cardio"}))
    assert node.satisfies(Attributes(set(Attributes.from_roles(["doctor"]).toks) | {"dept:neuro"}))
    assert not node.satisfies(Attributes(set(Attributes.from_roles(["doctor"]).toks) | {"dept:onco"}))
    assert not node.satisfies(Attributes(set(Attributes.from_roles(["nurse"]).toks) | {"dept:cardio"}))  # doctor missing


def test_dsl_paper_style_role_policy():
    # base paper writes role policies as "Role IN {Doctor, Nurse, Patient}";
    # the explicit DSL form satisfies Attributes.from_roles() end-to-end.
    node = build_tree("role:Doctor OR role:Nurse OR role:Patient")
    assert node.satisfies(Attributes.from_roles(["doctor"]))
    assert node.satisfies(Attributes.from_roles(["nurse"]))
    assert node.satisfies(Attributes.from_roles(["patient"]))
    assert not node.satisfies(Attributes.from_roles(["technician"]))


def test_dsl_quoted_leaf():
    node = build_tree('"role:Doctor"')
    assert isinstance(node, LeafNode)
    assert node.satisfies(Attributes(["role:doctor"]))


def test_dsl_case_insensitive_leaf_value():
    node = build_tree("role:Nurse")
    assert node.tok == "role:nurse"
    assert node.satisfies(Attributes(["role:NURSE"]))


def test_dsl_empty_raises():
    try:
        build_tree("")
    except ValueError as exc:
        assert "empty" in str(exc)
        return
    raise AssertionError("empty expression should raise ValueError")


def test_kof_literal_placeholder_raises():
    # "k-of(...)" is a doc placeholder, not a legal threshold: fail loudly
    try:
        build_tree("k-of(a:1,b:2)")
    except ValueError as exc:
        assert "placeholder" in str(exc)
        return
    raise AssertionError("literal 'k-of' should raise ValueError")


def test_kof_zero_and_one_child():
    one = build_tree("1-of(a:1)")
    assert one.threshold == 1
    assert not one.satisfies(Attributes([]))
    assert one.satisfies(Attributes(["a:1"]))
    two_with_one = build_tree("2-of(a:1)")
    assert two_with_one.threshold == 2
    assert not two_with_one.satisfies(Attributes(["a:1"]))  # 1<2, never satisfiable


# --------------------------------------------------------------------------- #
# Negatives / edge behaviour
# --------------------------------------------------------------------------- #


def test_neg_empty_attrs_fails_everything():
    node = build_tree("(a:1 OR b:2) AND 2-of(a:1,c:3)")
    assert not node.satisfies(Attributes([]))


def test_neg_wrong_value_matches_nothing():
    node = build_tree("role:Doctor AND clearance:2")
    assert not node.satisfies(Attributes(["role:doctor", "clearance:3"]))


def test_neg_threshold_not_reached_in_dsl():
    node = build_tree("2-of(a:1,b:2,c:3)")
    assert node.min_attrs() == 2
    assert not node.satisfies(Attributes(["b:2"]))


def test_neg_min_attrs_matches_structure():
    node = build_tree("role:Doctor AND (dept:Cardio OR dept:Neuro) AND 2-of(Clearance:2,role:Nurse)")
    assert node.min_attrs() == 1 + 1 + 2


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
        "module": "attributes",
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