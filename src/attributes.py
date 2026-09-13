# Copyright (C) 2026 SafeRAG-Improved Authors
# SPDX-License-Identifier: MIT
"""Attribute universe + access-tree (SafeRAG Section III-C: policy tagging).

SafeRAG tags each document d with an access tree T_d built from attribute
index terms (role, department, clearance) and evaluates it against the
caller's attribute set. Every term in a *query token* is itself an attribute
index term, so authorisation is checked per top-k candidate before the IPFE
inner product is ever decrypted (Algorithms 3-7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Attribute universe (SafeRAG Table I / Section III-C.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Attribute:
    """A single key:value attribute, e.g. ``role:Doctor``."""
    key: str
    value: str

    def __str__(self) -> str:
        return f"{self.key}:{self.value}"

    @classmethod
    def parse(cls, text: str) -> "Attribute":
        text = text.strip()
        if ":" not in text:
            raise ValueError(
                f"attribute must be explicit 'key:value', got {text!r}; "
                "write 'role:Doctor' (not 'Doctor') - the DSL never guesses a namespace")
        k, _, v = text.partition(":")
        return cls(k.strip().lower(), v.strip().lower())


@dataclass(frozen=True)
class Attributes:
    """Set of attribute tokens for a user/role. Immutable & hashable."""

    toks: frozenset[str] = field(default_factory=frozenset)

    def __init__(self, toks):
        # normalise all tokens to lower-case key:value
        norm = {t.strip().lower() for t in toks}
        object.__setattr__(self, "toks", frozenset(norm))

    @classmethod
    def from_roles(cls, roles):
        return cls({f"role:{r}" for r in roles})

    def __contains__(self, tok: str) -> bool:
        return tok in self.toks

    def __iter__(self):
        return iter(self.toks)

    def __len__(self) -> int:
        return len(self.toks)

    def has(self, prefix: str) -> bool:
        return any(t.startswith(prefix) for t in self.toks)

    def get(self, key: str) -> str | None:
        for t in self.toks:
            if ":" in t and t.split(":", 1)[0] == key:
                return t.split(":", 1)[1]
        return None

    def role(self) -> set[str]:
        return {t.split(":", 1)[1] for t in self.toks if t.startswith("role:")}


# --------------------------------------------------------------------------- #
# Access tree nodes (SafeRAG: AND / OR / k-of-n / attribute leaf)
# --------------------------------------------------------------------------- #


class AccessNode:
    def satisfies(self, attrs: Attributes) -> bool:
        raise NotImplementedError

    def min_attrs(self) -> int:
        raise NotImplementedError

    def __or__(self, other: "AccessNode") -> "OrNode":
        return OrNode([self, other])

    def __and__(self, other: "AccessNode") -> "AndNode":
        return AndNode([self, other])


@dataclass
class LeafNode(AccessNode):
    """Leaf: caller must hold attribute ``tok`` (e.g. ''role:Doctor'')."""

    tok: str
    value: str | None = None

    def __init__(self, attr: Attribute | str):
        if isinstance(attr, str):
            attr = Attribute.parse(attr)
        self.tok = f"{attr.key}:{attr.value}"
        self.value = attr.value

    @property
    def key(self) -> str:
        """Attribute key, derived from ``tok`` (kept out of dataclass eq/repr)."""
        return self.tok.split(":", 1)[0]

    def satisfies(self, attrs: Attributes) -> bool:
        return self.tok in attrs

    def min_attrs(self) -> int:
        return 1


@dataclass
class AndNode(AccessNode):
    children: list[AccessNode]

    def satisfies(self, attrs: Attributes) -> bool:
        return all(c.satisfies(attrs) for c in self.children)

    def min_attrs(self) -> int:
        return sum(c.min_attrs() for c in self.children)


@dataclass
class OrNode(AccessNode):
    children: list[AccessNode]

    def satisfies(self, attrs: Attributes) -> bool:
        return any(c.satisfies(attrs) for c in self.children)

    def min_attrs(self) -> int:
        return min(c.min_attrs() for c in self.children)


@dataclass
class ThresholdNode(AccessNode):
    """k-of-n gate: k leaves/children must be satisfied (Section III-C.3)."""

    threshold: int
    children: list[AccessNode]

    def __init__(self, threshold: int, children):
        self.threshold = threshold
        self.children = children

    def satisfies(self, attrs: Attributes) -> bool:
        return sum(c.satisfies(attrs) for c in self.children) >= self.threshold

    def min_attrs(self) -> int:
        return self.threshold


def k_of_n(k: int, *children: AccessNode) -> ThresholdNode:
    return ThresholdNode(k, list(children))


def build_tree(expr: str) -> AccessNode:
    """DSL: 'role:Doctor AND (dept:Cardio OR dept:Neuro) AND 2-of(Clearance:2,role:Nurse)'.

    Every attribute is explicit ``key:value``; colonless bare words raise
    ValueError (no namespace guessing - 'Doctor' vs 'Cardio' are never
    distinguished implicitly, only by their explicit key).
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("empty access-tree expression")
    return _parse_or(expr)


def _parse_or(s: str) -> AccessNode:
    parts = _split_top(s, "OR")
    nodes = [_parse_and(p) for p in parts]
    return OrNode(nodes) if len(nodes) > 1 else nodes[0]


def _parse_and(s: str) -> AccessNode:
    parts = _split_top(s, "AND")
    nodes = [_parse_atom(p) for p in parts]
    return AndNode(nodes) if len(nodes) > 1 else nodes[0]


def _parse_atom(s: str) -> AccessNode:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return _parse_or(s[1:-1])
    m = re.match(r"(?:k|(\d+))-of\(", s, re.IGNORECASE)
    if m:
        if m.group(1) is None:
            raise ValueError(
                f"literal 'k-of' placeholder in {s!r}; use an integer threshold "
                "(e.g. '2-of(...)') so the policy is well-defined")
        k = int(m.group(1))
        inner = s[s.index("(") + 1: s.rindex(")")]
        children = [_parse_atom(x) for x in _split_top(inner, ",")]
        return ThresholdNode(k, children)
    # leaf attribute token, strip quotes
    return LeafNode(Attribute.parse(s.strip('"').strip()))


def _split_top(s: str, sep: str) -> list[str]:
    """Split on separator not inside parentheses."""
    out, depth, cur = [], 0, ""
    i = 0
    tokens = re.split(r"(\(|\)|,|\bAND\b|\bOR\b)", s)  # noqa: SIM905
    for tok in tokens:
        if tok == "":
            continue
        if tok == "(":
            depth += 1
            cur += tok
        elif tok == ")":
            depth -= 1
            cur += tok
        elif depth == 0 and tok.strip() == sep:
            out.append(cur)
            cur = ""
        else:
            cur += tok
    out.append(cur)
    return [x for x in out if x.strip()]
