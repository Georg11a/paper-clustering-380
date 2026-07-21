#!/usr/bin/env python3
"""Deterministic, literature-informed first-pass coding of theory moves.

The four codes synthesize several strands of prior work; they are not presented
as a verbatim taxonomy from Gregor (2006) or Gregor and Jones (2007). The
implementation follows directed content analysis: explicit textual indicators
are coded per paper, then aggregated at cluster level. Ambiguous or weakly
supported clusters remain unclassified for human review.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PatternRule:
    name: str
    expression: str
    weight: int


@dataclass(frozen=True)
class TheoryMoveResult:
    key: str
    label: str
    score: int
    support_count: int
    paper_count: int
    matched_patterns: tuple[str, ...]
    category_scores: tuple[tuple[str, int], ...]
    category_support: tuple[tuple[str, int], ...]

    @property
    def support_text(self) -> str:
        if self.key == "unclear":
            if self.score:
                return (
                    f"No unambiguous move; best candidate has support from "
                    f"{self.support_count}/{self.paper_count} papers, weighted score {self.score}"
                )
            return f"No theory-move signal ({self.paper_count} papers reviewed)"
        return f"{self.support_count}/{self.paper_count} papers; weighted score {self.score}"

    @property
    def patterns_text(self) -> str:
        return "; ".join(self.matched_patterns) if self.matched_patterns else "No decisive pattern match"


THEORY_MOVE_LABELS = {
    "building": "Building New Theory",
    "borrowing": "Borrowing and Adapting Existing Theory",
    "testing": "Testing Theory Empirically",
    "meta_reflection": "Meta-Theoretical Reflection on Design",
    "unclear": "Unclear Theory Move — Requires Human Review",
}


# More specific categories win only as a final deterministic tie-breaker.
THEORY_MOVE_PRIORITY = {
    "meta_reflection": 4,
    "borrowing": 3,
    "testing": 2,
    "building": 1,
}


THEORY_MOVE_PATTERNS = {
    "building": (
        PatternRule(
            "explicit theory building/development",
            r"\btheor(?:y|ies)\s+(?:building|development|construction)\b|"
            r"\b(?:build|building|develop|developing|construct|constructing)\s+(?:an?\s+)?(?:new\s+)?theor(?:y|ies)\b",
            5,
        ),
        PatternRule(
            "proposes a new theory/framework",
            r"\b(?:propose|proposes|proposed|introduce|introduces|introduced|develop|develops|developed|"
            r"formulate|formulates|formulated|present|presents)\s+(?:an?\s+|the\s+)?(?:new\s+|novel\s+)?"
            r"(?:design\s+)?(?:theory|theoretical\s+framework|conceptual\s+framework|theoretical\s+model)\b",
            4,
        ),
        PatternRule(
            "new theoretical constructs/propositions",
            r"\b(?:new|novel)\s+(?:theoretical\s+)?(?:constructs?|conceptuali[sz]ations?|propositions?)\b|"
            r"\breconceptuali[sz](?:e|es|ed|ing|ation)\b",
            3,
        ),
    ),
    "borrowing": (
        PatternRule(
            "explicitly borrows/adapts theory",
            r"\b(?:borrow|borrows|borrowed|borrowing|adapt|adapts|adapted|adapting)\s+"
            r"(?:an?\s+|the\s+)?(?:existing\s+)?(?:theory|theoretical\s+framework|conceptual\s+framework)\b",
            5,
        ),
        PatternRule(
            "draws on a theory or theoretical lens",
            r"\b(?:draw|draws|drawing)\s+(?:directly\s+)?(?:on|from)\s+[^.;:]{0,80}\b"
            r"(?:theory|theories|theoretical\s+framework|theoretical\s+lens)\b",
            4,
        ),
        PatternRule(
            "applies/adopts an external theory",
            r"\b(?:apply|applies|applied|adopt|adopts|adopted|use|uses|used)\s+"
            r"(?:an?\s+|the\s+)?(?:existing\s+|external\s+|established\s+)?[^.;:]{0,60}\b"
            r"(?:theory|theoretical\s+lens)\b",
            4,
        ),
        PatternRule(
            "grounded/informed by theory",
            r"\b(?:grounded|rooted|informed)\s+(?:in|by)\s+[^.;:]{0,80}\b"
            r"(?:theory|theoretical\s+framework|theoretical\s+lens)\b",
            3,
        ),
        PatternRule(
            "reference discipline/kernel theory",
            r"\b(?:reference\s+discipline|kernel\s+theor(?:y|ies)|theor(?:y|ies)\s+from\s+(?:another|other)\s+(?:field|discipline))\b",
            4,
        ),
    ),
    "testing": (
        PatternRule(
            "testing a theory/framework/model",
            r"\btesting\s+(?:an?\s+|the\s+|our\s+)?(?:theory|theoretical\s+framework|conceptual\s+framework|framework|model)\b",
            5,
        ),
        PatternRule(
            "empirically tests/evaluates theory",
            r"\b(?:empirically\s+)?(?:test|tests|tested|testing|evaluate|evaluates|evaluated|validate|validates|validated)\s+"
            r"(?:an?\s+|the\s+|our\s+)?(?:proposed\s+|existing\s+)?(?:theory|theoretical\s+framework|conceptual\s+framework|model)\b",
            5,
        ),
        PatternRule(
            "tests hypotheses/propositions",
            r"\b(?:test|tests|tested|testing|support|supports|supported)\s+(?:the\s+|our\s+)?"
            r"(?:hypotheses|hypothesis|propositions?)\b|"
            r"\b(?:testable|empirically\s+testable)\s+(?:hypotheses|hypothesis|propositions?)\b",
            4,
        ),
        PatternRule(
            "empirical validation/evaluation of framework",
            r"\b(?:empirical|quantitative|experimental)\s+(?:validation|evaluation|test)\s+"
            r"(?:of\s+)?(?:an?\s+|the\s+|our\s+)?(?:theory|framework|model|propositions?)\b|"
            r"\b(?:theory|framework|model)\s+(?:was|is|were)\s+(?:empirically\s+)?(?:validated|tested|evaluated)\b",
            4,
        ),
        PatternRule(
            "hypothesis-driven empirical study",
            r"\b(?:hypothesis|hypotheses|testable\s+propositions?)\b.{0,100}\b"
            r"(?:empirical|experiment|survey|data|results?)\b|"
            r"\b(?:empirical|experiment|survey|data|results?)\b.{0,100}\b(?:hypothesis|hypotheses)\b",
            3,
        ),
    ),
    "meta_reflection": (
        PatternRule(
            "explicit meta-theory/metatheory",
            r"\b(?:meta[- ]?theor(?:y|ies|etical)|metatheor(?:y|ies|etical))\b",
            6,
        ),
        PatternRule(
            "philosophy/epistemology/ontology of design",
            r"\b(?:philosoph(?:y|ical)|epistemolog(?:y|ical)|ontolog(?:y|ical))\s+of\s+(?:design|design\s+research|design\s+theory)\b|"
            r"\bdesign\s+(?:philosoph(?:y|ical)|epistemolog(?:y|ical)|ontolog(?:y|ical))\b",
            5,
        ),
        PatternRule(
            "reflects on the nature/role/value of theory",
            r"\b(?:nature|role|status|value|meaning|foundations?)\s+of\s+(?:design\s+)?theor(?:y|ies)\b|"
            r"\bwhat\s+(?:is|counts\s+as|constitutes)\s+(?:a\s+)?(?:design\s+)?theory\b",
            5,
        ),
        PatternRule(
            "theorizing about design research",
            r"\b(?:theori[sz]ing|theori[sz]ation)\s+(?:in|about|within)\s+(?:design|design\s+research|design\s+science)\b|"
            r"\bcritical\s+(?:reflection|debate|analysis)\s+(?:on|of|about)\s+(?:design\s+)?theor(?:y|ies)\b",
            4,
        ),
    ),
}


TEXT_FIELDS = (
    "title",
    "abstract",
    "discussion_summary",
    "discussion_excerpt",
)


def _paper_text(paper: Mapping[str, object]) -> str:
    return " ".join(str(paper.get(field, "") or "") for field in TEXT_FIELDS).lower()


def classify_theory_move(papers: Iterable[Mapping[str, object]]) -> TheoryMoveResult:
    paper_list = list(papers)
    category_scores = Counter({key: 0 for key in THEORY_MOVE_PATTERNS})
    category_support = Counter({key: 0 for key in THEORY_MOVE_PATTERNS})
    matched = {key: Counter() for key in THEORY_MOVE_PATTERNS}

    for paper in paper_list:
        text = _paper_text(paper)
        for key, rules in THEORY_MOVE_PATTERNS.items():
            paper_score = 0
            for rule in rules:
                count = min(2, len(re.findall(rule.expression, text, flags=re.I | re.S)))
                if count:
                    paper_score += rule.weight * count
                    matched[key][rule.name] += count
            if paper_score:
                category_support[key] += 1
                category_scores[key] += min(paper_score, 12)

    ranked = sorted(
        THEORY_MOVE_PATTERNS,
        key=lambda key: (category_scores[key], category_support[key], THEORY_MOVE_PRIORITY[key]),
        reverse=True,
    )
    best = ranked[0]
    second = ranked[1]
    paper_count = len(paper_list)
    min_support = 1 if paper_count < 8 else max(2, math.ceil(paper_count * 0.10))
    ambiguous = (
        category_scores[second] > 0
        and category_scores[best] - category_scores[second] < 2
        and category_support[best] == category_support[second]
    )
    if category_scores[best] < 3 or category_support[best] < min_support or ambiguous:
        tentative_patterns = ()
        if category_scores[best]:
            tentative_patterns = (
                f"Best candidate: {THEORY_MOVE_LABELS[best]}",
                *(f"{name} ({count})" for name, count in matched[best].most_common(6)),
            )
        return TheoryMoveResult(
            key="unclear",
            label=THEORY_MOVE_LABELS["unclear"],
            score=category_scores[best],
            support_count=category_support[best],
            paper_count=paper_count,
            matched_patterns=tentative_patterns,
            category_scores=tuple((key, category_scores[key]) for key in ranked),
            category_support=tuple((key, category_support[key]) for key in ranked),
        )

    pattern_labels = tuple(
        f"{name} ({count})" for name, count in matched[best].most_common(6)
    )
    return TheoryMoveResult(
        key=best,
        label=THEORY_MOVE_LABELS[best],
        score=category_scores[best],
        support_count=category_support[best],
        paper_count=paper_count,
        matched_patterns=pattern_labels,
        category_scores=tuple((key, category_scores[key]) for key in ranked),
        category_support=tuple((key, category_support[key]) for key in ranked),
    )


def _self_test() -> None:
    cases = {
        "building": "We propose a new theoretical framework and develop a theory of collaborative design.",
        "borrowing": "We draw on sociocultural theory as a theoretical lens for craft preservation.",
        "testing": "We empirically test the proposed theory and evaluate its hypotheses with survey data.",
        "meta_reflection": "We examine the philosophy of design and the nature of design theory.",
        "unclear": "This paper describes a software interface for scheduling meetings.",
    }
    for expected, abstract in cases.items():
        result = classify_theory_move([{"abstract": abstract}])
        assert result.key == expected, (expected, result)
    print("theory_typology self-test: ok")


if __name__ == "__main__":
    _self_test()
