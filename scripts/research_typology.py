#!/usr/bin/env python3
"""Deterministic first-pass coding of research contributions and domains.

Contribution types adapt the six types used by Bai et al.'s HAID review from
Wobbrock and Kientz (2016), with Survey/Synthesis retained because this corpus
includes review papers. Application domains adapt HAID's 13-domain codebook.
Codes are assigned per paper and aggregated at cluster level; they remain
first-pass labels that require human validation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Rule:
    name: str
    expression: str
    weight: int


@dataclass(frozen=True)
class ContributionResult:
    primary_key: str
    primary_label: str
    secondary_key: str
    secondary_label: str
    score: int
    support_count: int
    paper_count: int
    matched_patterns: tuple[str, ...]

    @property
    def support_text(self) -> str:
        if self.primary_key == "unclear":
            return f"No reliable contribution signal ({self.paper_count} papers reviewed)"
        return f"{self.support_count}/{self.paper_count} papers; weighted score {self.score}"

    @property
    def patterns_text(self) -> str:
        return "; ".join(self.matched_patterns) if self.matched_patterns else "No decisive pattern match"


@dataclass(frozen=True)
class DomainResult:
    keys: tuple[str, ...]
    labels: tuple[str, ...]
    paper_count: int
    support: tuple[tuple[str, int], ...]
    matched_patterns: tuple[str, ...]

    @property
    def labels_text(self) -> str:
        return "; ".join(self.labels)

    @property
    def support_text(self) -> str:
        support_by_key = dict(self.support)
        return "; ".join(
            f"{label}: {support_by_key.get(key, 0)}/{self.paper_count} papers"
            for key, label in zip(self.keys, self.labels)
        )

    @property
    def patterns_text(self) -> str:
        return "; ".join(self.matched_patterns) if self.matched_patterns else "No specific domain signal"


CONTRIBUTION_LABELS = {
    "empirical": "Empirical Contribution",
    "algorithmic": "Algorithmic Contribution",
    "artifact": "Artifact Contribution",
    "methodological": "Methodological Contribution",
    "theoretical": "Theoretical Contribution",
    "dataset": "Dataset Contribution",
    "survey": "Survey/Synthesis Contribution",
    "unclear": "Unclear Contribution Type — Requires Human Review",
}


CONTRIBUTION_PRIORITY = {
    "dataset": 7,
    "algorithmic": 6,
    "artifact": 5,
    "methodological": 4,
    "theoretical": 3,
    "survey": 2,
    "empirical": 1,
}


CONTRIBUTION_RULES = {
    "empirical": (
        Rule(
            "conducts an empirical study",
            r"\b(?:we|this (?:paper|study))\s+(?:conduct|conducts|conducted|report|reports|reported)\s+"
            r"(?:an?\s+)?(?:controlled\s+|empirical\s+|qualitative\s+|quantitative\s+|mixed[- ]methods?\s+)?"
            r"(?:experiment|study|investigation|evaluation|case study|field study|user study|survey|interviews?)\b",
            5,
        ),
        Rule(
            "collects human-subject or observational data",
            r"\b(?:we|this (?:paper|study))\s+(?:interview|interviews|interviewed|survey|surveys|surveyed|observe|observes|observed|"
            r"evaluate|evaluates|evaluated|analy[sz]e|analy[sz]es|analy[sz]ed)\b[^.;]{0,100}\b"
            r"(?:participants?|users?|designers?|practitioners?|students?|data|interviews?|responses?)\b",
            4,
        ),
        Rule(
            "reports empirical findings",
            r"\b(?:our|the)\s+(?:findings|results|analysis)\s+(?:show|shows|showed|suggest|suggests|indicate|indicates|reveal|reveals)\b",
            3,
        ),
        Rule(
            "explicit empirical study design",
            r"\b(?:controlled experiment|human-subject experiment|user study|interview study|questionnaire study|"
            r"field study|case study|mixed[- ]methods? study|ethnograph(?:y|ic study)|observational study)\b",
            2,
        ),
    ),
    "algorithmic": (
        Rule(
            "contributes an algorithm or computational model",
            r"\b(?:we|this (?:paper|work))\s+(?:propose|proposes|present|presents|develop|develops|introduce|introduces)\s+"
            r"(?:an?\s+|the\s+)?(?:new\s+|novel\s+)?(?:algorithm|classifier|computational (?:approach|method|model)|"
            r"machine[- ]learning (?:approach|method|model)|optimization (?:approach|method)|neural (?:model|architecture|network))\b",
            6,
        ),
        Rule(
            "improves algorithmic performance",
            r"\b(?:state[- ]of[- ]the[- ]art|prediction accuracy|classification accuracy|model performance|computational efficiency|"
            r"optimization algorithm|deep learning|machine learning)\b",
            2,
        ),
        Rule(
            "explicit algorithmic output",
            r"\b(?:novel algorithm|optimization algorithm|classification model|prediction model|computational model|"
            r"machine[- ]learning model|neural network architecture)\b",
            3,
        ),
    ),
    "artifact": (
        Rule(
            "contributes an implemented artifact",
            r"\b(?:we|this (?:paper|work))\s+(?:present|presents|introduce|introduces|develop|develops|design|designs|implement|implements|build|builds)\s+"
            r"(?:an?\s+|the\s+)?(?:new\s+|novel\s+)?(?:system|tool|toolkit|interface|prototype|platform|application|dashboard|chatbot|robot)\b",
            6,
        ),
        Rule(
            "artifact implementation or deployment",
            r"\b(?:implemented|deployed|working prototype|interactive system|design tool|decision support system|visualization tool)\b",
            3,
        ),
        Rule(
            "explicit artifact output",
            r"\b(?:interactive prototype|software prototype|implemented system|web-based tool|mobile application|"
            r"design support tool|user interface|visualization system)\b",
            3,
        ),
    ),
    "methodological": (
        Rule(
            "contributes a method or methodology",
            r"\b(?:we|this (?:paper|work))\s+(?:propose|proposes|present|presents|introduce|introduces|develop|develops)\s+"
            r"(?:an?\s+|the\s+)?(?:new\s+|novel\s+)?(?:research method|design method|methodology|analytical approach|"
            r"evaluation method|evaluation framework|auditing method|process model|design process)\b",
            6,
        ),
        Rule(
            "contributes guidelines or principles",
            r"\b(?:we|this (?:paper|work|study))\s+(?:propose|proposes|present|presents|derive|derives|develop|develops|offer|offers)\s+"
            r"(?:an?\s+|the\s+|a set of\s+)?(?:design\s+)?(?:guidelines|principles|recommendations|heuristics|patterns)\b",
            5,
        ),
        Rule(
            "explicit methodological output",
            r"\b(?:design methodology|research methodology|evaluation methodology|evaluation framework|analytical framework|"
            r"design guidelines|design principles|design heuristics|design patterns|design method|design process|process model)\b",
            3,
        ),
    ),
    "theoretical": (
        Rule(
            "contributes theory or a conceptual framework",
            r"\b(?:we|this (?:paper|work))\s+(?:propose|proposes|present|presents|introduce|introduces|develop|develops|formulate|formulates)\s+"
            r"(?:an?\s+|the\s+)?(?:new\s+|novel\s+)?(?:design theory|theory|theoretical framework|conceptual framework|"
            r"conceptual model|normative framework|causal model)\b",
            6,
        ),
        Rule(
            "articulates conceptual or philosophical foundations",
            r"\b(?:conceptual|normative|philosophical|theoretical)\s+(?:foundation|foundations|account|explanation|model|framework)\b|"
            r"\b(?:formalize|formalizes|formalized|conceptualize|conceptualizes|conceptualized)\s+[^.;]{0,60}\b"
            r"(?:concept|construct|theory|accountability|fairness|knowledge)\b",
            4,
        ),
        Rule(
            "explicit theoretical output",
            r"\b(?:design theory|theoretical framework|conceptual framework|conceptual model|normative framework|"
            r"philosophy of design|meta[- ]theor(?:y|etical))\b",
            3,
        ),
    ),
    "dataset": (
        Rule(
            "introduces a dataset or benchmark",
            r"\b(?:we|this (?:paper|work))\s+(?:introduce|introduces|present|presents|release|releases|contribute|contributes|create|creates)\s+"
            r"(?:an?\s+|the\s+)?(?:new\s+|novel\s+|large[- ]scale\s+)?(?:dataset|data set|benchmark|corpus|test collection)\b",
            7,
        ),
        Rule(
            "dataset availability or benchmark task",
            r"\b(?:publicly available dataset|benchmark dataset|benchmark suite|annotated corpus|data repository)\b",
            3,
        ),
        Rule(
            "explicit dataset output",
            r"\b(?:new dataset|benchmark dataset|benchmark corpus|annotated dataset|evaluation benchmark)\b",
            3,
        ),
    ),
    "survey": (
        Rule(
            "systematic or scoping review",
            r"\b(?:systematic literature review|systematic review|scoping review|mapping review|meta[- ]analysis)\b",
            7,
        ),
        Rule(
            "literature review or research synthesis",
            r"\b(?:literature review|narrative review|integrative review|review of (?:the|existing|prior) literature|research synthesis)\b",
            5,
        ),
        Rule(
            "reviews or synthesizes prior work",
            r"\b(?:we|this (?:paper|work))\s+(?:review|reviews|survey|surveys|synthesi[sz]e|synthesi[sz]es|map|maps)\s+"
            r"(?:the\s+|existing\s+|prior\s+)?(?:literature|research|studies|work|field)\b",
            5,
        ),
    ),
}


DOMAIN_LABELS = {
    "healthcare": "Healthcare, Medicine, Surgery",
    "finance": "Finance, Business, Economy",
    "transportation": "Transportation, Mobility, Planning",
    "law": "Law, Democracy, Governance",
    "everyday": "Everyday, Employment, Public Service",
    "education": "Education, Teaching, Research",
    "manufacturing": "Manufacturing, Industry, Automation",
    "media": "Media, Communication, Entertainment",
    "environment": "Environment, Resource, Energy",
    "software": "Software, System, Cybersecurity",
    "defense": "Defense, Military, Emergency",
    "design": "Design, Creativity, Architecture",
    "generic": "Generic, Abstract, Domain-Agnostic",
}


DOMAIN_RULES = {
    "healthcare": (Rule("healthcare setting", r"\b(?:healthcare|health care|clinical|clinic|medical|medicine|patient|hospital|surgery|surgical|diagnosis|treatment)\b", 3),),
    "finance": (Rule("finance/business setting", r"\b(?:finance|financial|banking|business|economy|economic|loan|credit|investment|retail|market|entrepreneurship)\b", 3),),
    "transportation": (Rule("transportation/mobility setting", r"\b(?:transportation|transport|mobility|automotive|vehicle|driving|traffic|aviation|maritime|urban planning)\b", 3),),
    "law": (Rule("law/governance setting", r"\b(?:law|legal|court|judicial|democracy|governance|government|policy|public policy|regulation|regulatory)\b", 3),),
    "everyday": (Rule("everyday/employment/public-service setting", r"\b(?:employment|workplace|workforce|hiring|recruitment|public service|social care|caregiving|everyday life|daily life|aging|older adults?)\b", 3),),
    "education": (Rule("education/teaching/research setting", r"\b(?:education|educational|teaching|teacher|classroom|school|student|learning environment|university|academic research|peer review)\b", 3),),
    "manufacturing": (Rule("manufacturing/industry/automation setting", r"\b(?:manufacturing|industrial automation|industry 4\.0|factory|production line|supply chain|logistics|digital twin|robotic manufacturing)\b", 3),),
    "media": (Rule("media/communication/entertainment setting", r"\b(?:media|communication|journalism|news|social media|entertainment|game|gaming|sports?|content moderation|publishing)\b", 3),),
    "environment": (Rule("environment/resource/energy setting", r"\b(?:environment|environmental|climate|sustainability|sustainable|energy|resource management|agriculture|farming|water management|renewable)\b", 3),),
    "software": (Rule("software/system/cybersecurity setting", r"\b(?:software engineering|software development|programming|source code|code review|cybersecurity|cyber security|information security|software architecture|developer tools?)\b", 3),),
    "defense": (Rule("defense/military/emergency setting", r"\b(?:defense|defence|military|emergency response|disaster response|surveillance|air defense|hazardous situation)\b", 3),),
    "design": (Rule("design/creativity/architecture setting", r"\b(?:industrial design|graphic design|product design|architectural design|architecture|creative practice|creativity|craft|fashion design|ui/ux design)\b", 3),),
}


TEXT_FIELDS = ("title", "abstract", "discussion_summary", "discussion_excerpt")


def _paper_text(paper: Mapping[str, object]) -> str:
    return " ".join(str(paper.get(field, "") or "") for field in TEXT_FIELDS).lower()


def _minimum_support(paper_count: int) -> int:
    return 1 if paper_count < 8 else max(2, math.ceil(paper_count * 0.10))


def classify_contribution(papers: Iterable[Mapping[str, object]]) -> ContributionResult:
    paper_list = list(papers)
    scores = Counter({key: 0 for key in CONTRIBUTION_RULES})
    support = Counter({key: 0 for key in CONTRIBUTION_RULES})
    matched = {key: Counter() for key in CONTRIBUTION_RULES}

    for paper in paper_list:
        text = _paper_text(paper)
        paper_scores = Counter({key: 0 for key in CONTRIBUTION_RULES})
        paper_matches = {key: Counter() for key in CONTRIBUTION_RULES}
        for key, rules in CONTRIBUTION_RULES.items():
            for rule in rules:
                count = min(2, len(re.findall(rule.expression, text, flags=re.I | re.S)))
                if count:
                    paper_scores[key] += rule.weight * count
                    paper_matches[key][rule.name] += count

        paper_ranked = sorted(
            CONTRIBUTION_RULES,
            key=lambda key: (paper_scores[key], CONTRIBUTION_PRIORITY[key]),
            reverse=True,
        )
        paper_best, paper_second = paper_ranked[:2]
        if paper_scores[paper_best] < 3:
            continue
        selected = [paper_best]
        paper_equivalent = (
            paper_scores[paper_second] >= 3
            and abs(paper_scores[paper_best] - paper_scores[paper_second]) <= 1
        )
        if paper_equivalent:
            selected.append(paper_second)
        for key in selected:
            support[key] += 1
            scores[key] += min(paper_scores[key], 14)
            matched[key].update(paper_matches[key])

    ranked = sorted(
        CONTRIBUTION_RULES,
        key=lambda key: (scores[key], support[key], CONTRIBUTION_PRIORITY[key]),
        reverse=True,
    )
    best, second = ranked[:2]
    paper_count = len(paper_list)
    min_support = _minimum_support(paper_count)
    if scores[best] < 3 or support[best] < min_support:
        return ContributionResult(
            "unclear", CONTRIBUTION_LABELS["unclear"], "", "", scores[best], support[best],
            len(paper_list), (),
        )

    genuinely_equivalent = (
        scores[second] >= 3
        and support[second] >= min_support
        and abs(scores[best] - scores[second]) <= max(1, round(scores[best] * 0.08))
        and abs(support[best] - support[second]) <= 1
    )
    patterns = tuple(f"{name} ({count})" for name, count in matched[best].most_common(5))
    return ContributionResult(
        best,
        CONTRIBUTION_LABELS[best],
        second if genuinely_equivalent else "",
        CONTRIBUTION_LABELS[second] if genuinely_equivalent else "",
        scores[best],
        support[best],
        len(paper_list),
        patterns,
    )


def classify_domains(papers: Iterable[Mapping[str, object]], max_domains: int = 3) -> DomainResult:
    paper_list = list(papers)
    scores = Counter({key: 0 for key in DOMAIN_RULES})
    support = Counter({key: 0 for key in DOMAIN_RULES})
    matched = {key: Counter() for key in DOMAIN_RULES}

    for paper in paper_list:
        text = _paper_text(paper)
        for key, rules in DOMAIN_RULES.items():
            paper_score = 0
            for rule in rules:
                count = min(2, len(re.findall(rule.expression, text, flags=re.I | re.S)))
                if count:
                    paper_score += rule.weight * count
                    matched[key][rule.name] += count
            if paper_score:
                support[key] += 1
                scores[key] += min(paper_score, 8)

    paper_count = len(paper_list)
    min_support = 1 if paper_count <= 4 else max(2, math.ceil(paper_count * 0.15))
    ranked = sorted(DOMAIN_RULES, key=lambda key: (support[key], scores[key]), reverse=True)
    selected = [key for key in ranked if support[key] >= min_support and scores[key] >= 3][:max_domains]
    if not selected:
        return DomainResult(
            ("generic",), (DOMAIN_LABELS["generic"],), paper_count, (("generic", paper_count),), (),
        )

    pattern_labels = tuple(
        f"{DOMAIN_LABELS[key]}: {', '.join(name for name, _ in matched[key].most_common(2))}"
        for key in selected
    )
    return DomainResult(
        tuple(selected),
        tuple(DOMAIN_LABELS[key] for key in selected),
        paper_count,
        tuple((key, support[key]) for key in selected),
        pattern_labels,
    )


def _self_test() -> None:
    contribution_cases = {
        "empirical": "We conducted a controlled user study with 40 participants. Our findings show improved decisions.",
        "algorithmic": "We propose a novel machine-learning model for prediction accuracy.",
        "artifact": "We present an interactive system and implemented a working prototype.",
        "methodological": "We propose a new design method and derive design guidelines.",
        "theoretical": "We develop a new conceptual framework and theoretical account of design knowledge.",
        "dataset": "We introduce a new benchmark dataset and annotated corpus.",
        "survey": "We conduct a systematic literature review and synthesize prior research.",
    }
    for expected, abstract in contribution_cases.items():
        result = classify_contribution([{"abstract": abstract}])
        assert result.primary_key == expected, (expected, result)

    domains = classify_domains([
        {"abstract": "We study clinical diagnosis in hospitals and design tools for medical practitioners."}
    ])
    assert domains.keys == ("healthcare",), domains
    generic = classify_domains([{"abstract": "We present an interface for a general abstract problem."}])
    assert generic.keys == ("generic",), generic
    print("research_typology self-test: ok")


if __name__ == "__main__":
    _self_test()
