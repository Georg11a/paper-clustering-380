"""Source-evidence paragraph scorer for cluster summaries.

Standalone module (stdlib only; no imports from cluster_papers.py) that ranks
paragraphs from papers' extracted full text as candidate SOURCE EVIDENCE for a
cluster, and extracts short claim sentences for display.

Design goals, in order:
1. Contribution statements over related-work recitals. A paragraph that says
   "we define design patterns as..." is evidence about THIS paper; a paragraph
   that says "Alexander [3] defined patterns as..." is evidence about someone
   else's paper. First-person claim markers are boosted; citation-dense
   passages are penalized.
2. Auditable scores. Every score returns a per-component breakdown and
   human-readable reasons, so the explorer can show WHY a passage was chosen.
3. Copyright-safe display units. Evidence is surfaced as 1-2 sentences with
   provenance (paper title, paragraph index, sentence hash), not whole
   paragraphs.

Grounding literature: attribute-first generation (Slobodkin et al. 2024),
AIS attribution framework (Rashkin et al. 2023), sentence-level citation
display (Gao et al. 2023, ALCE), faithfulness as evidence-support ratio
(RAGAS; Es et al. 2023).

Integration sketch (for cluster_papers.py, done by the caller -- this module
does not modify existing code):

    from evidence_scorer import PaperParagraphs, select_cluster_evidence
    papers = [
        PaperParagraphs(
            paper_id=row["paper_id"], title=row["title"],
            paragraphs=str(row.get("extracted_context", "")).split("\n\n"),
            is_representative=row["representative_rank"] >= 0
                              and row["representative_rank"] <= 5,
        )
        for _, row in subset.iterrows()
    ]
    evidence = select_cluster_evidence(
        papers,
        cluster_terms=evidence_term_list,   # distinguishing evidence phrases
        keyword=design_keyword,             # e.g. "design patterns"
    )
    # evidence -> JSON payload -> SOURCE EVIDENCE section in the explorer;
    # summary sentences then cite evidence indices [E1][E2] and any claim
    # without a supporting item is dropped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Citation markers: numeric brackets [12] / [3, 4] / [3]-[7], author-year
# parentheticals (Author, 2020) / (Author et al. 2020), and bare "et al."
CITATION_PATTERNS = [
    re.compile(r"\[\d+(?:\s*[,;–-]\s*\d+)*\]"),
    re.compile(r"\([A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Za-z'’.-]+)*,?\s+(?:19|20)\d{2}[a-z]?\)"),
    re.compile(r"\bet\s+al\.?", re.IGNORECASE),
    re.compile(r"\((?:e\.g\.|cf\.)[^)]*\)", re.IGNORECASE),
]

# Related-work section cues in a paragraph's first sentence.
RELATED_WORK_CUES = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:related\s+work|background|prior\s+work|literature\s+review)\b",
    re.IGNORECASE,
)

# First-person / own-contribution claim markers. Kept multiword and anchored
# on we/our/this-paper so third-party attributions do not match.
FIRST_PERSON_CLAIM_PATTERNS = [
    "we define", "we propose", "we present", "we introduce", "we develop",
    "we describe", "we contribute", "we synthesize", "we organize",
    "we identify", "we derive", "we formalize", "we conceptualize",
    "we argue", "we distill", "our framework", "our approach", "our method",
    "our taxonomy", "our findings", "our results", "our contribution",
    "our analysis", "this paper presents", "this paper proposes",
    "this paper defines", "this paper introduces", "this paper contributes",
    "this article presents", "this study presents", "in this paper, we",
    "in this paper we", "in this work, we", "in this work we",
]

# Multiword, high-precision design-knowledge action patterns. Aligned with the
# tightened vocabulary in cluster_papers.py (bare "review"/"synthesis"/"survey"
# are excluded because they fire on related-work sections, questionnaire
# surveys, and design synthesis).
ACTION_EVIDENCE_PATTERNS = {
    "defines": [
        "is defined as", "are defined as", "we define", "can be defined as",
        "refers to", "is understood as", "conceptualized as", "we conceptualize",
    ],
    "organizes": [
        "a taxonomy of", "taxonomy for", "a classification of", "organized into",
        "categorized into", "we categorize", "an ontology of", "a typology of",
    ],
    "synthesizes": [
        "literature review", "systematic review", "scoping review",
        "meta-analysis", "survey of", "synthesis of prior", "state of the art",
        "review of existing", "landscape of",
    ],
    "translates": [
        "design implications", "actionable guidance", "guidelines for",
        "design recommendations", "implications for design", "heuristics for",
    ],
    "evaluates": [
        "we evaluate", "we evaluated", "empirical evaluation", "user study",
        "we validated", "we validate", "we tested", "controlled experiment",
    ],
    "represents": [
        "we represent", "representation of design", "we model", "formalized as",
        "we encode", "annotation scheme",
    ],
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")

GENERIC_ROOTS = {
    "design", "paper", "study", "research", "approach", "method", "result",
    "work", "use", "user", "system", "process", "base", "way", "form",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def canonical_root(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def phrase_roots(text: str, drop_generic: bool = True) -> set[str]:
    roots = {canonical_root(tok) for tok in _WORD.findall(text.lower())}
    if drop_generic:
        roots -= GENERIC_ROOTS
    return {r for r in roots if len(r) >= 3}


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(paragraph.strip()) if s.strip()]


def citation_density(paragraph: str) -> float:
    """Citation markers per 100 words."""
    words = max(1, len(_WORD.findall(paragraph.lower())))
    markers = sum(len(p.findall(paragraph)) for p in CITATION_PATTERNS)
    return 100.0 * markers / words


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ParagraphScore:
    total: float
    term_overlap: float
    first_person: float
    action: float
    keyword: float
    citation_penalty: float
    representative_bonus: float
    reasons: list[str] = field(default_factory=list)


def score_paragraph(
    paragraph: str,
    cluster_terms: list[str],
    keyword: str,
    is_representative: bool = False,
    citation_density_threshold: float = 1.5,
) -> ParagraphScore:
    """Score one paragraph as candidate source evidence for a cluster.

    Components (all visible in the returned breakdown):
    - term_overlap: root overlap with the cluster's distinguishing terms,
      weighted 3x for multiword phrase hits (frequent-vs-distinctive balance,
      Sievert & Shirley 2014).
    - first_person: own-contribution markers ("we define", "this paper
      presents"). The strongest signal that the passage states THIS paper's
      claim rather than reciting prior work.
    - action: multiword design-knowledge action patterns.
    - keyword: the cluster's design keyword appears.
    - citation_penalty: citation-dense passages (per-100-word density above
      threshold) are treated as related-work recitals and penalized; a
      related-work heading cue in the first sentence adds to the penalty.
    """
    low = paragraph.lower()
    reasons: list[str] = []

    term_score = 0.0
    para_roots = phrase_roots(paragraph)
    for term in cluster_terms:
        term_low = term.lower().strip()
        if not term_low:
            continue
        if " " in term_low and term_low in low:
            term_score += 3.0
            reasons.append(f'phrase match: "{term_low}"')
        else:
            overlap = phrase_roots(term_low) & para_roots
            if overlap:
                term_score += 1.0 * len(overlap)

    fp_score = 0.0
    for marker in FIRST_PERSON_CLAIM_PATTERNS:
        if marker in low:
            fp_score += 4.0
            reasons.append(f'own-claim marker: "{marker}"')
    fp_score = min(fp_score, 8.0)

    action_score = 0.0
    for action, patterns in ACTION_EVIDENCE_PATTERNS.items():
        for pattern in patterns:
            if pattern in low:
                action_score += 2.0
                reasons.append(f'action pattern ({action}): "{pattern}"')
    action_score = min(action_score, 6.0)

    kw_score = 0.0
    keyword_low = keyword.lower().strip()
    if keyword_low and keyword_low in low:
        kw_score = 2.0
    elif keyword_low and (phrase_roots(keyword_low, drop_generic=False) <= phrase_roots(paragraph, drop_generic=False)):
        kw_score = 1.0

    penalty = 0.0
    density = citation_density(paragraph)
    if density > citation_density_threshold:
        penalty += min(6.0, 1.5 * (density - citation_density_threshold))
        reasons.append(f"citation-dense ({density:.1f}/100w): likely related-work recital")
    sentences = split_sentences(paragraph)
    if sentences and RELATED_WORK_CUES.match(sentences[0]):
        penalty += 4.0
        reasons.append("related-work section cue in first sentence")

    rep_bonus = 1.5 if is_representative else 0.0

    total = term_score + fp_score + action_score + kw_score + rep_bonus - penalty
    return ParagraphScore(
        total=round(total, 2),
        term_overlap=round(term_score, 2),
        first_person=round(fp_score, 2),
        action=round(action_score, 2),
        keyword=round(kw_score, 2),
        citation_penalty=round(-penalty, 2),
        representative_bonus=rep_bonus,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------

@dataclass
class PaperParagraphs:
    paper_id: str
    title: str
    paragraphs: list[str]
    is_representative: bool = False


@dataclass
class EvidenceItem:
    paper_id: str
    paper_title: str
    sentences: str          # 1-2 claim sentences, the display unit
    paragraph_index: int    # index within the paper's paragraph list
    sentence_hash: str      # sha1 of sentences; stable locator across reruns
    score: float
    score_breakdown: ParagraphScore


def extract_claim_sentences(
    paragraph: str, cluster_terms: list[str], max_sentences: int = 2
) -> str:
    """Pick the 1-2 sentences that best carry the claim: own-claim markers
    first, then cluster-term root overlap. Keeps the display unit short."""
    sentences = split_sentences(paragraph)
    if not sentences:
        return paragraph[:300].strip()
    term_roots: set[str] = set()
    for term in cluster_terms:
        term_roots |= phrase_roots(term)

    def sentence_key(sentence: str) -> tuple[int, int]:
        low = sentence.lower()
        fp = any(marker in low for marker in FIRST_PERSON_CLAIM_PATTERNS)
        overlap = len(phrase_roots(sentence) & term_roots)
        return (1 if fp else 0, overlap)

    ranked = sorted(range(len(sentences)), key=lambda i: sentence_key(sentences[i]), reverse=True)
    chosen = sorted(ranked[:max_sentences])
    return " ".join(sentences[i] for i in chosen)


def select_cluster_evidence(
    papers: list[PaperParagraphs],
    cluster_terms: list[str],
    keyword: str,
    top_k: int = 6,
    max_per_paper: int = 2,
    min_score: float = 3.0,
) -> list[EvidenceItem]:
    """Rank all paragraphs across a cluster's papers; return top evidence.

    Constraints: at most max_per_paper items per paper (breadth over depth),
    near-duplicate sentences removed by root-set Jaccard > 0.8, and items
    below min_score dropped entirely -- a cluster with no qualifying evidence
    should show "no direct source evidence", not weak filler.
    """
    candidates: list[EvidenceItem] = []
    for paper in papers:
        for index, paragraph in enumerate(paper.paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < 80:
                continue
            score = score_paragraph(
                paragraph, cluster_terms, keyword, is_representative=paper.is_representative
            )
            if score.total < min_score:
                continue
            sentences = extract_claim_sentences(paragraph, cluster_terms)
            candidates.append(
                EvidenceItem(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    sentences=sentences,
                    paragraph_index=index,
                    sentence_hash=hashlib.sha1(
                        f"{paper.paper_id}|{index}|{sentences}".encode("utf-8")
                    ).hexdigest()[:12],
                    score=score.total,
                    score_breakdown=score,
                )
            )

    candidates.sort(key=lambda item: -item.score)

    selected: list[EvidenceItem] = []
    per_paper: dict[str, int] = {}
    seen_roots: list[set[str]] = []
    for item in candidates:
        if per_paper.get(item.paper_id, 0) >= max_per_paper:
            continue
        roots = phrase_roots(item.sentences)
        duplicate = any(
            roots and prior and len(roots & prior) / len(roots | prior) > 0.8
            for prior in seen_roots
        )
        if duplicate:
            continue
        selected.append(item)
        per_paper[item.paper_id] = per_paper.get(item.paper_id, 0) + 1
        seen_roots.append(roots)
        if len(selected) >= top_k:
            break
    return selected


# ---------------------------------------------------------------------------
# Faithfulness (RAGAS-style): share of summary sentences supported by evidence
# ---------------------------------------------------------------------------

def claim_supported(claim_sentence: str, evidence: list[EvidenceItem], threshold: float = 0.35) -> bool:
    claim_roots = phrase_roots(claim_sentence)
    if not claim_roots:
        return True  # no content-bearing roots; treat as connective text
    best = 0.0
    for item in evidence:
        ev_roots = phrase_roots(item.sentences)
        if ev_roots:
            best = max(best, len(claim_roots & ev_roots) / len(claim_roots))
    return best >= threshold


def faithfulness_score(summary: str, evidence: list[EvidenceItem]) -> float:
    """Fraction of summary sentences whose content roots are covered by at
    least one evidence item (Es et al. 2023, RAGAS faithfulness, adapted to a
    lexical, LLM-free setting)."""
    sentences = split_sentences(summary)
    if not sentences:
        return 1.0
    supported = sum(1 for s in sentences if claim_supported(s, evidence))
    return supported / len(sentences)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    contribution = (
        "In this paper, we define design patterns as reusable solutions that "
        "capture recurring interaction problems. Our framework organizes these "
        "patterns into a taxonomy of communication moves for human-robot "
        "interaction, derived from twelve field deployments."
    )
    related_work = (
        "Design patterns have been widely studied. Alexander et al. [3] defined "
        "patterns for architecture, and Borchers (2001) applied them to HCI [4], "
        "while Tidwell [5] and van Welie et al. [6] catalogued interface patterns "
        "(cf. [7], [8]). Kruschitz and Hitz [9] surveyed pattern languages."
    )
    terms = ["design patterns", "taxonomy", "human-robot interaction"]

    for name, para in [("contribution", contribution), ("related_work", related_work)]:
        s = score_paragraph(para, terms, keyword="design patterns", is_representative=True)
        print(f"== {name}: total={s.total}")
        for r in s.reasons:
            print("   -", r)

    papers = [
        PaperParagraphs("p1", "Patterns for HRI", [contribution, related_work], is_representative=True)
    ]
    ev = select_cluster_evidence(papers, terms, "design patterns")
    print("\nselected evidence items:", len(ev))
    for item in ev:
        print(f"[{item.sentence_hash}] {item.paper_title} (para {item.paragraph_index}, score {item.score})")
        print("   ", item.sentences[:160])

    good_summary = "This cluster defines design patterns and organizes them into a taxonomy for human-robot interaction."
    bad_summary = "This cluster focuses on sustainable fashion supply chains and circular economy business models."
    print("\nfaithfulness(good):", faithfulness_score(good_summary, ev))
    print("faithfulness(bad):", faithfulness_score(bad_summary, ev))
