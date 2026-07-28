#!/usr/bin/env python3
"""Build a reproducible, blinded paragraph-budget pilot.

The extraction unit is a normalized scientific-text passage rather than a
literal PDF paragraph. PDF text layers do not preserve paragraph boundaries
reliably, so long blocks are split into sentence groups and very short blocks
are excluded. Every unit retains its source order and nearest section heading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


FACETS = (
    "construct_definition",
    "purpose_background_theory",
    "method_evidence",
    "findings_results",
    "contribution_implications",
    "other",
)

SECTION_FACET_PATTERNS = (
    ("method_evidence", r"\b(method|methodology|materials?|procedure|study design|data|analysis|experiment|evaluation setup|case study)\b"),
    ("findings_results", r"\b(result|results|finding|findings|evaluation results|observations?)\b"),
    ("contribution_implications", r"\b(discussion|conclusion|conclusions|implication|implications|contribution|limitations?|future work)\b"),
    ("purpose_background_theory", r"\b(introduction|background|related work|literature review|theoretical background|motivation)\b"),
    ("construct_definition", r"\b(definition|definitions|conceptual framework|framework|model|theory|construct)\b"),
)

FACET_CUES = {
    "construct_definition": (
        "define", "defined as", "definition", "conceptualize", "conceptualisation",
        "conceptualization", "framework", "model", "construct", "theory consists",
    ),
    "purpose_background_theory": (
        "aim", "purpose", "motivat", "background", "prior work", "previous research",
        "research question", "we investigate", "we explore", "there is a need",
    ),
    "method_evidence": (
        "method", "we conducted", "we collected", "interview", "survey", "experiment",
        "case study", "analysis", "dataset", "participants", "procedure", "evaluate",
    ),
    "findings_results": (
        "result", "finding", "we found", "revealed", "demonstrate", "show that",
        "indicate", "evidence suggests", "observed",
    ),
    "contribution_implications": (
        "contribut", "implication", "we propose", "we present", "we introduce",
        "this work provides", "in conclusion", "future work", "limitation",
    ),
}

BOILERPLATE = re.compile(
    r"(copyright|all rights reserved|creative commons|isbn|issn|"
    r"permission to make digital|acm reference format|publisher'?s note|"
    r"downloaded from|authorized licensed use|published by|"
    r"international journal of|proceedings of the)",
    re.I,
)

TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{1,}", re.I)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])")

STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "these", "those", "was",
    "were", "are", "been", "being", "have", "has", "had", "into", "their", "there",
    "which", "when", "where", "what", "while", "about", "also", "than", "then",
    "they", "them", "such", "using", "used", "use", "our", "its", "not", "can",
    "may", "more", "between", "within", "through", "paper", "study", "research",
}


@dataclass
class Passage:
    paper_id: str
    passage_id: str
    order: int
    section: str
    text: str
    facet: str = "other"
    bm25: float = 0.0
    normalized_relevance: float = 0.0


def normalize_space(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\s+", " ", text).strip()


def is_heading(text: str) -> bool:
    value = normalize_space(text)
    if not value or len(value) > 110 or len(value.split()) > 14:
        return False
    if value.lower().startswith(("keywords:", "key words:", "figure ", "fig. ", "table ")):
        return False
    numbered = bool(re.match(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*)[.)]?\s+", value))
    core = re.sub(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*)[.)]?\s+", "", value)
    if not core or core.endswith((".", "?", "!", ";")):
        return False
    alpha = [c for c in core if c.isalpha()]
    if len(alpha) < 4:
        return False
    upper_ratio = sum(c.isupper() for c in alpha) / max(1, len(alpha))
    standard_heading = bool(
        re.fullmatch(
            r"(?:abstract|introduction|background|related work|literature review|"
            r"theoretical (?:background|framework)|conceptual (?:background|framework)|"
            r"methods?|methodology|materials?(?: and methods?)?|study design|"
            r"data(?: collection| analysis)?|analysis|experiments?|evaluation|"
            r"results?|findings?|results? and discussion|discussion(?: and conclusions?)?|"
            r"conclusions?|implications?|contributions?|limitations?|future work)",
            core,
            re.I,
        )
    )
    exact_special = core.casefold() in {"abstract", "references", "bibliography", "acknowledgments", "acknowledgements"}
    content_words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", core) if w.casefold() not in STOPWORDS]
    title_ratio = sum(w[0].isupper() for w in content_words) / max(1, len(content_words))
    numbered_heading = numbered and len(core.split()) <= 12 and title_ratio >= 0.5
    all_caps_heading = upper_ratio >= 0.72 and len(core.split()) <= 10
    return exact_special or standard_heading or all_caps_heading or numbered_heading


def sentence_groups(text: str, target_chars: int = 850, max_chars: int = 1400) -> list[str]:
    sentences = [normalize_space(x) for x in SENTENCE_RE.split(normalize_space(text)) if normalize_space(x)]
    if not sentences:
        return []
    groups: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) > max_chars:
            groups.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += len(sentence) + 1
        if size >= target_chars:
            groups.append(" ".join(current))
            current, size = [], 0
    if current:
        tail = " ".join(current)
        if groups and len(tail) < 180:
            groups[-1] = f"{groups[-1]} {tail}"
        else:
            groups.append(tail)
    return groups


def usable_text_quality(text: str) -> bool:
    compact = [c for c in text if not c.isspace()]
    if not compact or "�" in text:
        return False
    alpha_ratio = sum(c.isalpha() for c in compact) / len(compact)
    symbol_ratio = sum(unicodedata.category(c) in {"Sm", "So", "Sk"} for c in compact) / len(compact)
    return alpha_ratio >= 0.55 and symbol_ratio <= 0.04


def split_passages(paper_id: str, raw_text: str) -> list[Passage]:
    raw_text = raw_text.replace("\x00", " ").replace("\u00ad", "")
    raw_text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", raw_text)
    raw_lines = [normalize_space(line) for line in raw_text.splitlines()]
    repeated = Counter(line.casefold() for line in raw_lines if 5 <= len(line) <= 150)
    repeated_lines = {line for line, count in repeated.items() if count >= 3}
    passages: list[Passage] = []
    section = "front_matter"
    order = 0
    buffer: list[str] = []
    stopped = False

    def flush() -> None:
        nonlocal order, buffer
        block = normalize_space(" ".join(buffer))
        buffer = []
        if len(block) < 120:
            return
        for group in sentence_groups(block):
            word_count = len(TOKEN_RE.findall(group))
            if word_count < 35 or word_count > 280 or not usable_text_quality(group):
                continue
            order += 1
            passages.append(
                Passage(
                    paper_id=paper_id,
                    passage_id=f"{paper_id}_p{order:04d}",
                    order=order,
                    section=section,
                    text=group,
                )
            )

    for line in raw_lines:
        if stopped:
            break
        if not line:
            flush()
            continue
        if (
            line.casefold() in repeated_lines
            or BOILERPLATE.search(line)
            or re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", line)
            or line.lower().startswith(("figure ", "fig. ", "table "))
        ):
            continue
        abstract_match = re.match(r"^abstract\s*[:.—-]\s*(.+)$", line, re.I)
        if abstract_match:
            flush()
            section = "Abstract"
            buffer.append(abstract_match.group(1))
            continue
        if re.match(r"^(?:key\s*words|keywords)\s*:", line, re.I):
            flush()
            continue
        if re.fullmatch(r"(?:references|bibliography|references and notes|notes and references)", line, re.I):
            flush()
            stopped = True
            continue
        if is_heading(line):
            flush()
            core = re.sub(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*)[.)]?\s+", "", line).strip()
            if core.casefold() in {"references", "bibliography"}:
                stopped = True
                continue
            section = line[:110]
            continue
        buffer.append(line)
    flush()
    return passages


def tokens(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(text) if x.lower() not in STOPWORDS]


def keyword_query(keyword: str) -> list[str]:
    low = keyword.lower().strip()
    query = tokens(low)
    if low.startswith("design "):
        tail = low.removeprefix("design ").strip()
        query.extend(tokens(tail))
        if tail.endswith("s"):
            query.extend(tokens(tail[:-1]))
        else:
            query.extend(tokens(tail + "s"))
    query.extend(["design", "knowledge"])
    if low == "design theory":
        query.extend(["theory", "theories", "theoretical", "framework", "model", "construct"])
    return query


def section_facet(section: str) -> str | None:
    for facet, pattern in SECTION_FACET_PATTERNS:
        if re.search(pattern, section, re.I):
            return facet
    return None


def predict_facet(passage: Passage) -> str:
    base = section_facet(passage.section)
    low = passage.text.lower()
    scores = {facet: 0 for facet in FACETS[:-1]}
    if base:
        scores[base] += 3
    for facet, cues in FACET_CUES.items():
        scores[facet] += sum(1 for cue in cues if cue in low)
    best, score = max(scores.items(), key=lambda x: (x[1], x[0]))
    return best if score > 0 else "other"


def bm25_score(passages: list[Passage], query: list[str], k1: float = 1.5, b: float = 0.75) -> None:
    docs = [tokens(p.text) for p in passages]
    if not docs:
        return
    avgdl = sum(map(len, docs)) / len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    n = len(docs)
    for passage, doc in zip(passages, docs):
        tf = Counter(doc)
        score = 0.0
        for term in query:
            if not tf[term]:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(doc) / max(avgdl, 1))
            score += idf * tf[term] * (k1 + 1) / denom
        passage.bm25 = score
    max_score = max((p.bm25 for p in passages), default=0.0)
    for passage in passages:
        passage.normalized_relevance = passage.bm25 / max_score if max_score else 0.0
        passage.facet = predict_facet(passage)


def jaccard(a: str, b: str) -> float:
    aa, bb = set(tokens(a)), set(tokens(b))
    return len(aa & bb) / max(1, len(aa | bb))


def mean_pairwise_redundancy(passages: list[Passage]) -> float:
    pairs = [
        jaccard(passages[i].text, passages[j].text)
        for i in range(len(passages))
        for j in range(i + 1, len(passages))
    ]
    return sum(pairs) / len(pairs) if pairs else 0.0


def select_passages(
    passages: list[Passage],
    budget: int,
    facet_cap: int = 3,
    coverage_first: bool = True,
) -> list[Passage]:
    candidates = sorted(passages, key=lambda p: (-p.bm25, p.order))
    selected: list[Passage] = []
    counts = Counter()

    if coverage_first:
        # Pilot-only coverage initialization. The final representation uses a
        # soft diversity bonus instead of forcing one passage from every
        # automatically predicted facet.
        for facet in FACETS[:-1]:
            match = next((p for p in candidates if p.facet == facet and p.bm25 > 0), None)
            if match and len(selected) < budget:
                selected.append(match)
                counts[facet] += 1

    while len(selected) < budget:
        remaining = [
            p
            for p in candidates
            if p not in selected
            and (facet_cap <= 0 or counts[p.facet] < facet_cap)
        ]
        if not remaining:
            remaining = [p for p in candidates if p not in selected]
        if not remaining:
            break
        scored = []
        for p in remaining:
            redundancy = max((jaccard(p.text, s.text) for s in selected), default=0.0)
            diversity_bonus = 0.08 if counts[p.facet] == 0 else 0.0
            mmr = 0.72 * p.normalized_relevance - 0.28 * redundancy + diversity_bonus
            scored.append((mmr, p.bm25, -p.order, p))
        chosen = max(scored, key=lambda x: (x[0], x[1], x[2]))[-1]
        selected.append(chosen)
        counts[chosen.facet] += 1
    return sorted(selected, key=lambda p: p.order)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_code(value: str, prefix: str) -> str:
    return f"{prefix}{hashlib.sha256(value.encode()).hexdigest()[:6].upper()}"


def choose_calibration_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    ordered = sorted(rows, key=lambda r: int(r.get("canonical_text_chars") or 0))
    if len(ordered) <= size:
        return ordered
    rng = random.Random(seed)
    bins = [ordered[i::3] for i in range(3)]
    quotas = [size // 3] * 3
    for i in range(size % 3):
        quotas[2 - i] += 1
    sample: list[dict] = []
    forced_ids = {
        r["paper_id"] for r in ordered
        if r.get("analysis_unit_type") == "book" or str(r.get("text_source", "")).startswith("ocr")
    }
    for bucket, quota in zip(bins, quotas):
        forced = [r for r in bucket if r["paper_id"] in forced_ids]
        chosen = forced[:quota]
        pool = [r for r in bucket if r not in chosen]
        chosen.extend(rng.sample(pool, min(quota - len(chosen), len(pool))))
        sample.extend(chosen)
    return sorted(sample, key=lambda r: int(r.get("canonical_text_chars") or 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/batch1/canonical_analysis_input_284.csv")
    parser.add_argument("--output-dir", default="outputs/batch2/paragraph_budget_pilot")
    parser.add_argument("--keyword", default="Design Theory")
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 8, 12])
    parser.add_argument("--facet-cap", type=int, default=3)
    parser.add_argument(
        "--coverage-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--calibration-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--pool-alternates", type=int, default=4)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = repo / args.output_dir
    with (repo / args.input).open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r.get("keyword", "").casefold() == args.keyword.casefold()]
    if not rows:
        raise RuntimeError(f"No rows found for keyword {args.keyword!r}")

    all_passages: dict[str, list[Passage]] = {}
    selections: dict[tuple[str, int], list[Passage]] = {}
    candidate_rows: list[dict] = []
    subdoc_rows: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        text_path = Path(row["canonical_text_path"])
        if not text_path.is_absolute():
            text_path = repo / text_path
        passages = split_passages(row["paper_id"], text_path.read_text(encoding="utf-8", errors="replace"))
        # Experiment B already prepends metadata title + abstract. Exclude extracted
        # abstract copies so the passage budget measures additional full-text value.
        abstract = str(row.get("abstract", ""))
        passages = [
            p for p in passages
            if not (abstract and jaccard(p.text, abstract) >= 0.38)
        ]
        bm25_score(passages, keyword_query(args.keyword))
        all_passages[row["paper_id"]] = passages
        for passage in passages:
            candidate_rows.append(
                {
                    "paper_id": passage.paper_id,
                    "passage_id": passage.passage_id,
                    "source_order": passage.order,
                    "section": passage.section,
                    "predicted_facet": passage.facet,
                    "bm25_score": round(passage.bm25, 6),
                    "normalized_relevance": round(passage.normalized_relevance, 6),
                    "word_count": len(TOKEN_RE.findall(passage.text)),
                    "text": passage.text,
                }
            )
        for budget in args.budgets:
            selected = select_passages(
                passages,
                budget,
                args.facet_cap,
                args.coverage_first,
            )
            selections[(row["paper_id"], budget)] = selected
            subdoc_rows[budget].append(
                {
                    "paper_id": row["paper_id"],
                    "canonical_title": row["canonical_title"],
                    "keyword": row["keyword"],
                    "budget": budget,
                    "selected_count": len(selected),
                    "word_count": sum(len(TOKEN_RE.findall(p.text)) for p in selected),
                    "facet_count": len({p.facet for p in selected if p.facet != "other"}),
                    "mean_normalized_relevance": round(
                        sum(p.normalized_relevance for p in selected) / max(1, len(selected)), 6
                    ),
                    "mean_pairwise_redundancy": round(mean_pairwise_redundancy(selected), 6),
                    "passage_ids": ";".join(p.passage_id for p in selected),
                    "subdocument": "\n\n".join(p.text for p in selected),
                }
            )

    write_csv(
        output / "candidate_passages_design_theory.csv",
        candidate_rows,
        ["paper_id", "passage_id", "source_order", "section", "predicted_facet",
         "bm25_score", "normalized_relevance", "word_count", "text"],
    )
    for budget in args.budgets:
        write_csv(
            output / f"subdocuments_k{budget}_design_theory.csv",
            subdoc_rows[budget],
            ["paper_id", "canonical_title", "keyword", "budget", "selected_count",
             "word_count", "facet_count", "mean_normalized_relevance",
             "mean_pairwise_redundancy", "passage_ids", "subdocument"],
        )

    sample = choose_calibration_sample(rows, args.calibration_size, args.seed)
    sample_rows: list[dict] = []
    review_rows: list[dict] = []
    mapping_rows: list[dict] = []
    version_rows: list[dict] = []
    rng = random.Random(args.seed)
    max_budget = max(args.budgets)
    for index, row in enumerate(sample, 1):
        paper_id = row["paper_id"]
        paper_code = f"DT-{index:02d}"
        sample_rows.append(
            {
                "paper_code": paper_code,
                "paper_id": paper_id,
                "canonical_title": row["canonical_title"],
                "abstract": row["abstract"],
                "analysis_unit_type": row["analysis_unit_type"],
                "text_source": row["text_source"],
                "canonical_text_chars": row["canonical_text_chars"],
                "canonical_passage_count": len(all_passages[paper_id]),
            }
        )
        union = {p.passage_id: p for budget in args.budgets for p in selections[(paper_id, budget)]}
        alternates = [
            p for p in sorted(all_passages[paper_id], key=lambda x: (-x.bm25, x.order))
            if p.passage_id not in union
        ][: args.pool_alternates]
        pool = list(union.values()) + alternates
        rng.shuffle(pool)
        for review_order, passage in enumerate(pool, 1):
            review_id = stable_code(f"{paper_id}:{passage.passage_id}", "R-")
            review_rows.append(
                {
                    "review_id": review_id,
                    "paper_code": paper_code,
                    "paper_title": row["canonical_title"],
                    "keyword": row["keyword"],
                    "review_order": review_order,
                    "section": passage.section,
                    "passage_text": passage.text,
                    "relevance_0_2": "",
                    "human_facet": "",
                    "critical_evidence_yes_no": "",
                    "context_sufficiency_0_2": "",
                    "duplicate_group": "",
                    "include_yes_no": "",
                    "reviewer_notes": "",
                }
            )
            mapping_rows.append(
                {
                    "review_id": review_id,
                    "paper_code": paper_code,
                    "paper_id": paper_id,
                    "passage_id": passage.passage_id,
                    "source_order": passage.order,
                    "predicted_facet": passage.facet,
                    "bm25_score": round(passage.bm25, 6),
                    **{f"selected_k{budget}": passage in selections[(paper_id, budget)] for budget in args.budgets},
                }
            )
        aliases = list("ABC")
        rng.shuffle(aliases)
        for alias, budget in zip(aliases, args.budgets):
            selected = selections[(paper_id, budget)]
            version_rows.append(
                {
                    "paper_code": paper_code,
                    "version_alias": alias,
                    "paper_title": row["canonical_title"],
                    "keyword": row["keyword"],
                    "subdocument": "\n\n".join(p.text for p in selected),
                    "coverage_1_5": "",
                    "nonredundancy_1_5": "",
                    "usefulness_for_clustering_1_5": "",
                    "overall_preference_rank_1_3": "",
                    "reviewer_notes": "",
                }
            )
            mapping_rows.append(
                {
                    "review_id": "",
                    "paper_code": paper_code,
                    "paper_id": paper_id,
                    "passage_id": "",
                    "source_order": "",
                    "predicted_facet": "",
                    "bm25_score": "",
                    **{f"selected_k{x}": f"version_{alias}" if x == budget else "" for x in args.budgets},
                }
            )

    write_csv(output / "calibration_sample_15.csv", sample_rows, list(sample_rows[0]))
    write_csv(output / "blinded_paragraph_review.csv", review_rows, list(review_rows[0]))
    write_csv(output / "blinded_version_review.csv", version_rows, list(version_rows[0]))
    write_csv(output / "private_selection_mapping.csv", mapping_rows, list(mapping_rows[0]))

    summary_rows = []
    for budget in args.budgets:
        records = subdoc_rows[budget]
        summary_rows.append(
            {
                "budget": budget,
                "documents": len(records),
                "mean_selected_passages": round(sum(r["selected_count"] for r in records) / len(records), 2),
                "mean_words": round(sum(r["word_count"] for r in records) / len(records), 2),
                "mean_facets": round(sum(r["facet_count"] for r in records) / len(records), 2),
                "mean_normalized_relevance": round(
                    sum(r["mean_normalized_relevance"] for r in records) / len(records), 4
                ),
                "mean_pairwise_redundancy": round(
                    sum(r["mean_pairwise_redundancy"] for r in records) / len(records), 4
                ),
            }
        )
    write_csv(output / "automatic_budget_summary.csv", summary_rows, list(summary_rows[0]))

    readme = f"""# Design Theory paragraph-budget pilot

- Input documents: {len(rows)}
- Calibration documents: {len(sample)}
- Budgets: {', '.join(map(str, args.budgets))}
- Temporary facet cap for the budget test: {args.facet_cap}
- Coverage-first initialization: {args.coverage_first}
- Random seed: {args.seed}

This is a calibration pilot, not a final result. Reviewers should use the
blinded review files and must not inspect `private_selection_mapping.csv`
until annotation is complete.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote pilot to {output}")
    print(f"Design Theory documents: {len(rows)}")
    print(f"Candidate passages: {len(candidate_rows)}")
    print(f"Paragraph review rows: {len(review_rows)}")


if __name__ == "__main__":
    main()
