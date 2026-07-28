#!/usr/bin/env python3
"""Interpret fixed or candidate clusters with adapted class-based TF-IDF.

This script does not form or modify clusters. It aggregates the papers already
assigned to each cluster, extracts contrastive terms within each keyword
group, and links the statistical representation back to papers and passages.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.preprocessing import normalize


CUSTOM_STOP_WORDS = {
    "design",
    "paper",
    "papers",
    "article",
    "articles",
    "study",
    "studies",
    "research",
    "knowledge",
    "use",
    "uses",
    "used",
    "using",
    "project",
    "projects",
    "work",
    "based",
    "approach",
    "approaches",
    "method",
    "methods",
    "result",
    "results",
    "information",
    "system",
    "systems",
    "process",
    "processes",
    "model",
    "models",
    "framework",
    "frameworks",
    "problem",
    "problems",
    "different",
    "new",
    "set",
    "section",
    "figure",
    "table",
    "author",
    "authors",
    "et",
    "al",
    "dr",
    "dsr",
    "ui",
    # Corpus-specific shorthand/encoding artefacts. These are not meaningful
    # topic labels and should never be surfaced without an expansion.
    "dhsfx",
    "cpelds",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | CUSTOM_STOP_WORDS)
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
TOKEN_RE = re.compile(r"(?u)\b[\w][\w-]+\b")
PHRASE_REPLACEMENTS = (
    (re.compile(r"\bdesign\s+science\s+research\b", re.I), "design_science_research"),
    (re.compile(r"\bdesign\s+rationale\b", re.I), "design_rationale"),
    (re.compile(r"\bdesign\s+knowledge\b", re.I), "design_knowledge"),
    (re.compile(r"\bdesign\s+theor(?:y|ies)\b", re.I), "design_theory"),
    (re.compile(r"\bdesign\s+pattern(?:s)?\b", re.I), "design_patterns"),
    (re.compile(r"\bdesign\s+guideline(?:s)?\b", re.I), "design_guidelines"),
    (re.compile(r"\bdesign\s+principle(?:s)?\b", re.I), "design_principles"),
    (re.compile(r"\bhuman[-\s]+computer\s+interaction\b", re.I), "hci"),
    (re.compile(r"\buser\s+interface(?:s)?\b", re.I), "user_interface"),
    (re.compile(r"\bmachine\s+learning\b", re.I), "machine_learning"),
    (re.compile(r"\bartificial\s+intelligence\b", re.I), "ai"),
)


def normalize_text(value: object, keyword: str) -> str:
    text = YEAR_RE.sub(" ", str(value if value is not None else ""))
    text = re.sub(r"\bet\s+al\.?\b", " ", text, flags=re.I)
    text = re.sub(r"\bDSR\b", "design science research", text)
    text = re.sub(r"\bUI\b", "user interface", text)
    if keyword.casefold() == "design rationale":
        text = re.sub(r"\bDR\b", "design rationale", text)
    else:
        text = re.sub(r"\bDR\b", " ", text)
    for pattern, replacement in PHRASE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return " ".join(text.split()).casefold()


def display_term(term: str) -> str:
    displayed = term.replace("_", " ")
    tokens = displayed.split()
    return " ".join(
        token.upper() if token in {"hci", "ai"} else token
        for token in tokens
    )


def term_key(term: str) -> str:
    tokens = []
    for token in term.split():
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def select_terms(
    terms: np.ndarray,
    weights: np.ndarray,
    limit: int,
    blocked_terms: set[str],
) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index in np.argsort(-weights):
        weight = float(weights[index])
        if weight <= 0:
            break
        displayed = display_term(str(terms[index]))
        key = term_key(displayed)
        tokens = displayed.casefold().split()
        if key in blocked_terms:
            continue
        if any(term_key(token) in blocked_terms for token in tokens):
            continue
        if any(
            token not in {"hci", "ai"}
            and len(token.replace("-", "")) <= 3
            for token in tokens
        ):
            continue
        if key in seen:
            continue
        if any(
            key in existing or existing in key
            for existing in seen
            if len(key.split()) == len(existing.split())
        ):
            continue
        selected.append((displayed, weight))
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def passage_score(text: str, weighted_terms: list[tuple[str, float]]) -> float:
    normalized = normalize_text(text, "")
    score = 0.0
    for term, weight in weighted_terms:
        token = term.casefold().replace(" ", "_")
        count = normalized.count(token)
        if not count:
            count = normalized.count(term.casefold())
        score += min(count, 3) * weight
    return score / max(math.sqrt(len(TOKEN_RE.findall(normalized))), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-terms", type=int, default=15)
    parser.add_argument("--representative-papers", type=int, default=5)
    parser.add_argument("--representative-passages", type=int, default=3)
    args = parser.parse_args()

    papers = pd.read_csv(args.input).fillna("")
    assignments = pd.read_csv(args.assignments).fillna("")
    embeddings = normalize(np.load(args.embeddings), norm="l2")
    if len(papers) != len(embeddings):
        raise ValueError("Input rows and embeddings do not align.")
    if papers["paper_id"].duplicated().any():
        raise ValueError("Input contains duplicate paper IDs.")
    if assignments["paper_id"].duplicated().any():
        raise ValueError("Assignments contain duplicate paper IDs.")

    frame = papers.merge(
        assignments[
            [
                "paper_id",
                "cluster_id",
                "cluster_index",
                "group_k",
                "assignment_method",
                "assignment_status",
            ]
        ],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if frame["cluster_id"].eq("").any() or frame["cluster_id"].isna().any():
        raise ValueError("At least one paper lacks a cluster assignment.")
    embedding_by_id = {
        paper_id: embeddings[index]
        for index, paper_id in enumerate(papers["paper_id"].astype(str))
    }

    cluster_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    overlap_terms: dict[tuple[str, str], set[str]] = {}
    for keyword, keyword_frame in frame.groupby("keyword", sort=True):
        raw_keyword_text = " ".join(
            f"{row.title} {row.abstract} {row.subdocument}"
            for row in keyword_frame.itertuples()
        )
        raw_acronyms = {
            match.casefold()
            for match in re.findall(r"\b[A-Z][A-Z0-9-]{1,7}\b", raw_keyword_text)
            if match.casefold() not in {"hci", "ai"}
        }
        class_rows = []
        cluster_ids = sorted(keyword_frame["cluster_id"].unique())
        for cluster_id in cluster_ids:
            members = keyword_frame[keyword_frame["cluster_id"] == cluster_id]
            combined = " ".join(
                normalize_text(
                    f"{row.title} {row.abstract} {row.subdocument}",
                    str(keyword),
                )
                for row in members.itertuples()
            )
            class_rows.append(combined)

        vectorizer = CountVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 3),
            min_df=1,
            token_pattern=r"(?u)\b[\w][\w-]+\b",
        )
        counts = vectorizer.fit_transform(class_rows).astype(float)
        terms = vectorizer.get_feature_names_out()
        row_totals = np.asarray(counts.sum(axis=1)).ravel()
        normalized_tf = counts.multiply(
            1.0 / np.maximum(row_totals, 1.0)[:, None]
        )
        term_frequency = np.asarray(counts.sum(axis=0)).ravel()
        average_class_length = float(np.mean(row_totals))
        idf = np.log1p(
            average_class_length / np.maximum(term_frequency, 1.0)
        )
        ctfidf = normalized_tf.multiply(idf).toarray()

        for class_index, cluster_id in enumerate(cluster_ids):
            members = keyword_frame[
                keyword_frame["cluster_id"] == cluster_id
            ].copy()
            keyword_tokens = {
                token.casefold()
                for token in re.findall(r"[A-Za-z]+", str(keyword))
                if token.casefold() != "design"
            }
            blocked_raw = keyword_tokens | raw_acronyms | {
                str(keyword).casefold(),
                str(keyword).casefold().replace("design ", ""),
            }
            blocked_terms = {term_key(value) for value in blocked_raw}
            weighted_terms = select_terms(
                terms,
                ctfidf[class_index],
                args.top_terms,
                blocked_terms,
            )
            overlap_terms[(str(keyword), str(cluster_id))] = {
                term for term, _ in weighted_terms[:10]
            }
            for rank, (term, weight) in enumerate(weighted_terms, start=1):
                term_rows.append(
                    {
                        "keyword": keyword,
                        "cluster_id": cluster_id,
                        "rank": rank,
                        "term": term,
                        "ctfidf_weight": weight,
                    }
                )

            member_vectors = np.vstack(
                [
                    embedding_by_id[str(paper_id)]
                    for paper_id in members["paper_id"]
                ]
            )
            centroid = normalize(
                member_vectors.mean(axis=0, keepdims=True), norm="l2"
            )[0]
            similarities = member_vectors @ centroid
            members["centroid_similarity"] = similarities
            representatives = members.sort_values(
                "centroid_similarity", ascending=False
            ).head(args.representative_papers)

            passage_candidates: list[dict[str, object]] = []
            for row in members.itertuples():
                passages = [
                    " ".join(value.split())
                    for value in str(row.subdocument).split("\n\n")
                    if value.strip()
                ]
                passage_ids = [
                    value
                    for value in str(row.passage_ids).split(";")
                    if value
                ]
                for passage_index, passage in enumerate(passages):
                    passage_candidates.append(
                        {
                            "paper_id": row.paper_id,
                            "passage_id": (
                                passage_ids[passage_index]
                                if passage_index < len(passage_ids)
                                else f"{row.paper_id}-passage-{passage_index + 1}"
                            ),
                            "score": passage_score(passage, weighted_terms),
                            "text": passage,
                        }
                    )
            passage_candidates.sort(
                key=lambda item: float(item["score"]), reverse=True
            )
            selected_passages = []
            used_papers: set[str] = set()
            for candidate in passage_candidates:
                paper_id = str(candidate["paper_id"])
                if paper_id in used_papers:
                    continue
                selected_passages.append(candidate)
                used_papers.add(paper_id)
                if len(selected_passages) >= args.representative_passages:
                    break

            top_display = [term for term, _ in weighted_terms[:5]]
            descriptor = "; ".join(top_display[:3])
            cluster_rows.append(
                {
                    "keyword": keyword,
                    "cluster_id": cluster_id,
                    "paper_count": int(len(members)),
                    "assignment_status": members.iloc[0]["assignment_status"],
                    "interpretation_status": (
                        "frozen"
                        if members.iloc[0]["assignment_status"]
                        == "frozen_prior_pilot"
                        else "provisional_pending_cluster_review"
                    ),
                    "statistical_descriptor_not_final_label": descriptor,
                    "top_terms": " | ".join(
                        term for term, _ in weighted_terms
                    ),
                    "representative_paper_ids": " | ".join(
                        representatives["paper_id"].astype(str)
                    ),
                    "representative_titles": " | ".join(
                        representatives["title"].astype(str)
                    ),
                    "representative_passage_ids": " | ".join(
                        str(item["passage_id"])
                        for item in selected_passages
                    ),
                }
            )

    clusters = pd.DataFrame(cluster_rows)
    max_overlap: dict[tuple[str, str], float] = defaultdict(float)
    for keyword, group in clusters.groupby("keyword"):
        ids = group["cluster_id"].astype(str).tolist()
        for left in ids:
            for right in ids:
                if left == right:
                    continue
                left_terms = overlap_terms[(str(keyword), left)]
                right_terms = overlap_terms[(str(keyword), right)]
                union = left_terms | right_terms
                overlap = (
                    len(left_terms & right_terms) / len(union)
                    if union
                    else 0.0
                )
                max_overlap[(str(keyword), left)] = max(
                    max_overlap[(str(keyword), left)], overlap
                )
    clusters["maximum_sibling_top10_jaccard"] = [
        max_overlap[(str(row.keyword), str(row.cluster_id))]
        for row in clusters.itertuples()
    ]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clusters.sort_values(["keyword", "cluster_id"]).to_csv(
        out / "cluster_statistical_interpretations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(term_rows).sort_values(
        ["keyword", "cluster_id", "rank"]
    ).to_csv(
        out / "cluster_term_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "paper_count": int(len(frame)),
        "cluster_count": int(len(clusters)),
        "method": "adapted class-based TF-IDF within keyword groups",
        "cluster_membership_changed": False,
        "ngram_range": [1, 3],
        "abbreviation_display_policy": (
            "Expand abbreviations except HCI and AI; remove ambiguous DR."
        ),
        "generic_term_filtering": sorted(CUSTOM_STOP_WORDS),
        "interpretation_status_counts": clusters[
            "interpretation_status"
        ].value_counts().to_dict(),
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(clusters)} cluster interpretations and "
        f"{len(term_rows)} ranked terms to {out}"
    )


if __name__ == "__main__":
    main()
