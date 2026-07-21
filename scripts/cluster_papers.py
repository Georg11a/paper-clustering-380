#!/usr/bin/env python3
"""Cluster a paper CSV and create UMAP/summary outputs."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import umap
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from evidence_scorer import PaperParagraphs, faithfulness_score, select_cluster_evidence


DOMAIN_STOPWORDS = {
    "abstract",
    "acm",
    "acmdl",
    "acmreferenceformat",
    "activity",
    "adults",
    "agentcollaboration",
    "agentgen",
    "april",
    "aucdi",
    "barcelona",
    "based",
    "chi",
    "chiea",
    "clustering",
    "conference",
    "context",
    "copyright",
    "abstracts",
    "ccsconcepts",
    "centeredcomputing",
    "chair",
    "cyprus",
    "design",
    "designed",
    "designer",
    "designers",
    "designing",
    "different",
    "doi",
    "download",
    "downloads",
    "com",
    "edu",
    "extended",
    "facet",
    "figure",
    "forexample",
    "group",
    "isbn",
    "issn",
    "iui",
    "http",
    "https",
    "hci",
    "information",
    "knowledge",
    "making",
    "new",
    "newyork",
    "ny",
    "cid",
    "org",
    "paper",
    "papers",
    "paphos",
    "permission",
    "proceedings",
    "processes",
    "publication",
    "published",
    "publisher",
    "project",
    "research",
    "researcher",
    "researchers",
    "republicofkorea",
    "seoul",
    "set",
    "signal",
    "signals",
    "spain",
    "study",
    "studies",
    "support",
    "supported",
    "supporting",
    "system",
    "systems",
    "table",
    "terms",
    "time",
    "title",
    "type",
    "university",
    "usa",
    "vol",
    "volume",
    "workshop",
    "use",
    "used",
    "user",
    "users",
    "using",
    "venue",
    "view",
    "year",
    "york",
    "keyword",
}
STOPWORDS = sorted(ENGLISH_STOP_WORDS.union(DOMAIN_STOPWORDS))

LABEL_PREFERRED_PHRASES = {
    "dark patterns",
    "design futures",
    "design guidelines",
    "design patterns",
    "design rationale",
    "design science",
    "ethical design",
    "transparent rag",
    "value-sensitive design",
    "value-sensitive",
}
APPLICATION_CONTEXT_PHRASES = {
    "behavior change",
    "co-design",
    "generative ai",
    "hri",
    "human-ai collaboration",
    "participatory design",
    "social networking sites",
    "social robots",
    "user-centered design",
}
GENERIC_LABEL_PHRASES = {
    "framework",
    "frameworks",
    "guidelines",
    "patterns",
    "principles",
    "method",
    "methods",
    "model",
    "system",
    "tool",
    "design guidelines",
    "design methods",
    "design patterns",
}

DESIGN_KNOWLEDGE_FORM_PATTERNS = {
    "tacit design knowledge": ["tacit knowledge", "tacit design knowledge", "craft knowledge", "expertise"],
    "design rationale": ["design rationale", "rationale", "decision rationale", "traceability"],
    "design theory": ["design theory", "theory", "theoretical framework", "mid-range theory"],
    "design patterns": ["design pattern", "design patterns", "pattern language", "reusable solution"],
    "design guidelines": ["design guideline", "design guidelines", "guideline", "guidelines"],
    "design heuristics": ["design heuristic", "design heuristics", "heuristic", "heuristics"],
    "design principles": ["design principle", "design principles", "principle", "principles"],
    "design frameworks": ["design framework", "design frameworks", "framework", "frameworks"],
    "design methods": ["design method", "design methods", "methodology", "methodologies"],
    "design rules": ["design rule", "design rules", "rule", "rules"],
    "design expertise": ["design expertise", "expertise", "expert knowledge"],
    "design procedures": ["design procedure", "design procedures", "procedure", "procedures"],
    "design knowledge": ["design knowledge", "knowledge base", "knowledge system"],
}

DESIGN_KNOWLEDGE_ACTION_PATTERNS = {
    "defines": ["define", "defines", "defined", "definition", "conceptualize", "conceptualizes", "conceptualization"],
    "organizes": ["organize", "organizes", "organized", "taxonomy", "typology", "classification", "categorize", "catalog"],
    "translates": ["translate", "translates", "operationalize", "operationalizes", "actionable", "apply", "application"],
    "represents": ["represent", "represents", "representation", "document", "documentation", "trace", "traceable"],
    "captures": ["capture", "captures", "codify", "codifies", "externalize", "transfer", "share", "communicate"],
    "adapts": ["adapt", "adapts", "adapted", "tailor", "domain-specific", "context-specific"],
    "evaluates": ["evaluate", "evaluates", "validated", "validation", "empirical", "evidence", "assessment"],
    # Multiword-only triggers: bare "synthesis" is ambiguous in design corpora
    # (design synthesis = generating designs, not aggregating literature) and
    # bare "review"/"survey" fire on related-work sections and questionnaire
    # surveys. Precision over recall; unmatched clusters fall to other actions
    # or the honest "unclassified" fallback.
    "synthesizes": [
        "literature review", "systematic review", "scoping review",
        "meta-analysis", "state of the art", "survey of", "surveying",
        "literature synthesis", "research synthesis", "knowledge synthesis",
        "synthesis of prior", "landscape of", "review of existing",
    ],
}

ACTION_DISPLAY = {
    "unclassified": "contributes",
    "defines": "defines and conceptualizes",
    "organizes": "organizes and classifies",
    "translates": "translates into actionable guidance",
    "represents": "represents and documents",
    "captures": "captures and transfers",
    "adapts": "adapts to a specific context",
    "evaluates": "evaluates with evidence",
    "synthesizes": "synthesizes prior work on",
}

ACTION_NOUN_PHRASES = {
    "unclassified": "a mix of contribution moves",
    "defines": "definition and conceptualization",
    "organizes": "organization and classification",
    "translates": "translation into actionable guidance",
    "represents": "representation and documentation",
    "captures": "capture and transfer",
    "adapts": "context-specific adaptation",
    "evaluates": "evidence-based evaluation",
    "synthesizes": "synthesis of prior work",
}

ACTION_LABEL_TEMPLATES = {
    "defines": "Defines {form} Through Conceptual Framing",
    "organizes": "Organizes {form} into Taxonomies or Frameworks",
    "translates": "Translates {form} into Actionable Design Guidance",
    "represents": "Represents {form} as Traceable Decision Knowledge",
    "captures": "Frames {form} as Expertise to Capture and Transfer",
    "adapts": "Adapts {form} for Domain-Specific Use",
    "evaluates": "Evaluates {form} Through Empirical Evidence",
    "synthesizes": "Synthesizes {form} into Shared Design Constructs",
}

CLUSTER_DESCRIPTOR_TEMPLATES = {
    "unclassified": "{form} with mixed contribution signals",
    "defines": "{form} through definition and conceptual framing",
    "organizes": "{form} as organized categories or frameworks",
    "translates": "{form} as actionable design guidance",
    "represents": "{form} as documented representations",
    "captures": "{form} as captured and transferable knowledge",
    "adapts": "{form} in context-specific use",
    "evaluates": "{form} through evidence and evaluation",
    "synthesizes": "{form} through prior-work synthesis",
}

ACTION_ROLE_QUALIFIERS = {
    "defines and conceptualizes": "Conceptualized",
    "organizes and classifies": "Organized",
    "translates into actionable guidance": "Actionable",
    "represents and documents": "Documented",
    "captures and transfers": "Transferable",
    "adapts to a specific context": "Contextualized",
    "evaluates with evidence": "Evaluated",
    "synthesizes prior work on": "Synthesized",
}

CONCEPTUAL_ROLE_PATTERNS = {
    "Evaluation Criteria or Evidence Base": [
        "assessment",
        "criteria",
        "detect",
        "diagnose",
        "evaluate",
        "evaluation",
        "evidence",
        "heuristic",
        "quality",
        "smell",
        "trustworthy",
        "usability",
        "validated",
    ],
    "Reusable Design Pattern Language": [
        "catalog",
        "gallery",
        "pattern language",
        "patterns",
        "reusable",
        "reuse",
        "solution",
    ],
    "Traceable Decision Rationale": [
        "argumentation",
        "decision",
        "document",
        "documentation",
        "rationale",
        "record",
        "trace",
        "traceable",
    ],
    "Conceptual Framework": [
        "conceptual framework",
        "conceptualization",
        "define",
        "definition",
        "framework",
        "model",
        "theoretical",
        "theory",
    ],
    "Actionable Design Guidance": [
        "actionable",
        "apply",
        "guidance",
        "guideline",
        "method",
        "operationalize",
        "practice",
        "recommendation",
    ],
    "Taxonomy or Classification System": [
        "classification",
        "classify",
        "ontology",
        "taxonomy",
        "typology",
    ],
    "Tacit or Situated Expertise": [
        "craft",
        "expertise",
        "expert",
        "professional",
        "situated",
        "tacit",
    ],
    "Representational Modeling Schema": [
        "constraint",
        "formalize",
        "language",
        "modeling",
        "notation",
        "representation",
        "schema",
    ],
    "Empirical Evidence Base": [
        "case study",
        "empirical",
        "experiment",
        "literature review",
        "study",
        "systematic review",
        "user study",
    ],
}


DESIGN_KNOWLEDGE_KEYWORDS = [
    "Design knowledge",
    "Design patterns",
    "Design guidelines",
    "Design heuristics",
    "Design procedures",
    "Design expertise",
    "Design methods",
    "Design rules",
    "Design principles",
    "Design rationale",
    "Design theory",
    "Design frameworks",
]
DESIGN_KNOWLEDGE_KEYWORD_LOOKUP = {
    re.sub(r"\s+", " ", keyword.strip().lower()): keyword for keyword in DESIGN_KNOWLEDGE_KEYWORDS
}
DESIGN_KNOWLEDGE_KEYWORD_TERMS = {
    "Design knowledge": ["design knowledge", "knowledge"],
    "Design patterns": ["design pattern", "design patterns", "pattern", "patterns"],
    "Design guidelines": ["design guideline", "design guidelines", "guideline", "guidelines"],
    "Design heuristics": ["design heuristic", "design heuristics", "heuristic", "heuristics"],
    "Design procedures": ["design procedure", "design procedures", "procedure", "procedures"],
    "Design expertise": ["design expertise", "expertise", "expert", "experts"],
    "Design methods": ["design method", "design methods", "method", "methods"],
    "Design rules": ["design rule", "design rules", "rule", "rules"],
    "Design principles": ["design principle", "design principles", "principle", "principles"],
    "Design rationale": ["design rationale", "rationale", "rationales"],
    "Design theory": ["design theory", "theory", "theoretical"],
    "Design frameworks": ["design framework", "design frameworks", "framework", "frameworks"],
}


def canonical_design_keyword(value: object) -> str:
    return DESIGN_KNOWLEDGE_KEYWORD_LOOKUP.get(normalize_phrase(str(value or "")), str(value or "").strip())


def split_keyword_values(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"\s*(?:;|\||,)\s*", text)
    return [canonical_design_keyword(part) for part in parts if part.strip()]


def row_matches_focus_keyword(row: pd.Series, focus_keyword: str) -> bool:
    focus = canonical_design_keyword(focus_keyword)
    keywords = split_keyword_values(row.get("keyword", ""))
    return any(normalize_phrase(keyword) == normalize_phrase(focus) for keyword in keywords)


def keyword_terms(keyword: str) -> list[str]:
    canonical = canonical_design_keyword(keyword)
    terms = DESIGN_KNOWLEDGE_KEYWORD_TERMS.get(canonical, [canonical.lower()])
    return list(dict.fromkeys([canonical.lower(), *terms]))


def paragraph_relevance_score(paragraph: str, terms: list[str]) -> int:
    low = normalize_phrase(paragraph)
    score = 0
    for term in terms:
        term_low = normalize_phrase(term)
        if not term_low:
            continue
        count = len(re.findall(rf"(?<![a-z0-9]){re.escape(term_low)}(?![a-z0-9])", low))
        score += count * (4 if " " in term_low else 1)
    return score


def split_context_paragraphs(text: object) -> list[str]:
    raw = str(text or "").strip()
    if not raw or raw.lower() == "nan":
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    if paragraphs:
        return paragraphs
    return [raw]


def keyword_conditioned_context(row: pd.Series, focus_keyword: str | None, max_paragraphs: int = 6) -> tuple[str, int]:
    if not focus_keyword:
        context = str(row.get("extracted_context", "") or "")
        return context, len(split_context_paragraphs(context))

    terms = keyword_terms(focus_keyword)
    paragraphs = split_context_paragraphs(row.get("extracted_context", ""))
    scored = [
        (paragraph_relevance_score(paragraph, terms), len(paragraph), paragraph)
        for paragraph in paragraphs
    ]
    matched = [(score, length, paragraph) for score, length, paragraph in scored if score > 0]
    matched.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [paragraph for _, _, paragraph in matched[:max_paragraphs]]
    if not selected:
        selected = paragraphs[: min(len(paragraphs), max_paragraphs)]
    return "\n\n".join(selected), len(selected)


def paper_text(row: pd.Series, focus_keyword: str | None = None) -> str:
    context, _ = keyword_conditioned_context(row, focus_keyword)
    parts = [
        str(row.get("title", "")),
        str(row.get("abstract", "")),
        context,
        str(row.get("discussion_summary", "")),
        f"focus keyword: {focus_keyword}" if focus_keyword else "",
        f"keyword: {row.get('keyword', '')}",
        f"venue: {row.get('venue', '')}",
        f"year: {row.get('year', '')}",
    ]
    return ". ".join(p for p in parts if p and p != "nan")


FACET_VIEW_TERMS = {
    "context": [
        "accessibility",
        "aging",
        "automotive",
        "children",
        "climate",
        "collaboration",
        "education",
        "elder",
        "family",
        "game",
        "health",
        "healthcare",
        "industry",
        "learning",
        "mobile",
        "privacy",
        "robot",
        "software",
        "sustainability",
        "visualization",
        "xr",
        "ai",
    ],
    "knowledge-type": [
        keyword.lower() for keyword in DESIGN_KNOWLEDGE_KEYWORDS
    ],
    "method": [
        "case study",
        "co-design",
        "content analysis",
        "ethnographic",
        "experiment",
        "expert evaluation",
        "focus group",
        "interview",
        "participatory design",
        "prototype",
        "research through design",
        "survey",
        "systematic literature review",
        "systematic review",
        "thematic analysis",
        "user study",
        "workshop",
    ],
    "target-user": [
        "children",
        "designers",
        "developers",
        "domain experts",
        "families",
        "older adults",
        "parents",
        "patients",
        "practitioners",
        "students",
        "teachers",
        "users with disabilities",
        "visually impaired",
    ],
    "purpose": [
        "accessibility",
        "behavior change",
        "collaboration",
        "decision making",
        "ethical reflection",
        "evaluation",
        "knowledge capture",
        "learning",
        "privacy",
        "recommendations",
        "support design practice",
        "trust",
        "value sensitive",
    ],
}


def matched_view_terms(text: str, terms: list[str], max_terms: int = 10) -> list[str]:
    normalized = normalize_phrase(text)
    matches = []
    for term in terms:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")
        if pattern.search(normalized):
            matches.append(term)
    return matches[:max_terms]


def paper_text_for_view(row: pd.Series, view: str, focus_keyword: str | None = None) -> str:
    if view == "overall":
        return paper_text(row, focus_keyword)

    context, _ = keyword_conditioned_context(row, focus_keyword)
    base_text = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("abstract", "")),
            context,
            str(row.get("discussion_summary", "")),
        ]
    )
    terms = matched_view_terms(base_text, FACET_VIEW_TERMS[view])
    keyword = str(row.get("keyword", ""))
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))

    if focus_keyword:
        terms = [canonical_design_keyword(focus_keyword)] + terms

    if view == "knowledge-type":
        normalized_keyword = normalize_phrase(keyword)
        canonical_keyword = DESIGN_KNOWLEDGE_KEYWORD_LOOKUP.get(normalized_keyword, keyword)
        terms = [canonical_keyword] + [
            DESIGN_KNOWLEDGE_KEYWORD_LOOKUP.get(normalize_phrase(term), phrase_title(term))
            for term in terms
        ]

    # Keep facet text compact so clustering emphasizes the requested lens rather
    # than collapsing back into broad overall HCI similarity.
    facet_line = ", ".join(dict.fromkeys(term for term in terms if term))
    if not facet_line:
        facet_line = f"{keyword}. {title}"
    return f"Clustering view: {view}. Facet terms: {facet_line}. Title: {title}. Abstract signal: {abstract[:500]}"


def embedding_chunks(text: str, max_chars: int = 1800) -> list[str]:
    text = str(text or "").strip()
    paragraph_parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraph_parts) > 1:
        chunks = []
        for part in paragraph_parts:
            chunks.extend(embedding_chunks(part, max_chars=max_chars))
        return chunks
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def ollama_embed_inputs(
    inputs: list[str],
    model: str,
    embed_url: str,
    legacy_url: str,
    session: requests.Session,
) -> list[list[float]]:
    vectors = []
    for text in inputs:
        resp = session.post(embed_url, json={"model": model, "input": text}, timeout=120)
        if resp.status_code == 404:
            resp = session.post(legacy_url, json={"model": model, "prompt": text}, timeout=120)
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
            continue
        resp.raise_for_status()
        payload = resp.json()
        vectors.append(payload["embeddings"][0])
    return vectors


def ollama_embeddings(texts: list[str], model: str, host: str) -> np.ndarray:
    vectors = []
    embed_url = f"{host.rstrip('/')}/api/embed"
    legacy_url = f"{host.rstrip('/')}/api/embeddings"
    session = requests.Session()
    session.trust_env = False
    for i, text in enumerate(texts, start=1):
        chunks = embedding_chunks(text)
        chunk_vectors = np.asarray(ollama_embed_inputs(chunks, model, embed_url, legacy_url, session), dtype=np.float32)
        vectors.append(chunk_vectors.mean(axis=0))
        if i % 25 == 0:
            print(f"embedded {i}/{len(texts)} papers with {model}")
    return normalize(np.asarray(vectors, dtype=np.float32))


def tfidf_embeddings(texts: list[str], max_features: int, n_components: int) -> tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b(?:ai|ux|xr|[a-zA-Z]{3,})\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    dims = max(2, min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=dims, random_state=42)
    vectors = normalize(svd.fit_transform(tfidf))
    return vectors, vectorizer


def choose_dbscan(vectors: np.ndarray, min_samples: int) -> DBSCAN:
    distances = pairwise_distances(vectors, metric="cosine")
    kth = np.sort(distances, axis=1)[:, min(min_samples, len(vectors) - 1)]
    eps_candidates = np.quantile(kth, [0.35, 0.45, 0.55, 0.65, 0.75])
    best_model = None
    best_score = -1
    for eps in eps_candidates:
        model = DBSCAN(eps=float(eps), min_samples=min_samples, metric="cosine")
        labels = model.fit_predict(vectors)
        clusters = [x for x in set(labels) if x != -1]
        noise_ratio = float(np.mean(labels == -1))
        score = len(clusters) - abs(noise_ratio - 0.15) * 4
        if 2 <= len(clusters) <= 20 and score > best_score:
            best_model, best_score = model, score
    return best_model or DBSCAN(eps=float(np.quantile(kth, 0.55)), min_samples=min_samples, metric="cosine")


def cluster_labels(
    vectors: np.ndarray,
    method: str,
    k: int,
    min_samples: int,
    dbscan_eps: float | None,
    min_cluster_size: int,
) -> np.ndarray:
    if method == "kmeans":
        if k < 2:
            return np.zeros(len(vectors), dtype=int)
        return KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(vectors)
    if method == "dbscan":
        if dbscan_eps is not None:
            return DBSCAN(eps=dbscan_eps, min_samples=min_samples, metric="cosine").fit_predict(vectors)
        return choose_dbscan(vectors, min_samples).fit_predict(vectors)
    if method == "hdbscan":
        # Vectors are L2-normalized when created/loaded; Euclidean distance on
        # normalized vectors preserves cosine-neighborhood ordering well enough
        # for HDBSCAN while avoiding all-noise results from sklearn's cosine mode.
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            copy=True,
        ).fit_predict(vectors)
    raise ValueError(f"Unsupported clustering method: {method}")


def kmeans_silhouette_selection(vectors: np.ndarray, k_values: list[int]) -> tuple[int, pd.DataFrame]:
    rows = []
    n = len(vectors)
    for k in k_values:
        if k < 2 or k >= n:
            continue
        labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(vectors)
        avg_score = float(silhouette_score(vectors, labels, metric="cosine"))
        sample_scores = silhouette_samples(vectors, labels, metric="cosine")
        sizes = pd.Series(labels).value_counts().sort_index()
        rows.append(
            {
                "k": k,
                "average_silhouette": avg_score,
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "max_cluster_ratio": float(sizes.max() / n),
                "cluster_sizes": "; ".join(f"{idx}:{count}" for idx, count in sizes.items()),
                "negative_silhouette_count": int((sample_scores < 0).sum()),
                "negative_silhouette_ratio": float((sample_scores < 0).mean()),
            }
        )
    if not rows:
        raise ValueError("No valid k values for silhouette selection. Need at least 3 papers and k between 2 and n-1.")
    table = pd.DataFrame(rows).sort_values("k")
    best_score = float(table["average_silhouette"].max())

    # Select k directly from the silhouette diagnostic. If several candidates
    # are nearly tied, prefer the smaller k for readability, but do not impose
    # a minimum cluster-size or balance floor; those floors made large overall
    # views collapse into an artificially coarse two-cluster solution.
    tie_margin = max(0.002, abs(best_score) * 0.02)
    table["close_to_best"] = table["average_silhouette"] >= best_score - tie_margin
    table["selection_candidate"] = table["close_to_best"]
    candidates = table[table["selection_candidate"]]
    selected_k = int(candidates.sort_values(["k"]).iloc[0]["k"])
    return selected_k, table


def dynamic_k_cap(n_papers: int) -> int:
    """Conservative K-means cap for small keyword-conditioned corpora."""
    if n_papers < 4:
        return max(0, n_papers - 1)
    if n_papers < 10:
        return 2
    if n_papers < 20:
        return 3
    if n_papers < 35:
        return 4
    if n_papers < 60:
        return 5
    if n_papers < 90:
        return 6
    return 8


def representative_stats(vectors: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.full(len(labels), np.nan)
    ranks = np.full(len(labels), -1)
    medoid_ranks = np.full(len(labels), -1)
    for label in sorted(set(labels)):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        center = normalize(vectors[idx].mean(axis=0, keepdims=True))[0]
        d = cosine_distances(vectors[idx], center.reshape(1, -1)).ravel()
        order = np.argsort(d)
        distances[idx] = d
        ranks[idx[order]] = np.arange(1, len(idx) + 1)

        intra = cosine_distances(vectors[idx])
        medoid_order = np.argsort(intra.mean(axis=1))
        medoid_ranks[idx[medoid_order]] = np.arange(1, len(idx) + 1)
    return distances, ranks, medoid_ranks


def nearest_papers(vectors: np.ndarray, df: pd.DataFrame, top_n: int = 5) -> list[list[dict[str, object]]]:
    distances = cosine_distances(vectors)
    nearest = []
    for i in range(len(df)):
        order = np.argsort(distances[i])
        items = []
        for j in order:
            if i == j:
                continue
            items.append(
                {
                    "paper_id": str(df.iloc[j].get("paper_id", "")),
                    "title": str(df.iloc[j].get("title", "")),
                    "cluster": int(df.iloc[j].get("cluster", -1)),
                    "distance": float(distances[i, j]),
                }
            )
            if len(items) >= top_n:
                break
        nearest.append(items)
    return nearest


def lda_topics(texts: list[str], n_topics: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    vectorizer = CountVectorizer(
        stop_words=STOPWORDS,
        min_df=2,
        max_features=5000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b(?:ai|ux|xr|[a-zA-Z]{3,})\b",
    )
    counts = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    probs = lda.fit_transform(counts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    topic_words = []
    for topic in lda.components_:
        ranked = terms[np.argsort(topic)[::-1]].tolist()
        cleaned = _clean_label_terms(ranked, 8)
        topic_words.append(", ".join(cleaned or ranked[:8]))
    return probs.argmax(axis=1), probs.max(axis=1), topic_words, terms.tolist()


def _clean_label_terms(terms: list[str], limit: int) -> list[str]:
    selected = []
    selected_tokens = set()
    selected_roots = set()
    for term in terms:
        tokens = term.split()
        if any(token in DOMAIN_STOPWORDS for token in tokens):
            continue
        if _phrase_has_dangling_tail(term):
            continue
        roots = {canonical_label_token(token) for token in tokens}
        if roots & selected_roots:
            continue
        if len(tokens) == 1 and (
            token_is_too_generic(tokens[0])
            or token_is_too_generic(canonical_label_token(tokens[0]))
            or tokens[0] in GENERIC_RAKE_SINGLETONS
        ):
            continue
        if len(tokens) == 1 and any(tokens[0] in phrase.split() for phrase in selected if " " in phrase):
            continue
        overlap = len(set(tokens) & selected_tokens)
        if overlap and len(tokens) == 1:
            continue
        selected.append(term)
        selected_tokens.update(tokens)
        selected_roots.update(roots)
        if len(selected) >= limit:
            break
    return selected


def canonical_label_token(token: str) -> str:
    aliases = {
        "robots": "robot",
        "agents": "agent",
        "participants": "participant",
    "methods": "method",
        "participant": "participant",
        "practices": "practice",
        "guidelines": "guideline",
        "patterns": "pattern",
        "principles": "principle",
        "interfaces": "interface",
        "games": "game",
    }
    if token in aliases:
        return aliases[token]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def token_is_too_generic(token: str) -> bool:
    return token in {
        "approach",
        "activity",
        "case",
        "context",
        "care",
        "cultural",
        "data",
        "development",
        "evaluation",
        "experience",
        "framework",
        "group",
        "human",
        "information",
        "interaction",
        "knowledge",
        "learning",
        "making",
        "method",
        "model",
        "people",
        "participant",
        "participants",
        "principle",
        "principles",
        "sns",
        "process",
        "processe",
        "processes",
        "project",
        "researcher",
        "researchers",
        "results",
        "role",
        "set",
        "support",
        "technology",
        "theory",
        "work",
        "china",
    }


FACET_PATTERNS = {
    "population_or_context": [
        "accessibility",
        "aging",
        "automotive",
        "caregiving",
        "climate",
        "classroom",
        "education",
        "finance",
        "health",
        "healthcare",
        "industrial design",
        "privacy",
        "school",
        "software industry",
        "student",
        "students",
        "sustainability",
        "teacher",
        "teachers",
        "teaching",
        "workplace",
        "xr",
    ],
    "stakeholder_or_population": [
        "blind",
        "caregivers",
        "children",
        "clinicians",
        "designers",
        "developers",
        "domain experts",
        "elderly",
        "family",
        "older adults",
        "parents",
        "patients",
        "players",
        "practitioners",
        "visual impairment",
        "visually impaired",
    ],
    "method_or_lens": [
        "participatory design",
        "user-centered design",
        "value-sensitive design",
        "value-sensitive",
        "heuristic evaluation",
        "design science",
        "learning analytics",
        "ontology",
        "co-design",
        "case study",
        "ethnographic",
        "interview",
        "prototype",
        "systematic review",
        "thematic analysis",
        "user study",
        "workshop",
    ],
    "artifact_or_domain": [
        "dashboard",
        "mobile application",
        "game",
        "interface",
        "knowledge system",
        "conversational agent",
        "prototype",
        "smartwatch",
        "social robot",
        "tool",
        "xr",
        "robot",
        "visualization system",
    ],
    "contribution_or_outcome": [
        "empirical",
        "artifact",
        "methodological",
        "theoretical",
        "algorithmic",
        "dataset",
        "taxonomy",
    ],
}


CONTRIBUTION_TYPE_PATTERNS = {
    "Empirical": [
        "case study",
        "controlled experiment",
        "empirical",
        "ethnographic",
        "field study",
        "focus group",
        "interview",
        "mixed methods",
        "observational",
        "questionnaire",
        "survey",
        "user study",
        "workshop",
    ],
    "Artifact/System": [
        "application",
        "artifact",
        "chatbot",
        "dashboard",
        "design tool",
        "implemented",
        "interface",
        "platform",
        "prototype",
        "robot",
        "system",
        "tool",
        "visualization tool",
    ],
    "Methodological": [
        "analytical approach",
        "design guideline",
        "design guidelines",
        "design method",
        "design pattern",
        "design patterns",
        "design principle",
        "design principles",
        "evaluation framework",
        "framework",
        "guideline",
        "method",
        "methodology",
        "process model",
        "recommendation",
    ],
    "Theoretical": [
        "conceptual framework",
        "conceptual model",
        "design rationale",
        "design theory",
        "model",
        "normative",
        "rationale",
        "theoretical",
        "theory",
    ],
    "Algorithmic": [
        "algorithm",
        "algorithmic",
        "classifier",
        "computational approach",
        "large language model",
        "machine learning",
        "model performance",
        "optimization",
        "prediction",
    ],
    "Dataset/Benchmark": [
        "benchmark",
        "corpus",
        "data set",
        "dataset",
    ],
    "Taxonomy/Review": [
        "catalog",
        "classification",
        "literature review",
        "scoping review",
        "systematic literature review",
        "systematic review",
        "taxonomy",
    ],
}


FACET_DISPLAY_LABELS = {
    "accessibility": "Accessibility",
    "aging": "Aging",
    "ai": "AI",
    "automotive": "Automotive / Mobility",
    "blind": "Blind or Low-Vision Users",
    "caregivers": "Caregivers",
    "caregiving": "Caregiving",
    "children": "Children",
    "climate": "Climate / Environment",
    "classroom": "Education / Learning",
    "clinicians": "Clinicians",
    "designers": "Designers",
    "developers": "Developers",
    "domain experts": "Domain Experts",
    "education": "Education / Learning",
    "elderly": "Older Adults",
    "family": "Families",
    "finance": "Finance / Business",
    "health": "Healthcare",
    "healthcare": "Healthcare",
    "industrial design": "Industrial Design",
    "older adults": "Older Adults",
    "parents": "Parents",
    "patients": "Patients",
    "players": "Players",
    "practitioners": "Practitioners",
    "privacy": "Privacy / Security",
    "school": "Education / Learning",
    "software industry": "Software Industry",
    "student": "Education / Learning",
    "students": "Education / Learning",
    "sustainability": "Sustainability",
    "teacher": "Education / Learning",
    "teachers": "Education / Learning",
    "teaching": "Education / Learning",
    "users": "Users",
    "visual impairment": "Blind or Low-Vision Users",
    "visually impaired": "Blind or Low-Vision Users",
    "workplace": "Workplace",
    "xr": "XR",
}


def format_facet_values(values: list[str]) -> str:
    formatted = []
    seen = set()
    for value in values:
        label = FACET_DISPLAY_LABELS.get(value, phrase_title(value))
        key = normalize_phrase(label)
        if key in seen:
            continue
        formatted.append(label)
        seen.add(key)
    return ", ".join(formatted)


def normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def row_analysis_context(row: pd.Series) -> str:
    return str(row.get("keyword_conditioned_context", row.get("extracted_context", "")))


def find_facet_matches_for_cluster(subset: pd.DataFrame, facet: str, max_items: int = 4) -> list[str]:
    paper_texts = [
        normalize_phrase(f"{row.get('title', '')} {row.get('abstract', '')} {row_analysis_context(row)}")
        for _, row in subset.iterrows()
    ]
    min_count = 1 if len(paper_texts) <= 4 else 2
    scored = []
    for phrase in FACET_PATTERNS[facet]:
        pattern = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
        count = sum(bool(pattern.search(text)) for text in paper_texts)
        share = count / max(len(paper_texts), 1)
        if count >= min_count or share >= 0.25:
            scored.append((count, share, phrase))
    scored.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    return [phrase for _, _, phrase in scored[:max_items]]


def infer_contribution_types_for_cluster(subset: pd.DataFrame, max_items: int = 3) -> list[str]:
    paper_texts = [
        normalize_phrase(f"{row.get('title', '')} {row.get('abstract', '')} {row_analysis_context(row)}")
        for _, row in subset.iterrows()
    ]
    scored = []
    for contribution_type, phrases in CONTRIBUTION_TYPE_PATTERNS.items():
        count = 0
        weighted_hits = 0
        for text in paper_texts:
            matched = False
            for phrase in phrases:
                pattern = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
                if pattern.search(text):
                    matched = True
                    weighted_hits += 2 if " " in phrase else 1
            if matched:
                count += 1
        share = count / max(len(paper_texts), 1)
        if count:
            scored.append((count, weighted_hits, share, contribution_type))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [contribution_type for _, _, _, contribution_type in scored[:max_items]]


def score_pattern_group(text: str, patterns: dict[str, list[str]], max_items: int = 3) -> list[tuple[str, int]]:
    scored = []
    for label, phrases in patterns.items():
        score = 0
        for phrase in phrases:
            pattern = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
            hits = len(pattern.findall(text))
            score += hits * (3 if " " in phrase else 1)
        if score:
            scored.append((label, score))
    scored.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return scored[:max_items]


def infer_conceptual_roles(
    text: str,
    facets: dict[str, list[str]],
    contribution_types: list[str],
    max_items: int = 3,
) -> list[str]:
    scored = []
    facet_text = normalize_phrase(
        " ".join(
            [
                " ".join(facets.get("method_or_lens", [])),
                " ".join(facets.get("artifact_or_domain", [])),
                " ".join(contribution_types),
            ]
        )
    )
    combined_text = f"{text} {facet_text}"
    for role, phrases in CONCEPTUAL_ROLE_PATTERNS.items():
        score = 0
        for phrase in phrases:
            pattern = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
            hits = len(pattern.findall(combined_text))
            score += hits * (3 if " " in phrase else 1)
        if score:
            scored.append((role, score))
    scored.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    return [role for role, _ in scored[:max_items]]


def cluster_analysis_text(subset: pd.DataFrame) -> str:
    return normalize_phrase(
        " ".join(
            f"{row.get('title', '')} {row.get('abstract', '')} "
            f"{row.get('primary_reason', '')} {row.get('val_reason', '')} {row_analysis_context(row)}"
            for _, row in subset.iterrows()
        )
    )


def focus_form_from_keyword(subset: pd.DataFrame) -> str:
    keywords = dominant_design_keywords(subset, limit=1)
    if not keywords:
        return "design knowledge"
    keyword = keywords[0].lower()
    unique_keywords = {str(value).strip().lower() for value in subset.get("keyword", []) if str(value).strip()}
    if len(unique_keywords) == 1 and keyword.startswith("design "):
        return keyword
    return DESIGN_KNOWLEDGE_FORM_PATTERNS.get(keyword, [keyword])[0]


def conservative_cluster_descriptor(cluster_id: int, claim: dict[str, str]) -> str:
    form = claim.get("design_knowledge_form") or "Design Knowledge"
    action_key = claim.get("design_knowledge_action_key") or ""
    template = CLUSTER_DESCRIPTOR_TEMPLATES.get(action_key, "{form}")
    return f"Cluster {cluster_id}: {template.format(form=form)}"


GENERIC_DESCRIPTOR_EVIDENCE = {
    "case study",
    "design science",
    "design knowledge",
    "design theory",
    "design theories",
    "design method",
    "design methods",
    "design pattern",
    "design patterns",
    "design principle",
    "design principles",
    "design guideline",
    "design guidelines",
    "design rationale",
    "design rules",
    "design heuristics",
    "unknown theory",
}

LOW_SIGNAL_DESCRIPTOR_STARTS = {
    "define",
    "unknown",
    "explainable",
    "cognitive",
    # verb-initial title fragments read poorly after "around"
    "promote", "promoting", "describe", "describing", "generate", "generating",
    "trace", "tracing", "use", "using", "make", "making", "improve", "improving",
    "explore", "exploring", "understand", "understanding", "support", "supporting",
    "enable", "enabling", "develop", "developing", "propose", "proposing",
    "toward", "towards",
}


DANGLING_MODIFIER_TAILS = {
    "centred", "centered", "centric", "based", "driven", "oriented",
    "aware", "focused", "led", "informed", "related", "specific",
    "enabled", "assisted", "supported", "grounded",
}


def _phrase_has_dangling_tail(phrase: str) -> bool:
    """A phrase ending in a bare modifier ("human centred", "evidence based")
    is a decapitated noun phrase missing its head noun; it should not be shown
    as a standalone evidence term or descriptor qualifier."""
    tokens = normalize_phrase(phrase).split()
    return bool(tokens) and tokens[-1] in DANGLING_MODIFIER_TAILS


def _suffix_phrase_is_readable(phrase: str) -> bool:
    """Reject opaque or mangled fragments as descriptor qualifiers.

    Title n-grams have stopwords stripped by the vectorizer, so trigrams like
    'ways knowing hri' or acronym pairs like 'cpm pdd' read as machine noise.
    A qualifier is accepted only if every token is a known acronym or a word
    of at least four letters, and non-facet fragments longer than two tokens
    are rejected outright.
    """
    tokens = normalize_phrase(phrase).split()
    if not tokens or len(tokens) > 3:
        return False
    unknown_short = [t for t in tokens if t not in DISPLAY_ACRONYMS and len(t) < 4]
    if unknown_short:
        return False
    if _phrase_has_dangling_tail(phrase):
        return False
    acronym_count = sum(1 for t in tokens if t in DISPLAY_ACRONYMS)
    return acronym_count < len(tokens)


def descriptor_evidence_suffix(summary: dict[str, str]) -> str:
    form = normalize_phrase(str(summary.get("design_knowledge_form", "")))

    def usable(term: str, readability_check: bool = True) -> bool:
        normalized = normalize_phrase(term)
        if not normalized or normalized in GENERIC_DESCRIPTOR_EVIDENCE:
            return False
        if normalized.split()[0] in LOW_SIGNAL_DESCRIPTOR_STARTS:
            return False
        if normalized == form or len(normalized.split()) > 5:
            return False
        if readability_check and not _suffix_phrase_is_readable(term):
            return False
        return True

    # Curated facet phrases come from pattern dictionaries and read naturally,
    # so they are preferred over raw title n-grams.
    facet_candidates = []
    for field in ["facet_artifact_or_domain", "facet_population_or_context", "facet_method_or_lens"]:
        facet_candidates.extend(
            term.strip() for term in str(summary.get(field, "")).split(",") if term.strip()
        )
    for term in facet_candidates:
        normalized = normalize_phrase(term)
        if normalized and normalized not in GENERIC_DESCRIPTOR_EVIDENCE and normalized != form:
            return term

    evidence_terms = [
        term.strip()
        for term in str(summary.get("distinguishing_evidence_terms", "")).split(",")
        if term.strip()
    ]
    for term in evidence_terms:
        if usable(term):
            return term
    for term in evidence_terms:
        if usable(term, readability_check=False):
            return term
    return ""


def infer_design_knowledge_claim(
    subset: pd.DataFrame,
    facets: dict[str, list[str]],
    contribution_types: list[str],
) -> dict[str, str]:
    text = cluster_analysis_text(subset)
    focus_form = focus_form_from_keyword(subset)
    forms = score_pattern_group(text, DESIGN_KNOWLEDGE_FORM_PATTERNS, max_items=3)
    actions = score_pattern_group(text, DESIGN_KNOWLEDGE_ACTION_PATTERNS, max_items=3)
    conceptual_roles = infer_conceptual_roles(text, facets, contribution_types)

    # In per-keyword views, the form should remain anchored to the selected keyword.
    # Other detected forms are kept as supporting context, not allowed to rename the cluster.
    form = focus_form
    action = actions[0][0] if actions else "unclassified"
    conceptual_role = conceptual_roles[0] if conceptual_roles else ""
    secondary_actions = [item[0] for item in actions[1:3]]
    secondary_forms = [item[0] for item in forms[:3] if item[0] != form]

    if conceptual_role:
        label = f"{phrase_title(form)} as {conceptual_role}"
    else:
        label = ACTION_LABEL_TEMPLATES.get(action, "Frames {form} as Design Knowledge").format(
            form=phrase_title(form)
        )
    action_text = ACTION_DISPLAY.get(action, action)

    representative_titles = [
        str(title)
        for title in subset.sort_values(["representative_rank", "medoid_rank"]).get("title", pd.Series(dtype=str)).head(3)
        if str(title).strip()
    ]
    rep_sentence = ""
    if representative_titles:
        rep_sentence = " Representative papers include " + "; ".join(representative_titles[:2]) + "."

    form_detail = ""
    if secondary_forms:
        form_detail = f" Related forms include {', '.join(phrase_title(item) for item in secondary_forms)}."

    role_detail = ""
    if conceptual_role:
        role_detail = f" Conceptually, it treats {form} as {conceptual_role.lower()}."

    action_detail = ""
    if secondary_actions:
        action_detail = (
            " The shared move also involves "
            + " and ".join(ACTION_NOUN_PHRASES.get(item, item) for item in secondary_actions)
            + "."
        )

    facet_detail = []
    if facets.get("method_or_lens"):
        facet_detail.append("methods or lenses such as " + format_facet_values(facets["method_or_lens"][:3]))
    if facets.get("population_or_context"):
        facet_detail.append("contexts such as " + format_facet_values(facets["population_or_context"][:3]))
    facet_sentence = ""
    if facet_detail:
        facet_sentence = " Application and method terms are treated as supporting facets, especially " + " and ".join(facet_detail) + "."

    contribution_sentence = ""
    if contribution_types:
        contribution_sentence = " Its contribution pattern is coded as " + ", ".join(contribution_types[:3]) + "."

    summary = (
        f"This cluster {action_text} {form}, emphasizing how a design-knowledge construct is made explicit, "
        f"organized, or put to work rather than only where it is applied."
        f"{role_detail}{form_detail}{action_detail}{contribution_sentence}{facet_sentence}{rep_sentence}"
    )

    return {
        "cluster_label_candidate": label,
        "cluster_summary_candidate": summary,
        "design_knowledge_form": phrase_title(form),
        "design_knowledge_action": ACTION_DISPLAY.get(action, action),
        "design_knowledge_action_key": action,
        "design_knowledge_contribution": f"{ACTION_DISPLAY.get(action, action)} {phrase_title(form)}",
        "design_knowledge_role": conceptual_role,
        "_action_key": action,
        "_rep_titles": representative_titles[:2],
    }


def preferred_keyphrases_for_cluster(subset: pd.DataFrame, keyphrases: list[str], max_items: int = 4) -> list[str]:
    cluster_text = normalize_phrase(
        " ".join(
            f"{row.get('title', '')} {row.get('abstract', '')} {row_analysis_context(row)}"
            for _, row in subset.iterrows()
        )
    )
    candidates = []
    for phrase in sorted(LABEL_PREFERRED_PHRASES, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")
        if pattern.search(cluster_text):
            candidates.append(phrase)
    for phrase in keyphrases:
        if " " in phrase and phrase not in candidates and phrase not in APPLICATION_CONTEXT_PHRASES:
            candidates.append(phrase)
    for phrase in keyphrases:
        if phrase not in candidates and phrase not in APPLICATION_CONTEXT_PHRASES:
            candidates.append(phrase)

    selected = []
    roots_seen = set()
    for phrase in candidates:
        tokens = phrase.split()
        roots = {canonical_label_token(token) for token in tokens}
        if roots & roots_seen:
            continue
        if len(tokens) == 1 and token_is_too_generic(tokens[0]):
            continue
        selected.append(phrase)
        roots_seen.update(roots)
        if len(selected) >= max_items:
            break
    return selected


def add_label_phrase(label_parts: list[str], roots_seen: set[str], phrase: str, allow_generic: bool = False) -> None:
    phrase = phrase.strip()
    if not phrase:
        return
    if phrase in label_parts:
        return
    if not allow_generic and phrase in GENERIC_LABEL_PHRASES:
        return
    tokens = phrase.split()
    roots = {canonical_label_token(token) for token in tokens}
    if roots & roots_seen:
        return
    if len(tokens) == 1 and token_is_too_generic(tokens[0]) and not allow_generic:
        return
    label_parts.append(phrase)
    roots_seen.update(roots)


DISPLAY_ACRONYMS = {
    "ai": "AI",
    "ar": "AR",
    "cpm": "CPM",
    "cscw": "CSCW",
    "dsr": "DSR",
    "hci": "HCI",
    "hri": "HRI",
    "iot": "IoT",
    "lda": "LDA",
    "ml": "ML",
    "nlp": "NLP",
    "pdd": "PDD",
    "ui": "UI",
    "ux": "UX",
    "vr": "VR",
    "xr": "XR",
}


def phrase_title(phrase: str) -> str:
    special = {
        "ai": "AI",
        "hci": "HCI",
        "human-ai": "Human-AI",
        "human-ai collaboration": "Human-AI Collaboration",
        "social networking sites": "Social Networking Sites",
        "value-sensitive": "Value-Sensitive Design",
        "value-sensitive design": "Value-Sensitive Design",
    }
    if phrase in special:
        return special[phrase]
    small = {"and", "or", "of", "for", "in", "with", "to"}

    def render(word: str) -> str:
        if word.lower() in DISPLAY_ACRONYMS:
            return DISPLAY_ACRONYMS[word.lower()]
        if word in small:
            return word
        return word[:1].upper() + word[1:]

    return " ".join(render(word) for word in phrase.split())


def dominant_design_keywords(subset: pd.DataFrame, limit: int = 3) -> list[str]:
    keywords = []
    for keyword in subset.get("keyword", pd.Series(dtype=str)).fillna(""):
        canonical = DESIGN_KNOWLEDGE_KEYWORD_LOOKUP.get(normalize_phrase(str(keyword)))
        if canonical:
            keywords.append(canonical)
    if not keywords:
        return []
    counts = pd.Series(keywords).value_counts()
    return counts.head(limit).index.tolist()


GENERIC_RAKE_SINGLETONS = {
    "seen", "term", "terms", "ways", "means", "forms", "kind", "kinds", "basis",
    "makes", "made", "given", "taken", "shown", "found", "presented",
}


def rake_title_phrases(titles: list[str], max_phrases: int = 12) -> list[str]:
    """RAKE keyphrases over representative titles (Rose et al. 2010).

    Titles are split at punctuation and function-word stopwords; contiguous
    content-word runs keep their natural surface form ("generative rational
    heuristics", "synchronous e-learning") instead of the mangled fragments
    produced by stopword-stripped n-grams ("unknown theory", "ways knowing
    hri"). Phrases are scored by summed word degree/frequency. Phrase length
    is capped at three words and bare single words must be substantive:
    1-3-grams cover ~95% of human-assigned keywords across 17 KE benchmark
    datasets and gold keywords are overwhelmingly noun phrases (Altuncu et
    al. 2025, arXiv:2211.05031).
    """
    runs: list[list[str]] = []
    for title in titles:
        for fragment in re.split(r"[^A-Za-z0-9\-'\s]+", str(title).lower()):
            current: list[str] = []
            for word in fragment.split():
                word = word.strip("-'")
                is_stop = (
                    not word
                    or word in ENGLISH_STOP_WORDS
                    or (len(word) < 3 and word not in DISPLAY_ACRONYMS)
                )
                if is_stop:
                    if current:
                        runs.append(current)
                    current = []
                else:
                    current.append(word)
            if current:
                runs.append(current)

    freq: dict[str, int] = {}
    degree: dict[str, int] = {}
    for run in runs:
        for word in run:
            freq[word] = freq.get(word, 0) + 1
            degree[word] = degree.get(word, 0) + len(run) - 1

    scored: list[tuple[str, float]] = []
    for run in runs:
        if not 1 <= len(run) <= 3:
            continue
        if len(run) == 1 and (
            (len(run[0]) < 4 and run[0] not in DISPLAY_ACRONYMS)
            or run[0] in GENERIC_RAKE_SINGLETONS
        ):
            continue
        phrase = " ".join(run)
        score = sum((degree[word] + freq[word]) / freq[word] for word in run)
        scored.append((phrase, score))
    scored.sort(key=lambda item: -item[1])

    seen: set[str] = set()
    ordered: list[str] = []
    for phrase, _ in scored:
        if phrase not in seen:
            seen.add(phrase)
            ordered.append(phrase)
    return ordered[:max_phrases]


def representative_title_terms(subset: pd.DataFrame, limit: int = 4) -> list[str]:
    titles = [
        str(title)
        for title in subset.sort_values(["representative_rank", "medoid_rank"]).get("title", pd.Series(dtype=str)).head(6)
        if str(title).strip() and not any(marker in str(title) for marker in ["√", "�", "‚", "Ä", "ê"])
    ]
    if not titles:
        return []
    return _clean_label_terms(rake_title_phrases(titles), limit)


def distinguishing_evidence_terms(
    subset: pd.DataFrame,
    keyphrases: list[str],
    facets: dict[str, list[str]],
    contribution_types: list[str],
    limit: int = 4,
) -> str:
    evidence_candidates = []
    evidence_candidates.extend(representative_title_terms(subset, limit=limit))
    for facet in ["population_or_context", "artifact_or_domain", "method_or_lens"]:
        evidence_candidates.extend(facets.get(facet, [])[:2])
    evidence_candidates.extend(contribution_types[:2])
    preferred = preferred_keyphrases_for_cluster(subset, keyphrases, max_items=limit + 2)
    label_parts = []
    roots_seen: set[str] = set()
    candidate_phrases = sorted(
        evidence_candidates + preferred + [phrase for phrase in keyphrases if len(phrase.split()) > 1],
        key=lambda item: (len(item.split()) < 2, item in GENERIC_LABEL_PHRASES, len(item)),
    )
    for phrase in candidate_phrases:
        tokens = phrase.split()
        if len(set(tokens)) < len(tokens):
            continue
        if normalize_phrase(phrase) in {"design rules", "rules"}:
            continue
        add_label_phrase(label_parts, roots_seen, phrase, allow_generic=True)
        if len(label_parts) >= limit:
            break
    if not label_parts and len(subset):
        title_words = normalize_phrase(str(subset.iloc[0].get("title", ""))).split()
        label_parts = [word for word in title_words if not token_is_too_generic(word)][:limit]
    return format_facet_values(label_parts)


def deduplicate_conceptual_cluster_labels(summaries: dict[int, dict[str, str]]) -> None:
    labels: dict[str, list[int]] = {}
    for cluster_id, summary in summaries.items():
        labels.setdefault(str(summary.get("cluster_label_candidate", "")), []).append(cluster_id)
    for label, cluster_ids in labels.items():
        if not label or len(cluster_ids) < 2:
            continue
        for cluster_id in cluster_ids:
            summary = summaries[cluster_id]
            role = str(summary.get("design_knowledge_role", "") or "")
            form = str(summary.get("design_knowledge_form", "") or "")
            action = str(summary.get("design_knowledge_action", "") or "")
            qualifier = ACTION_ROLE_QUALIFIERS.get(action, "")
            if not role or not form or not qualifier or role == "n/a":
                continue
            if role.startswith(qualifier):
                continue
            summary["cluster_label_candidate"] = f"{form} as {qualifier} {role}"



def _split_terms(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _phrase_roots(phrase: str) -> set[str]:
    return {
        canonical_label_token(token)
        for token in normalize_phrase(phrase).split()
        if token and not token_is_too_generic(token)
    }


def _unique_phrases(cluster_id: int, phrase_lists: dict[int, list[str]]) -> list[str]:
    """Phrases whose lexical roots appear in this cluster's list only."""
    other_roots: set[str] = set()
    for other_id, phrases in phrase_lists.items():
        if other_id == cluster_id:
            continue
        for phrase in phrases:
            other_roots |= _phrase_roots(phrase)
    unique = []
    for phrase in phrase_lists.get(cluster_id, []):
        roots = _phrase_roots(phrase)
        if roots and not (roots & other_roots):
            unique.append(phrase)
    return unique


CONTRASTIVE_PRIVATE_KEYS = ["_action_key", "_rep_titles", "_contexts", "_methods", "_contributions"]


def apply_contrastive_summary_pass(summaries: dict[int, dict]) -> None:
    """Rewrite cluster summaries so each is discriminative within its view.

    A cluster label/summary should both describe the cluster and distinguish
    it from sibling clusters (Manning et al., IIR ch.17 cluster labeling;
    Sievert & Shirley 2014; Grootendorst 2022). Cluster-unique evidence leads
    each summary; view-shared properties (e.g. an action frame shared by most
    clusters) are stated once and briefly instead of repeated as boilerplate.
    Descriptor deduplication is handled separately by
    distinguish_duplicate_cluster_descriptors.
    """
    real_ids = [cid for cid in summaries if cid != -1]
    if len(real_ids) < 2:
        for cid in real_ids:
            for key in CONTRASTIVE_PRIVATE_KEYS:
                summaries[cid].pop(key, None)
        return

    evidence_lists = {cid: _split_terms(summaries[cid].get("distinguishing_evidence_terms")) for cid in real_ids}
    context_lists = {cid: _split_terms(summaries[cid].get("_contexts")) for cid in real_ids}
    method_lists = {cid: _split_terms(summaries[cid].get("_methods")) for cid in real_ids}

    action_counts = Counter(str(summaries[cid].get("_action_key") or "") for cid in real_ids)
    majority_action = action_counts.most_common(1)[0][0]
    contribution_sets = {cid: frozenset(_split_terms(summaries[cid].get("_contributions"))) for cid in real_ids}
    majority_contribution = Counter(contribution_sets.values()).most_common(1)[0][0]

    for cid in sorted(real_ids):
        summary = summaries[cid]
        action_key = str(summary.get("_action_key") or "")
        form_lower = str(summary.get("design_knowledge_form") or "design knowledge").lower()
        evidence = evidence_lists.get(cid, [])
        unique_evidence = _unique_phrases(cid, evidence_lists)
        unique_contexts = _unique_phrases(cid, context_lists)
        unique_methods = _unique_phrases(cid, method_lists)

        sentences: list[str] = []
        if unique_evidence:
            sentences.append(
                f"Within this keyword view, this cluster is set apart by "
                f"{format_facet_values(unique_evidence[:3])}."
            )
        elif evidence:
            sentences.append(
                f"This cluster is anchored by {format_facet_values(evidence[:3])}, "
                "signals it partly shares with neighboring clusters."
            )

        action_noun = ACTION_NOUN_PHRASES.get(action_key, "synthesis of prior work")
        if action_key and action_key != majority_action:
            sentences.append(
                f"Unlike most clusters in this view, its dominant design-knowledge move is {action_noun}."
            )
        else:
            connective = "on" if action_noun.endswith("prior work") else "of"
            sentences.append(
                f"Its dominant move, {action_noun} {connective} {form_lower}, is shared across most of "
                "this view, so the distinguishing terms above carry the interpretive weight."
            )

        if unique_contexts:
            sentences.append(
                f"It concentrates on contexts such as {format_facet_values(unique_contexts[:2])} "
                "that do not anchor the sibling clusters."
            )
        if unique_methods:
            sentences.append(
                f"Methodologically, it leans on {format_facet_values(unique_methods[:2])}."
            )

        contributions = _split_terms(summary.get("_contributions"))
        if contributions and contribution_sets[cid] != majority_contribution:
            sentences.append(
                f"Its contribution coding ({', '.join(contributions)}) also departs from the view majority."
            )

        rep_titles = summary.get("_rep_titles") or []
        if rep_titles:
            sentences.append("Representative papers: " + "; ".join(rep_titles[:2]) + ".")

        summary["cluster_summary_candidate"] = " ".join(sentences)
        for key in CONTRASTIVE_PRIVATE_KEYS:
            summary.pop(key, None)


def distinguish_duplicate_cluster_descriptors(summaries: dict[int, dict[str, str]]) -> None:
    labels: dict[str, list[int]] = {}
    for cluster_id, summary in summaries.items():
        label = str(summary.get("cluster_label_candidate", "")).strip()
        if not label or label == "Unclustered papers":
            continue
        label_without_cluster = re.sub(r"^Cluster\s+-?\d+:\s*", "", label, flags=re.I)
        labels.setdefault(label_without_cluster, []).append(cluster_id)
    for cluster_ids in labels.values():
        if len(cluster_ids) < 2:
            continue
        used_suffixes: set[str] = set()
        for cluster_id in cluster_ids:
            summary = summaries[cluster_id]
            suffix = descriptor_evidence_suffix(summary)
            normalized_suffix = normalize_phrase(suffix)
            if not suffix or normalized_suffix in used_suffixes:
                continue
            used_suffixes.add(normalized_suffix)
            summary["cluster_label_candidate"] = f"{summary['cluster_label_candidate']} around {suffix}"


def build_cluster_summaries(df: pd.DataFrame, text_view: str = "overall") -> dict[int, dict[str, str]]:
    summaries = {}
    for label in sorted(set(df["cluster"])):
        subset = df[df["cluster"] == label].sort_values(["representative_rank", "medoid_rank"])
        if label == -1:
            summaries[int(label)] = {
                "cluster_label_candidate": "Unclustered papers",
                "cluster_summary_candidate": (
                    "HDBSCAN did not assign these papers to a dense cluster. "
                    "They are retained for inspection but should not be interpreted as a thematic cluster."
                ),
                "design_knowledge_form": "n/a",
                "design_knowledge_action": "n/a",
                "design_knowledge_role": "n/a",
                "design_knowledge_contribution": "Not interpreted as a cluster theme.",
                "facet_population_or_context": "",
                "facet_stakeholder_or_population": "",
                "facet_method_or_lens": "",
                "facet_artifact_or_domain": "",
                "facet_contribution_or_outcome": "",
                "distinguishing_evidence_terms": "",
            }
            continue
        facets = {facet: find_facet_matches_for_cluster(subset, facet) for facet in FACET_PATTERNS}
        contribution_types = infer_contribution_types_for_cluster(subset)
        design_claim = infer_design_knowledge_claim(subset, facets, contribution_types)
        keyphrases = [
            phrase.strip()
            for phrase in str(subset.iloc[0].get("cluster_theme_terms", "")).split(",")
            if phrase.strip()
        ]
        preferred = preferred_keyphrases_for_cluster(subset, keyphrases)
        label_parts = []
        roots_seen: set[str] = set()
        for phrase in preferred:
            add_label_phrase(label_parts, roots_seen, phrase)
            if len(label_parts) >= 3:
                break
        for facet in ["population_or_context", "stakeholder_or_population", "artifact_or_domain", "method_or_lens"]:
            for phrase in facets[facet]:
                add_label_phrase(label_parts, roots_seen, phrase)
                if len(label_parts) >= 3:
                    break
            if len(label_parts) >= 3:
                break
        for phrase in contribution_types:
            add_label_phrase(label_parts, roots_seen, phrase, allow_generic=True)
            if len(label_parts) >= 3:
                break
        for phrase in keyphrases:
            add_label_phrase(label_parts, roots_seen, phrase, allow_generic=True)
            if len(label_parts) >= 3:
                break
        if text_view == "knowledge-type":
            keyword_label_parts = dominant_design_keywords(subset)
            if keyword_label_parts:
                label_parts = keyword_label_parts
        if not label_parts:
            label_parts = keyphrases[:3]
        cluster_label = conservative_cluster_descriptor(int(label), design_claim)
        evidence_terms = distinguishing_evidence_terms(subset, keyphrases, facets, contribution_types)

        context = format_facet_values(facets["population_or_context"][:3]) or ", ".join(keyphrases[:2]) or "the selected papers"
        stakeholders = format_facet_values(facets["stakeholder_or_population"][:3])
        method = format_facet_values(facets["method_or_lens"][:3])
        contribution = ", ".join(contribution_types[:3])
        artifacts = format_facet_values(facets["artifact_or_domain"][:3])
        clauses = [f"This cluster focuses on {context}"]
        if stakeholders:
            clauses.append(f"involving {stakeholders}")
        if method:
            clauses.append(f"using {method} as the main methodological or conceptual lens")
        if artifacts:
            clauses.append(f"with recurring attention to {artifacts}")
        if contribution:
            clauses.append(f"with contribution types coded as {contribution}")
        summary = design_claim["cluster_summary_candidate"] or ", ".join(clauses) + "."

        summaries[int(label)] = {
            "cluster_label_candidate": cluster_label,
            "cluster_summary_candidate": summary,
            "design_knowledge_form": design_claim["design_knowledge_form"],
            "design_knowledge_action": design_claim["design_knowledge_action"],
            "design_knowledge_role": design_claim["design_knowledge_role"],
            "design_knowledge_contribution": design_claim["design_knowledge_contribution"],
            "facet_population_or_context": format_facet_values(facets["population_or_context"]),
            "facet_stakeholder_or_population": format_facet_values(facets["stakeholder_or_population"]),
            "facet_method_or_lens": format_facet_values(facets["method_or_lens"]),
            "facet_artifact_or_domain": format_facet_values(facets["artifact_or_domain"]),
            "facet_contribution_or_outcome": ", ".join(contribution_types),
            "distinguishing_evidence_terms": evidence_terms,
            "_action_key": design_claim.get("_action_key", ""),
            "_rep_titles": design_claim.get("_rep_titles", []),
            "_contexts": facets["population_or_context"][:4],
            "_methods": facets["method_or_lens"][:4],
            "_contributions": contribution_types[:3],
        }
    apply_contrastive_summary_pass(summaries)
    distinguish_duplicate_cluster_descriptors(summaries)
    return summaries


def top_terms_by_cluster(texts: list[str], labels: np.ndarray, top_n: int = 8) -> dict[int, str]:
    unique_labels = sorted(label for label in set(labels) if label != -1)
    if not unique_labels:
        return {-1: "noise / outliers"}

    cluster_docs = []
    for label in unique_labels:
        cluster_docs.append(" ".join(texts[i] for i in np.where(labels == label)[0]))

    vectorizer = CountVectorizer(
        stop_words=STOPWORDS,
        min_df=1,
        max_features=8000,
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b(?:ai|ux|xr|[a-zA-Z]{3,})\b",
    )
    counts = vectorizer.fit_transform(cluster_docs).astype(float)
    terms = np.asarray(vectorizer.get_feature_names_out())

    # Class-based TF-IDF: each cluster is treated as one combined document.
    tf = counts / np.maximum(counts.sum(axis=1), 1)
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(1 + len(unique_labels) / np.maximum(df, 1))
    ctfidf = tf.multiply(idf).toarray()

    labels_to_terms = {}
    for row_idx, label in enumerate(unique_labels):
        weights = ctfidf[row_idx]
        ranked = terms[np.argsort(weights)[::-1]].tolist()
        cleaned = _clean_label_terms(ranked, top_n)
        labels_to_terms[label] = ", ".join(cleaned or ranked[:top_n])

    for label in sorted(set(labels)):
        if label == -1:
            labels_to_terms[label] = "noise / outliers"
    return labels_to_terms


def write_summary(df: pd.DataFrame, cluster_terms: dict[int, str], topic_words: list[str], out: Path) -> None:
    lines = ["# Clustering Summary", ""]
    lines.append("## Cluster Themes")
    for label in sorted(cluster_terms):
        subset = df[df["cluster"] == label]
        lines.append("")
        lines.append(f"### Cluster {label} ({len(subset)} papers)")
        if len(subset):
            lines.append(f"Label candidate: {subset.iloc[0].get('cluster_label_candidate', '')}")
            lines.append(f"Distinguishing evidence: {subset.iloc[0].get('distinguishing_evidence_terms', '')}")
            lines.append(f"Summary candidate: {subset.iloc[0].get('cluster_summary_candidate', '')}")
        lines.append(f"Theme words: {cluster_terms[label]}")
        reps = subset.sort_values(["representative_rank", "medoid_rank"]).head(3)
        lines.append("")
        lines.append("Representative papers:")
        for _, row in reps.iterrows():
            lines.append(f"- {row.get('year', '')}: {row.get('title', '')}")
    lines.append("")
    lines.append("## LDA Topics")
    for i, words in enumerate(topic_words):
        lines.append(f"- Topic {i}: {words}")
    out.write_text("\n".join(lines), encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_comma_terms(value: object) -> list[str]:
    terms = []
    for term in str(value or "").split(","):
        term = term.strip()
        if term and term.lower() != "nan":
            terms.append(term)
    return terms


def build_cluster_source_evidence(subset: pd.DataFrame, label: int) -> dict[str, object]:
    if label == -1 or subset.empty:
        return {"items": [], "faithfulness_score": None}

    first = subset.iloc[0]
    cluster_terms = []
    cluster_terms.extend(split_comma_terms(first.get("distinguishing_evidence_terms", "")))
    cluster_terms.extend(split_comma_terms(first.get("cluster_theme_terms", "")))
    cluster_terms.extend(split_comma_terms(first.get("facet_contribution_or_outcome", "")))
    dominant_keywords = dominant_design_keywords(subset, limit=2)
    cluster_terms.extend(dominant_keywords)

    unique_terms = []
    seen_terms = set()
    for term in cluster_terms:
        key = normalize_phrase(term)
        if not key or key in seen_terms:
            continue
        unique_terms.append(term)
        seen_terms.add(key)
    cluster_terms = unique_terms

    keyword = dominant_keywords[0] if dominant_keywords else str(first.get("keyword", "") or "")
    papers = []
    for _, row in subset.iterrows():
        paragraphs = split_context_paragraphs(row_analysis_context(row))
        if not paragraphs:
            continue
        papers.append(
            PaperParagraphs(
                paper_id=str(row.get("paper_id", "")),
                title=str(row.get("title", "")),
                paragraphs=paragraphs,
                is_representative=bool(row.get("is_representative_top3", False)),
            )
        )

    evidence = select_cluster_evidence(
        papers,
        cluster_terms=cluster_terms,
        keyword=keyword,
        top_k=6,
        max_per_paper=2,
        min_score=3.0,
    )
    summary = str(first.get("cluster_summary_candidate", "") or "")
    return {
        "items": [
            {
                "id": f"E{index}",
                "paper_id": item.paper_id,
                "paper_title": item.paper_title,
                "sentences": item.sentences,
                "paragraph_index": item.paragraph_index,
                "sentence_hash": item.sentence_hash,
                "score": item.score,
                "score_breakdown": asdict(item.score_breakdown),
            }
            for index, item in enumerate(evidence, start=1)
        ],
        "faithfulness_score": round(faithfulness_score(summary, evidence), 2) if summary else None,
    }


def write_dashboard(df: pd.DataFrame, out: Path, title: str) -> None:
    cluster_counts = df["cluster"].value_counts().sort_index().to_dict()
    cluster_summaries = []
    for cluster, count in cluster_counts.items():
        subset = df[df["cluster"] == cluster].sort_values(["representative_rank", "medoid_rank"])
        source_evidence = build_cluster_source_evidence(subset, int(cluster))
        cluster_summaries.append(
            {
                "cluster": int(cluster),
                "count": int(count),
                "theme": str(subset.iloc[0]["cluster_theme_terms"]) if len(subset) else "",
                "label": str(subset.iloc[0]["cluster_label_candidate"]) if len(subset) else "",
                "summary": str(subset.iloc[0]["cluster_summary_candidate"]) if len(subset) else "",
                "source_evidence": source_evidence["items"],
                "source_faithfulness_score": source_evidence["faithfulness_score"],
                "representatives": subset.head(3)[["title", "year", "paper_id"]].to_dict("records"),
            }
        )

    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "paper_id": str(row.get("paper_id", "")),
                "title": str(row.get("title", "")),
                "authors": str(row.get("authors", "")),
                "abstract": str(row.get("abstract", "")),
                "discussion_summary": str(row.get("discussion_summary", "")),
                "discussion_excerpt": str(row.get("discussion_excerpt", "")),
                "discussion_found": truthy(row.get("discussion_found", False)),
                "discussion_paragraph_count": int(row.get("discussion_paragraph_count", 0) or 0),
                "year": str(row.get("year", "")),
                "venue": str(row.get("venue", "")),
                "doi": str(row.get("doi", "")),
                "url": str(row.get("url", "")),
                "keyword": str(row.get("keyword", "")),
                "cluster": int(row.get("cluster", -1)),
                "cluster_theme_terms": str(row.get("cluster_theme_terms", "")),
                "cluster_label_candidate": str(row.get("cluster_label_candidate", "")),
                "distinguishing_evidence_terms": str(row.get("distinguishing_evidence_terms", "")),
                "cluster_summary_candidate": str(row.get("cluster_summary_candidate", "")),
                "design_knowledge_form": str(row.get("design_knowledge_form", "")),
                "design_knowledge_action": str(row.get("design_knowledge_action", "")),
                "design_knowledge_role": str(row.get("design_knowledge_role", "")),
                "design_knowledge_contribution": str(row.get("design_knowledge_contribution", "")),
                "facet_population_or_context": str(row.get("facet_population_or_context", "")),
                "facet_stakeholder_or_population": str(row.get("facet_stakeholder_or_population", "")),
                "facet_method_or_lens": str(row.get("facet_method_or_lens", "")),
                "facet_artifact_or_domain": str(row.get("facet_artifact_or_domain", "")),
                "facet_contribution_or_outcome": str(row.get("facet_contribution_or_outcome", "")),
                "umap_x": float(row.get("umap_x", 0)),
                "umap_y": float(row.get("umap_y", 0)),
                "distance_to_centroid": None
                if pd.isna(row.get("distance_to_centroid"))
                else float(row.get("distance_to_centroid")),
                "representative_rank": int(row.get("representative_rank", -1)),
                "is_representative_top3": bool(row.get("is_representative_top3", False)),
                "medoid_rank": int(row.get("medoid_rank", -1)),
                "lda_topic": int(row.get("lda_topic", -1)),
                "lda_topic_probability": float(row.get("lda_topic_probability", 0)),
                "lda_topic_words": str(row.get("lda_topic_words", "")),
                "nearest_papers": row.get("nearest_papers", []),
            }
        )

    payload = {
        "title": title,
        "papers": records,
        "clusters": cluster_summaries,
        "keywords": sorted(str(x) for x in df["keyword"].dropna().unique()),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #1f2a44;
      --muted: #6b7a90;
      --soft: #8b9bb0;
      --line: #dde6f0;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f8fafd;
      --accent: #256fbd;
      --accent-soft: #eaf3ff;
      --shadow: 0 18px 45px rgba(31, 42, 68, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 16px 20px 12px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; min-width: 0; font-weight: 750; }}
    .controls {{
      display: grid;
      grid-template-columns: 180px 210px;
      gap: 10px;
      align-items: center;
    }}
    input, select, button {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 10px;
      background: white;
      color: var(--ink);
      font-size: 14px;
    }}
    button {{ background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; }}
    label.toggle {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    label.toggle input {{ width: 16px; height: 16px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(600px, 1fr) minmax(420px, 470px);
      gap: 12px;
      padding: 12px;
      min-height: calc(100vh - 92px);
      align-items: stretch;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .plot-panel {{ display: grid; grid-template-rows: auto 1fr auto; height: calc(100vh - 116px); min-height: 690px; }}
    .status {{
      padding: 11px 16px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      background: var(--panel-soft);
    }}
    svg {{ width: 100%; height: 100%; min-height: 560px; background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%); }}
    .axis {{ stroke: #e7edf5; stroke-width: 1; }}
    .point {{ stroke: white; stroke-width: 1.8; cursor: pointer; opacity: 0.86; transition: opacity 120ms ease; }}
    .point:hover {{ opacity: 1; stroke: #1f2a44; stroke-width: 2.6; }}
    .point.dim {{ opacity: 0.14; }}
    .point.selected {{ stroke: #111827; stroke-width: 3; opacity: 1; }}
    .legend, .cluster-list {{ padding: 10px 12px; border-top: 1px solid var(--line); }}
    .legend {{ max-height: 150px; overflow-y: auto; background: #ffffff; }}
    .legend-items {{ display: flex; flex-wrap: wrap; gap: 7px 10px; }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border: 1px solid #e7edf5;
      border-radius: 999px;
      background: var(--panel-soft);
      font-size: 12px;
      color: var(--muted);
      max-width: 100%;
    }}
    .swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .details {{ padding: 0; overflow: auto; height: calc(100vh - 116px); min-height: 690px; align-self: stretch; }}
    .details-inner {{ padding: 18px 18px 22px; }}
    .paper-heading {{
      position: sticky;
      top: 0;
      z-index: 1;
      padding: 18px 18px 14px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    .paper-heading h2 {{ font-size: 19px; margin: 0 0 8px; line-height: 1.22; font-weight: 760; }}
    .details h2 {{ font-size: 18px; margin: 0 0 8px; line-height: 1.25; }}
    .meta {{ color: var(--muted); font-size: 13px; line-height: 1.45; margin-bottom: 10px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 0; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      margin: 0;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #31577f;
      font-size: 12px;
      line-height: 1.25;
    }}
    .abstract {{ line-height: 1.55; font-size: 14px; color: #2f3b52; }}
    .insight-card {{
      margin-top: 12px;
      padding: 13px 14px;
      border: 1px solid #e4ebf4;
      border-radius: 8px;
      background: var(--panel-soft);
    }}
    .insight-card.primary {{
      background: #ffffff;
      border-left: 4px solid var(--accent);
    }}
    details.discussion-full,
    details.lexical-evidence {{
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 11px;
      background: #fbfcfe;
    }}
    details.discussion-full summary,
    details.lexical-evidence summary {{
      cursor: pointer;
      color: #31577f;
      font-weight: 700;
      font-size: 13px;
    }}
    .evidence-block {{ margin-top: 8px; }}
    .source-evidence-item {{
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }}
    .source-evidence-item:first-child {{ border-top: 0; padding-top: 0; }}
    .source-evidence-quote {{
      line-height: 1.45;
      font-size: 13px;
      color: #2f3b52;
      margin: 6px 0;
    }}
    .source-evidence-reasons {{
      margin: 6px 0 0;
      padding-left: 16px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .discussion-list {{
      margin: 6px 0 0;
      padding-left: 18px;
      line-height: 1.45;
      font-size: 14px;
      color: #2f3b52;
    }}
    .discussion-list li {{ margin: 0 0 7px; }}
    .section-title {{ margin: 0 0 7px; font-weight: 760; font-size: 12px; text-transform: uppercase; color: var(--soft); letter-spacing: 0.02em; }}
    .paper-link {{ display: block; color: var(--accent); text-decoration: none; margin: 7px 0; font-size: 13px; line-height: 1.35; }}
    .cluster-card {{ border-top: 1px solid var(--line); padding: 10px 12px; }}
    .cluster-card button {{ height: 28px; padding: 0 8px; margin-top: 6px; font-size: 12px; }}
    @media (max-width: 980px) {{
      header {{ display: block; }}
      h1 {{ margin-bottom: 12px; }}
      .controls, main {{ grid-template-columns: 1fr; }}
      .plot-panel {{ min-height: 560px; }}
    }}
  </style>
</head>
<body>
  <div style="display:none" aria-hidden="true">
      <select id="clusterFilter"><option value="all">All clusters</option></select>
  </div>
  <main>
    <section class="panel plot-panel">
      <div class="status" id="status"></div>
      <svg id="plot" role="img" aria-label="UMAP scatter plot"></svg>
      <div class="legend"><div class="legend-items" id="legend"></div></div>
    </section>
    <aside class="panel details" id="details">
      <div class="details-inner">
        <h2>Select a paper</h2>
        <p class="meta">Click a point to inspect metadata, cluster interpretation, evidence terms, representative rank, and nearest papers.</p>
        <div id="clusterCards"></div>
      </div>
    </aside>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('payload').textContent);
    const papers = data.papers;
    const palette = ['#f05a71','#4c78a8','#f58518','#54a24b','#b279a2','#72b7b2','#eeca3b','#ff9da6','#9d755d','#59a14f','#edc948','#76b7b2'];
    const colorFor = c => palette[Math.abs(Number(c)) % palette.length];
    const clusterName = c => Number(c) === -1 ? 'Unclustered papers' : `Cluster ${{c}}`;
    const clusterLegendLabel = c => {{
      const label = String(c.label || c.theme || '').trim();
      if (!label) return clusterName(c.cluster);
      return /^Cluster\\s+-?\\d+:/i.test(label) ? label : `${{clusterName(c.cluster)}}: ${{label}}`;
    }};
    const state = {{ selected: papers[0]?.paper_id || null }};
    const svg = document.getElementById('plot');
    const details = document.getElementById('details');
    const status = document.getElementById('status');

    function initControls() {{
      const clusterFilter = document.getElementById('clusterFilter');
      data.clusters.forEach(c => {{
        const opt = document.createElement('option');
        opt.value = c.cluster;
        opt.textContent = `${{clusterName(c.cluster)}} (${{c.count}})`;
        clusterFilter.appendChild(opt);
      }});
      ['clusterFilter'].forEach(id => {{
        document.getElementById(id).addEventListener('input', render);
        document.getElementById(id).addEventListener('change', render);
      }});
      renderLegend();
    }}

    function filteredPapers() {{
      const cluster = document.getElementById('clusterFilter').value;
      return papers.filter(p => {{
        return (cluster === 'all' || String(p.cluster) === cluster);
      }});
    }}

    function renderLegend() {{
      const legend = document.getElementById('legend');
      legend.innerHTML = '';
      data.clusters.forEach(c => {{
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<span class="swatch" style="background:${{colorFor(c.cluster)}}"></span> ${{escapeHtml(clusterLegendLabel(c))}}`;
        legend.appendChild(item);
      }});
      const shapeNote = document.createElement('span');
      shapeNote.className = 'legend-item';
      shapeNote.innerHTML = '<span>◆ Discussion detected</span><span>● No explicit discussion</span>';
      legend.appendChild(shapeNote);
    }}

    function render() {{
      const shown = filteredPapers();
      status.textContent = `${{shown.length}} of ${{papers.length}} papers shown; ${{data.clusters.length}} clusters in this view. Axes are UMAP 1 and UMAP 2 coordinates, not interpretable variables.`;
      drawPlot(shown);
      renderDetails(papers.find(p => p.paper_id === state.selected) || shown[0] || papers[0]);
    }}

    function drawPlot(shown) {{
      const width = svg.clientWidth || 900;
      const height = svg.clientHeight || 560;
      const pad = 44;
      const xs = papers.map(p => p.umap_x), ys = papers.map(p => p.umap_y);
      let minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
      const xRange = Math.max(0.0001, maxX - minX);
      const yRange = Math.max(0.0001, maxY - minY);
      minX -= xRange * 0.08;
      maxX += xRange * 0.08;
      minY -= yRange * 0.08;
      maxY += yRange * 0.08;
      const sx = x => pad + ((x - minX) / Math.max(0.0001, maxX - minX)) * (width - pad * 2);
      const sy = y => height - pad - ((y - minY) / Math.max(0.0001, maxY - minY)) * (height - pad * 2);
      const shownIds = new Set(shown.map(p => p.paper_id));
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = '';
      for (let i = 0; i < 6; i++) {{
        const x = pad + i * (width - pad * 2) / 5;
        const y = pad + i * (height - pad * 2) / 5;
        svg.insertAdjacentHTML('beforeend', `<line class="axis" x1="${{x}}" y1="${{pad}}" x2="${{x}}" y2="${{height-pad}}"></line>`);
        svg.insertAdjacentHTML('beforeend', `<line class="axis" x1="${{pad}}" y1="${{y}}" x2="${{width-pad}}" y2="${{y}}"></line>`);
      }}
      svg.insertAdjacentHTML('beforeend', `<text x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle" fill="#65758b" font-size="12">UMAP 1</text>`);
      svg.insertAdjacentHTML('beforeend', `<text x="16" y="${{height / 2}}" text-anchor="middle" fill="#65758b" font-size="12" transform="rotate(-90 16 ${{height / 2}})">UMAP 2</text>`);
      papers.forEach(p => {{
        const visible = shownIds.has(p.paper_id);
        const selected = p.paper_id === state.selected;
        const x = sx(p.umap_x), y = sy(p.umap_y);
        const size = selected ? 7.5 : 6.5;
        const cls = `point ${{visible ? '' : 'dim'}} ${{selected ? 'selected' : ''}}`;
        const fill = colorFor(p.cluster);
        const node = document.createElementNS('http://www.w3.org/2000/svg', p.discussion_found ? 'path' : 'circle');
        node.setAttribute('class', cls);
        node.setAttribute('fill', fill);
        if (p.discussion_found) {{
          const d = `M ${{x}} ${{y - size}} L ${{x + size}} ${{y}} L ${{x}} ${{y + size}} L ${{x - size}} ${{y}} Z`;
          node.setAttribute('d', d);
        }} else {{
          node.setAttribute('cx', x);
          node.setAttribute('cy', y);
          node.setAttribute('r', size);
        }}
        node.addEventListener('click', () => {{
          state.selected = p.paper_id;
          render();
        }});
        node.appendChild(document.createElementNS('http://www.w3.org/2000/svg', 'title')).textContent = p.title;
        svg.appendChild(node);
      }});
    }}

    function clusterRecord(clusterId) {{
      return data.clusters.find(c => String(c.cluster) === String(clusterId));
    }}

    function renderSourceEvidence(clusterId) {{
      const cluster = clusterRecord(clusterId);
      const evidence = cluster?.source_evidence || [];
      const score = cluster?.source_faithfulness_score;
      const scorePill = Number.isFinite(Number(score)) ?
        `<span class="pill">${{Math.round(Number(score) * 100)}}% summary support</span>` : '';
      if (!evidence.length) {{
        return `
          <div class="insight-card">
            <div class="section-title">Source Evidence ${{scorePill}}</div>
            <div class="meta">No direct source evidence selected from extracted full text for this cluster.</div>
          </div>
        `;
      }}
      const items = evidence.map(item => {{
        const reasons = item.score_breakdown?.reasons || [];
        const reasonList = reasons.slice(0, 4).map(reason => `<li>${{escapeHtml(reason)}}</li>`).join('');
        return `
          <div class="source-evidence-item">
            <div class="pill-row">
              <span class="pill">${{escapeHtml(item.id)}}</span>
              <span class="pill">score ${{Number(item.score).toFixed(1)}}</span>
              <span class="pill">para ${{Number(item.paragraph_index) + 1}}</span>
              <span class="pill">hash ${{escapeHtml(item.sentence_hash)}}</span>
            </div>
            <div class="source-evidence-quote">${{escapeHtml(item.sentences)}}</div>
            <a class="paper-link" href="#" data-paper="${{escapeAttr(item.paper_id)}}">${{escapeHtml(item.paper_title)}}</a>
            ${{reasonList ? `<ul class="source-evidence-reasons">${{reasonList}}</ul>` : ''}}
          </div>
        `;
      }}).join('');
      return `
        <div class="insight-card">
          <div class="section-title">Source Evidence ${{scorePill}}</div>
          <div class="evidence-block">${{items}}</div>
        </div>
      `;
    }}

    function renderDetails(p) {{
      if (!p) return;
      const doi = p.doi ? `<a class="paper-link" href="https://doi.org/${{escapeHtml(p.doi)}}" target="_blank">DOI: ${{escapeHtml(p.doi)}}</a>` : '';
      const url = p.url ? `<a class="paper-link" href="${{escapeAttr(p.url)}}" target="_blank">Open paper URL</a>` : '';
      const nearest = (p.nearest_papers || []).map(n =>
        `<a class="paper-link" href="#" data-paper="${{escapeAttr(n.paper_id)}}">${{clusterName(n.cluster)}}, distance ${{Number(n.distance).toFixed(3)}}: ${{escapeHtml(n.title)}}</a>`
      ).join('');
      const discussionSummary = p.discussion_summary ?
        `<div class="insight-card"><div class="section-title">Discussion Summary</div>${{renderDiscussionSummary(p.discussion_summary)}}</div>` : '';
      const discussionExcerpt = '';
      const discussionStatus = p.discussion_found ?
        `<span class="pill">Discussion detected: ${{p.discussion_paragraph_count}} paragraphs</span>` :
        `<span class="pill">No explicit Discussion section detected</span>`;
      const sourceEvidence = renderSourceEvidence(p.cluster);
      details.innerHTML = `
        <div class="paper-heading">
          <h2>${{escapeHtml(p.title)}}</h2>
          <div class="meta">${{escapeHtml(p.authors)}}<br>${{escapeHtml(p.year)}} · ${{escapeHtml(p.venue)}}</div>
          <div class="pill-row">
            <span class="pill">${{clusterName(p.cluster)}}</span>
            <span class="pill">${{escapeHtml(p.keyword)}}</span>
            <span class="pill">Rep rank ${{p.representative_rank}}</span>
            <span class="pill">Medoid rank ${{p.medoid_rank}}</span>
            <span class="pill">LDA topic ${{p.lda_topic}} (${{Math.round(p.lda_topic_probability * 100)}}%)</span>
            ${{discussionStatus}}
          </div>
        </div>
        <div class="details-inner">
        <div class="insight-card primary">
          <div class="section-title">Cluster Descriptor</div>
          <div class="abstract">${{escapeHtml(p.cluster_label_candidate)}}</div>
        </div>
        <div class="insight-card">
          <div class="section-title">Distinguishing Evidence</div>
          <div class="abstract">${{escapeHtml(p.distinguishing_evidence_terms || p.cluster_theme_terms || 'n/a')}}</div>
        </div>
        <div class="insight-card">
          <div class="section-title">Cluster Summary Candidate</div>
          <div class="abstract">${{escapeHtml(p.cluster_summary_candidate)}}</div>
        </div>
        ${{sourceEvidence}}
        <div class="insight-card">
          <div class="section-title">Design-Knowledge Contribution</div>
          <div class="pill-row">
            <span class="pill">Form: ${{escapeHtml(p.design_knowledge_form || 'n/a')}}</span>
            <span class="pill">Action: ${{escapeHtml(p.design_knowledge_action || 'n/a')}}</span>
            <span class="pill">Auto role: ${{escapeHtml(p.design_knowledge_role || 'n/a')}}</span>
          </div>
          <div class="abstract" style="margin-top:8px">${{escapeHtml(p.design_knowledge_contribution || '')}}</div>
        </div>
        <div class="insight-card">
          <div class="section-title">Paper-Oriented Facets</div>
          <div class="pill-row">
            <span class="pill">Context/Domain: ${{escapeHtml(p.facet_population_or_context || 'n/a')}}</span>
            <span class="pill">Population/Stakeholder: ${{escapeHtml(p.facet_stakeholder_or_population || 'n/a')}}</span>
            <span class="pill">Method/Lens: ${{escapeHtml(p.facet_method_or_lens || 'n/a')}}</span>
            <span class="pill">Artifact/System: ${{escapeHtml(p.facet_artifact_or_domain || 'n/a')}}</span>
            <span class="pill">Contribution Type: ${{escapeHtml(p.facet_contribution_or_outcome || 'n/a')}}</span>
          </div>
        </div>
        <details class="lexical-evidence">
          <summary>Show lexical evidence</summary>
          <div class="evidence-block">
            <div class="section-title">Cluster Theme Evidence</div>
            <div class="abstract">${{escapeHtml(p.cluster_theme_terms)}}</div>
            <div class="section-title">Secondary Topic-Model Evidence</div>
            <div class="abstract">${{escapeHtml(p.lda_topic_words)}}</div>
          </div>
        </details>
        <div class="insight-card">
          <div class="section-title">Abstract</div>
          <div class="abstract">${{escapeHtml(p.abstract)}}</div>
        </div>
        ${{discussionSummary}}
        ${{discussionExcerpt}}
        <div class="insight-card">
          <div class="section-title">Links</div>
          ${{doi}}${{url}}
        </div>
        <div class="insight-card">
          <div class="section-title">Nearest Papers by Cosine Distance</div>
          ${{nearest || '<div class="meta">No nearest-paper data available.</div>'}}
        </div>
        </div>
      `;
      details.querySelectorAll('[data-paper]').forEach(a => {{
        a.addEventListener('click', evt => {{
          evt.preventDefault();
          state.selected = a.getAttribute('data-paper');
          render();
        }});
      }});
      details.scrollTop = 0;
    }}

    function renderDiscussionSummary(value) {{
      const items = String(value || '').split('\\n').map(line => line.replace(/^[-•]\\s*/, '').trim()).filter(Boolean);
      if (!items.length) return '<div class="meta">No discussion summary available.</div>';
      return `<ul class="discussion-list">${{items.map(item => `<li>${{escapeHtml(item)}}</li>`).join('')}}</ul>`;
    }}

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function escapeAttr(value) {{ return escapeHtml(value).replace(/`/g, '&#96;'); }}

    initControls();
    render();
    window.addEventListener('resize', render);
  </script>
</body>
</html>"""
    out.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/paper_sample_100.csv")
    parser.add_argument("--outdir", default="outputs/sample_100")
    parser.add_argument("--method", choices=["dbscan", "hdbscan", "kmeans"], default="dbscan")
    parser.add_argument("--embedding", choices=["ollama", "tfidf-svd"], default="ollama")
    parser.add_argument(
        "--text-view",
        choices=["overall", "context", "knowledge-type", "method", "target-user", "purpose"],
        default="overall",
        help="Choose which paper representation to embed for clustering.",
    )
    parser.add_argument(
        "--focus-keyword",
        default=None,
        help=(
            "Run keyword-conditioned clustering for one design-knowledge keyword. "
            "Rows whose keyword column does not include this value are filtered out, "
            "and extracted_context is re-weighted toward paragraphs mentioning this keyword."
        ),
    )
    parser.add_argument(
        "--keyword-context-paragraphs",
        type=int,
        default=6,
        help="Maximum extracted_context paragraphs to keep per paper for --focus-keyword runs.",
    )
    parser.add_argument(
        "--min-papers-to-cluster",
        type=int,
        default=6,
        help="Minimum papers required after --focus-keyword filtering.",
    )
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--select-k", action="store_true", help="Choose k for k-means using average silhouette score.")
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--dbscan-eps", type=float, default=None)
    parser.add_argument("--lda-topics", type=int, default=10)
    parser.add_argument("--embedding-cache", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input).fillna("")
    focus_keyword = canonical_design_keyword(args.focus_keyword) if args.focus_keyword else None
    if focus_keyword:
        before_count = len(df)
        df = df[df.apply(lambda row: row_matches_focus_keyword(row, focus_keyword), axis=1)].copy()
        if len(df) < args.min_papers_to_cluster:
            raise ValueError(
                f"--focus-keyword {focus_keyword!r} matched {len(df)} papers from {before_count}; "
                f"need at least {args.min_papers_to_cluster}. Lower --min-papers-to-cluster or summarize without clustering."
            )
        print(f"filtered to {len(df)}/{before_count} papers for focus keyword: {focus_keyword}")

    context_records = df.apply(
        lambda row: keyword_conditioned_context(row, focus_keyword, args.keyword_context_paragraphs),
        axis=1,
    )
    df["keyword_conditioned_context"] = [context for context, _ in context_records]
    df["keyword_conditioned_paragraph_count"] = [count for _, count in context_records]
    df["focus_keyword"] = focus_keyword or ""
    df["paper_text"] = df.apply(lambda row: paper_text_for_view(row, args.text_view, focus_keyword), axis=1)
    texts = df["paper_text"].tolist()

    cache_path = Path(args.embedding_cache) if args.embedding_cache else None
    if cache_path and cache_path.exists():
        vectors = np.load(cache_path)
        embedding_label = args.ollama_model if args.embedding == "ollama" else "tfidf-svd"
        print(f"loaded embeddings from {cache_path}")
    else:
        if args.embedding == "ollama":
            vectors = ollama_embeddings(texts, args.ollama_model, args.ollama_host)
            embedding_label = args.ollama_model
        else:
            vectors, _ = tfidf_embeddings(texts, max_features=8000, n_components=50)
            embedding_label = "tfidf-svd"
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vectors)
            print(f"saved embeddings to {cache_path}")

    selected_k = args.k
    silhouette_table = None
    if args.select_k:
        if args.method != "kmeans":
            raise ValueError("--select-k is currently supported only with --method kmeans")
        dynamic_k_max = min(args.k_max, dynamic_k_cap(len(df)), len(df) - 1)
        dynamic_k_min = min(args.k_min, dynamic_k_max)
        dynamic_k_min = max(2, dynamic_k_min)
        k_values = list(range(dynamic_k_min, dynamic_k_max + 1))
        selected_k, silhouette_table = kmeans_silhouette_selection(vectors, k_values)
        silhouette_table.to_csv(outdir / "silhouette_scores.csv", index=False)
        if selected_k >= 2:
            selected_row = silhouette_table[silhouette_table["k"] == selected_k].iloc[0]
            selected_score = f"{selected_row['average_silhouette']:.4f}"
            selection_note = (
                "Selected k is the highest-silhouette candidate; when scores are nearly tied, "
                "the smaller k is used for readability."
            )
        else:
            selected_score = "n/a"
            selection_note = "No valid multi-cluster candidate was available, so this view is kept as one interpretive group."
        (outdir / "silhouette_summary.md").write_text(
            "\n".join(
                [
                    "# K-Means Silhouette Selection",
                    "",
                    f"- Requested k range: {args.k_min}-{args.k_max}",
                    f"- Dynamic k range after paper-count cap: {dynamic_k_min}-{dynamic_k_max}",
                    f"- Selected k: {selected_k}",
                    f"- Selected average silhouette score: {selected_score}",
                    f"- Distance metric for silhouette: cosine",
                    f"- Selection note: {selection_note}",
                    "",
                    "Silhouette analysis evaluates whether each paper is closer to papers in its own cluster than to papers in neighboring clusters. It is a diagnostic signal for k selection, not a replacement for human interpretation.",
                ]
            ),
            encoding="utf-8",
        )
        fig_sil = px.bar(
            silhouette_table,
            x="k",
            y="average_silhouette",
            hover_data=["min_cluster_size", "max_cluster_size", "negative_silhouette_count", "cluster_sizes"],
            title="K-Means Silhouette Analysis",
        )
        fig_sil.update_layout(template="plotly_white", xaxis_dtick=1)
        fig_sil.write_html(outdir / "silhouette_scores.html", include_plotlyjs="cdn")
        print(
            f"selected k={selected_k} using silhouette analysis "
            f"(dynamic candidate range {dynamic_k_min}-{dynamic_k_max} for {len(df)} papers)"
        )
    elif args.method == "kmeans":
        selected_k = min(args.k, dynamic_k_cap(len(df)), len(df) - 1)
        selected_k = max(2, selected_k)
        print(f"using k={selected_k} for K-means after dynamic paper-count cap")

    labels = cluster_labels(vectors, args.method, selected_k, args.min_samples, args.dbscan_eps, args.min_cluster_size)
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=12, min_dist=0.08)
    coords = reducer.fit_transform(vectors)
    distances, ranks, medoid_ranks = representative_stats(vectors, labels)
    lda_topic, lda_prob, topic_words, _ = lda_topics(texts, args.lda_topics)
    cluster_terms = top_terms_by_cluster(texts, labels)

    df["embedding_model"] = embedding_label
    df["text_view"] = args.text_view
    df["cluster_method"] = args.method
    df["cluster"] = labels
    df["cluster_theme_terms"] = df["cluster"].map(cluster_terms)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]
    df["distance_to_centroid"] = distances
    df["representative_rank"] = ranks
    df["is_representative_top3"] = ranks <= 3
    df["medoid_rank"] = medoid_ranks
    df["lda_topic"] = lda_topic
    df["lda_topic_probability"] = lda_prob
    df["lda_topic_words"] = [topic_words[i] for i in lda_topic]
    cluster_summaries = build_cluster_summaries(df, args.text_view)
    for field in [
        "cluster_label_candidate",
        "distinguishing_evidence_terms",
        "cluster_summary_candidate",
        "design_knowledge_form",
        "design_knowledge_action",
        "design_knowledge_role",
        "design_knowledge_contribution",
        "facet_population_or_context",
        "facet_stakeholder_or_population",
        "facet_method_or_lens",
        "facet_artifact_or_domain",
        "facet_contribution_or_outcome",
    ]:
        df[field] = df["cluster"].map(lambda label: cluster_summaries[int(label)][field])
    df["nearest_papers"] = nearest_papers(vectors, df)

    csv_out = outdir / "clustered_papers.csv"
    df.drop(columns=["paper_text"]).to_csv(csv_out, index=False)

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color=df["cluster"].astype(str),
        symbol="is_representative_top3",
        hover_name="title",
        hover_data=["year", "venue", "keyword", "cluster_theme_terms", "representative_rank", "lda_topic_words"],
        title=f"Paper corpus UMAP ({args.embedding}, {args.method}, {args.text_view} view)",
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.82})
    fig.update_layout(legend_title_text="cluster", template="plotly_white")
    fig.write_html(outdir / "umap_clusters.html", include_plotlyjs="cdn")

    write_summary(df, cluster_terms, topic_words, outdir / "cluster_summary.md")
    write_dashboard(df, outdir / "paper_explorer.html", f"Paper Corpus Explorer ({embedding_label}, {args.method}, {args.text_view} view)")
    metadata = {
        "input": args.input,
        "text_view": args.text_view,
        "focus_keyword": focus_keyword,
        "keyword_context_paragraphs": args.keyword_context_paragraphs if focus_keyword else None,
        "embedding": args.embedding,
        "embedding_model": embedding_label,
        "method": args.method,
        "k": selected_k if args.method == "kmeans" else None,
        "k_selection": "silhouette" if args.select_k else "manual",
        "k_min": args.k_min if args.select_k else None,
        "k_max": args.k_max if args.select_k else None,
        "dbscan_eps": args.dbscan_eps,
        "min_samples": args.min_samples,
        "min_cluster_size": args.min_cluster_size if args.method == "hdbscan" else None,
        "papers": len(df),
        "clusters": sorted(int(x) for x in set(labels)),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {csv_out}")
    print(f"Wrote {outdir / 'umap_clusters.html'}")
    print(f"Wrote {outdir / 'paper_explorer.html'}")
    print(f"Wrote {outdir / 'cluster_summary.md'}")


if __name__ == "__main__":
    main()
