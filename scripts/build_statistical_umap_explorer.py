#!/usr/bin/env python3
"""Build the existing UMAP explorer UI from the frozen statistical pipeline.

The script preserves the original interactive dashboard while replacing its
legacy TF-IDF/K-means runs with the 282-paper BGE-M3/Spectral assignments and
adapted class-based TF-IDF cluster interpretations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from ftfy import fix_encoding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from cluster_papers import split_context_paragraphs, write_dashboard
from evidence_scorer import PaperParagraphs, select_cluster_evidence, split_sentences
from theory_typology import classify_theory_move


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def clean_metadata(value: object) -> str:
    return fix_encoding(str(value if value is not None else "")).replace(
        "MoÃàller", "Möller"
    )


DISPLAY_ACRONYMS = {
    "ai": "AI",
    "hci": "HCI",
    "ui": "UI",
    "ux": "UX",
    "c-k": "C-K",
}


def display_phrase(value: str) -> str:
    words = []
    for word in value.split():
        words.append(DISPLAY_ACRONYMS.get(word.casefold(), word))
    phrase = " ".join(words)
    return phrase[:1].upper() + phrase[1:]


def normalize_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def phrase_roots(value: str) -> set[str]:
    roots = set()
    for word in normalize_words(value).split():
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        roots.add(word)
    return roots


def build_cluster_keyphrases(base: pd.DataFrame) -> dict[str, list[str]]:
    """Extract contrastive multiword labels within each predefined keyword group."""
    output: dict[str, list[str]] = {}
    for keyword, keyword_frame in base.groupby("keyword", sort=True):
        cluster_ids = sorted(keyword_frame["cluster_id"].unique())

        class_texts = []
        cluster_papers: dict[str, list[str]] = {}
        for cluster_id in cluster_ids:
            subset = keyword_frame[keyword_frame["cluster_id"] == cluster_id]
            paper_texts = (
                subset["title"].astype(str)
                + " "
                + subset["abstract"].astype(str)
                + " "
                + subset["subdocument"].astype(str)
            ).tolist()
            cluster_papers[cluster_id] = paper_texts
            class_texts.append(" ".join(paper_texts))

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(2, 5),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
            max_features=120_000,
        )
        counts = vectorizer.fit_transform(class_texts).toarray().astype(float)
        terms = np.asarray(vectorizer.get_feature_names_out())
        term_frequency = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        average_class_length = max(float(counts.sum(axis=1).mean()), 1.0)
        inverse_class_frequency = np.log1p(
            average_class_length / np.maximum(counts.sum(axis=0), 1.0)
        )
        scores = term_frequency * inverse_class_frequency

        keyword_roots = phrase_roots(str(keyword))
        for cluster_position, cluster_id in enumerate(cluster_ids):
            papers = cluster_papers[cluster_id]
            paper_matrix = vectorizer.transform(papers)
            support = np.asarray((paper_matrix > 0).sum(axis=0)).ravel()
            adjusted = scores[cluster_position] * np.power(
                support / max(len(papers), 1), 0.15
            )
            adjusted *= np.asarray(
                [1.0 + 0.05 * min(len(term.split()) - 2, 2) for term in terms]
            )
            chosen: list[str] = []
            chosen_roots: list[set[str]] = []
            for term_index in np.argsort(adjusted)[::-1]:
                term = str(terms[term_index]).strip()
                roots = phrase_roots(term)
                if adjusted[term_index] <= 0:
                    break
                if roots == keyword_roots:
                    continue
                if len(papers) >= 5 and support[term_index] < 2:
                    continue
                if len(roots) < 2 or len(roots) < len(term.split()):
                    continue
                if any(
                    noisy in term
                    for noisy in (
                        "et al",
                        "figure ",
                        "table ",
                        "copyright",
                        "proceedings",
                    )
                ):
                    continue
                if any(
                    roots <= prior or prior <= roots or len(roots & prior) / len(roots | prior) > 0.55
                    for prior in chosen_roots
                ):
                    continue
                chosen.append(display_phrase(term))
                chosen_roots.append(roots)
                if len(chosen) == 2:
                    break
            output[cluster_id] = chosen or [display_phrase(str(keyword))]
    return output


def phrase_label(keyword: str, phrases: list[str]) -> str:
    if len(phrases) == 1:
        return f"{keyword} — {phrases[0]}"
    return f"{keyword} — {phrases[0]} and {phrases[1]}"


def cluster_evidence(
    subset: pd.DataFrame, terms: list[str]
) -> list:
    papers = [
        PaperParagraphs(
            paper_id=str(row["paper_id"]),
            title=str(row["title"]),
            paragraphs=(
                [str(row["abstract"]).strip()]
                if str(row["abstract"]).strip()
                else split_context_paragraphs(row["subdocument"])
            ),
            is_representative=bool(row.get("is_representative_top3", False)),
        )
        for _, row in subset.iterrows()
    ]
    return select_cluster_evidence(
        papers,
        cluster_terms=terms,
        keyword=str(subset.iloc[0]["keyword"]),
        top_k=6,
        max_per_paper=2,
        min_score=3.0,
    )


def extractive_summary(evidence: list) -> str:
    if not evidence:
        return "No sufficiently supported representative sentences were found."
    selected = []
    used_papers = set()
    ranked = sorted(
        enumerate(evidence, start=1),
        key=lambda pair: (
            pair[1].score_breakdown.first_person + pair[1].score_breakdown.action,
            1 if pair[1].paragraph_index == 0 else 0,
            pair[1].score,
        ),
        reverse=True,
    )
    for evidence_index, item in ranked:
        if item.paper_id in used_papers:
            continue
        candidates = []
        for sentence in split_sentences(item.sentences):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            sentence = re.sub(r"\s*©\s*\d{4}.*$", "", sentence).strip()
            sentence = re.sub(
                r"(?<=[a-z)])(?=(?:Objective|Purpose|Findings|Method):)",
                " ",
                sentence,
            )
            word_count = len(sentence.split())
            low = sentence.casefold()
            if not 10 <= word_count <= 70:
                continue
            if any(
                cue in low
                for cue in (
                    "paper is organized",
                    "article is organized",
                    "remainder of this paper",
                    "in the following section",
                    "figure ",
                    "table ",
                )
            ):
                continue
            action_score = sum(
                cue in low
                for cue in (
                    "we define",
                    "we propose",
                    "we present",
                    "we introduce",
                    "we develop",
                    "we identify",
                    "we examine",
                    "we analyze",
                    "we evaluate",
                    "we test",
                    "we argue",
                    "this paper",
                    "this article",
                    "this study",
                    "our framework",
                    "our findings",
                )
            )
            candidates.append((action_score, word_count, sentence))
        if not candidates:
            continue
        candidates.sort(key=lambda row: (row[0], -abs(row[1] - 34)), reverse=True)
        selected.append((evidence_index, candidates[0][2]))
        used_papers.add(item.paper_id)
        if len(selected) == 2:
            break
    if not selected:
        item = evidence[0]
        sentence = split_sentences(item.sentences)[0] if item.sentences else ""
        selected = [(1, sentence)]
    return " ".join(
        f"{sentence} [E{evidence_index}]"
        for evidence_index, sentence in selected
    )


def add_geometry(frame: pd.DataFrame, vectors: np.ndarray, seed: int) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    if len(frame) == 1:
        coordinates = np.zeros((1, 2), dtype=float)
    else:
        neighbors = min(15, len(frame) - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=max(2, neighbors),
            min_dist=0.12,
            metric="cosine",
            random_state=seed,
            transform_seed=seed,
        )
        coordinates = reducer.fit_transform(vectors)
    frame["umap_x"] = coordinates[:, 0]
    frame["umap_y"] = coordinates[:, 1]

    frame["distance_to_centroid"] = 0.0
    frame["representative_rank"] = 1
    frame["medoid_rank"] = 1
    frame["is_representative_top3"] = False
    frame["nearest_papers"] = [[] for _ in range(len(frame))]

    all_distances = cosine_distances(vectors)
    for position, row in frame.iterrows():
        nearest = np.argsort(all_distances[position])
        frame.at[position, "nearest_papers"] = [
            {
                "paper_id": str(frame.iloc[index]["paper_id"]),
                "title": str(frame.iloc[index]["title"]),
                "cluster": int(frame.iloc[index]["cluster"]),
                "distance": float(all_distances[position, index]),
            }
            for index in nearest
            if index != position
        ][:5]

    for cluster in sorted(frame["cluster"].unique()):
        indices = np.flatnonzero(frame["cluster"].to_numpy() == cluster)
        cluster_vectors = vectors[indices]
        centroid = normalize(cluster_vectors.mean(axis=0, keepdims=True))[0]
        distances = 1 - cluster_vectors @ centroid
        order = np.argsort(distances)
        for rank, local_index in enumerate(order, start=1):
            frame_index = indices[local_index]
            frame.at[frame_index, "distance_to_centroid"] = float(distances[local_index])
            frame.at[frame_index, "representative_rank"] = rank
            frame.at[frame_index, "medoid_rank"] = rank
            frame.at[frame_index, "is_representative_top3"] = rank <= 3
    return frame


def prepare_scope(
    base: pd.DataFrame,
    vectors: np.ndarray,
    interpretations: pd.DataFrame,
    keyphrases_by_cluster: dict[str, list[str]],
    keyword: str | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    mask = np.ones(len(base), dtype=bool)
    if keyword is not None:
        mask = base["keyword"].eq(keyword).to_numpy()
    frame = base.loc[mask].copy().reset_index(drop=True)
    scope_vectors = vectors[mask]

    cluster_ids = sorted(frame["cluster_id"].unique())
    cluster_number = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    frame["cluster"] = frame["cluster_id"].map(cluster_number).astype(int)

    interpretation_by_id = interpretations.set_index("cluster_id")
    frame["cluster_theme_terms"] = frame["cluster_id"].map(
        lambda cluster_id: " | ".join(
            term.strip()
            for term in str(interpretation_by_id.loc[cluster_id, "top_terms"]).split("|")
            if term.strip()
        )
    )
    frame["cluster_label_candidate"] = frame["cluster_id"].map(
        lambda cluster_id: (
            f"{phrase_label(str(interpretation_by_id.loc[cluster_id, 'keyword']), keyphrases_by_cluster[cluster_id])} "
            f"[{cluster_id}]"
        )
    )
    frame["distinguishing_evidence_terms"] = frame["cluster_id"].map(
        lambda cluster_id: ", ".join(keyphrases_by_cluster[cluster_id][:6])
    )
    frame["design_knowledge_form"] = frame["cluster_id"].map(
        lambda cluster_id: phrase_label(
            str(interpretation_by_id.loc[cluster_id, "keyword"]),
            keyphrases_by_cluster[cluster_id],
        )
    )
    frame["contribution_type"] = "Provisional interpretation"
    frame["contribution_type_support"] = frame["cluster_id"].map(
        lambda cluster_id: (
            "Assignment frozen; interpretation requires reviewer confirmation"
            if str(interpretation_by_id.loc[cluster_id, "interpretation_status"]) == "frozen"
            else "Assignment frozen; interpretation requires confirmation by at least two reviewers"
        )
    )
    frame["contribution_type_definition"] = (
        "Contrastive phrase label with source-grounded extractive summary."
    )
    frame["primary_application_domain"] = "Not assigned by Path 1"
    frame["application_domain_support"] = "Not evaluated in this statistical interpretation"
    frame["application_domain_definitions"] = "Not applicable"
    frame["facet_population_or_context"] = "Predefined keyword group"
    frame["facet_stakeholder_or_population"] = "Not assigned"
    frame["facet_method_or_lens"] = "BGE-M3 embedding + Spectral clustering"
    frame["facet_artifact_or_domain"] = "Not assigned"
    frame["facet_contribution_or_outcome"] = "Source-grounded cluster interpretation"
    frame["theory_move_key"] = "not_applicable"
    frame["theory_move"] = ""
    frame["theory_move_support"] = ""
    frame["theory_move_patterns"] = ""
    frame["lda_topic"] = -1
    frame["lda_topic_probability"] = 0.0
    frame["lda_topic_words"] = frame["cluster_theme_terms"]
    frame["hdbscan_peripheral"] = False
    frame["discussion_found"] = False
    frame["discussion_paragraph_count"] = 0
    frame["discussion_summary"] = ""
    frame["discussion_excerpt"] = ""
    frame["keyword_conditioned_context"] = frame.apply(
        lambda row: (
            str(row["abstract"]).strip()
            if str(row["abstract"]).strip()
            else str(row["subdocument"])
        ),
        axis=1,
    )

    frame = add_geometry(frame, scope_vectors, seed)
    frame["cluster_summary_candidate"] = ""
    for cluster_id, subset in frame.groupby("cluster_id", sort=True):
        terms = keyphrases_by_cluster[cluster_id]
        evidence = cluster_evidence(subset, terms)
        frame.loc[
            frame["cluster_id"] == cluster_id, "cluster_summary_candidate"
        ] = extractive_summary(evidence)

        if str(subset.iloc[0]["keyword"]).casefold() == "design theory":
            theory = classify_theory_move(
                [
                    {
                        "title": row["title"],
                        "abstract": row["abstract"],
                    }
                    for _, row in subset.iterrows()
                ]
            )
            mask_for_cluster = frame["cluster_id"] == cluster_id
            frame.loc[mask_for_cluster, "theory_move_key"] = theory.key
            frame.loc[mask_for_cluster, "theory_move"] = theory.label
            frame.loc[mask_for_cluster, "theory_move_support"] = theory.support_text
            frame.loc[mask_for_cluster, "theory_move_patterns"] = theory.patterns_text
    return frame, scope_vectors


def write_scope_summary(frame: pd.DataFrame, out: Path, label: str) -> None:
    lines = [
        f"# {label}: Statistical Bottom-Up Cluster Interpretations",
        "",
        "Assignments: BGE-M3 + within-keyword Spectral clustering.",
        "Interpretation: adapted class-based TF-IDF over analysis-frozen memberships.",
        "",
    ]
    for cluster, subset in frame.groupby("cluster", sort=True):
        first = subset.iloc[0]
        lines.extend(
            [
                f"## {first['cluster_label_candidate']} ({len(subset)} papers)",
                "",
                str(first["cluster_summary_candidate"]),
                "",
            ]
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def adapt_dashboard_copy(path: Path) -> None:
    """Relabel legacy dashboard fields without changing its visual structure."""
    page = path.read_text(encoding="utf-8")
    page = page.replace(
        "opt.textContent = `${clusterName(c.cluster)} (${c.count})`;",
        "const readableLabel = String(c.label || clusterName(c.cluster)).trim(); "
        "opt.textContent = `${readableLabel} · ${c.count} ${c.count === 1 ? 'paper' : 'papers'}`;",
    )
    page = page.replace(
        "const clusterName = c => Number(c) === -1 ? 'Unclustered papers' : `Cluster ${c}`;",
        "const clusterName = c => { "
        "if (Number(c) === -1) return 'Unclustered papers'; "
        "const meta = data.clusters.find(item => Number(item.cluster) === Number(c)); "
        "const match = String(meta?.label || '').match(/\\[([^\\]]+)\\]$/); "
        "return match ? match[1] : `Cluster ${c}`; };",
    )
    page = page.replace(
        "const clusterLegendLabel = c => {\n"
        "      const label = String(c.label || c.theme || '').trim();\n"
        "      if (!label) return clusterName(c.cluster);\n"
        "      return /^Cluster\\s+-?\\d+:/i.test(label) ? label : `${clusterName(c.cluster)}: ${label}`;\n"
        "    };",
        "const readableClusterLabel = c => { "
        "const label = String(c.label || c.theme || '').trim(); "
        "const withoutId = label.replace(/\\s*\\[[^\\]]+\\]\\s*$/, '').trim(); "
        "return `${withoutId || clusterName(c.cluster)} · ${c.count} "
        "${c.count === 1 ? 'paper' : 'papers'} [${clusterName(c.cluster)}]`; };\n"
        "    const clusterLegendLabel = c => readableClusterLabel(c);",
    )
    page = page.replace(
        "const readableLabel = String(c.label || clusterName(c.cluster)).trim(); "
        "opt.textContent = `${readableLabel} · ${c.count} ${c.count === 1 ? 'paper' : 'papers'}`;",
        "opt.textContent = readableClusterLabel(c);",
    )
    page = page.replace(
        '<span class="pill">LDA topic ${p.lda_topic} '
        '(${Math.round(p.lda_topic_probability * 100)}%)</span>',
        '<span class="pill">Path 1 interpretation</span>',
    )
    page = page.replace(
        '<div class="section-title">Cluster Descriptor</div>',
        '<div class="section-title">Topic Phrase</div>',
    )
    page = page.replace(
        '<div class="section-title">Distinguishing Evidence</div>',
        '<div class="section-title">Key Phrases</div>',
    )
    page = page.replace(
        '<div class="section-title">Cluster Summary Candidate</div>',
        '<div class="section-title">Prose Summary</div>',
    )
    page = page.replace(
        '<div class="section-title">Research Contribution &amp; Domain</div>',
        '<div class="section-title">Fine-Grained Form &amp; Review Status</div>',
    )
    page = page.replace(
        '<span class="pill">Primary: ${escapeHtml(p.contribution_type || \'n/a\')}</span>',
        '<span class="pill">Status: ${escapeHtml(p.contribution_type || \'n/a\')}</span>',
    )
    page = page.replace(
        '<span class="pill">Primary domain: ${escapeHtml(p.primary_application_domain || p.application_domains || \'n/a\')}</span>',
        '',
    )
    page = page.replace(
        '<div class="meta" style="margin-top:8px">Contribution support: ${escapeHtml(p.contribution_type_support || \'n/a\')}</div>',
        '<div class="meta" style="margin-top:8px">Review status: ${escapeHtml(p.contribution_type_support || \'n/a\')}</div>',
    )
    for line in (
        '<div class="meta">Subtype support: ${escapeHtml(p.contribution_subtype_support || \'n/a\')}</div>',
        '<div class="meta">Contribution patterns: ${escapeHtml(p.contribution_type_patterns || \'n/a\')}</div>',
        '<div class="meta">Domain support: ${escapeHtml(p.application_domain_support || \'n/a\')}</div>',
        '<div class="meta">Domain definition: ${escapeHtml(p.application_domain_definitions || \'n/a\')}</div>',
    ):
        page = page.replace(line, "")
    page = page.replace(
        '<div class="section-title" style="margin-top:14px">Path 1 Theory Move</div>',
        '<div class="section-title" style="margin-top:14px">Theory Move Candidate</div>',
    )
    page = page.replace(
        '<div class="section-title">Paper-Oriented Facets</div>',
        '<div class="section-title">Pipeline Context</div>',
    )
    page = page.replace(
        '<div class="section-title">Secondary Topic-Model Evidence</div>',
        '<div class="section-title">Class-Based TF-IDF Evidence</div>',
    )
    page = "\n".join(line.rstrip() for line in page.splitlines()) + "\n"
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--interpretations", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    inputs = pd.read_csv(args.input).fillna("")
    metadata = pd.read_csv(args.metadata).fillna("")
    for column in metadata.select_dtypes(include=["object", "string"]).columns:
        metadata[column] = metadata[column].map(clean_metadata)
    assignments = pd.read_csv(args.assignments).fillna("")
    interpretations = pd.read_csv(args.interpretations).fillna("")
    vectors = normalize(np.load(args.embeddings), norm="l2")
    if len(inputs) != len(vectors):
        raise ValueError("Input rows and embedding rows do not align.")

    metadata_columns = ["paper_id", "authors", "year", "venue", "doi", "url"]
    base = inputs.merge(
        metadata[metadata_columns].drop_duplicates("paper_id"),
        on="paper_id",
        how="left",
        validate="one_to_one",
    ).merge(
        assignments[["paper_id", "cluster_id"]],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if base["cluster_id"].eq("").any():
        raise ValueError("At least one paper lacks a frozen cluster assignment.")
    keyphrases_by_cluster = build_cluster_keyphrases(base)

    out_root = Path(args.out)
    scopes: list[tuple[str, str | None]] = [("all", None)]
    scopes.extend((slugify(keyword), keyword) for keyword in sorted(base["keyword"].unique()))
    for offset, (slug, keyword) in enumerate(scopes):
        frame, _ = prepare_scope(
            base,
            vectors,
            interpretations,
            keyphrases_by_cluster,
            keyword,
            args.seed + offset,
        )
        scope_out = out_root / slug
        scope_out.mkdir(parents=True, exist_ok=True)
        label = keyword or "All keyword groups"
        write_dashboard(
            frame,
            scope_out / "paper_explorer.html",
            f"{label} · Statistical bottom-up interpretation",
        )
        adapt_dashboard_copy(scope_out / "paper_explorer.html")
        write_scope_summary(frame, scope_out / "cluster_summary.md", label)
        frame.to_csv(scope_out / "clustered_papers.csv", index=False, encoding="utf-8-sig")
        print(f"Wrote {label}: {len(frame)} papers, {frame['cluster'].nunique()} clusters")


if __name__ == "__main__":
    main()
