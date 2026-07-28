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
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

from cluster_papers import write_dashboard


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def clean_metadata(value: object) -> str:
    return fix_encoding(str(value if value is not None else "")).replace(
        "MoÃàller", "Möller"
    )


def summary_sentence(row: pd.Series) -> str:
    terms = [term.strip() for term in str(row["top_terms"]).split("|") if term.strip()]
    representatives = [
        title.strip()
        for title in str(row["representative_titles"]).split("|")
        if title.strip()
    ]
    sentence = (
        f"Within the predefined {row['keyword']} group, {row.name} is "
        f"statistically distinguished by {', '.join(terms[:3])}."
    )
    if len(terms) > 3:
        sentence += f" Additional contrastive terms include {', '.join(terms[3:8])}."
    if representatives:
        sentence += " Representative papers include " + "; ".join(representatives[:2]) + "."
    sentence += (
        " This is an analysis-frozen statistical interpretation, not a final "
        "editorial topic label."
    )
    return sentence


def human_descriptor(value: object) -> str:
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    if not parts:
        return "Topic description pending"
    parts[0] = parts[0][:1].upper() + parts[0][1:]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} & {parts[1]}"
    return f"{', '.join(parts[:-1])} & {parts[-1]}"


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
            f"{interpretation_by_id.loc[cluster_id, 'keyword']} — "
            f"{human_descriptor(interpretation_by_id.loc[cluster_id, 'statistical_descriptor_not_final_label'])} "
            f"[{cluster_id}]"
        )
    )
    frame["distinguishing_evidence_terms"] = frame["cluster_theme_terms"]
    frame["cluster_summary_candidate"] = frame["cluster_id"].map(
        lambda cluster_id: summary_sentence(interpretation_by_id.loc[cluster_id])
    )
    frame["design_knowledge_form"] = frame["keyword"]
    frame["contribution_type"] = "Statistical bottom-up interpretation"
    frame["contribution_type_support"] = frame["cluster_id"].map(
        lambda cluster_id: (
            "Prior-pilot assignment retained"
            if str(interpretation_by_id.loc[cluster_id, "interpretation_status"]) == "frozen"
            else "Analysis frozen; confirmation by at least two reviewers pending"
        )
    )
    frame["contribution_type_definition"] = (
        "Contrastive cluster explanation using adapted class-based TF-IDF; "
        "not a predefined codebook classification."
    )
    frame["primary_application_domain"] = "Not assigned by Path 1"
    frame["application_domain_support"] = "Not evaluated in this statistical interpretation"
    frame["application_domain_definitions"] = "Not applicable"
    frame["facet_population_or_context"] = "Predefined keyword group"
    frame["facet_stakeholder_or_population"] = "Not assigned"
    frame["facet_method_or_lens"] = "BGE-M3 embedding + Spectral clustering"
    frame["facet_artifact_or_domain"] = "Not assigned"
    frame["facet_contribution_or_outcome"] = "Adapted class-based TF-IDF interpretation"
    frame["theory_move_key"] = "not_applicable"
    frame["lda_topic"] = -1
    frame["lda_topic_probability"] = 0.0
    frame["lda_topic_words"] = frame["cluster_theme_terms"]
    frame["hdbscan_peripheral"] = False
    frame["discussion_found"] = False
    frame["discussion_paragraph_count"] = 0
    frame["discussion_summary"] = ""
    frame["discussion_excerpt"] = ""
    return add_geometry(frame, scope_vectors, seed), scope_vectors


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
        '<span class="pill">Statistical bottom-up interpretation</span>',
    )
    page = page.replace(
        '<div class="section-title">Cluster Summary Candidate</div>',
        '<div class="section-title">Statistical Bottom-Up Cluster Summary</div>',
    )
    page = page.replace(
        '<div class="section-title">Research Contribution &amp; Domain</div>',
        '<div class="section-title">Interpretation Method &amp; Status</div>',
    )
    page = page.replace(
        '<div class="section-title">Paper-Oriented Facets</div>',
        '<div class="section-title">Pipeline Context</div>',
    )
    page = page.replace(
        '<div class="section-title">Secondary Topic-Model Evidence</div>',
        '<div class="section-title">Class-Based TF-IDF Evidence</div>',
    )
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

    out_root = Path(args.out)
    scopes: list[tuple[str, str | None]] = [("all", None)]
    scopes.extend((slugify(keyword), keyword) for keyword in sorted(base["keyword"].unique()))
    for offset, (slug, keyword) in enumerate(scopes):
        frame, _ = prepare_scope(base, vectors, interpretations, keyword, args.seed + offset)
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
