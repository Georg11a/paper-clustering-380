#!/usr/bin/env python3
"""Build the three meeting-requested global clustering comparison views.

The views hold the cleaned 282-paper corpus and frozen BGE-M3 representation
constant while comparing:

1. raw-space K-Means, DBSCAN, and HDBSCAN;
2. UMAP-before-clustering K-Means, DBSCAN, and HDBSCAN; and
3. the UMAP + HDBSCAN workflow shared by Zhicheng.

The generated explorers reuse the project's existing dashboard UI. Density
noise remains label -1 and is never reassigned to a nearby cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from build_statistical_umap_explorer import adapt_dashboard_copy
from cluster_papers import write_dashboard


SEED = 20260730


def partition_metrics(
    original_vectors: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int]:
    kept = labels >= 0
    kept_labels = labels[kept]
    counts = (
        np.unique(kept_labels, return_counts=True)[1]
        if len(kept_labels)
        else np.asarray([], dtype=int)
    )
    valid = len(kept_labels) >= 3 and 2 <= len(counts) < len(kept_labels)
    return {
        "cluster_count": int(len(counts)),
        "noise_count": int(np.sum(~kept)),
        "noise_fraction": float(np.mean(~kept)),
        "coverage": float(np.mean(kept)),
        "smallest_cluster": int(counts.min()) if len(counts) else 0,
        "largest_cluster": int(counts.max()) if len(counts) else 0,
        "silhouette_original_cosine": (
            float(
                silhouette_score(
                    original_vectors[kept],
                    kept_labels,
                    metric="cosine",
                )
            )
            if valid
            else float("nan")
        ),
    }


def candidate_score(metrics: dict[str, float | int]) -> tuple[int, float, float]:
    cluster_count = int(metrics["cluster_count"])
    smallest = int(metrics["smallest_cluster"])
    structurally_usable = int(2 <= cluster_count <= 15 and smallest >= 5)
    silhouette = float(metrics["silhouette_original_cosine"])
    if not np.isfinite(silhouette):
        silhouette = -1.0
    coverage = float(metrics["coverage"])
    return structurally_usable, silhouette + 0.15 * coverage, coverage


def choose_kmeans(
    clustering_vectors: np.ndarray,
    original_vectors: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, float | int], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    candidates = []
    for k in [2, 3, 5, 6, 8, 10, 12]:
        labels = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(
            clustering_vectors
        )
        metrics = partition_metrics(original_vectors, labels)
        row = {"config": f"kmeans_k{k}", "algorithm": "kmeans", **metrics}
        rows.append(row)
        candidates.append((candidate_score(metrics), row["config"], labels, metrics))
    _, config, labels, metrics = max(candidates, key=lambda item: item[0])
    return str(config), labels, metrics, rows


def dbscan_eps_candidates(
    vectors: np.ndarray,
    min_samples: int,
    metric: str,
) -> np.ndarray:
    neighbors = NearestNeighbors(
        n_neighbors=min(min_samples, len(vectors)),
        metric=metric,
    ).fit(vectors)
    distances = neighbors.kneighbors(vectors, return_distance=True)[0][:, -1]
    quantiles = np.quantile(distances, np.linspace(0.25, 0.92, 28))
    return np.unique(np.round(quantiles, 6))


def choose_dbscan(
    clustering_vectors: np.ndarray,
    original_vectors: np.ndarray,
    metric: str,
) -> tuple[str, np.ndarray, dict[str, float | int], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    candidates = []
    for min_samples in [3, 5, 8, 10]:
        for eps in dbscan_eps_candidates(clustering_vectors, min_samples, metric):
            labels = DBSCAN(
                eps=float(eps),
                min_samples=min_samples,
                metric=metric,
            ).fit_predict(clustering_vectors)
            metrics = partition_metrics(original_vectors, labels)
            config = f"dbscan_eps{eps:.4f}_ms{min_samples}"
            row = {
                "config": config,
                "algorithm": "dbscan",
                "eps": float(eps),
                "min_samples": min_samples,
                **metrics,
            }
            rows.append(row)
            candidates.append((candidate_score(metrics), config, labels, metrics))
    _, config, labels, metrics = max(candidates, key=lambda item: item[0])
    return str(config), labels, metrics, rows


def choose_hdbscan(
    clustering_vectors: np.ndarray,
    original_vectors: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, float | int], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    candidates = []
    for min_cluster_size in [5, 8, 10, 12, 15, 20]:
        for min_samples in [1, 3, 5, 8, 10]:
            if min_samples > min_cluster_size:
                continue
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                copy=True,
            ).fit_predict(clustering_vectors)
            metrics = partition_metrics(original_vectors, labels)
            config = f"hdbscan_mcs{min_cluster_size}_ms{min_samples}"
            row = {
                "config": config,
                "algorithm": "hdbscan",
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                **metrics,
            }
            rows.append(row)
            candidates.append((candidate_score(metrics), config, labels, metrics))
    _, config, labels, metrics = max(candidates, key=lambda item: item[0])
    return str(config), labels, metrics, rows


def cluster_terms(frame: pd.DataFrame, labels: np.ndarray) -> dict[int, list[str]]:
    text = (
        frame["title"].fillna("")
        + " "
        + frame["abstract"].fillna("")
        + " "
        + frame["subdocument"].fillna("")
    )
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=12_000,
    )
    matrix = vectorizer.fit_transform(text)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    output: dict[int, list[str]] = {}
    for label in sorted(set(int(value) for value in labels)):
        indices = np.flatnonzero(labels == label)
        weights = np.asarray(matrix[indices].mean(axis=0)).ravel()
        output[label] = vocabulary[np.argsort(-weights)[:12]].tolist()
    return output


def enrich_for_dashboard(
    papers: pd.DataFrame,
    original_vectors: np.ndarray,
    labels: np.ndarray,
    coordinates: np.ndarray,
    configuration_label: str,
) -> pd.DataFrame:
    frame = papers.copy()
    frame["cluster"] = labels.astype(int)
    terms_by_cluster = cluster_terms(frame, labels)

    metadata_path = Path("data/final_advancing_list.csv")
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path).fillna("")
        metadata_columns = [
            column
            for column in ["paper_id", "authors", "year", "venue", "doi", "url"]
            if column in metadata.columns
        ]
        frame = frame.merge(
            metadata[metadata_columns].drop_duplicates("paper_id"),
            on="paper_id",
            how="left",
        )
    for column in ["authors", "year", "venue", "doi", "url"]:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("")

    distances = pairwise_distances(original_vectors, metric="cosine")
    nearest_records: list[list[dict[str, object]]] = []
    for index in range(len(frame)):
        order = np.argsort(distances[index])
        nearest_records.append(
            [
                {
                    "paper_id": str(frame.iloc[neighbor]["paper_id"]),
                    "title": str(frame.iloc[neighbor]["title"]),
                    "cluster": int(labels[neighbor]),
                    "distance": float(distances[index, neighbor]),
                }
                for neighbor in order
                if neighbor != index
            ][:5]
        )
    frame["nearest_papers"] = nearest_records

    frame["distance_to_centroid"] = 0.0
    frame["representative_rank"] = -1
    frame["medoid_rank"] = -1
    frame["is_representative_top3"] = False
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        centroid = normalize(
            original_vectors[indices].mean(axis=0, keepdims=True),
            norm="l2",
        )[0]
        cluster_distances = 1.0 - original_vectors[indices] @ centroid
        order = np.argsort(cluster_distances)
        for rank, position in enumerate(order, start=1):
            row_index = indices[position]
            frame.at[row_index, "distance_to_centroid"] = float(
                cluster_distances[position]
            )
            frame.at[row_index, "representative_rank"] = rank
            frame.at[row_index, "medoid_rank"] = rank
            frame.at[row_index, "is_representative_top3"] = rank <= 3

    frame["umap_x"] = coordinates[:, 0]
    frame["umap_y"] = coordinates[:, 1]
    frame["discussion_found"] = False
    frame["discussion_paragraph_count"] = 0
    frame["discussion_summary"] = ""
    frame["discussion_excerpt"] = ""
    frame["hdbscan_peripheral"] = False
    frame["lda_topic"] = -1
    frame["lda_topic_probability"] = 0.0
    frame["contribution_type"] = "Exploratory global comparison"
    frame["contribution_type_support"] = (
        "Candidate assignment; preserve density noise for review."
    )
    frame["contribution_type_definition"] = (
        "All 282 papers clustered together using frozen BGE-M3 embeddings."
    )
    frame["design_knowledge_form"] = configuration_label
    frame["theory_move_key"] = "not_applicable"

    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        terms = terms_by_cluster[int(label)]
        representatives = (
            frame.iloc[indices]
            .sort_values("representative_rank")
            .head(3)["title"]
            .tolist()
        )
        if int(label) == -1:
            topic = "Noise / weak affinity"
            summary = (
                f"{len(indices)} papers were left unassigned by the density "
                "model. They are retained as evidence of weak affinity rather "
                "than treated as an algorithm failure."
            )
        else:
            topic = " · ".join(term.title() for term in terms[:3])
            summary = (
                f"This exploratory cluster contains {len(indices)} papers. "
                f"Frequent terms include {', '.join(terms[:6])}. "
                f"Representative papers include {'; '.join(representatives)}."
            )
        mask = frame["cluster"].eq(int(label))
        frame.loc[mask, "cluster_theme_terms"] = " | ".join(terms)
        frame.loc[mask, "cluster_label_candidate"] = topic
        frame.loc[mask, "distinguishing_evidence_terms"] = ", ".join(terms[:6])
        frame.loc[mask, "cluster_summary_candidate"] = summary
        frame.loc[mask, "lda_topic_words"] = " | ".join(terms)

    return frame


def adapt_global_dashboard(path: Path) -> None:
    adapt_dashboard_copy(path)
    page = path.read_text(encoding="utf-8")
    page = page.replace("Unclustered papers", "Noise / weak affinity")
    page = page.replace(
        "const colorFor = c => palette[Math.abs(Number(c)) % palette.length];",
        "const colorFor = c => Number(c) === -1 ? '#9aa6b2' : "
        "palette[Math.abs(Number(c)) % palette.length];",
    )
    page = page.replace(
        "      const shapeNote = document.createElement('span');\n"
        "      shapeNote.className = 'legend-item';\n"
        "      shapeNote.innerHTML = '<span>◆ Discussion detected</span><span>● No explicit discussion</span>';\n"
        "      legend.appendChild(shapeNote);\n",
        "",
    )
    page = page.replace(
        "      const discussionStatus = p.discussion_found ?\n"
        "        `<span class=\"pill\">Discussion detected: ${p.discussion_paragraph_count} paragraphs</span>` :\n"
        "        `<span class=\"pill\">No explicit Discussion section detected</span>`;",
        "      const discussionStatus = '';",
    )
    page = page.replace(
        "      status.textContent = `${shown.length} of ${papers.length} papers shown; "
        "${data.clusters.length} clusters in this view. Axes are UMAP 1 and UMAP 2 "
        "coordinates, not interpretable variables.`;",
        "      const nonNoiseClusters = data.clusters.filter(c => Number(c.cluster) >= 0).length;\n"
        "      const noiseCount = papers.filter(p => Number(p.cluster) === -1).length;\n"
        "      status.textContent = `${shown.length} of ${papers.length} papers shown; "
        "${nonNoiseClusters} clusters; ${noiseCount} noise / weak-affinity papers. "
        "Axes are UMAP 1 and UMAP 2 coordinates, not interpretable variables.`;",
    )
    page = page.replace("Path 1 interpretation", "Global comparison")
    page = page.replace(
        "Fine-Grained Form &amp; Review Status",
        "Configuration &amp; Review Status",
    )
    page = page.replace(
        "Assignments: BGE-M3 + within-keyword Spectral clustering.",
        "Assignments: all papers combined; density noise retained as -1.",
    )
    path.write_text(page, encoding="utf-8")


def write_summary(
    path: Path,
    title: str,
    config: str,
    metrics: dict[str, float | int],
    frame: pd.DataFrame,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Selected configuration: `{config}`",
        f"- Papers: {len(frame)}",
        f"- Non-noise clusters: {metrics['cluster_count']}",
        f"- Noise papers: {metrics['noise_count']} ({metrics['noise_fraction']:.1%})",
        f"- Coverage: {metrics['coverage']:.1%}",
        f"- Original-space cosine silhouette: {metrics['silhouette_original_cosine']:.4f}",
        "",
        "Noise is retained as weak-affinity information and is not reassigned.",
        "",
    ]
    for cluster, subset in frame.groupby("cluster", sort=True):
        label = subset.iloc[0]["cluster_label_candidate"]
        lines.extend(
            [
                f"## {label} ({len(subset)} papers)",
                "",
                str(subset.iloc[0]["cluster_summary_candidate"]),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_view(
    out_root: Path,
    relative: str,
    title: str,
    config: str,
    metrics: dict[str, float | int],
    labels: np.ndarray,
    papers: pd.DataFrame,
    vectors: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, object]:
    out = out_root / relative
    out.mkdir(parents=True, exist_ok=True)
    display = f"{title} · {config}"
    frame = enrich_for_dashboard(papers, vectors, labels, coordinates, display)
    write_dashboard(frame, out / "paper_explorer.html", display)
    adapt_global_dashboard(out / "paper_explorer.html")
    public_columns = [
        "paper_id",
        "keyword",
        "title",
        "authors",
        "year",
        "venue",
        "doi",
        "url",
        "cluster",
        "cluster_label_candidate",
        "distinguishing_evidence_terms",
        "cluster_summary_candidate",
        "distance_to_centroid",
        "representative_rank",
        "is_representative_top3",
        "umap_x",
        "umap_y",
    ]
    frame[public_columns].to_csv(out / "clustered_papers.csv", index=False)
    write_summary(out / "cluster_summary.md", title, config, metrics, frame)
    return {
        "title": title,
        "config": config,
        "path": relative,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-paper-count", type=int, default=282)
    args = parser.parse_args()

    papers = pd.read_csv(args.input).fillna("")
    vectors = normalize(np.load(args.embeddings), norm="l2")
    if (
        len(papers) != args.expected_paper_count
        or papers["paper_id"].nunique() != args.expected_paper_count
        or len(vectors) != args.expected_paper_count
    ):
        raise ValueError("Expected the cleaned 282-paper corpus and aligned embeddings.")

    layout = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.08,
        metric="cosine",
        random_state=SEED,
        n_jobs=1,
    ).fit_transform(vectors)
    reduced = umap.UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
        n_jobs=1,
    ).fit_transform(vectors)
    zhicheng_reduced = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
        n_jobs=1,
    ).fit_transform(vectors)

    out_root = Path(args.out)
    metric_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []

    selectors = {
        "kmeans": choose_kmeans,
        "dbscan": choose_dbscan,
        "hdbscan": choose_hdbscan,
    }
    selected: dict[tuple[str, str], tuple[str, np.ndarray, dict[str, float | int]]] = {}

    for space, clustering_vectors, metric in [
        ("raw", vectors, "cosine"),
        ("umap", reduced, "euclidean"),
    ]:
        for algorithm, selector in selectors.items():
            if algorithm == "dbscan":
                config, labels, metrics, candidates = selector(
                    clustering_vectors,
                    vectors,
                    metric,
                )
            else:
                config, labels, metrics, candidates = selector(
                    clustering_vectors,
                    vectors,
                )
            for row in candidates:
                metric_rows.append({"space": space, **row})
            selected[(space, algorithm)] = (config, labels, metrics)
            manifest.append(
                export_view(
                    out_root,
                    f"{space}/{algorithm}",
                    (
                        f"All papers · {algorithm.upper()} · Raw BGE-M3"
                        if space == "raw"
                        else f"UMAP before clustering · {algorithm.upper()}"
                    ),
                    config,
                    metrics,
                    labels,
                    papers,
                    vectors,
                    layout,
                )
            )

    zh_config, zh_labels, zh_metrics, zh_candidates = choose_hdbscan(
        zhicheng_reduced,
        vectors,
    )
    for row in zh_candidates:
        metric_rows.append({"space": "zhicheng_umap10", **row})
    manifest.append(
        export_view(
            out_root,
            "zhicheng_umap_hdbscan",
            "Zhicheng workflow · UMAP 10D + HDBSCAN",
            zh_config,
            zh_metrics,
            zh_labels,
            papers,
            vectors,
            layout,
        )
    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame["selected"] = False
    for (space, algorithm), (config, _, _) in selected.items():
        mask = (
            metrics_frame["space"].eq(space)
            & metrics_frame["algorithm"].eq(algorithm)
            & metrics_frame["config"].eq(config)
        )
        metrics_frame.loc[mask, "selected"] = True
    zhicheng_mask = (
        metrics_frame["space"].eq("zhicheng_umap10")
        & metrics_frame["algorithm"].eq("hdbscan")
        & metrics_frame["config"].eq(zh_config)
    )
    metrics_frame.loc[zhicheng_mask, "selected"] = True
    metrics_frame.to_csv(out_root / "configuration_metrics.csv", index=False)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "paper_count": len(papers),
                "input": args.input,
                "embeddings": args.embeddings,
                "umap_clustering": {
                    "comparison_n_components": 5,
                    "zhicheng_n_components": 10,
                    "n_neighbors": 15,
                    "min_dist": 0.0,
                    "metric": "cosine",
                    "seed": SEED,
                },
                "noise_policy": "retain label -1; never nearest-cluster reassignment",
                "views": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(manifest).to_string(index=False))
    print(f"Wrote global comparison explorers to {out_root}")


if __name__ == "__main__":
    main()
