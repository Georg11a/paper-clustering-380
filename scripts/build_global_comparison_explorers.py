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
import html
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from build_statistical_umap_explorer import adapt_dashboard_copy
from cluster_papers import write_dashboard


SEED = 20260730
METADATA_PATH = Path("data/final_advancing_list.csv")
FORBIDDEN_CLUSTER_INPUT_COLUMNS = {
    "cluster", "cluster_id", "cluster_label", "focus_keyword", "keyword",
    "keyword_group", "keyword_query", "label", "query", "search_keyword",
}


def load_neutral_papers(path: Path) -> pd.DataFrame:
    """Load paper rows without exposing retrieval labels to clustering.

    A neutral chunk CSV is aggregated in first-occurrence order, which is the
    same order used by Stage 01 when pooling the frozen embedding matrix.
    """
    source = pd.read_csv(path).fillna("")
    leaked = sorted(set(source.columns) & FORBIDDEN_CLUSTER_INPUT_COLUMNS)
    if leaked:
        raise ValueError(f"Forbidden clustering-input columns: {leaked}")
    if {"paper_id", "chunk_index", "chunk"} <= set(source.columns):
        order = source["paper_id"].drop_duplicates().tolist()
        rows = []
        for paper_id in order:
            subset = source[source["paper_id"].eq(paper_id)].copy()
            subset["chunk_index"] = pd.to_numeric(subset["chunk_index"])
            subset = subset.sort_values("chunk_index")
            first = str(subset.iloc[0]["chunk"])
            title, separator, abstract = first.partition("[SEP]")
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": " ".join(title.split()),
                    "abstract": " ".join(abstract.split()) if separator else "",
                    "subdocument": "\n\n".join(subset["chunk"].astype(str)),
                }
            )
        return pd.DataFrame(rows)
    required = {"paper_id", "title", "abstract", "subdocument"}
    if not required <= set(source.columns):
        raise ValueError(f"Missing neutral input columns: {sorted(required - set(source.columns))}")
    return source


def load_discussion_metadata(path: Path | None) -> pd.DataFrame:
    """Load optional, display-only Discussion fields keyed by paper_id.

    These fields are joined only after clustering, so they cannot affect the
    embeddings, UMAP coordinates, configuration selection, or assignments.
    """
    columns = [
        "paper_id",
        "discussion_found",
        "discussion_paragraph_count",
        "discussion_summary",
        "discussion_excerpt",
    ]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)

    source = pd.read_csv(path).fillna("")
    if "paper_id" not in source.columns:
        raise ValueError(f"Discussion metadata has no paper_id column: {path}")
    for column in columns[1:]:
        if column not in source.columns:
            source[column] = ""
    source = source[columns].drop_duplicates("paper_id", keep="last").copy()
    source["discussion_found"] = (
        source["discussion_found"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    source["discussion_paragraph_count"] = (
        pd.to_numeric(source["discussion_paragraph_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return source


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


def valid_partition(labels: np.ndarray) -> bool:
    kept = labels >= 0
    unique = np.unique(labels[kept])
    return int(kept.sum()) >= 3 and 2 <= len(unique) < int(kept.sum())


def fit_selected(config: str, vectors: np.ndarray, metric: str) -> np.ndarray:
    if config.startswith("kmeans_"):
        k = int(re.search(r"k(\d+)$", config).group(1))
        return KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(vectors)
    if config.startswith("dbscan_"):
        match = re.search(r"eps([0-9.]+)_ms(\d+)$", config)
        return DBSCAN(
            eps=float(match.group(1)), min_samples=int(match.group(2)), metric=metric
        ).fit_predict(vectors)
    if config.startswith("hdbscan_"):
        match = re.search(r"mcs(\d+)_ms(\d+)$", config)
        return HDBSCAN(
            min_cluster_size=int(match.group(1)),
            min_samples=int(match.group(2)),
            metric="euclidean",
            copy=True,
        ).fit_predict(vectors)
    raise ValueError(config)


def subsample_stability(
    config: str,
    vectors: np.ndarray,
    metric: str,
    repeats: int = 10,
    fraction: float = 0.8,
) -> float:
    """Mean overlap ARI across repeated 80% subsamples in a fixed space."""
    rng = np.random.default_rng(SEED)
    size = int(round(len(vectors) * fraction))
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(repeats):
        indices = np.sort(rng.choice(len(vectors), size=size, replace=False))
        runs.append((indices, fit_selected(config, vectors[indices], metric)))
    scores = []
    for (left_i, left_y), (right_i, right_y) in combinations(runs, 2):
        _, left_pos, right_pos = np.intersect1d(
            left_i, right_i, return_indices=True
        )
        if valid_partition(left_y[left_pos]) and valid_partition(right_y[right_pos]):
            scores.append(adjusted_rand_score(left_y[left_pos], right_y[right_pos]))
    return float(np.mean(scores)) if scores else float("nan")


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
            config = f"dbscan_eps{eps:.6f}_ms{min_samples}"
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


def zhicheng_hdbscan_sweep(
    original_vectors: np.ndarray,
    cached_reductions: dict[tuple[int, int], np.ndarray],
    shared_labels: np.ndarray,
) -> list[dict[str, object]]:
    """Fixed-seed parameter inspection around the shared UMAP workflow."""
    rows: list[dict[str, object]] = []
    for components in [5, 10]:
        for neighbors in [5, 10, 15, 30]:
            key = (components, neighbors)
            reduced = cached_reductions.get(key)
            if reduced is None:
                reduced = umap.UMAP(
                    n_components=components,
                    n_neighbors=neighbors,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=SEED,
                    n_jobs=1,
                ).fit_transform(original_vectors)
            for min_cluster_size in [5, 8, 10, 12, 15, 20]:
                for min_samples in [1, 3, 5, 8, 10]:
                    if min_samples > min_cluster_size:
                        continue
                    labels = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric="euclidean",
                        copy=True,
                    ).fit_predict(reduced)
                    rows.append(
                        {
                            "space": f"umap{components}",
                            "algorithm": "hdbscan",
                            "config": (
                                f"nn{neighbors}_hdbscan_mcs{min_cluster_size}_ms{min_samples}"
                            ),
                            "n_components": components,
                            "n_neighbors": neighbors,
                            "min_cluster_size": min_cluster_size,
                            "min_samples": min_samples,
                            **partition_metrics(original_vectors, labels),
                            "ari_vs_zhicheng_shared": float(
                                adjusted_rand_score(shared_labels, labels)
                            ),
                            "is_zhicheng_shared": bool(
                                components == 10
                                and neighbors == 15
                                and min_cluster_size == 8
                                and min_samples == 1
                            ),
                        }
                    )
    return rows


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
    discussion_metadata: pd.DataFrame,
) -> pd.DataFrame:
    frame = papers.copy()
    frame["cluster"] = labels.astype(int)
    terms_by_cluster = cluster_terms(frame, labels)

    metadata_path = METADATA_PATH
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path).fillna("")
        metadata_columns = [
            column
            for column in ["paper_id", "keyword", "authors", "year", "venue", "doi", "url"]
            if column in metadata.columns
        ]
        frame = frame.merge(
            metadata[metadata_columns].drop_duplicates("paper_id"),
            on="paper_id",
            how="left",
        )
    for column in ["keyword", "authors", "year", "venue", "doi", "url"]:
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
    if len(discussion_metadata):
        frame = frame.merge(
            discussion_metadata,
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
    for column, default in [
        ("discussion_found", False),
        ("discussion_paragraph_count", 0),
        ("discussion_summary", ""),
        ("discussion_excerpt", ""),
    ]:
        if column not in frame:
            frame[column] = default
        frame[column] = frame[column].fillna(default)
    frame["discussion_found"] = frame["discussion_found"].astype(bool)
    frame["discussion_paragraph_count"] = (
        pd.to_numeric(frame["discussion_paragraph_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
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
        "const readableClusterLabel = c => { const label = clusterTopicLabel(c.label || c.theme); return `${label || clusterName(c.cluster)} · ${c.count} ${c.count === 1 ? 'paper' : 'papers'}`; };",
        "const readableClusterLabel = c => {\n"
        "      const label = clusterTopicLabel(c.label || c.theme);\n"
        "      const name = clusterName(c.cluster);\n"
        "      const displayName = Number(c.cluster) === -1 ? (label || name) : `${name} · ${label || 'Untitled'}`;\n"
        "      return `${displayName} · ${c.count} ${c.count === 1 ? 'paper' : 'papers'}`;\n"
        "    };",
    )
    page = page.replace(
        "      const discussionStatus = p.discussion_found ?\n"
        "        `<span class=\"pill\">Discussion detected: ${p.discussion_paragraph_count} paragraphs</span>` :\n"
        "        `<span class=\"pill\">No explicit Discussion section detected</span>`;",
        "      const discussionStatus = p.discussion_found ?\n"
        "        `<span class=\"pill\">Discussion detected: ${p.discussion_paragraph_count} paragraphs</span>` : '';",
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
        f"- 80% subsample overlap stability (mean ARI): {metrics.get('stability_ari', float('nan')):.4f}",
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
    discussion_metadata: pd.DataFrame,
) -> dict[str, object]:
    out = out_root / relative
    out.mkdir(parents=True, exist_ok=True)
    display = f"{title} · {config}"
    frame = enrich_for_dashboard(
        papers,
        vectors,
        labels,
        coordinates,
        display,
        discussion_metadata,
    )
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
        "discussion_found",
        "discussion_paragraph_count",
        "discussion_summary",
        "discussion_excerpt",
    ]
    frame[public_columns].to_csv(out / "clustered_papers.csv", index=False)
    write_summary(out / "cluster_summary.md", title, config, metrics, frame)
    return {
        "title": title,
        "config": config,
        "path": relative,
        **metrics,
    }


def build_unified_index(
    out_root: Path,
    manifest: list[dict[str, object]],
    zhicheng_rows: list[dict[str, object]],
) -> None:
    views = {str(item["path"]): item for item in manifest}
    matrix_rows = []
    for algorithm in ["kmeans", "dbscan", "hdbscan"]:
        cells = []
        for space in ["raw", "umap5", "umap10"]:
            item = views[f"{space}/{algorithm}"]
            cells.append(
                "<td><b>{clusters}</b> clusters · {noise:.0%} noise<br>"
                "sil {sil:.3f} · ARI {ari:.3f}</td>".format(
                    clusters=item["cluster_count"],
                    noise=item["noise_fraction"],
                    sil=item["silhouette_original_cosine"],
                    ari=item["stability_ari"],
                )
            )
        matrix_rows.append(
            f"<tr><th>{html.escape(algorithm.upper())}</th>{''.join(cells)}</tr>"
        )

    hdb_rows = list(zhicheng_rows)
    hdb_rows.sort(
        key=lambda row: (
            str(row.get("space")),
            -float(row.get("silhouette_original_cosine", -1) or -1),
        )
    )
    hdb_table = "".join(
        "<tr><td>{space}</td><td>{nn}</td><td>{config}</td><td>{clusters}</td><td>{noise:.1%}</td>"
        "<td>{coverage:.1%}</td><td>{sil:.3f}</td><td>{ari:.3f}</td></tr>".format(
            space=html.escape(str(row["space"])),
            nn=row["n_neighbors"],
            config=html.escape(str(row["config"])),
            clusters=row["cluster_count"],
            noise=float(row["noise_fraction"]),
            coverage=float(row["coverage"]),
            sil=float(row["silhouette_original_cosine"]),
            ari=float(row["ari_vs_zhicheng_shared"]),
        )
        for row in hdb_rows
    )

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Global clustering comparison · 282 papers</title>
<style>
:root{--ink:#17233d;--muted:#657089;--line:#dbe2ee;--blue:#356ae6;--bg:#f5f7fb}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 Inter,system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:24px 28px 16px;background:white;border-bottom:1px solid var(--line)}h1{margin:0 0 6px;font-size:24px}p{margin:5px 0;color:var(--muted)}
.tabs{display:flex;gap:8px;margin-top:18px}.tab{border:1px solid var(--line);background:#fff;padding:10px 14px;border-radius:10px;cursor:pointer}.tab.active{color:white;background:var(--blue);border-color:var(--blue)}
main{padding:18px 28px}.panel{display:none}.panel.active{display:block}.toolbar{display:flex;gap:12px;align-items:center;margin-bottom:12px}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px}select{padding:9px;border:1px solid var(--line);border-radius:8px}iframe{width:100%;height:720px;border:1px solid var(--line);border-radius:14px;background:white}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.scroll{max-height:420px;overflow:auto}
.note{border-left:4px solid #e4a72b;padding-left:12px}.tag{display:inline-block;border-radius:99px;background:#eaf0ff;padding:3px 8px;color:#315cc5}
</style></head><body>
<header><h1>Global clustering comparison</h1><p>282 papers · neutral R_cent text · frozen BGE-M3 1024D embeddings</p>
<p>Retrieval keywords are joined back only for diagnosis; they do not participate in embedding, reduction, or assignment.</p>
<div class="tabs"><button class="tab active" data-panel="raw">1 · All papers together</button><button class="tab" data-panel="umap">2 · UMAP before clustering</button><button class="tab" data-panel="zhicheng">3 · UMAP + HDBSCAN</button></div></header>
<main>
<section id="raw" class="panel active"><div class="toolbar"><label>Algorithm <select id="rawAlgo"><option value="kmeans">K-Means</option><option value="dbscan">DBSCAN</option><option value="hdbscan">HDBSCAN</option></select></label></div><iframe id="rawFrame" src="raw/kmeans/paper_explorer.html"></iframe></section>
<section id="umap" class="panel"><div class="card"><h2>Raw vs UMAP comparison</h2><table><thead><tr><th>Algorithm</th><th>Raw 1024D</th><th>UMAP 5D</th><th>UMAP 10D</th></tr></thead><tbody>__MATRIX__</tbody></table><p>Silhouette is always measured in the original 1024D cosine space. ARI is overlap agreement across repeated 80% subsamples in the stated clustering space.</p></div><div class="toolbar"><label>Space <select id="umapSpace"><option value="umap5">UMAP 5D</option><option value="umap10">UMAP 10D</option></select></label><label>Algorithm <select id="umapAlgo"><option value="kmeans">K-Means</option><option value="dbscan">DBSCAN</option><option value="hdbscan">HDBSCAN</option></select></label></div><iframe id="umapFrame" src="umap5/kmeans/paper_explorer.html"></iframe></section>
<section id="zhicheng" class="panel"><div class="card note"><b>Zhicheng shared configuration</b><p>UMAP: n_components=10, n_neighbors=15, min_dist=0, cosine. HDBSCAN: Euclidean distance in reduced space. Noise remains −1 and is interpreted as weak affinity, never reassigned.</p></div><iframe src="zhicheng_umap_hdbscan/paper_explorer.html"></iframe><div class="card"><h2>HDBSCAN parameter inspection</h2><p>ARI compares each assignment with the marked shared configuration; it measures sensitivity, not accuracy.</p><div class="scroll"><table><thead><tr><th>Space</th><th>n_neighbors</th><th>Configuration</th><th>Clusters</th><th>Noise</th><th>Coverage</th><th>Raw-space silhouette</th><th>ARI vs shared</th></tr></thead><tbody>__HDB__</tbody></table></div></div></section>
</main><script>
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.panel).classList.add('active')}));
rawAlgo.addEventListener('change',()=>rawFrame.src=`raw/${rawAlgo.value}/paper_explorer.html`);
function updateUmap(){umapFrame.src=`${umapSpace.value}/${umapAlgo.value}/paper_explorer.html`}umapSpace.addEventListener('change',updateUmap);umapAlgo.addEventListener('change',updateUmap);
</script></body></html>"""
    page = page.replace("__MATRIX__", "".join(matrix_rows)).replace("__HDB__", hdb_table)
    (out_root / "index.html").write_text(page, encoding="utf-8")


def replace_standalone_entry_with_main_ui(out_root: Path) -> None:
    """Keep one project-wide UI; result folders contain data views, not a new shell."""
    (out_root / "index.html").write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=../../index.html">
<title>Opening the main explorer</title></head><body>
<p>These results use the existing explorer interface. <a href="../../index.html">Open the main explorer</a>.</p>
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-paper-count", type=int, default=282)
    parser.add_argument(
        "--metadata",
        default="data/final_advancing_list.csv",
        help="Display-only paper metadata joined after clustering by paper_id.",
    )
    parser.add_argument(
        "--discussion-metadata",
        default="data/fulltext_context_confirmed_284_only.csv",
        help=(
            "Optional post-clustering Discussion metadata CSV. Pass an empty "
            "string to omit Discussion cards."
        ),
    )
    args = parser.parse_args()

    global METADATA_PATH
    METADATA_PATH = Path(args.metadata)

    papers = load_neutral_papers(Path(args.input)).fillna("")
    discussion_metadata = load_discussion_metadata(
        Path(args.discussion_metadata) if args.discussion_metadata else None
    )
    vectors = normalize(np.load(args.embeddings), norm="l2")
    if (
        len(papers) != args.expected_paper_count
        or papers["paper_id"].nunique() != args.expected_paper_count
        or len(vectors) != args.expected_paper_count
    ):
        raise ValueError(
            f"Expected {args.expected_paper_count} unique papers and aligned embeddings."
        )

    layout = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.08,
        metric="cosine",
        random_state=SEED,
        n_jobs=1,
    ).fit_transform(vectors)
    reduced_spaces = {
        f"umap{components}": umap.UMAP(
            n_components=components,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=SEED,
            n_jobs=1,
        ).fit_transform(vectors)
        for components in [5, 10]
    }

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
        ("umap5", reduced_spaces["umap5"], "euclidean"),
        ("umap10", reduced_spaces["umap10"], "euclidean"),
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
            metrics["stability_ari"] = subsample_stability(
                config, clustering_vectors, metric
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
                        else f"{space.upper()} before clustering · {algorithm.upper()}"
                    ),
                    config,
                    metrics,
                    labels,
                    papers,
                    vectors,
                    layout,
                    discussion_metadata,
                )
            )

    # Page 3 is the fixed configuration shared by Zhicheng, not the
    # automatically selected Page 2 HDBSCAN candidate. These happened to be
    # identical for the 282-paper run (mcs=8, min_samples=1), which previously
    # masked the distinction.
    zh_config = "hdbscan_mcs8_ms1"
    zh_labels = fit_selected(zh_config, reduced_spaces["umap10"], "euclidean")
    zh_metrics = partition_metrics(vectors, zh_labels)
    zh_metrics["stability_ari"] = subsample_stability(
        zh_config, reduced_spaces["umap10"], "euclidean"
    )
    zh_candidates = [
        row for row in metric_rows
        if row.get("space") == "umap10" and row.get("algorithm") == "hdbscan"
    ]
    for row in zh_candidates:
        metric_rows.append({"space": "zhicheng_umap10", **{k: v for k, v in row.items() if k != "space"}})
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
            discussion_metadata,
        )
    )
    zhicheng_sweep = zhicheng_hdbscan_sweep(
        vectors,
        {
            (5, 15): reduced_spaces["umap5"],
            (10, 15): reduced_spaces["umap10"],
        },
        zh_labels,
    )
    pd.DataFrame(zhicheng_sweep).to_csv(
        out_root / "zhicheng_hdbscan_parameter_sweep.csv", index=False
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
    replace_standalone_entry_with_main_ui(out_root)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "paper_count": len(papers),
                "input": args.input,
                "embeddings": args.embeddings,
                "umap_clustering": {
                    "comparison_n_components": [5, 10],
                    "zhicheng_n_components": 10,
                    "n_neighbors": 15,
                    "min_dist": 0.0,
                    "metric": "cosine",
                    "seed": SEED,
                },
                "noise_policy": "retain label -1; never nearest-cluster reassignment",
                "ground_truth_required": False,
                "selection_note": (
                    "Selected configurations are representative views, not ground-truth winners. "
                    "Comparison uses internal geometry, coverage, noise, and subsample stability."
                ),
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
