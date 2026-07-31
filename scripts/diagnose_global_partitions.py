#!/usr/bin/env python3
"""Diagnose the seven saved global partitions without rerunning clustering."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import adjusted_rand_score


def aligned_partition(frame: pd.DataFrame, paper_ids: pd.Series, source: Path) -> np.ndarray:
    required = {"paper_id", "cluster"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if frame["paper_id"].duplicated().any():
        raise ValueError(f"{source} contains duplicate paper_id values")
    indexed = frame.set_index("paper_id")
    absent = set(paper_ids) - set(indexed.index)
    extra = set(indexed.index) - set(paper_ids)
    if absent or extra:
        raise ValueError(
            f"{source} paper IDs do not align; missing={len(absent)}, extra={len(extra)}"
        )
    return indexed.loc[paper_ids, "cluster"].astype(int).to_numpy()


def load_partitions(
    manifest_path: Path,
    paper_ids: pd.Series,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Load labels from each manifest view's clustered_papers.csv."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    parts: dict[str, np.ndarray] = {}
    public_meta: pd.DataFrame | None = None

    for view in manifest.get("views", []):
        name = str(view.get("title") or view.get("config") or view.get("path"))
        relative = view.get("path")
        if not relative:
            raise ValueError(f"Manifest view has no path: {view}")
        source = root / str(relative) / "clustered_papers.csv"
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        parts[name] = aligned_partition(frame, paper_ids, source)
        if public_meta is None:
            keep = [c for c in ("paper_id", "authors", "year", "venue", "doi") if c in frame]
            public_meta = frame[keep].drop_duplicates("paper_id")

    if len(parts) != 7:
        raise ValueError(f"Expected 7 manifest views, found {len(parts)}")
    assert public_meta is not None
    return parts, public_meta, manifest


def canonical_partition_signature(labels: np.ndarray) -> str:
    """Hash a partition independent of arbitrary cluster label numbers."""
    same = labels[:, None] == labels[None, :]
    return hashlib.sha256(np.packbits(same).tobytes()).hexdigest()


def partition_ari(parts: dict[str, np.ndarray], out: Path) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(parts, 2):
        la, lb = parts[a], parts[b]
        mask = (la != -1) & (lb != -1)
        shared = adjusted_rand_score(la[mask], lb[mask]) if mask.sum() > 1 else np.nan
        rows.append(
            {
                "config_a": a,
                "config_b": b,
                "ari_full": adjusted_rand_score(la, lb),
                "ari_shared_nonnoise": shared,
                "n_shared": int(mask.sum()),
            }
        )
    result = pd.DataFrame(rows).sort_values("ari_full", ascending=False)
    result.to_csv(out / "partition_ari.csv", index=False, float_format="%.6f")
    names = list(parts)
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for row in rows:
        matrix.loc[row["config_a"], row["config_b"]] = row["ari_full"]
        matrix.loc[row["config_b"], row["config_a"]] = row["ari_full"]
    matrix.to_csv(out / "partition_ari_matrix.csv", float_format="%.6f")

    groups: dict[str, list[str]] = {}
    for name, labels in parts.items():
        groups.setdefault(canonical_partition_signature(labels), []).append(name)
    exact = [names for names in groups.values() if len(names) > 1]
    (out / "exact_partition_groups.json").write_text(
        json.dumps(exact, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Pairwise partition ARI ===")
    print(result.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if exact:
        print("\n=== Exact partitions (label-permutation invariant) ===")
        for names in exact:
            print(f"{len(names)} identical views:")
            for name in names:
                print(f"  - {name}")
    return result


def keyword_ari(
    parts: dict[str, np.ndarray], meta: pd.DataFrame, out: Path
) -> pd.DataFrame:
    keyword = meta["keyword"].astype("category").cat.codes.to_numpy()
    rows = []
    for name, labels in parts.items():
        mask = labels != -1
        rows.append(
            {
                "config": name,
                "ari_vs_keyword_full": adjusted_rand_score(keyword, labels),
                "ari_vs_keyword_nonnoise": (
                    adjusted_rand_score(keyword[mask], labels[mask])
                    if mask.sum() > 1
                    else np.nan
                ),
                "coverage": mask.mean(),
            }
        )
    result = pd.DataFrame(rows).sort_values("ari_vs_keyword_full", ascending=False)
    result.to_csv(out / "keyword_ari.csv", index=False, float_format="%.6f")
    print("\n=== ARI versus original keyword groups ===")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return result


def dominant_partition(parts: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    groups: dict[str, list[str]] = {}
    labels_by_signature: dict[str, np.ndarray] = {}
    for name, labels in parts.items():
        nonnoise = labels[labels != -1]
        if len(np.unique(nonnoise)) != 2:
            continue
        signature = canonical_partition_signature(labels)
        groups.setdefault(signature, []).append(name)
        labels_by_signature[signature] = labels
    if not groups:
        raise ValueError("No two-cluster partition found")
    signature, names = max(groups.items(), key=lambda item: len(item[1]))
    return names, labels_by_signature[signature]


def numeric_summary(frame: pd.DataFrame, column: str) -> str:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    working = frame[["dominant_group"]].copy()
    working[column] = numeric
    grouped = working.groupby("dominant_group", observed=True)[column]
    summary = grouped.agg(["count", "mean", "median", "std", "min", "max"])
    lines = [summary.to_string(float_format=lambda x: f"{x:.3f}")]
    overall_std = numeric.std()
    if overall_std and np.isfinite(overall_std):
        means = grouped.mean()
        effect = (means.get("small", np.nan) - means.get("large", np.nan)) / overall_std
        lines.append(f"standardized mean difference (small-large)/overall_sd = {effect:.3f}")
    if numeric.nunique(dropna=False) <= 20:
        lines.append("counts:\n" + pd.crosstab(numeric, frame["dominant_group"]).to_string())
    return "\n".join(lines)


def categorical_summary(frame: pd.DataFrame, column: str) -> str:
    counts = pd.crosstab(frame[column].fillna("<missing>"), frame["dominant_group"])
    within = pd.crosstab(
        frame[column].fillna("<missing>"), frame["dominant_group"], normalize="columns"
    )
    note = ""
    if len(counts) > 40:
        keep = counts.sum(axis=1).sort_values(ascending=False).head(30).index
        counts = counts.loc[keep]
        within = within.loc[keep]
        note = f"Top 30 of {frame[column].nunique(dropna=False)} values by frequency shown.\n"
    return note + "counts:\n" + counts.to_string() + "\n\nwithin-group proportions:\n" + within.round(3).to_string()


def dominant_split_crosstab(
    parts: dict[str, np.ndarray], meta: pd.DataFrame, emb: np.ndarray, out: Path
) -> pd.DataFrame:
    configs, labels = dominant_partition(parts)
    counts = pd.Series(labels[labels != -1]).value_counts()
    small_label = int(counts.idxmin())
    group = np.where(labels == small_label, "small", np.where(labels == -1, "noise", "large"))

    audit = meta.copy()
    audit["dominant_group"] = pd.Categorical(group, categories=["small", "large", "noise"])
    audit["title_length"] = audit["title"].fillna("").astype(str).str.len()
    audit["abstract_length"] = audit["abstract"].fillna("").astype(str).str.len()
    audit["subdocument_length"] = audit["subdocument"].fillna("").astype(str).str.len()
    audit["combined_input_length"] = (
        audit["title_length"] + audit["abstract_length"] + audit["subdocument_length"]
    )
    audit["has_selected_passages"] = audit["selected_count"].fillna(0).astype(float) > 0
    audit["has_canonical_text_path"] = audit["canonical_text_path"].fillna("").ne("")
    audit["embedding_l2_norm"] = np.linalg.norm(emb, axis=1)

    lines = [
        "DOMINANT PARTITION DIAGNOSTIC",
        "================================",
        f"Exact partition appears in {len(configs)} views:",
        *[f"  - {name}" for name in configs],
        "",
        f"Cluster sizes: {sorted(counts.tolist())}",
        f"Small-cluster label in reference view: {small_label}",
        "",
    ]

    numeric = [
        "selected_count",
        "budget",
        "year",
        "title_length",
        "abstract_length",
        "subdocument_length",
        "combined_input_length",
        "embedding_l2_norm",
    ]
    categorical = [
        "keyword",
        "venue",
        "has_selected_passages",
        "has_canonical_text_path",
    ]
    for column in numeric:
        if column in audit and audit[column].notna().any():
            lines.extend([f"--- {column} ---", numeric_summary(audit, column), ""])
    for column in categorical:
        if column in audit and audit[column].notna().any():
            lines.extend([f"--- {column} ---", categorical_summary(audit, column), ""])

    is_patterns = audit["keyword"].eq("Design Patterns")
    is_small = audit["dominant_group"].eq("small")
    contingency = np.asarray(
        [
            [int((is_patterns & is_small).sum()), int((~is_patterns & is_small).sum())],
            [int((is_patterns & ~is_small).sum()), int((~is_patterns & ~is_small).sum())],
        ]
    )
    fisher = fisher_exact(contingency)
    lines.extend(
        [
            "--- Design Patterns enrichment ---",
            f"2x2 table [[patterns_small, other_small], [patterns_large, other_large]] = {contingency.tolist()}",
            f"odds ratio = {fisher.statistic:.3f}",
            f"Fisher exact p = {fisher.pvalue:.6g}",
            "Interpretation: strong association, not proof that keyword conditioning caused the split.",
            "",
        ]
    )

    members = audit[audit["dominant_group"] == "small"].copy()
    member_columns = [
        "paper_id",
        "keyword",
        "title",
        "year",
        "venue",
        "selected_count",
        "abstract_length",
        "subdocument_length",
        "combined_input_length",
    ]
    members[member_columns].to_csv(out / "cluster55_members.csv", index=False)
    raw_text = "\n".join(lines)
    clean_text = "\n".join(line.rstrip() for line in raw_text.splitlines()).rstrip() + "\n"
    (out / "cluster55_crosstab.txt").write_text(clean_text, encoding="utf-8")

    print("\n=== Dominant split headline ===")
    print(f"Exact views: {len(configs)}; sizes: {sorted(counts.tolist())}")
    for column in ("selected_count", "subdocument_length", "combined_input_length", "year"):
        if column in audit:
            print(f"\n{column}:\n{numeric_summary(audit, column)}")
    print("\nkeyword counts:\n" + pd.crosstab(audit["keyword"], audit["dominant_group"]).to_string())
    return audit


def nearest_pairs(meta: pd.DataFrame, emb: np.ndarray, out: Path, top: int = 30) -> pd.DataFrame:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains zero-norm rows")
    vectors = emb / norms
    similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -np.inf)
    upper = np.triu_indices_from(similarity, k=1)
    order = np.argsort(similarity[upper])[::-1][:top]
    left, right = upper[0][order], upper[1][order]
    result = pd.DataFrame(
        {
            "cosine": similarity[left, right],
            "id_a": meta["paper_id"].to_numpy()[left],
            "title_a": meta["title"].to_numpy()[left],
            "id_b": meta["paper_id"].to_numpy()[right],
            "title_b": meta["title"].to_numpy()[right],
        }
    )
    result.to_csv(out / "nearest_pairs.csv", index=False, float_format="%.8f")
    print("\n=== Top embedding-neighbor pairs ===")
    print(result.head(15).to_string(index=False, max_colwidth=46, float_format=lambda x: f"{x:.6f}"))
    return result


def pareto_front(manifest: dict, out: Path) -> pd.DataFrame:
    """Mark selected views nondominated on silhouette, coverage, and size balance."""
    rows = []
    for view in manifest.get("views", []):
        largest = float(view.get("largest_cluster") or np.nan)
        smallest = float(view.get("smallest_cluster") or np.nan)
        rows.append(
            {
                "config": view.get("title"),
                "silhouette_original_cosine": view.get("silhouette_original_cosine"),
                "coverage": view.get("coverage"),
                "size_balance_min_over_max": smallest / largest if largest else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    values = frame[["silhouette_original_cosine", "coverage", "size_balance_min_over_max"]].to_numpy(float)
    dominated = np.zeros(len(frame), dtype=bool)
    for i, candidate in enumerate(values):
        for j, other in enumerate(values):
            if i != j and np.all(other >= candidate) and np.any(other > candidate):
                dominated[i] = True
                break
    frame["pareto_nondominated"] = ~dominated
    frame.to_csv(out / "selected_view_pareto.csv", index=False, float_format="%.6f")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="outputs/diagnostics/global_partitions_20260731")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.input)
    embeddings = np.load(args.embeddings)
    if len(meta) != len(embeddings):
        raise ValueError(f"Input rows {len(meta)} != embedding rows {len(embeddings)}")
    if meta["paper_id"].duplicated().any():
        raise ValueError("Input contains duplicate paper IDs")

    parts, public_meta, manifest = load_partitions(Path(args.manifest), meta["paper_id"])
    meta = meta.merge(public_meta, on="paper_id", how="left", validate="one_to_one")
    print(f"papers={len(meta)} embeddings={embeddings.shape} configurations={len(parts)}")

    partition_ari(parts, out)
    keyword_ari(parts, meta, out)
    dominant_split_crosstab(parts, meta, embeddings, out)
    nearest_pairs(meta, embeddings, out)
    pareto_front(manifest, out)
    print(f"\nWrote diagnostics to {out}")


if __name__ == "__main__":
    main()
