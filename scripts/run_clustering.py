#!/usr/bin/env python3
"""Run keyword-conditioned clustering over the final advancing-paper CSV."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"paper_id", "title", "abstract", "keyword"}


def normalize_keyword(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in slug.split("_") if part)


def dynamic_k_cap(n_papers: int) -> int:
    """Keep K-means from over-splitting small keyword groups.

    K-means always returns the requested number of clusters. For small or
    homogeneous keyword groups, a large k creates repeated interpretation
    labels rather than meaningful themes.
    """
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


def choose_k_bounds(n_papers: int, k_min: int, k_max: int) -> tuple[int, int]:
    upper = min(k_max, dynamic_k_cap(n_papers), n_papers - 1)
    lower = min(k_min, upper)
    lower = max(2, lower)
    return lower, upper


def run_keyword_method(
    input_csv: Path,
    outdir: Path,
    keyword: str,
    method: str,
    args: argparse.Namespace,
    n_papers: int,
) -> Path:
    method_outdir = outdir / "by_keyword" / safe_slug(keyword) / method
    command = [
        sys.executable,
        "scripts/cluster_papers.py",
        "--input",
        str(input_csv),
        "--outdir",
        str(method_outdir),
        "--focus-keyword",
        keyword,
        "--method",
        method,
        "--embedding",
        args.embedding,
        "--text-view",
        args.text_view,
        "--min-papers-to-cluster",
        str(args.min_papers_to_cluster),
        "--keyword-context-paragraphs",
        str(args.keyword_context_paragraphs),
        "--lda-topics",
        str(min(args.lda_topics, max(2, n_papers - 1))),
    ]
    if method == "kmeans":
        if args.select_k:
            k_min, k_max = choose_k_bounds(n_papers, args.k_min, args.k_max)
            command.extend(["--select-k", "--k-min", str(k_min), "--k-max", str(k_max)])
        else:
            command.extend(["--k", str(min(args.k, n_papers - 1))])
    if method == "hdbscan":
        command.extend(
            [
                "--min-cluster-size",
                str(min(args.min_cluster_size, max(2, n_papers // 3))),
                "--min-samples",
                str(min(args.min_samples, max(2, n_papers // 4))),
            ]
        )
    if method == "dbscan":
        command.extend(["--min-samples", str(min(args.min_samples, max(2, n_papers // 4)))])
        if args.dbscan_eps is not None:
            command.extend(["--dbscan-eps", str(args.dbscan_eps)])
    if args.embedding == "ollama":
        command.extend(["--ollama-model", args.ollama_model, "--ollama-host", args.ollama_host])

    print(f"Running {method} for {keyword} ({n_papers} papers)")
    subprocess.run(command, check=True)
    return method_outdir / "clustered_papers.csv"


def too_few_rows(group: pd.DataFrame, keyword: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = group.copy()
    rows["focus_keyword"] = keyword
    rows["cluster_method"] = ""
    rows["cluster_status"] = "too_few_papers"
    rows["cluster"] = ""
    rows["cluster_label_candidate"] = ""
    rows["contribution_type_key"] = ""
    rows["contribution_type"] = ""
    rows["contribution_type_secondary"] = ""
    rows["contribution_type_patterns"] = ""
    rows["contribution_type_support"] = ""
    rows["application_domains"] = ""
    rows["application_domain_patterns"] = ""
    rows["application_domain_support"] = ""
    rows["theory_move_key"] = ""
    rows["theory_move"] = ""
    rows["theory_move_patterns"] = ""
    rows["theory_move_support"] = ""
    rows["cluster_summary_candidate"] = (
        f"Skipped clustering because this keyword group has fewer than "
        f"{args.min_papers_to_cluster} papers."
    )
    rows["is_representative_top3"] = False
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final_advancing_list.csv")
    parser.add_argument("--output", default="outputs/clustering_results.csv")
    parser.add_argument("--summary-output", default="outputs/keyword_summary.csv")
    parser.add_argument("--run-metadata", default="outputs/run_metadata.json")
    parser.add_argument("--methods", nargs="+", default=["kmeans", "hdbscan"], choices=["kmeans", "hdbscan", "dbscan"])
    parser.add_argument("--embedding", choices=["tfidf-svd", "ollama"], default="tfidf-svd")
    parser.add_argument(
        "--text-view",
        choices=["overall", "context", "knowledge-type", "method", "target-user", "purpose"],
        default="overall",
    )
    parser.add_argument("--min-papers-to-cluster", type=int, default=10)
    parser.add_argument("--keyword-context-paragraphs", type=int, default=6)
    parser.add_argument("--select-k", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--dbscan-eps", type=float, default=None)
    parser.add_argument("--lda-topics", type=int, default=6)
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)
    summary_csv = Path(args.summary_output)
    metadata_path = Path(args.run_metadata)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df["keyword"] = df["keyword"].map(normalize_keyword)
    combined = []
    summary_rows = []
    grouped = sorted(df.groupby("keyword"), key=lambda item: (-len(item[1]), item[0].lower()))

    for keyword, group in grouped:
        n_papers = len(group)
        if n_papers < args.min_papers_to_cluster:
            combined.append(too_few_rows(group, keyword, args))
            summary_rows.append(
                {
                    "keyword": keyword,
                    "paper_count": n_papers,
                    "status": "too_few_papers",
                    "methods": "",
                    "output_dir": "",
                }
            )
            print(f"Skipping {keyword}: {n_papers} papers (< {args.min_papers_to_cluster})")
            continue

        method_outputs = []
        for method in args.methods:
            clustered_csv = run_keyword_method(input_csv, output_csv.parent, keyword, method, args, n_papers)
            method_df = pd.read_csv(clustered_csv).fillna("")
            method_df["cluster_status"] = "clustered"
            combined.append(method_df)
            method_outputs.append(str(clustered_csv.parent))
        summary_rows.append(
            {
                "keyword": keyword,
                "paper_count": n_papers,
                "status": "clustered",
                "methods": "|".join(args.methods),
                "output_dir": "|".join(method_outputs),
            }
        )

    combined_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    combined_df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    metadata = {
        "input": str(input_csv),
        "output": str(output_csv),
        "summary_output": str(summary_csv),
        "paper_count": int(len(df)),
        "keyword_count": int(df["keyword"].nunique()),
        "methods": args.methods,
        "embedding": args.embedding,
        "text_view": args.text_view,
        "min_papers_to_cluster": args.min_papers_to_cluster,
        "keyword_counts": df["keyword"].value_counts().to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
