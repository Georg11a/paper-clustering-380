#!/usr/bin/env python3
"""Dependency-light K=5/8/12 clustering sensitivity test.

Uses the real pilot CSVs, a single shared TF-IDF vocabulary, a single shared
randomized-SVD projection, and a K-means cluster count selected on K8 then
fixed across budgets. Numeric labels are aligned to K8 only for the readable
paper-switch table; ARI and NMI are label-permutation invariant.

Runs two views:
  1. subdocument_only -- isolates the effect of paragraph budget;
  2. final_input -- title + abstract + subdocument, the planned Experiment B
     representation.

Only NumPy and pandas are required. HDBSCAN is intentionally left to Batch 3
rather than replaced by a non-equivalent home-grown density algorithm.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BUDGETS = ("K5", "K8", "K12")
TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{1,}", re.I)
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "these", "those",
    "was", "were", "are", "been", "being", "have", "has", "had", "into",
    "their", "there", "which", "when", "where", "what", "while", "about",
    "also", "than", "then", "they", "them", "such", "using", "used", "use",
    "our", "its", "not", "can", "may", "more", "between", "within", "through",
}


def tokenize(text: str) -> list[str]:
    words = [x.lower() for x in TOKEN_RE.findall(text)]
    words = [x for x in words if x not in STOPWORDS]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def load_inputs(paths: dict[str, Path], manifest_path: Path):
    manifest = pd.read_csv(manifest_path).fillna("")
    metadata = manifest.set_index("paper_id")[["abstract"]]
    frames, sets = {}, []
    for name, path in paths.items():
        frame = pd.read_csv(path).fillna("")
        if frame["paper_id"].duplicated().any():
            raise ValueError(f"Duplicate paper IDs in {path}")
        frame = frame.set_index("paper_id").join(metadata, how="left")
        frames[name] = frame
        sets.append(set(frame.index))
    ids = sorted(set.intersection(*sets))
    if len(ids) < 5:
        raise ValueError(f"Only {len(ids)} IDs shared across budgets")
    return ids, {name: frame.loc[ids] for name, frame in frames.items()}


def documents_for_view(frames, view: str):
    output = {}
    for name, frame in frames.items():
        if view == "subdocument_only":
            output[name] = frame["subdocument"].astype(str).tolist()
        else:
            output[name] = [
                "\n\n".join(
                    value for value in (
                        str(row["canonical_title"]),
                        str(row["abstract"]),
                        str(row["subdocument"]),
                    )
                    if value.strip()
                )
                for _, row in frame.iterrows()
            ]
    return output


def shared_tfidf(documents, max_features=8000):
    all_docs = [tokenize(doc) for name in BUDGETS for doc in documents[name]]
    n_docs = len(all_docs)
    df = Counter()
    total = Counter()
    for doc in all_docs:
        counts = Counter(doc)
        df.update(counts)
        total.update(counts)
    eligible = [
        term for term, freq in df.items()
        if freq >= 2 and freq / n_docs <= 0.95
    ]
    eligible.sort(key=lambda term: (total[term] * math.log((1 + n_docs) / (1 + df[term])), term), reverse=True)
    vocab = {term: i for i, term in enumerate(eligible[:max_features])}
    matrix = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for row, doc in enumerate(all_docs):
        counts = Counter(term for term in doc if term in vocab)
        for term, count in counts.items():
            matrix[row, vocab[term]] = (1 + math.log(count)) * (math.log((1 + n_docs) / (1 + df[term])) + 1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms, len(vocab)


def randomized_svd_projection(matrix, components=50, seed=11):
    rng = np.random.default_rng(seed)
    rank = min(components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    width = min(matrix.shape[1], rank + 10)
    omega = rng.normal(size=(matrix.shape[1], width))
    y = matrix @ omega
    for _ in range(2):
        y = matrix @ (matrix.T @ y)
    q, _ = np.linalg.qr(y, mode="reduced")
    small = q.T @ matrix
    u_hat, singular, _ = np.linalg.svd(small, full_matrices=False)
    reduced = (q @ u_hat[:, :rank]) * singular[:rank]
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1
    explained = float(np.sum(singular[:rank] ** 2) / max(np.sum(singular ** 2), 1e-12))
    return reduced / norms, rank, explained


def shared_space(documents, seed=11):
    matrix, vocab_size = shared_tfidf(documents)
    reduced, dimensions, explained = randomized_svd_projection(matrix, seed=seed)
    size = len(documents["K5"])
    vectors = {
        name: reduced[i * size : (i + 1) * size]
        for i, name in enumerate(BUDGETS)
    }
    return vectors, {
        "vocabulary_size": vocab_size,
        "svd_dimensions": dimensions,
        "relative_singular_energy": explained,
    }


def kmeans(vectors, k, seed, n_init=20, max_iter=300):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [int(rng.integers(len(vectors)))]
        while len(centers) < k:
            distances = np.min(
                np.stack([np.sum((vectors - vectors[i]) ** 2, axis=1) for i in centers]),
                axis=0,
            )
            distances[centers] = 0
            total = distances.sum()
            candidate = int(rng.choice(len(vectors), p=distances / total)) if total > 0 else int(rng.integers(len(vectors)))
            if candidate not in centers:
                centers.append(candidate)
        centroids = vectors[centers].copy()
        labels = np.full(len(vectors), -1, dtype=int)
        for _ in range(max_iter):
            new_labels = np.argmin(
                np.sum((vectors[:, None, :] - centroids[None, :, :]) ** 2, axis=2),
                axis=1,
            )
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = vectors[labels == cluster]
                if len(members):
                    centroid = members.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    centroids[cluster] = centroid / norm if norm else centroid
                else:
                    centroids[cluster] = vectors[int(rng.integers(len(vectors)))]
        inertia = float(np.sum((vectors - centroids[labels]) ** 2))
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy())
    return best[1]


def pairwise_cosine_distance(vectors):
    return 1 - np.clip(vectors @ vectors.T, -1, 1)


def silhouette(vectors, labels):
    distance = pairwise_cosine_distance(vectors)
    values = []
    for i, label in enumerate(labels):
        same = np.where(labels == label)[0]
        same = same[same != i]
        if len(same) == 0:
            values.append(0.0)
            continue
        a = float(distance[i, same].mean())
        b = min(
            float(distance[i, labels == other].mean())
            for other in set(labels.tolist())
            if other != label
        )
        values.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(values))


def comb2(value):
    return value * (value - 1) / 2


def contingency(left, right):
    left_values = sorted(set(left.tolist()))
    right_values = sorted(set(right.tolist()))
    table = np.zeros((len(left_values), len(right_values)), dtype=int)
    for i, a in enumerate(left_values):
        for j, b in enumerate(right_values):
            table[i, j] = int(np.sum((left == a) & (right == b)))
    return table


def ari(left, right):
    table = contingency(left, right)
    sum_cells = sum(comb2(x) for x in table.ravel())
    sum_rows = sum(comb2(x) for x in table.sum(axis=1))
    sum_cols = sum(comb2(x) for x in table.sum(axis=0))
    total = comb2(len(left))
    expected = sum_rows * sum_cols / total if total else 0
    maximum = 0.5 * (sum_rows + sum_cols)
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def nmi(left, right):
    table = contingency(left, right).astype(float)
    n = table.sum()
    pi, pj = table.sum(axis=1), table.sum(axis=0)
    mutual = 0.0
    for i, j in itertools.product(range(table.shape[0]), range(table.shape[1])):
        if table[i, j] > 0:
            mutual += table[i, j] / n * math.log((table[i, j] * n) / (pi[i] * pj[j]))
    h_left = -sum((x / n) * math.log(x / n) for x in pi if x > 0)
    h_right = -sum((x / n) * math.log(x / n) for x in pj if x > 0)
    return float(mutual / math.sqrt(h_left * h_right)) if h_left and h_right else 1.0


def choose_k(vectors, seeds, k_min, k_max):
    rows = []
    for k in range(k_min, min(k_max, len(vectors) - 1) + 1):
        runs = [kmeans(vectors, k, seed) for seed in seeds]
        silhouettes = [silhouette(vectors, labels) for labels in runs]
        seed_aris = [ari(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        sizes = sorted(int(np.sum(runs[0] == value)) for value in set(runs[0].tolist()))
        rows.append({
            "k": k,
            "mean_silhouette": float(np.mean(silhouettes)),
            "mean_seed_ari": float(np.mean(seed_aris)),
            "admissible": min(sizes) >= 2 and max(sizes) / len(vectors) <= 0.70,
            "cluster_sizes_seed_0": ";".join(map(str, sizes)),
        })
    table = pd.DataFrame(rows)
    pool = table[table["admissible"]]
    if pool.empty:
        pool = table
    best = pool.sort_values(["mean_silhouette", "mean_seed_ari", "k"], ascending=[False, False, True]).iloc[0]
    return int(best["k"]), table


def bootstrap_ari(left, right, repetitions=1000, seed=20260727):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(left), len(left))
        if len(set(left[indices])) >= 2 and len(set(right[indices])) >= 2:
            values.append(ari(left[indices], right[indices]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def align_to_reference(labels, reference):
    source = sorted(set(labels.tolist()))
    target = sorted(set(reference.tolist()))
    table = contingency(labels, reference)
    best_score, best_perm = -1, None
    for perm in itertools.permutations(target, len(source)):
        score = sum(table[i, target.index(perm[i])] for i in range(len(source)))
        if score > best_score:
            best_score, best_perm = score, perm
    mapping = dict(zip(source, best_perm))
    return np.array([mapping[x] for x in labels], dtype=int)


def run_view(view, ids, frames, output, seeds, k_min, k_max):
    documents = documents_for_view(frames, view)
    vectors, embedding = shared_space(documents, seed=seeds[0])
    selected_k, k_table = choose_k(vectors["K8"], seeds, k_min, k_max)
    labels = {name: kmeans(vectors[name], selected_k, seeds[0]) for name in BUDGETS}
    agreements = []
    for left_name, right_name in itertools.combinations(BUDGETS, 2):
        low, high = bootstrap_ari(labels[left_name], labels[right_name])
        agreements.append({
            "view": view,
            "pair": f"{left_name}_vs_{right_name}",
            "method": "kmeans_fixed_k",
            "ari": ari(labels[left_name], labels[right_name]),
            "ari_ci_low": low,
            "ari_ci_high": high,
            "nmi": nmi(labels[left_name], labels[right_name]),
        })
    similarity_rows = []
    for left_name, right_name in itertools.combinations(BUDGETS, 2):
        same_paper = np.sum(vectors[left_name] * vectors[right_name], axis=1)
        similarity_rows.append({
            "view": view,
            "pair": f"{left_name}_vs_{right_name}",
            "mean_same_paper_cosine": float(np.mean(same_paper)),
            "median_same_paper_cosine": float(np.median(same_paper)),
            "minimum_same_paper_cosine": float(np.min(same_paper)),
        })
    aligned = {name: align_to_reference(value, labels["K8"]) for name, value in labels.items()}
    paper_rows = []
    titles = frames["K8"]["canonical_title"].astype(str).tolist()
    for index, paper_id in enumerate(ids):
        values = [int(aligned[name][index]) for name in BUDGETS]
        paper_rows.append({
            "view": view,
            "paper_id": paper_id,
            "title": titles[index],
            "K5_cluster_aligned": values[0],
            "K8_cluster_reference": values[1],
            "K12_cluster_aligned": values[2],
            "switched_between_any_budget": len(set(values)) > 1,
        })
    k_table.insert(0, "view", view)
    k_table.to_csv(output / f"{view}_k_selection.csv", index=False)
    pd.DataFrame(paper_rows).sort_values(
        ["switched_between_any_budget", "paper_id"], ascending=[False, True]
    ).to_csv(output / f"{view}_paper_cluster_by_budget.csv", index=False)
    return {
        "view": view,
        "selected_k_from_K8": selected_k,
        "embedding": embedding,
        "agreement": agreements,
        "same_paper_similarity": similarity_rows,
        "switch_count": int(sum(row["switched_between_any_budget"] for row in paper_rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k5", default="outputs/batch2/paragraph_budget_pilot/subdocuments_k5_design_theory.csv")
    parser.add_argument("--k8", default="outputs/batch2/paragraph_budget_pilot/subdocuments_k8_design_theory.csv")
    parser.add_argument("--k12", default="outputs/batch2/paragraph_budget_pilot/subdocuments_k12_design_theory.csv")
    parser.add_argument("--manifest", default="outputs/batch1/canonical_analysis_input_284.csv")
    parser.add_argument("--out", default="outputs/batch2/paragraph_budget_sensitivity")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    ids, frames = load_inputs(
        {"K5": Path(args.k5), "K8": Path(args.k8), "K12": Path(args.k12)},
        Path(args.manifest),
    )
    results = [
        run_view(view, ids, frames, output, args.seeds, args.k_min, args.k_max)
        for view in ("subdocument_only", "final_input")
    ]
    pd.DataFrame(row for result in results for row in result["agreement"]).to_csv(
        output / "budget_agreement.csv", index=False
    )
    pd.DataFrame(row for result in results for row in result["same_paper_similarity"]).to_csv(
        output / "same_paper_similarity.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps({"n_papers": len(ids), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Compared {len(ids)} Design Theory papers.")
    for result in results:
        print(f"\n{result['view']}: fixed k={result['selected_k_from_K8']}; switches={result['switch_count']}/{len(ids)}")
        for row in result["agreement"]:
            print(
                f"  {row['pair']:12s} ARI={row['ari']:.3f} "
                f"95% CI [{row['ari_ci_low']:.3f}, {row['ari_ci_high']:.3f}] "
                f"NMI={row['nmi']:.3f}"
            )
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
