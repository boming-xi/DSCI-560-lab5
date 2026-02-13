#!/usr/bin/env python3
"""
Cluster Reddit messages using TF-IDF embeddings and K-Means.

Outputs:
- cluster_report.json / .txt with keywords + sample messages
- cluster_assignments.csv with per-document cluster label
- cluster_plot.png (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from typing import Dict, List, Tuple

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_posts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def get_text(post: Dict, text_field: str) -> str:
    if text_field in post and post.get(text_field):
        return str(post.get(text_field))
    title = post.get("title") or ""
    body = post.get("selftext") or post.get("body") or post.get("clean_text") or ""
    return f"{title} {body}".strip()


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_report_txt(path: str, report: Dict) -> None:
    lines: List[str] = []
    lines.append(f"K = {report['k']}")
    lines.append(f"Documents = {report['num_docs']}")
    lines.append("")
    for cluster in report["clusters"]:
        lines.append(f"Cluster {cluster['cluster']} (size={cluster['size']})")
        lines.append(f"Keywords: {', '.join(cluster['keywords'])}")
        lines.append("Samples:")
        for sample in cluster["samples"]:
            title = sample.get("title") or ""
            preview = sample.get("preview") or ""
            lines.append(f"- {sample.get('id')}: {title}")
            lines.append(f"  {preview}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster Reddit messages.")
    parser.add_argument("--input", required=True, help="Path to clean.json or clean.jsonl.")
    parser.add_argument("--text-field", default="clean_text", help="Field to cluster on.")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters.")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--samples-per-cluster", type=int, default=3)
    parser.add_argument("--output-dir", default="cluster_output")
    parser.add_argument("--save-plot", action="store_true")
    parser.add_argument("--plot-method", choices=["pca", "tsne"], default="pca")
    args = parser.parse_args()

    posts = load_posts(args.input)
    if not posts:
        raise SystemExit("No posts found in input.")

    texts: List[str] = []
    meta: List[Dict] = []
    for post in posts:
        text = get_text(post, args.text_field)
        if not text:
            continue
        texts.append(text)
        meta.append(post)

    if len(texts) < args.k:
        args.k = max(1, len(texts))

    min_df = args.min_df
    max_df = args.max_df
    if isinstance(min_df, int) and min_df > len(texts):
        min_df = 1
    if len(texts) <= 1:
        max_df = 1.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=min_df,
        max_df=max_df,
    )
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    feature_names = vectorizer.get_feature_names_out()
    centroids = kmeans.cluster_centers_

    ensure_dir(args.output_dir)

    # Build report using correct similarity computation
    report = {"k": int(max(labels) + 1), "num_docs": len(texts), "clusters": []}
    for cluster_id in range(report["k"]):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices:
            continue

        centroid = centroids[cluster_id]
        top_idx = centroid.argsort()[::-1][: args.top_terms]
        keywords = [feature_names[i] for i in top_idx]

        # Similarity to centroid for sample selection
        sims = cosine_similarity(X[indices], centroid.reshape(1, -1)).ravel()
        ranked = sorted(zip(indices, sims), key=lambda x: x[1], reverse=True)
        samples = []
        for doc_idx, score in ranked[: args.samples_per_cluster]:
            post = meta[doc_idx]
            preview = textwrap.shorten(
                (post.get(args.text_field) or get_text(post, args.text_field)),
                width=200,
                placeholder="...",
            )
            samples.append(
                {
                    "id": post.get("id"),
                    "title": post.get("title"),
                    "preview": preview,
                    "score": round(float(score), 4),
                }
            )

        report["clusters"].append(
            {
                "cluster": cluster_id,
                "size": len(indices),
                "keywords": keywords,
                "samples": samples,
            }
        )

    report_path = os.path.join(args.output_dir, "cluster_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    write_report_txt(os.path.join(args.output_dir, "cluster_report.txt"), report)

    # Save assignments
    csv_path = os.path.join(args.output_dir, "cluster_assignments.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "title", "cluster", "clean_text"])
        for post, label in zip(meta, labels):
            writer.writerow(
                [
                    post.get("id"),
                    post.get("title"),
                    label,
                    post.get(args.text_field) or get_text(post, args.text_field),
                ]
            )

    # Visualization
    if args.save_plot and len(texts) > 2:
        import matplotlib.pyplot as plt

        if args.plot_method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            points = reducer.fit_transform(X.toarray())
        else:
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, random_state=42, init="pca")
            points = reducer.fit_transform(X.toarray())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=12)
        plt.title("Reddit Post Clusters")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "cluster_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
