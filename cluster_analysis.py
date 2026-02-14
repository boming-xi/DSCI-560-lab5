from __future__ import annotations

import argparse
import csv
import json
import os
import textwrap
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


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
    lines.append(f"Silhouette Score = {report.get('silhouette_score', 'N/A')}")
    lines.append("")

    for cluster in report["clusters"]:
        lines.append(f"Cluster {cluster['cluster']} (size={cluster['size']})")
        lines.append(f"Keywords: {', '.join(cluster['keywords'])}")
        lines.append("Samples:")
        for sample in cluster["samples"]:
            lines.append(f"- {sample.get('id')}: {sample.get('title')}")
            lines.append(f"  {sample.get('preview')}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# select the best k automatically
def find_best_k(X, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1
    scores = {}

    max_possible_k = min(k_max, X.shape[0] - 1)

    for k in range(k_min, max_possible_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster Reddit messages.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-field", default="clean_text")
    parser.add_argument("--k", type=int, default=8)
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
        raise SystemExit("No posts found.")

    texts = []
    meta = []

    for post in posts:
        text = get_text(post, args.text_field)
        if text:
            texts.append(text)
            meta.append(post)

    if len(texts) < 3:
        raise SystemExit("Not enough documents.")

    # TF-IDF for keyword extraction
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
    )
    X_tfidf = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Sentence-BERT for semantic embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, show_progress_bar=True)
    dynamic_k_max = min(args.k, max(2, len(texts)//10))
    best_k, best_score, k_scores = find_best_k(X, 2, dynamic_k_max)


    print("Silhouette scores:")
    for k, score in k_scores.items():
        print(f"K={k}: {score:.4f}")

    print(f"Best K selected: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    ensure_dir(args.output_dir)

    report = {
        "k": int(best_k),
        "num_docs": len(texts),
        "silhouette_score": round(float(best_score), 4),
        "clusters": [],
    }

    # TF-IDF for extracting keywords
    tfidf_centroids = []
    for cluster_id in range(best_k):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices:
            tfidf_centroids.append(None)
            continue
        centroid = np.mean(X_tfidf[indices].toarray(), axis=0)
        tfidf_centroids.append(centroid)

    for cluster_id in range(best_k):
        indices = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices:
            continue

        centroid = tfidf_centroids[cluster_id]
        top_idx = centroid.argsort()[::-1][: args.top_terms]
        keywords = [feature_names[i] for i in top_idx]

        sims = cosine_similarity(X[indices], X[indices].mean(axis=0).reshape(1, -1)).ravel()
        ranked = sorted(zip(indices, sims), key=lambda x: x[1], reverse=True)

        # Measure internal similarity of this cluster
        cluster_vectors = X[indices]
        if len(indices) > 1:
            sim_matrix = cosine_similarity(cluster_vectors)
            upper_triangle = sim_matrix[np.triu_indices(len(indices), k=1)]
            avg_similarity = float(np.mean(upper_triangle))
        else:
            avg_similarity = 1.0

        samples = []
        for doc_idx, score in ranked[: args.samples_per_cluster]:
            post = meta[doc_idx]
            preview = textwrap.shorten(
                get_text(post, args.text_field),
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
                "avg_intra_similarity": round(avg_similarity, 4),
                "keywords": keywords,
                "samples": samples,
            }
        )

    with open(os.path.join(args.output_dir, "cluster_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

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

    if args.save_plot and len(texts) > 2:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Dimensionality reduction(SBERT embeddings)
        pca = PCA(n_components=2, random_state=42)
        points = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            cmap="tab10",
            s=25,
            alpha=0.8,
        )

        plt.title(f"SBERT Cluster Visualization (K={best_k})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, "cluster_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"PCA plot saved to {plot_path}")

        # cluster size distribution
        cluster_sizes = [cluster["size"] for cluster in report["clusters"]]
        cluster_ids = [cluster["cluster"] for cluster in report["clusters"]]

        plt.figure(figsize=(6, 4))
        plt.bar(cluster_ids, cluster_sizes)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Documents")
        plt.title("Cluster Size Distribution")
        plt.tight_layout()

        size_plot_path = os.path.join(args.output_dir, "cluster_size_distribution.png")
        plt.savefig(size_plot_path, dpi=150)
        plt.close()

        print(f"Cluster size distribution saved to {size_plot_path}")

    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
