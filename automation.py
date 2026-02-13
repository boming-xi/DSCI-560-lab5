from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import collect_store
import preprocess


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def save_json(path: str, data: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def run_update(args: argparse.Namespace) -> List[Dict]:
    log("Fetching data...")
    posts = collect_store.fetch_posts(
        subreddit=args.subreddit,
        total=args.limit,
        sort=args.sort,
        time_filter=args.time_filter,
        max_per_request=args.max_per_request,
        pause=args.pause,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )

    if args.raw_output:
        save_json(args.raw_output, posts)
        log(f"Raw data saved to {args.raw_output}")

    log("Processing data...")
    processed_rows: List[Tuple] = []
    processed_posts: List[Dict] = []
    for post in posts:
        processed = preprocess.preprocess_post(
            post,
            max_keywords=args.max_keywords,
            drop_irrelevant=args.drop_irrelevant,
            enable_ocr=args.enable_ocr,
            ocr_lang=args.ocr_lang,
        )
        if processed is None:
            continue
        processed_posts.append(processed)
        processed_rows.append(collect_store.format_row(processed))

    if args.clean_output:
        save_json(args.clean_output, processed_posts)
        log(f"Clean data saved to {args.clean_output}")

    log("Updating database...")
    if args.db_type == "sqlite":
        conn = collect_store.connect_sqlite(args.sqlite_path)
    else:
        mysql_args = collect_store.resolve_mysql_args(args)
        if not mysql_args["database"]:
            raise SystemExit("MySQL database name is required. Set --mysql-database or MYSQL_DATABASE.")
        conn = collect_store.connect_mysql(**mysql_args)  # type: ignore[arg-type]

    collect_store.create_table(conn, args.db_type)
    collect_store.insert_rows(conn, args.db_type, processed_rows)
    conn.close()
    log("Database update complete.")

    return processed_posts


def build_model(posts: List[Dict], args: argparse.Namespace):
    texts: List[str] = []
    meta: List[Dict] = []
    for post in posts:
        text = post.get(args.text_field) or ""
        if not text:
            title = post.get("title") or ""
            body = post.get("body") or post.get("selftext") or ""
            text = f"{title} {body}".strip()
        if text:
            texts.append(text)
            meta.append(post)

    if not texts:
        return None, None, None, []

    k = min(args.k, len(texts))
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
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return (vectorizer, kmeans, (X, labels), meta)


def display_cluster(
    query: str,
    vectorizer,
    kmeans,
    X,
    labels,
    meta: List[Dict],
    args: argparse.Namespace,
) -> None:
    query_vec = vectorizer.transform([query])
    cluster_id = int(kmeans.predict(query_vec)[0])
    centroid = kmeans.cluster_centers_[cluster_id]

    indices = [i for i, label in enumerate(labels) if label == cluster_id]
    if not indices:
        log(f"No documents found for cluster {cluster_id}.")
        return

    feature_names = vectorizer.get_feature_names_out()
    top_idx = centroid.argsort()[::-1][: args.top_terms]
    keywords = [feature_names[i] for i in top_idx]

    sims = cosine_similarity(X[indices], centroid.reshape(1, -1)).ravel()
    ranked = sorted(zip(indices, sims), key=lambda x: x[1], reverse=True)

    log(f"Closest cluster: {cluster_id} (keywords: {', '.join(keywords)})")
    print("Top messages:")
    for doc_idx, score in ranked[: args.samples_per_cluster]:
        post = meta[doc_idx]
        preview = textwrap.shorten(
            post.get(args.text_field) or "",
            width=200,
            placeholder="...",
        )
        print(f"- {post.get('id')} | {post.get('title')}")
        print(f"  {preview} (score={score:.4f})")

    if args.save_query_plot:
        try:
            import matplotlib.pyplot as plt

            values = [centroid[i] for i in top_idx]
            plt.figure(figsize=(6, 4))
            plt.bar(keywords, values)
            plt.title(f"Cluster {cluster_id} Top Keywords")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            os.makedirs(args.output_dir, exist_ok=True)
            plot_path = os.path.join(args.output_dir, f"cluster_{cluster_id}_keywords.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log(f"Keyword plot saved to {plot_path}")
        except Exception as exc:
            log(f"Plotting failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate data collection and clustering.")
    parser.add_argument("interval_minutes", type=int, help="Update interval in minutes.")
    parser.add_argument("--subreddit", default="tech")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sort", choices=["new", "hot", "top"], default="new")
    parser.add_argument(
        "--time-filter",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="week",
    )
    parser.add_argument("--max-per-request", type=int, default=100)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--user-agent", default=collect_store.DEFAULT_USER_AGENT)

    parser.add_argument("--raw-output", default="raw.json")
    parser.add_argument("--clean-output", default="clean.json")

    parser.add_argument("--db-type", choices=["sqlite", "mysql"], default="sqlite")
    parser.add_argument("--sqlite-path", default="reddit.db")
    parser.add_argument("--mysql-host", default=None)
    parser.add_argument("--mysql-port", type=int, default=None)
    parser.add_argument("--mysql-user", default=None)
    parser.add_argument("--mysql-password", default=None)
    parser.add_argument("--mysql-database", default=None)

    parser.add_argument("--max-keywords", type=int, default=8)
    parser.add_argument("--drop-irrelevant", action="store_true")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument("--ocr-lang", default="eng")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--text-field", default="clean_text")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.8)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--samples-per-cluster", type=int, default=3)
    parser.add_argument("--save-query-plot", action="store_true")
    parser.add_argument("--output-dir", default="cluster_output")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interval_seconds = max(1, args.interval_minutes) * 60

    log("Automation started.")
    next_run = time.time()

    while True:
        if time.time() >= next_run:
            try:
                processed_posts = run_update(args)
            except Exception as exc:
                log(f"Error during update: {exc}")
                processed_posts = []

            vectorizer, kmeans, matrix_bundle, meta = build_model(processed_posts, args)
            if vectorizer is None:
                log("No data available for clustering.")
                next_run = time.time() + interval_seconds
                continue

            X, labels = matrix_bundle
            next_run = time.time() + interval_seconds

        remaining = int(next_run - time.time())
        if remaining <= 0:
            continue

        try:
            query = input(f"[next update in {remaining}s] Enter keywords/message (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            log("Exiting automation.")
            return 0

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            log("Exiting automation.")
            return 0

        try:
            display_cluster(query, vectorizer, kmeans, X, labels, meta, args)
        except Exception as exc:
            log(f"Query error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
