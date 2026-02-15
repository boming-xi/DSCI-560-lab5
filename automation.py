from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import subprocess
import json
import time
from typing import Dict, List

import collect_store
from preprocess import preprocess_post
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import textwrap


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{timestamp}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("interval_minutes", type=int)
    parser.add_argument("--subreddit", default="tech")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sort", default="new")
    parser.add_argument("--sqlite-path", default="reddit.db")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--text-field", default="clean_text")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--samples-per-cluster", type=int, default=3)
    return parser.parse_args()


def run_collect_store(args):
    cmd = [
        "python",
        "collect_store.py",
        "--subreddit", args.subreddit,
        "--limit", str(args.limit),
        "--sort", args.sort,
        "--db-type", "sqlite",
        "--sqlite-path", args.sqlite_path,
    ]

    subprocess.run(cmd, check=True)


def load_from_db(db_path):
    conn = collect_store.connect_sqlite(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM reddit_posts")
    columns = [c[0] for c in cur.description]
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append(dict(zip(columns, row)))
    return results


def build_model(posts, args):
    texts = []
    meta = []

    for post in posts:
        raw_text = post.get(args.text_field)
        if raw_text is None:
            raw_text = ""
        text = str(raw_text).strip()
        if text:
            texts.append(text)
            meta.append(post)

    if not texts:
        return None, None, None, None

    k = min(args.k, len(texts))

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
    )

    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    return vectorizer, kmeans, (X, labels), meta


def main():

    args = parse_args()
    interval_seconds = max(1, args.interval_minutes) * 60

    log("Automation started.")

    next_run = time.time()

    vectorizer = None
    kmeans = None
    X = None
    labels = None
    meta = None

    while True:

        if time.time() >= next_run:

            try:
                log("Running collect_store...")
                run_collect_store(args)

                posts = load_from_db(args.sqlite_path)

                result = build_model(posts, args)

                if result[0] is not None:
                    vectorizer, kmeans, bundle, meta = result
                    X, labels = bundle

                    dense = X.toarray().tolist()

                    conn = collect_store.connect_sqlite(args.sqlite_path)
                    collect_store.update_embeddings(conn, meta, dense)
                    collect_store.update_cluster_ids(conn, meta, labels)
                    conn.close()

                    log("Model rebuilt and embeddings saved.")

            except Exception as e:
                log(f"Update error: {e}")

            next_run = time.time() + interval_seconds

        remaining = int(next_run - time.time())
        if remaining <= 0:
            continue

        query = input(f"[next update in {remaining}s] Enter keywords/message (or 'quit'): ").strip()

        if query.lower() in {"quit", "exit"}:
            log("Exiting automation.")
            return

        if vectorizer is None:
            log("Model not ready yet.")
            continue

        query_vec = vectorizer.transform([query])
        cluster_id = int(kmeans.predict(query_vec)[0])
        log(f"Closest cluster: {cluster_id}")

if __name__ == "__main__":
    main()
