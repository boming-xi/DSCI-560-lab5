from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

import preprocess


DEFAULT_USER_AGENT = "DSCI-560-lab5/1.0 (contact: student)"


def fetch_posts(
    subreddit: str,
    total: int,
    sort: str = "new",
    time_filter: Optional[str] = None,
    max_per_request: int = 100,
    pause: float = 1.0,
    timeout: int = 10,
    user_agent: str = DEFAULT_USER_AGENT,
) -> List[Dict]:
    posts: List[Dict] = []
    after: Optional[str] = None

    while len(posts) < total:
        limit = min(max_per_request, total - len(posts))
        params: Dict[str, object] = {"limit": limit}
        if after:
            params["after"] = after
        if sort == "top" and time_filter:
            params["t"] = time_filter

        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": user_agent},
                params=params,
                timeout=timeout,
            )
        except requests.exceptions.RequestException:
            time.sleep(pause)
            continue

        if resp.status_code == 429:
            time.sleep(max(pause, 2.0))
            continue
        if resp.status_code >= 400:
            break

        try:
            payload = resp.json()
        except ValueError:
            break

        data = payload.get("data", {})
        children = data.get("children", [])
        if not children:
            break

        for child in children:
            post_data = child.get("data", {})
            permalink = post_data.get("permalink") or ""
            if permalink and permalink.startswith("/"):
                permalink = f"https://www.reddit.com{permalink}"
            post = {
                "id": post_data.get("id"),
                "subreddit": post_data.get("subreddit"),
                "title": post_data.get("title"),
                "selftext": post_data.get("selftext"),
                "author": post_data.get("author"),
                "created_utc": post_data.get("created_utc"),
                "url": post_data.get("url"),
                "permalink": permalink,
                "score": post_data.get("score"),
                "num_comments": post_data.get("num_comments"),
                "is_self": post_data.get("is_self"),
                "over_18": post_data.get("over_18"),
                "thumbnail": post_data.get("thumbnail"),
            }
            posts.append(post)
            if len(posts) >= total:
                break

        after = data.get("after")
        if not after:
            break
        time.sleep(pause)

    return posts


def connect_sqlite(path: str):
    import sqlite3

    conn = sqlite3.connect(path)
    return conn


def connect_mysql(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
):
    try:
        import mysql.connector  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mysql-connector-python is required for MySQL. "
            "Install with: pip install mysql-connector-python"
        ) from exc

    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


def create_table(conn, db_type: str) -> None:
    cursor = conn.cursor()
    if db_type == "sqlite":
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id TEXT PRIMARY KEY,
                subreddit TEXT,
                title TEXT,
                body TEXT,
                clean_text TEXT,
                ocr_text TEXT,
                created_utc REAL,
                created_iso_utc TEXT,
                author_masked TEXT,
                keywords TEXT,
                topic TEXT,
                is_irrelevant INTEGER,
                raw TEXT,
                permalink TEXT,
                url TEXT,
                score INTEGER,
                num_comments INTEGER,
                retrieved_utc TEXT
            )
            """
        )
    else:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id VARCHAR(32) PRIMARY KEY,
                subreddit VARCHAR(64),
                title TEXT,
                body LONGTEXT,
                clean_text LONGTEXT,
                ocr_text LONGTEXT,
                created_utc DOUBLE,
                created_iso_utc VARCHAR(64),
                author_masked VARCHAR(64),
                keywords TEXT,
                topic VARCHAR(64),
                is_irrelevant TINYINT,
                raw LONGTEXT,
                permalink TEXT,
                url TEXT,
                score INT,
                num_comments INT,
                retrieved_utc VARCHAR(64)
            ) CHARACTER SET utf8mb4
            """
        )
    conn.commit()


def format_row(processed: Dict) -> Tuple:
    raw = processed.get("raw") or {}
    keywords = processed.get("keywords") or []
    retrieved_utc = datetime.now(timezone.utc).isoformat()

    return (
        processed.get("id") or raw.get("id"),
        processed.get("subreddit") or raw.get("subreddit"),
        processed.get("title"),
        processed.get("body"),
        processed.get("clean_text"),
        processed.get("ocr_text"),
        processed.get("created_utc"),
        processed.get("created_iso_utc"),
        processed.get("author_masked"),
        json.dumps(keywords, ensure_ascii=False),
        processed.get("topic"),
        int(bool(processed.get("is_irrelevant"))),
        json.dumps(raw, ensure_ascii=False),
        raw.get("permalink"),
        raw.get("url"),
        raw.get("score"),
        raw.get("num_comments"),
        retrieved_utc,
    )


def insert_rows(conn, db_type: str, rows: List[Tuple]) -> None:
    if not rows:
        return

    columns = (
        "id, subreddit, title, body, clean_text, ocr_text, created_utc, created_iso_utc, "
        "author_masked, keywords, topic, is_irrelevant, raw, permalink, url, score, "
        "num_comments, retrieved_utc"
    )
    placeholder = "?" if db_type == "sqlite" else "%s"
    placeholders = ", ".join([placeholder] * 18)

    if db_type == "sqlite":
        sql = f"INSERT OR REPLACE INTO reddit_posts ({columns}) VALUES ({placeholders})"
    else:
        update_clause = (
            "subreddit=VALUES(subreddit), "
            "title=VALUES(title), "
            "body=VALUES(body), "
            "clean_text=VALUES(clean_text), "
            "ocr_text=VALUES(ocr_text), "
            "created_utc=VALUES(created_utc), "
            "created_iso_utc=VALUES(created_iso_utc), "
            "author_masked=VALUES(author_masked), "
            "keywords=VALUES(keywords), "
            "topic=VALUES(topic), "
            "is_irrelevant=VALUES(is_irrelevant), "
            "raw=VALUES(raw), "
            "permalink=VALUES(permalink), "
            "url=VALUES(url), "
            "score=VALUES(score), "
            "num_comments=VALUES(num_comments), "
            "retrieved_utc=VALUES(retrieved_utc)"
        )
        sql = (
            f"INSERT INTO reddit_posts ({columns}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {update_clause}"
        )

    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and store Reddit posts.")
    parser.add_argument("--subreddit", default="tech", help="Subreddit name (default: tech).")
    parser.add_argument("--limit", type=int, required=True, help="Number of posts to fetch.")
    parser.add_argument("--sort", choices=["new", "hot", "top"], default="new")
    parser.add_argument(
        "--time-filter",
        choices=["hour", "day", "week", "month", "year", "all"],
        default="week",
        help="Only used when sort=top (default: week).",
    )
    parser.add_argument("--max-per-request", type=int, default=100)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)

    parser.add_argument("--raw-output", help="Optional path to save raw JSON.")

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
    return parser.parse_args()


def resolve_mysql_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "host": args.mysql_host or os.environ.get("MYSQL_HOST", "localhost"),
        "port": args.mysql_port or int(os.environ.get("MYSQL_PORT", "3306")),
        "user": args.mysql_user or os.environ.get("MYSQL_USER", ""),
        "password": args.mysql_password or os.environ.get("MYSQL_PASSWORD", ""),
        "database": args.mysql_database or os.environ.get("MYSQL_DATABASE", ""),
    }


def main() -> int:
    args = parse_args()

    posts = fetch_posts(
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
        with open(args.raw_output, "w", encoding="utf-8") as handle:
            json.dump(posts, handle, ensure_ascii=False, indent=2)

    processed_rows: List[Tuple] = []
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
        processed_rows.append(format_row(processed))

    if args.db_type == "sqlite":
        conn = connect_sqlite(args.sqlite_path)
    else:
        mysql_args = resolve_mysql_args(args)
        if not mysql_args["database"]:
            raise SystemExit("MySQL database name is required. Set --mysql-database or MYSQL_DATABASE.")
        conn = connect_mysql(**mysql_args)  # type: ignore[arg-type]

    create_table(conn, args.db_type)
    insert_rows(conn, args.db_type, processed_rows)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
