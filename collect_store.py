from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime,timezone
from typing import Dict,Iterable,List,Optional,Tuple
import requests
import preprocess

DEFAULT_USER_AGENT="DSCI-560-lab5/1.0 (contact: student)"

def fetch_posts(
    subreddit:str,
    total:int,
    sort:str="new",
    time_filter:Optional[str]=None,
    max_per_request:int=100,
    pause:float=1.0,
    timeout:int=10,
    user_agent:str=DEFAULT_USER_AGENT,
)->List[Dict]:

    #accumulate posts until reaching total
    posts:List[Dict]=[]
    after_token:Optional[str]=None

    while len(posts)<total:
        request_limit=min(max_per_request,total-len(posts))
        request_params:Dict[str,object]={"limit":request_limit}
        if after_token:
            request_params["after"]=after_token
        #time filter only applies when sort=top
        if sort=="top" and time_filter:
            request_params["t"]=time_filter
        request_url=f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        try:
            response=requests.get(
                request_url,
                headers={"User-Agent":user_agent},
                params=request_params,
                timeout=timeout,
            )
        except requests.exceptions.RequestException:
            #network hiccup, wait and retry
            time.sleep(pause)
            continue
        if response.status_code==429:
            #rate limited
            time.sleep(max(pause,2.0))
            continue
        if response.status_code>=400:
            break
        try:
            payload=response.json()
        except ValueError:
            break
        data_block=payload.get("data",{})
        children=data_block.get("children",[])
        if not children:
            break
        for child in children:
            raw_post_data=child.get("data",{})
            permalink=raw_post_data.get("permalink") or ""
            #normalize permalink to full url
            if permalink and permalink.startswith("/"):
                permalink=f"https://www.reddit.com{permalink}"
            post_record={
                "id":raw_post_data.get("id"),
                "subreddit":raw_post_data.get("subreddit"),
                "title":raw_post_data.get("title"),
                "selftext":raw_post_data.get("selftext"),
                "author":raw_post_data.get("author"),
                "created_utc":raw_post_data.get("created_utc"),
                "url":raw_post_data.get("url"),
                "permalink":permalink,
                "score":raw_post_data.get("score"),
                "num_comments":raw_post_data.get("num_comments"),
                "is_self":raw_post_data.get("is_self"),
                "over_18":raw_post_data.get("over_18"),
                "thumbnail":raw_post_data.get("thumbnail"),
            }
            posts.append(post_record)
            if len(posts)>=total:
                break
        after_token=data_block.get("after")
        if not after_token:
            break
        time.sleep(pause)
    return posts

def connect_sqlite(database_path:str):
    import sqlite3
    #simple sqlite connection
    database_connection=sqlite3.connect(database_path)
    return database_connection

def connect_mysql(
    host:str,
    port:int,
    user:str,
    password:str,
    database:str,
):
    try:
        import mysql.connector
    except Exception as error:
        raise RuntimeError(
            "mysql-connector-python is required. "
            "Install with: pip install mysql-connector-python"
        ) from error
    return mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database, )

def create_table(database_connection,db_type:str)->None:
    database_cursor=database_connection.cursor()
    #two schemas depending on backend
    if db_type=="sqlite":
        database_cursor.execute(
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
                retrieved_utc TEXT,
                embedding TEXT,
                cluster_id INTEGER
            )
            """
        )
    else:
        database_cursor.execute(
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
                retrieved_utc VARCHAR(64),
                embedding TEXT,
                cluster_id INTEGER
            ) CHARACTER SET utf8mb4
            """
        )

    database_connection.commit()

def format_row(processed:Dict)->Tuple:
    raw_block=processed.get("raw") or {}
    keyword_list=processed.get("keywords") or []
    retrieved_timestamp=datetime.now(timezone.utc).isoformat()
    return (
        processed.get("id") or raw_block.get("id"),
        processed.get("subreddit") or raw_block.get("subreddit"),
        processed.get("title"),
        processed.get("body"),
        processed.get("clean_text"),
        processed.get("ocr_text"),
        processed.get("created_utc"),
        processed.get("created_iso_utc"),
        processed.get("author_masked"),
        json.dumps(keyword_list,ensure_ascii=False),
        processed.get("topic"),
        int(bool(processed.get("is_irrelevant"))),
        json.dumps(raw_block,ensure_ascii=False),
        raw_block.get("permalink"),
        raw_block.get("url"),
        raw_block.get("score"),
        raw_block.get("num_comments"),
        retrieved_timestamp,
    )

def insert_rows(database_connection,db_type:str,row_values:List[Tuple])->None:
    if not row_values:
        return
    column_list=(
        "id, subreddit, title, body, clean_text, ocr_text, created_utc, created_iso_utc, "
        "author_masked, keywords, topic, is_irrelevant, raw, permalink, url, score, "
        "num_comments, retrieved_utc"
    )
    placeholder="?" if db_type=="sqlite" else "%s"
    placeholder_block=", ".join([placeholder]*18)
    if db_type=="sqlite":
        sql_statement=f"INSERT OR REPLACE INTO reddit_posts ({column_list}) VALUES ({placeholder_block})"
    else:
        update_clause=(
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
        sql_statement=(
            f"INSERT INTO reddit_posts ({column_list}) VALUES ({placeholder_block}) "
            f"ON DUPLICATE KEY UPDATE {update_clause}"
        )
    database_cursor=database_connection.cursor()
    database_cursor.executemany(sql_statement,row_values)
    database_connection.commit()

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

def update_embeddings(database_connection,posts,embeddings):
    database_cursor=database_connection.cursor()
    for post_record,embedding_vector in zip(posts,embeddings):
        embedding_json=json.dumps(embedding_vector,ensure_ascii=False)
        database_cursor.execute(
            "UPDATE reddit_posts SET embedding = ? WHERE id = ?",
            (embedding_json,post_record.get("id")),
        )
    database_connection.commit()

def update_cluster_ids(database_connection,posts,cluster_labels):
    database_cursor=database_connection.cursor()
    for post_record,cluster_label in zip(posts,cluster_labels):
        database_cursor.execute(
            "UPDATE reddit_posts SET cluster_id = ? WHERE id = ?",
            (int(cluster_label),post_record.get("id")),
        )
    database_connection.commit()

if __name__=="__main__":
    raise SystemExit(main())
