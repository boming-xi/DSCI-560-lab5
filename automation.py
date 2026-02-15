from __future__ import annotations

import os
#limit sklearn or openmp to single thread
#otherwise kmeans + tfidf may spawn too many workers
os.environ["OMP_NUM_THREADS"]="1"
import argparse
import subprocess
import json
import time
from typing import Dict,List
import collect_store
from preprocess import preprocess_post
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import textwrap


def log(message:str)->None:
    #simple utc logger
    #helps see when each update cycle runs
    timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())
    print(f"[{timestamp}] {message}")

def parse_args()->argparse.Namespace:
    #basic runtime config
    #interval is required, others optional
    parser=argparse.ArgumentParser()
    parser.add_argument("interval_minutes",type=int)
    #data source config
    parser.add_argument("--subreddit",default="tech")
    parser.add_argument("--limit",type=int,default=200)
    parser.add_argument("--sort",default="new")
    #db config
    parser.add_argument("--sqlite-path",default="reddit.db")
    #clustering params
    parser.add_argument("--k",type=int,default=5)
    parser.add_argument("--text-field",default="clean_text")
    parser.add_argument("--max-features",type=int,default=5000)
    #not used yet but kept for later analysis
    parser.add_argument("--top-terms",type=int,default=8)
    parser.add_argument("--samples-per-cluster",type=int,default=3)
    return parser.parse_args()

def run_collect_store(args):
    #call external collector
    #basically refresh db before rebuilding model
    cmd=[
        "python",
        "collect_store.py",
        "--subreddit",args.subreddit,
        "--limit",str(args.limit),
        "--sort",args.sort,
        "--db-type","sqlite",
        "--sqlite-path",args.sqlite_path,
    ]

    subprocess.run(cmd,check=True)

def load_from_db(db_path):
    #pull everything from reddit_posts
    #no filtering here (might add later)
    database_connection=collect_store.connect_sqlite(db_path)
    database_cursor=database_connection.cursor()
    database_cursor.execute("SELECT * FROM reddit_posts")
    column_names=[column[0] for column in database_cursor.description]
    query_results=database_cursor.fetchall()
    database_connection.close()

    results=[]
    for row in query_results:
        post_directory=dict(zip(column_names,row))
        results.append(post_directory)

    return results


def build_model(posts,args):

    #collect usable text
    texts=[]
    meta=[]
    for post in posts:
        raw_text=post.get(args.text_field)
        #just guard against null
        if raw_text is None:
            raw_text=""
        text=str(raw_text).strip()
        #skip empty after cleaning
        if text:
            texts.append(text)
            meta.append(post)
    #nothing to train on
    if not texts:
        return None,None,None,None
    #k should not exceed sample size
    k=min(args.k,len(texts))
    #tfidf representation
    #english stop words for quick cleanup
    vectorizer=TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
    )

    X=vectorizer.fit_transform(texts)
    #kmeans clustering
    #random_state fixed for reproducibility
    kmeans=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels=kmeans.fit_predict(X)
    #return everything needed downstream
    return vectorizer,kmeans,(X,labels),meta

def main():
    args=parse_args()
    #convert minutes to seconds
    interval_seconds=max(1,args.interval_minutes)*60
    log("Automation started.")
    next_run=time.time()
    #model state
    #will be overwritten every cycle
    vectorizer=None
    kmeans=None
    X=None
    labels=None
    meta=None
    while True:
        #check if it's time to refresh
        if time.time()>=next_run:
            try:
                log("Running collect_store...")
                #step 1: refresh data
                run_collect_store(args)
                #step 2: reload database
                posts=load_from_db(args.sqlite_path)
                #step 3: rebuild clustering model
                result=build_model(posts,args)
                if result[0] is not None:
                    vectorizer,kmeans,bundle,post_metadata=result
                    X,labels=bundle
                    #convert sparse matrix to dense list
                    dense_embeddings=X.toarray().tolist()
                    database_connection=collect_store.connect_sqlite(
                        args.sqlite_path )
                    collect_store.update_embeddings(
                        database_connection,
                        post_metadata,
                        dense_embeddings)
                    collect_store.update_cluster_ids(
                        database_connection,
                        post_metadata,
                        labels)
                    database_connection.close()
                    log("Model rebuilt and embedding saved")
            except Exception as error:
                #avoid breaking the loop completely
                log(f"Update error: {error}")
            #schedule next run
            next_run=time.time()+interval_seconds
        remaining=int(next_run-time.time())
        if remaining<=0:
            continue
        #interactive query mode
        query=input(
            f"[next update in {remaining}s] Enter keywords/message (or 'quit'): "
        ).strip()
        if query.lower() in {"quit","exit"}:
            log("Exiting automation.")
            return
        #model not ready yet
        if vectorizer is None:
            log("Model not ready yet.")
            continue
        #predict cluster for user query
        query_vector=vectorizer.transform([query])
        cluster_id=int(
            kmeans.predict(query_vector)[0])
        log(f"Closest cluster: {cluster_id}")


if __name__=="__main__":
    main()
