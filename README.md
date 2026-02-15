# DSCI-560 Lab 5 – Data Collection, Storage, and Preprocessing (r/tech)

This repository contains **data collection**, **storage**, and **preprocessing** for Reddit posts.  
The chosen topic/subreddit is: `r/tech` (https://www.reddit.com/r/tech/).

---
## 1) Environment Setup

Requires Python 3.8+.

Install required packages:
```bash
pip install requests scikit-learn sentence-transformers matplotlib
```

Optional (for MySQL storage):
```bash
pip install mysql-connector-python
```

Optional (for OCR on images):
```bash
pip install pytesseract pillow
```

> **Notes:**
> - `sentence-transformers` automatically installs PyTorch if not already available.
> - The clustering module uses Sentence-BERT (`all-MiniLM-L6-v2`) for semantic embeddings.
> - `matplotlib` is required for PCA visualization and cluster plots.

---
## 2) Data Collection + Storage (Recommended)

Use `collect_store.py` to **fetch from r/tech**, preprocess, and store in a database.

### SQLite (default)
```bash
python collect_store.py --subreddit tech --limit 200 --db-type sqlite --sqlite-path reddit.db --raw-output raw.json
```

### MySQL
Set credentials (env or CLI):
```bash
export MYSQL_HOST="localhost"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="your_password"
export MYSQL_DATABASE="reddit_db"
```

Then run:
```bash
python collect_store.py --subreddit tech --limit 200 --db-type mysql --raw-output raw.json
```

### Large requests (5000+ posts)
The script automatically paginates multiple API calls.  
You can slow it down if needed:
```bash
python collect_store.py --subreddit tech --limit 5000 --pause 1.5 --max-per-request 100
```


## 3) Run Preprocessing

Basic run:
```bash
python preprocess.py --input raw.json --output clean.json
```

Write JSON Lines output:
```bash
python preprocess.py --input raw.jsonl --output clean.jsonl --jsonl
```

Drop irrelevant/advertisement posts:
```bash
python preprocess.py --input raw.json --output clean.json --drop-irrelevant
```

Enable OCR on images:
```bash
python preprocess.py --input raw.json --output clean.json --enable-ocr
```

Limit keywords per post:
```bash
python preprocess.py --input raw.json --output clean.json --max-keywords 8
```

## 4) Forum Analysis & Clustering (Part 4)

This step clusters the cleaned messages and shows keywords, statistics, and visualizations per cluster.

**Embedding method:**
- **Sentence-BERT** (`all-MiniLM-L6-v2`) for semantic document embeddings.
- **TF-IDF** is retained for keyword extraction and interpretability.

**Clustering method:**
- Automatic K selection using silhouette score (range: 2 to `--k`).
- K-Means clustering applied to semantic embeddings.

Run clustering on the cleaned data:
```bash
python cluster_analysis.py --input clean.json --save-plot
```

Output files (under `cluster_output/` by default):
- `cluster_report.json` / `cluster_report.txt` — cluster keywords, silhouette score, and sample messages
- `cluster_assignments.csv` — document → cluster mapping
- `cluster_plot.png` — PCA visualization of semantic clusters (if `--save-plot`)
- `cluster_size_distribution.png` — bar chart of cluster sizes (if `--save-plot`)

Common options:
```bash
python cluster_analysis.py --input clean.json --k 8 --top-terms 10 --samples-per-cluster 3 --save-plot
```

> **Notes:**
> - Clustering is performed using Sentence-BERT semantic embeddings.
> - The optimal number of clusters (K) is automatically selected using silhouette score.
> - If the dataset is small, the script dynamically limits the maximum K value.
> - If your dataset is very small (< 10 posts), increase post count or reduce `--k`.

---
## 5) Automation (Part 5)

This script periodically runs data collection → preprocessing → storage, and
accepts query text between updates to find the closest cluster.

Run every 5 minutes (SQLite example):
```bash
python automation.py 5 --subreddit tech --limit 200 --db-type sqlite --sqlite-path reddit.db --save-query-plot
```

Run every 5 minutes (MySQL example):
```bash
python automation.py 5 --subreddit tech --limit 200 --db-type mysql --save-query-plot
```

While waiting for the next update, enter a message or keywords in the prompt.
The script will:
- find the closest cluster
- show top messages
- save a keyword bar chart in `cluster_output/` (when `--save-query-plot` is set)

---
## 6) Reminder About Topic

The chosen subreddit/topic for this assignment is:
**r/tech** — https://www.reddit.com/r/tech/
