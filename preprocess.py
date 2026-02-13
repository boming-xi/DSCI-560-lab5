#!/usr/bin/env python3
"""
Preprocess Reddit posts for storage and analysis.

Features:
- Remove HTML tags, URLs, and special characters.
- Filter or flag promoted/advertisement content.
- Convert timestamps to ISO 8601 (UTC).
- Mask usernames for privacy.
- Extract keywords and infer a simple topic label.
- Optional OCR on local image files.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


IRRELEVANT_PATTERNS = [
    re.compile(r"\bpromoted\b", re.IGNORECASE),
    re.compile(r"\bsponsored\b", re.IGNORECASE),
    re.compile(r"\badvertis(e|ement|ing)\b", re.IGNORECASE),
    re.compile(r"\bsubscribe\b", re.IGNORECASE),
    re.compile(r"\bclick here\b", re.IGNORECASE),
    re.compile(r"\bbuy now\b", re.IGNORECASE),
    re.compile(r"\bdeal\b", re.IGNORECASE),
    re.compile(r"\bdiscount\b", re.IGNORECASE),
    re.compile(r"\bpromo code\b", re.IGNORECASE),
    re.compile(r"\bfree trial\b", re.IGNORECASE),
]


TOPIC_KEYWORDS = {
    "security": {"security", "vuln", "breach", "malware", "ransomware", "phish", "cve"},
    "ai": {"ai", "ml", "model", "llm", "neural", "training", "inference"},
    "hardware": {"cpu", "gpu", "chip", "hardware", "server", "laptop"},
    "software": {"software", "bug", "release", "update", "patch", "version"},
    "business": {"startup", "funding", "ipo", "acquisition", "earnings", "revenue"},
}


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")


def strip_html(text: str) -> str:
    return HTML_TAG_PATTERN.sub(" ", text)


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = strip_html(text)
    text = remove_urls(text)
    text = NON_WORD_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_irrelevant(text: str) -> bool:
    return any(pattern.search(text) for pattern in IRRELEVANT_PATTERNS)


def mask_username(username: Optional[str]) -> Optional[str]:
    if not username:
        return None
    salt = os.environ.get("USERNAME_SALT", "dsci560")
    digest = hashlib.sha256(f"{salt}:{username}".encode("utf-8")).hexdigest()
    return f"user_{digest[:12]}"


def to_iso_utc(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t and t not in STOPWORDS and len(t) > 2]


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    sorted_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in sorted_tokens[:max_keywords]]


def infer_topic(tokens: Iterable[str]) -> str:
    scores: Dict[str, int] = {}
    token_set = set(tokens)
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = len(token_set.intersection(keywords))
    best_topic, best_score = "other", 0
    for topic, score in scores.items():
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic


def try_import_ocr() -> Tuple[Optional[object], Optional[object]]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        return pytesseract, Image
    except Exception:
        return None, None


def extract_ocr_text(image_paths: Iterable[str], ocr_lang: str = "eng") -> str:
    pytesseract, Image = try_import_ocr()
    if pytesseract is None or Image is None:
        return ""

    texts: List[str] = []
    for path in image_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as img:
                text = pytesseract.image_to_string(img, lang=ocr_lang)
                if text:
                    texts.append(text)
        except Exception:
            continue
    return " ".join(texts).strip()


def preprocess_post(
    post: Dict,
    max_keywords: int = 8,
    drop_irrelevant: bool = False,
    enable_ocr: bool = False,
    ocr_lang: str = "eng",
) -> Optional[Dict]:
    title = post.get("title") or ""
    body = post.get("selftext") or post.get("body") or ""
    author = post.get("author")
    created_utc = post.get("created_utc") or post.get("created")

    image_paths = post.get("image_paths") or []
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    ocr_text = extract_ocr_text(image_paths, ocr_lang=ocr_lang) if enable_ocr else ""

    raw_text = f"{title} {body}".strip()
    if not raw_text and ocr_text:
        raw_text = ocr_text

    cleaned_text = normalize_text(raw_text)
    cleaned_ocr = normalize_text(ocr_text) if ocr_text else ""

    combined_for_keywords = " ".join([cleaned_text, cleaned_ocr]).strip()

    if is_irrelevant(combined_for_keywords):
        if drop_irrelevant:
            return None
        irrelevant_flag = True
    else:
        irrelevant_flag = False

    keywords = extract_keywords(combined_for_keywords, max_keywords=max_keywords)
    topic = infer_topic(keywords)

    return {
        "id": post.get("id"),
        "subreddit": post.get("subreddit"),
        "title": title,
        "body": body,
        "clean_text": cleaned_text,
        "ocr_text": cleaned_ocr or None,
        "created_utc": created_utc,
        "created_iso_utc": to_iso_utc(created_utc),
        "author_masked": mask_username(author),
        "keywords": keywords,
        "topic": topic,
        "is_irrelevant": irrelevant_flag,
        "raw": post,
    }


def load_posts(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def save_posts(path: str, posts: List[Dict], json_lines: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        if json_lines:
            for post in posts:
                handle.write(json.dumps(post, ensure_ascii=False) + "\n")
        else:
            json.dump(posts, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Reddit posts.")
    parser.add_argument("--input", required=True, help="Path to input JSON or JSONL.")
    parser.add_argument("--output", required=True, help="Path to output JSON or JSONL.")
    parser.add_argument("--jsonl", action="store_true", help="Write output as JSON Lines.")
    parser.add_argument("--max-keywords", type=int, default=8, help="Max keywords per post.")
    parser.add_argument("--drop-irrelevant", action="store_true", help="Drop ads/promoted posts.")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR on image_paths.")
    parser.add_argument("--ocr-lang", default="eng", help="OCR language (default: eng).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    posts = load_posts(args.input)
    processed: List[Dict] = []
    for post in posts:
        item = preprocess_post(
            post,
            max_keywords=args.max_keywords,
            drop_irrelevant=args.drop_irrelevant,
            enable_ocr=args.enable_ocr,
            ocr_lang=args.ocr_lang,
        )
        if item is not None:
            processed.append(item)
    save_posts(args.output, processed, json_lines=args.jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
