"""
download_datasets.py — downloads HOVER and FEVER datasets to data/ directory.

Run once before building dataset cases:
    python3 download_datasets.py
"""

import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HOVER_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json"
HOVER_PATH = DATA_DIR / "hover_train_release_v1.1.json"

FEVER_URL = "https://fever.ai/download/fever/train.jsonl"
FEVER_PATH = DATA_DIR / "fever_train.jsonl"


def download(url: str, path: Path, label: str) -> None:
    if path.exists():
        print(f"[skip] {label} already exists: {path}")
        return
    print(f"[download] {label} → {path}")
    print(f"  URL: {url}")
    urllib.request.urlretrieve(url, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  done: {size_mb:.1f} MB")


if __name__ == "__main__":
    download(HOVER_URL, HOVER_PATH, "HOVER")
    download(FEVER_URL, FEVER_PATH, "FEVER")
    print("\nDatasets downloaded to data/")
