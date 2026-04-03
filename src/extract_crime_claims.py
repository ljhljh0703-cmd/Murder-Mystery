"""
extract_crime_claims.py — extracts crime-related claims from HOVER/FEVER datasets
and saves them as a filtered dataset.

This does NOT generate game cases. It only:
  1. Loads HOVER/FEVER datasets
  2. Filters for crime-related claims using keyword matching
  3. Saves the filtered claims as a new dataset file

Output: data/filtered_crime_claims.json

Usage:
    python3 src/extract_crime_claims.py
"""

import json
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"
HOVER_PATH = DATA_DIR / "hover_train_release_v1.1.json"
FEVER_PATH = DATA_DIR / "fever_train.jsonl"
OUTPUT_PATH = DATA_DIR / "filtered_crime_claims.json"

# ---------------------------------------------------------------------------
# Crime keyword filters
# ---------------------------------------------------------------------------

CRIME_KEYWORDS_STRICT = [
    "murder",
    "killed",
    "assassin",
    "homicide",
    "manslaughter",
    "kidnap",
    "abduct",
    "ransom",
    "robbery",
    "theft",
    "fraud",
    "shoot",
    "shot",
    "gun",
    "bomb",
    "arson",
    "poison",
    "assassination",
    "massacre",
    "살인",
    "암살",
    "유괴",
    "납치",
    "폭탄",
    "강도",
    "사기",
    "총격",
]

CRIME_KEYWORDS_BROAD = [
    "suspect",
    "convict",
    "verdict",
    "trial",
    "sentence",
    "FBI",
    "CIA",
    "detective",
    "police",
    "arrest",
    "prison",
    "수사",
    "체포",
    "유죄",
    "재판",
]

_ALL_CRIME_RE = re.compile(
    "|".join(re.escape(k) for k in CRIME_KEYWORDS_STRICT + CRIME_KEYWORDS_BROAD),
    re.IGNORECASE,
)

_STRICT_CRIME_RE = re.compile(
    "|".join(re.escape(k) for k in CRIME_KEYWORDS_STRICT),
    re.IGNORECASE,
)


def _is_crime_related(text: str) -> bool:
    return bool(_ALL_CRIME_RE.search(text))


def _is_strict_crime(text: str) -> bool:
    return bool(_STRICT_CRIME_RE.search(text))


# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------


def load_and_filter() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    hover_count = 0
    fever_count = 0

    if HOVER_PATH.exists():
        with open(HOVER_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            claim = item.get("claim", "")
            if _is_crime_related(claim):
                hover_count += 1
                claims.append(
                    {
                        "source": "hover",
                        "uid": str(item.get("uid", "")),
                        "claim": claim,
                        "label": item.get("label", ""),
                        "is_strict_crime": _is_strict_crime(claim),
                        "supporting_facts": item.get("supporting_facts", []),
                    }
                )
        print(f"  HOVER total: {len(data)} | crime-related: {hover_count}")

    if FEVER_PATH.exists():
        fever_total = 0
        with open(FEVER_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                fever_total += 1
                claim = item.get("claim", "")
                if _is_crime_related(claim):
                    fever_count += 1
                    claims.append(
                        {
                            "source": "fever",
                            "uid": str(item.get("id", "")),
                            "claim": claim,
                            "label": item.get("label", ""),
                            "is_strict_crime": _is_strict_crime(claim),
                            "evidence": item.get("evidence", []),
                        }
                    )
        print(f"  FEVER total: {fever_total} | crime-related: {fever_count}")

    return claims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Extract Crime-Related Claims from HOVER/FEVER")
    print("=" * 60)

    print("\nLoading and filtering...")
    claims = load_and_filter()

    if not claims:
        print("\nNo crime-related claims found.")
        exit(1)

    # Statistics
    strict_count = sum(1 for c in claims if c["is_strict_crime"])
    hover_count = sum(1 for c in claims if c["source"] == "hover")
    fever_count = sum(1 for c in claims if c["source"] == "fever")

    label_dist: dict[str, int] = {}
    for c in claims:
        label_dist[c["label"]] = label_dist.get(c["label"], 0) + 1

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total crime-related claims: {len(claims)}")
    print(f"    Strict crime keywords:    {strict_count}")
    print(f"    Broad crime keywords:     {len(claims) - strict_count}")
    print(f"  By source:")
    print(f"    HOVER: {hover_count}")
    print(f"    FEVER: {fever_count}")
    print(f"  By label:")
    for label, count in sorted(label_dist.items()):
        print(f"    {label}: {count}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(claims, f, ensure_ascii=False, indent=2)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Done! {len(claims)} claims saved ({size_mb:.1f} MB)")

    # Show sample
    print(f"\nSample claims (first 5):")
    for c in claims[:5]:
        strict_tag = " [STRICT]" if c["is_strict_crime"] else " [BROAD]"
        print(f"  [{c['source']}] [{c['label']}]{strict_tag}")
        print(f"    {c['claim'][:100]}...")
        print()
