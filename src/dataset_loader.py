"""
dataset_loader.py — loads crime-related claims from HOVER and FEVER datasets,
and converts them into verification-ready structures.

HOVER: hover_train_release_v1.1.json
  schema: {"uid", "claim", "label": "SUPPORTED"|"NOT_SUPPORTED",
           "supporting_facts": [{"key": wiki_title, "idx": sent_id}]}

FEVER: train.jsonl
  schema: {"id", "label": "SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO",
           "claim", "evidence": [[[ann_id, ev_id, "WikiPage", sent_id]]]}

Neither dataset includes actual Wikipedia sentence text — only page title +
sentence index.  This loader:
  1. Extracts claims + labels (the verification targets)
  2. Filters for crime-related content
  3. Returns claims in a unified schema ready for Fact Verification
"""

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Crime-related keyword filter
# ---------------------------------------------------------------------------

CRIME_KEYWORDS = [
    "murder",
    "kill",
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
    "살인",
    "암살",
    "유괴",
    "납치",
    "폭탄",
    "강도",
    "사기",
    "총격",
    "수사",
    "체포",
    "유죄",
    "재판",
]

_CRIME_RE = re.compile(
    "|".join(re.escape(k) for k in CRIME_KEYWORDS),
    re.IGNORECASE,
)


def _is_crime_related(text: str) -> bool:
    return bool(_CRIME_RE.search(text))


# ---------------------------------------------------------------------------
# HOVER loader
# ---------------------------------------------------------------------------


def load_hover(path: str | Path, max_claims: int = 500) -> list[dict[str, Any]]:
    """
    Load crime-related claims from HOVER JSON.

    Returns unified claim dicts:
      {
        "source":  "hover",
        "uid":     str,
        "claim":   str,
        "label":   "SUPPORTED" | "REFUTED",          # normalized
        "evidence_refs": [{"title": str, "sent_id": int}],
        "hop_count": int,                            # number of supporting docs
      }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HOVER file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results: list[dict[str, Any]] = []
    for item in data:
        claim = item.get("claim", "")
        if not _is_crime_related(claim):
            continue

        raw_label = item.get("label", "")
        normalized_label = "SUPPORTED" if raw_label == "SUPPORTED" else "REFUTED"

        evidence_refs = [
            {"title": sf["key"], "sent_id": sf["idx"]}
            for sf in item.get("supporting_facts", [])
        ]

        results.append(
            {
                "source": "hover",
                "uid": str(item.get("uid", "")),
                "claim": claim,
                "label": normalized_label,
                "evidence_refs": evidence_refs,
                "hop_count": len(evidence_refs),
            }
        )
        if len(results) >= max_claims:
            break

    return results


# ---------------------------------------------------------------------------
# FEVER loader
# ---------------------------------------------------------------------------


def load_fever(path: str | Path, max_claims: int = 500) -> list[dict[str, Any]]:
    """
    Load crime-related claims from FEVER JSONL.

    Returns unified claim dicts:
      {
        "source":   "fever",
        "uid":      str,
        "claim":    str,
        "label":    "SUPPORTED" | "REFUTED" | "NOT_ENOUGH_INFO",
        "evidence_refs": [{"title": str, "sent_id": int}],
        "hop_count": int,
      }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FEVER file not found: {path}")

    results: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            claim = item.get("claim", "")
            if not _is_crime_related(claim):
                continue

            raw_label = item.get("label", "")
            if raw_label == "SUPPORTS":
                normalized_label = "SUPPORTED"
            elif raw_label == "REFUTES":
                normalized_label = "REFUTED"
            else:
                normalized_label = "NOT_ENOUGH_INFO"

            evidence_refs: list[dict] = []
            for ann_group in item.get("evidence", []):
                for ev in ann_group:
                    if len(ev) >= 4 and ev[2] is not None:
                        evidence_refs.append({"title": ev[2], "sent_id": int(ev[3])})

            results.append(
                {
                    "source": "fever",
                    "uid": str(item.get("id", "")),
                    "claim": claim,
                    "label": normalized_label,
                    "evidence_refs": evidence_refs,
                    "hop_count": len(evidence_refs),
                }
            )
            if len(results) >= max_claims:
                break

    return results


# ---------------------------------------------------------------------------
# Convert dataset claims → clue-format dicts (for FAISS indexing)
# ---------------------------------------------------------------------------


def claims_to_clues(
    claims: list[dict[str, Any]],
    case_id: str = "dataset",
) -> list[dict[str, Any]]:
    """
    Wrap raw HOVER/FEVER claims into clue dicts for FAISS indexing.
    The claim text becomes the indexed document.
    """
    clues: list[dict[str, Any]] = []
    for item in claims:
        hop_level = 2 if item.get("hop_count", 1) > 1 else 1
        clues.append(
            {
                "id": f"{item['source']}_{item['uid']}",
                "text": item["claim"],
                "hop": hop_level,
                "tags": [item["source"], item["label"].lower()],
                "case_id": case_id,
                "case_title": f"[{item['source'].upper()}] Dataset",
                "label": item["label"],
                "evidence_refs": item.get("evidence_refs", []),
                "entities": [],
            }
        )
    return clues


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def summarise(claims: list[dict[str, Any]]) -> str:
    total = len(claims)
    sources: dict[str, int] = {}
    labels: dict[str, int] = {}
    for c in claims:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
        labels[c["label"]] = labels.get(c["label"], 0) + 1
    src_str = ", ".join(f"{k}:{v}" for k, v in sorted(sources.items()))
    lbl_str = ", ".join(f"{k}:{v}" for k, v in sorted(labels.items()))
    return f"{total} claims ({src_str}) | labels: {lbl_str}"
