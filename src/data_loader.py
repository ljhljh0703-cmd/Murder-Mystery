"""
data_loader.py — loads cases from cases.json
"""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"
CASES_PATH = DATA_DIR / "cases.json"


def load_cases() -> list[dict[str, Any]]:
    """Load all cases from data/cases.json."""
    with open(CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_case_by_id(case_id: str) -> dict[str, Any] | None:
    """Return a single case by its id, or None if not found."""
    for case in load_cases():
        if case["id"] == case_id:
            return case
    return None


def get_all_clues(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten all clues from all cases into a single list."""
    all_clues: list[dict[str, Any]] = []
    for case in cases:
        for clue in case.get("clues", []):
            enriched = dict(clue)
            enriched["case_id"] = case["id"]
            enriched["case_title"] = case["title"]
            all_clues.append(enriched)
    return all_clues


def get_all_claims(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten all claims from all cases into a single list."""
    all_claims: list[dict[str, Any]] = []
    for case in cases:
        for claim in case.get("claims", []):
            enriched = dict(claim)
            enriched["case_id"] = case["id"]
            enriched["case_title"] = case["title"]
            all_claims.append(enriched)
    return all_claims
