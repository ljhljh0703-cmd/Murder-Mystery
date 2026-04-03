"""
dataset_to_cases.py — converts HOVER/FEVER crime claims into direct verification challenges.

Instead of using LLM to generate game cases (which fails with llama3:8b),
this script creates simple verification challenges directly from dataset claims.

Each challenge:
  - Shows the claim text to the user
  - User answers: SUPPORTED (사실) or REFUTED (거짓)
  - System retrieves multi-hop evidence and verifies

Output: data/dataset_verifications.json — compatible with the game system.

Usage:
    python3 src/dataset_to_cases.py
"""

import json
import re
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data"
HOVER_PATH = DATA_DIR / "hover_train_release_v1.1.json"
FEVER_PATH = DATA_DIR / "fever_train.jsonl"
OUTPUT_PATH = DATA_DIR / "dataset_cases.json"

# Number of verification challenges to create
MAX_CHALLENGES = 5

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
# Load claims
# ---------------------------------------------------------------------------


def load_all_claims() -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []

    if HOVER_PATH.exists():
        with open(HOVER_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            claim = item.get("claim", "")
            if _is_crime_related(claim):
                claims.append(
                    {
                        "source": "hover",
                        "uid": str(item.get("uid", "")),
                        "claim": claim,
                        "label": item.get("label", ""),
                        "supporting_facts": item.get("supporting_facts", []),
                    }
                )
        print(
            f"  HOVER: {len([c for c in claims if c['source'] == 'hover'])} crime claims"
        )

    if FEVER_PATH.exists():
        with open(FEVER_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                claim = item.get("claim", "")
                if _is_crime_related(claim):
                    claims.append(
                        {
                            "source": "fever",
                            "uid": str(item.get("id", "")),
                            "claim": claim,
                            "label": item.get("label", ""),
                            "evidence": item.get("evidence", []),
                        }
                    )
        print(
            f"  FEVER: {len([c for c in claims if c['source'] == 'fever'])} crime claims"
        )

    return claims


# ---------------------------------------------------------------------------
# Rule-based clustering
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "To",
    "For",
    "Of",
    "And",
    "But",
    "Not",
    "Was",
    "Were",
    "Is",
    "Are",
    "It",
    "That",
    "This",
    "With",
    "By",
    "From",
    "He",
    "She",
    "They",
    "His",
    "Her",
    "Their",
    "Its",
    "Who",
    "Which",
    "What",
    "When",
    "Where",
    "How",
    "Why",
    "Did",
    "Does",
    "Do",
    "Has",
    "Have",
    "Had",
    "Been",
    "Being",
    "Would",
    "Could",
    "Should",
    "Will",
    "Can",
    "May",
    "Might",
    "Must",
    "Shall",
    "After",
    "Before",
    "During",
    "Between",
    "Through",
    "Under",
    "Over",
    "Above",
    "Below",
    "Up",
    "Down",
    "Out",
    "Off",
    "Into",
    "Onto",
    "Upon",
    "Within",
    "Also",
    "Very",
    "Just",
    "More",
    "Most",
    "Some",
    "Any",
    "All",
    "Each",
    "Every",
    "Both",
    "Few",
    "Many",
    "Much",
    "Such",
    "Only",
    "Other",
    "Another",
    "Own",
    "Same",
    "So",
    "Than",
    "Too",
    "Yet",
    "However",
    "Although",
    "Though",
    "While",
    "Whereas",
    "Despite",
    "According",
    "About",
    "Against",
    "Among",
    "Around",
    "Since",
    "Until",
    "Unless",
    "Whether",
    "Without",
    "SUPPORTED",
    "NOT_SUPPORTED",
    "SUPPORTS",
    "REFUTES",
    "REFUTED",
    "NOT_ENOUGH_INFO",
    "hover",
    "fever",
    "I",
    "Me",
    "We",
    "You",
    "One",
    "No",
    "Yes",
}


def _extract_entities(text: str) -> set[str]:
    words = re.findall(r"[A-Z][a-zA-Z']+", text)
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def _rule_based_cluster(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claim_entities = [(c["uid"], _extract_entities(c["claim"])) for c in claims]

    parent: dict[str, str] = {uid: uid for uid, _ in claim_entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(claim_entities)):
        for j in range(i + 1, len(claim_entities)):
            uid_i, ent_i = claim_entities[i]
            uid_j, ent_j = claim_entities[j]
            if len(ent_i & ent_j) >= 2:
                union(uid_i, uid_j)

    groups_map: dict[str, list[str]] = {}
    for uid, _ in claim_entities:
        groups_map.setdefault(find(uid), []).append(uid)

    groups = []
    for root, uids in groups_map.items():
        if len(uids) < 3:
            continue
        group_claims = [c for c in claims if c["uid"] in uids]
        if not any(_is_strict_crime(c["claim"]) for c in group_claims):
            continue
        groups.append(
            {
                "group_id": f"group_{len(groups) + 1:03d}",
                "case_name": group_claims[0]["claim"][:80] + "...",
                "claim_uids": uids,
                "claims": group_claims,
            }
        )

    groups.sort(key=lambda g: len(g["claim_uids"]), reverse=True)
    print(f"  Found {len(groups)} crime-related case groups")
    return groups


# ---------------------------------------------------------------------------
# Convert group → game case (no LLM, direct mapping)
# ---------------------------------------------------------------------------


def group_to_case(group: dict[str, Any], case_index: int) -> dict[str, Any]:
    """
    Convert a claim group into a game case WITHOUT LLM.

    Strategy:
    - Use the most informative SUPPORTED claim as the "narrative" basis
    - Use all claims as verification targets
    - Extract entities from claims for hop chain
    - Set answer from the most common entities in SUPPORTED claims
    """
    group_claims = group["claims"]
    supported = [c for c in group_claims if c["label"] in ("SUPPORTED", "SUPPORTS")]
    refuted = [
        c for c in group_claims if c["label"] in ("REFUTED", "NOT_SUPPORTED", "REFUTES")
    ]

    # Generate case name from shared entities
    all_entities: set[str] = set()
    for c in group_claims:
        all_entities |= _extract_entities(c["claim"])
    case_name = ", ".join(sorted(all_entities)[:3]) if all_entities else "Unknown Case"

    # Build narrative from the first supported claim (trimmed to not give away answer)
    narrative_claim = supported[0]["claim"] if supported else group_claims[0]["claim"]
    # Truncate to create mystery — remove the key entity if possible
    narrative = (
        f"The following claims relate to a case involving {case_name}. "
        f"Review the evidence and determine which claims are true."
    )

    # Build clues from supported claims
    clues = []
    for i, c in enumerate(supported[:5], 1):
        entities = list(_extract_entities(c["claim"]))
        clues.append(
            {
                "id": f"clue_{case_index:03d}_{i:02d}",
                "text": c["claim"],
                "hop": 1 if i <= 2 else 2,
                "tags": [c["source"], "supported"],
                "entities": entities[:5],
                "source_claims": [c["uid"]],
            }
        )

    # Build claims array (verification targets)
    claims_arr = []
    for i, c in enumerate(group_claims[:6], 1):
        label = "SUPPORTED" if c["label"] in ("SUPPORTED", "SUPPORTS") else "REFUTED"
        evidence_ids = [cl["id"] for cl in clues[:2]]
        claims_arr.append(
            {
                "id": f"claim_{case_index:03d}_{chr(96 + i)}",
                "text": c["claim"],
                "label": label,
                "evidence_clue_ids": evidence_ids,
            }
        )

    # Build answer from most common entities in supported claims
    entity_freq: dict[str, int] = {}
    for c in supported:
        for e in _extract_entities(c["claim"]):
            entity_freq[e] = entity_freq.get(e, 0) + 1
    top_entities = sorted(entity_freq.items(), key=lambda x: -x[1])[:3]
    who = top_entities[0][0] if top_entities else "Unknown"
    how = "Unknown"
    where = top_entities[1][0] if len(top_entities) > 1 else "Unknown"

    # Briefing
    briefing_claims = [c["claim"] for c in supported[:3]]
    briefing = "Supported claims: " + " | ".join(briefing_claims)

    return {
        "id": f"dataset_case_{case_index:03d}",
        "title": f"Dataset Case: {case_name[:50]}",
        "subtitle": f"{group['case_name'][:60]}",
        "difficulty": "중간",
        "narrative": narrative,
        "clues": clues,
        "claims": claims_arr,
        "answer": {
            "who": who,
            "mastermind": None,
            "how": how,
            "where": where,
            "motive": "Unknown",
        },
        "answer_aliases": {
            "who": [who],
            "how": [how],
            "where": [where],
        },
        "briefing": briefing,
        "source_claims": group["claim_uids"],
        "is_dataset_case": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Dataset → Verification Challenges (no LLM)")
    print(f"MAX_CHALLENGES = {MAX_CHALLENGES}")
    print("=" * 60)

    print("\n[Step 1] Loading crime claims...")
    all_claims = load_all_claims()
    if not all_claims:
        print("  No crime claims found. Run download_datasets.py first.")
        exit(1)
    print(f"  Total: {len(all_claims)} crime claims")

    print("\n[Step 2] Clustering claims...")
    groups = _rule_based_cluster(all_claims)
    if not groups:
        print("  No groups found. Exiting.")
        exit(1)

    if len(groups) > MAX_CHALLENGES:
        print(f"  Limiting to {MAX_CHALLENGES} groups (of {len(groups)} found)")
        groups = groups[:MAX_CHALLENGES]

    print(f"\n[Step 3] Converting {len(groups)} groups to game cases...")
    cases = []
    for i, group in enumerate(groups, 1):
        case = group_to_case(group, i)
        cases.append(case)
        print(f"  [{i}] {case['title'][:60]} ({len(group['claim_uids'])} claims)")

    print(f"\n[Step 4] Saving {len(cases)} cases to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"  Done! {len(cases)} dataset cases created.")
    print("\nNext: run  python3 build_index.py  to index the new cases.")
