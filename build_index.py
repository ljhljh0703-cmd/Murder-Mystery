"""
build_index.py — one-time script to embed all clues and claims, write FAISS index.

Run this before starting the game:
    python3 build_index.py
"""

from pathlib import Path

from src.data_loader import load_cases, get_all_clues, get_all_claims
from src.indexer import build_index


def claims_to_index_clues(claims: list[dict]) -> list[dict]:
    """Convert case claims into clue-format for FAISS indexing."""
    clues = []
    for claim in claims:
        clue_ids = claim.get("evidence_clue_ids", [])
        clues.append(
            {
                "id": claim["id"],
                "text": claim["text"],
                "hop": len(clue_ids) if len(clue_ids) > 1 else 1,
                "tags": ["claim", claim.get("label", "").lower()],
                "case_id": claim["case_id"],
                "case_title": claim["case_title"],
                "label": claim.get("label", ""),
                "evidence_clue_ids": clue_ids,
                "entities": [],
            }
        )
    return clues


if __name__ == "__main__":
    print("케이스 데이터 로드 중...")
    cases = load_cases()

    # Clues (evidence documents)
    clues = get_all_clues(cases)
    print(f"  단서: {len(clues)}개")

    # Claims (verification targets)
    claims = get_all_claims(cases)
    claim_clues = claims_to_index_clues(claims)
    print(f"  클레임 (검증 대상): {len(claim_clues)}개")

    all_docs = clues + claim_clues
    print(f"\n총 {len(all_docs)}개 문서 인덱싱 중...")
    build_index(all_docs)
    print("인덱스 빌드 완료. 이제 python3 main.py 로 게임을 시작하세요.")
