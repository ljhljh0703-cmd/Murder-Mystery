"""
retriever.py — Entity-bridged multi-hop retrieval.

진짜 multi-hop verification 구조:
  Hop 1: query → 엔티티 추출 → 엔티티를 공유하는 clue 문서 탐색
  Hop 2: hop-1 문서에서 새 엔티티 수집 → 그 엔티티를 포함하는 다른 문서 탐색
         (= 문서 간 공유 엔티티를 "다리"로 삼아 정보를 연결)

이렇게 하면 단순 유사도 검색 2번이 아니라,
"A 문서의 엔티티 X → X를 언급하는 B 문서" 형태의 실제 체인이 형성된다.

반환 구조:
  {
    "hop1": [{"clue": ..., "matched_entities": [...], "hop": 1}],
    "hop2": [{"clue": ..., "matched_entities": [...], "hop": 2,
              "bridge_entity": str, "bridged_from": clue_id}],
    "chain": [hop1_clue, hop2_clue, ...]   ← 논리적 순서의 체인
  }
"""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.indexer import load_index, EMBED_MODEL_NAME

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _embed(text: str) -> np.ndarray:
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)
    return vec.astype(np.float32)


def _extract_entities_from_query(query: str, case_clues: list[dict]) -> list[str]:
    """
    쿼리 텍스트에 언급된 엔티티를 clue.entities 목록과 매칭하여 추출.
    완전 일치 + 부분 일치 모두 허용.
    """
    query_lower = query.lower()
    matched: list[str] = []
    seen: set[str] = set()
    for clue in case_clues:
        for ent in clue.get("entities", []):
            if ent in seen:
                continue
            if ent.lower() in query_lower or query_lower in ent.lower():
                matched.append(ent)
                seen.add(ent)
    return matched


def _clues_sharing_entities(
    entities: list[str],
    candidate_clues: list[dict],
    exclude_ids: set[str],
) -> list[dict]:
    """
    주어진 엔티티 목록 중 하나 이상을 포함하는 clue를 반환.
    매칭된 엔티티 목록도 함께 반환한다.
    """
    results = []
    for clue in candidate_clues:
        if clue["id"] in exclude_ids:
            continue
        clue_entities = set(clue.get("entities", []))
        matched = [e for e in entities if e in clue_entities]
        if matched:
            results.append(
                {
                    "clue": clue,
                    "matched_entities": matched,
                    "entity_overlap": len(matched),
                }
            )
    # 엔티티 겹침 수 내림차순 정렬
    results.sort(key=lambda x: x["entity_overlap"], reverse=True)
    return results


def _semantic_fallback(
    query: str,
    case_indices: list[int],
    index,
    meta: list[dict],
    exclude_ids: set[str],
    k: int,
) -> list[dict]:
    """엔티티 매칭 결과가 없을 때 사용하는 유사도 기반 보조 검색."""
    query_vec = _embed(query)
    _, faiss_ids = index.search(query_vec, len(meta))
    results = []
    for fid in faiss_ids[0]:
        if fid not in case_indices:
            continue
        clue = meta[fid]
        if clue["id"] in exclude_ids:
            continue
        results.append(
            {
                "clue": clue,
                "matched_entities": [],
                "entity_overlap": 0,
            }
        )
        if len(results) >= k:
            break
    return results


def retrieve(
    case: dict[str, Any],
    user_query: str,
    hop1_k: int = 2,
    hop2_k: int = 2,
) -> dict[str, Any]:
    """
    Entity-bridged 2-hop retrieval.

    Parameters
    ----------
    case       : 전체 케이스 dict (clues, claims 포함)
    user_query : 플레이어의 자유 텍스트 (힌트 요청 or 추리)
    hop1_k     : hop-1에서 반환할 최대 문서 수
    hop2_k     : hop-1 문서 당 hop-2에서 탐색할 최대 문서 수

    Returns
    -------
    {
        "hop1":  [HopResult, ...],
        "hop2":  [HopResult, ...],
        "chain": [HopResult, ...],   # hop1 + hop2, 체인 순서
        "bridge_entities": [str, ...],  # hop 간 연결에 쓰인 엔티티들
    }

    HopResult = {
        "clue": dict,
        "matched_entities": [str],
        "hop": int,
        "bridge_entity": str | None,   # hop2만
        "bridged_from": str | None,    # hop2만 — 연결 출발 clue id
    }
    """
    index, meta = load_index()
    case_id = case["id"]
    case_clues: list[dict] = case.get("clues", [])

    # case에 속하는 FAISS 인덱스 번호 목록
    case_indices = [i for i, c in enumerate(meta) if c.get("case_id") == case_id]

    seen_ids: set[str] = set()
    hop1_results: list[dict[str, Any]] = []
    hop2_results: list[dict[str, Any]] = []
    bridge_entities: list[str] = []

    # ── Hop 1: 쿼리 엔티티 → 매칭 문서 ──────────────────────────────────
    query_entities = _extract_entities_from_query(user_query, case_clues)

    if query_entities:
        candidates = _clues_sharing_entities(query_entities, case_clues, seen_ids)
    else:
        # 엔티티 미매칭 → 유사도 폴백
        candidates = _semantic_fallback(
            user_query, case_indices, index, meta, seen_ids, hop1_k
        )

    for item in candidates[:hop1_k]:
        clue = item["clue"]
        seen_ids.add(clue["id"])
        hop1_results.append(
            {
                "clue": clue,
                "matched_entities": item["matched_entities"],
                "hop": 1,
                "bridge_entity": None,
                "bridged_from": None,
            }
        )

    # ── Hop 2: hop-1 문서의 새 엔티티 → 연결 문서 ───────────────────────
    for h1 in hop1_results:
        h1_entities = set(h1["clue"].get("entities", []))
        # 이미 쿼리에서 가져온 엔티티는 제외 → "새" 엔티티만 브리지로 사용
        new_entities = [e for e in h1_entities if e not in query_entities]

        if not new_entities:
            new_entities = list(h1_entities)  # 새 엔티티 없으면 전체 사용

        hop2_candidates = _clues_sharing_entities(new_entities, case_clues, seen_ids)

        if not hop2_candidates:
            hop2_candidates = _semantic_fallback(
                h1["clue"]["text"], case_indices, index, meta, seen_ids, hop2_k
            )

        added = 0
        for item in hop2_candidates:
            if added >= hop2_k:
                break
            clue = item["clue"]
            # 실제 연결 브리지 엔티티: h1과 h2가 공유하는 엔티티
            h2_entities = set(clue.get("entities", []))
            bridge = list(h1_entities & h2_entities)
            bridge_str = bridge[0] if bridge else None

            seen_ids.add(clue["id"])
            if bridge_str and bridge_str not in bridge_entities:
                bridge_entities.append(bridge_str)

            hop2_results.append(
                {
                    "clue": clue,
                    "matched_entities": item["matched_entities"],
                    "hop": 2,
                    "bridge_entity": bridge_str,
                    "bridged_from": h1["clue"]["id"],
                }
            )
            added += 1

    chain = hop1_results + hop2_results
    return {
        "hop1": hop1_results,
        "hop2": hop2_results,
        "chain": chain,
        "bridge_entities": bridge_entities,
    }
