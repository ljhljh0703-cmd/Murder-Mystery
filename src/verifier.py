"""
verifier.py — Rule-based Fact Verification engine.

LLM(hallucination-prone) 대신 증거 문서의 실제 텍스트와 answer_aliases를
직접 비교하여 who/how/where를 개별 검증한다.

검증 로직:
  1. 플레이어 입력을 answer_aliases와 매칭 (부분 일치 허용)
  2. 매칭되지 않으면 증거 문서 텍스트에서 키워드 검색
  3. 증거 문서에 정답 키워드가 있으면 SUPPORTED, 없으면 REFUTED
"""

import re
from typing import Any

FIELD_LABELS = {
    "who": "범인",
    "how": "수단",
    "where": "장소",
}


def _normalize(text: str) -> str:
    """텍스트 정규화: 소문자, 공백/특수문자 제거."""
    return re.sub(r"[\s\-\.\(\)\[\]']", "", text).lower()


def _alias_match(user_input: str, aliases: list[str]) -> bool:
    """사용자 입력이 alias 목록 중 하나와 일치하는지 확인."""
    user_norm = _normalize(user_input)
    for alias in aliases:
        alias_norm = _normalize(alias)
        if alias_norm in user_norm or user_norm in alias_norm:
            return True
    return False


def _evidence_supports(user_input: str, evidence_chain: list[dict[str, Any]]) -> bool:
    """
    증거 문서 텍스트에 사용자 입력의 핵심 키워드가 포함되어 있는지 확인.
    """
    user_norm = _normalize(user_input)
    # 3글자 이상의 단어만 추출
    keywords = [w for w in re.findall(r"[a-zA-Z가-힣]+", user_input) if len(w) >= 2]

    if not keywords:
        return False

    for item in evidence_chain:
        clue = item.get("clue", item)
        text_norm = _normalize(clue.get("text", ""))
        for kw in keywords:
            kw_norm = _normalize(kw)
            if kw_norm in text_norm:
                return True
    return False


def verify_claim(
    case: dict[str, Any],
    guess: dict[str, str],
    evidence_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    플레이어의 who/how/where 추리를 개별적으로 검증.

    Returns
    -------
    {
        "fields": {
            "who": {"label": "SUPPORTED", "reasoning": "...", "confidence": 100},
            "how": {"label": "REFUTED", "reasoning": "...", "confidence": 80},
            "where": {"label": "SUPPORTED", "reasoning": "...", "confidence": 90},
        },
        "score": 66,
        "correct_count": 2,
        "total_count": 3,
    }
    """
    aliases = case.get("answer_aliases", {})
    answer = case.get("answer", {})
    correct_answer = case.get("answer", {})

    fields_result: dict[str, dict[str, Any]] = {}
    correct_count = 0

    for field in ["who", "how", "where"]:
        user_val = guess.get(field, "").strip()
        field_aliases = aliases.get(field, [])
        correct_val = correct_answer.get(field, "")

        if not user_val or user_val.lower() in ("모름", "모르겠어", "unknown"):
            fields_result[field] = {
                "label": "NOT_ENOUGH_INFO",
                "reasoning": "입력 없음",
                "confidence": 0,
            }
            continue

        # 1차: alias 직접 매칭
        if _alias_match(user_val, field_aliases):
            fields_result[field] = {
                "label": "SUPPORTED",
                "reasoning": f"{FIELD_LABELS[field]}이(가) 공식 기록과 일치합니다.",
                "confidence": 100,
            }
            correct_count += 1
            continue

        # 2차: 증거 문서에서 키워드 검색
        if _evidence_supports(user_val, evidence_chain):
            # 증거에 키워드가 있지만 alias 매칭 실패 → 부분 정답
            fields_result[field] = {
                "label": "SUPPORTED",
                "reasoning": f"증거에 관련 키워드가 있으나, 공식 기록과 완전히 일치하지는 않습니다. 정답: {correct_val}",
                "confidence": 60,
            }
            correct_count += 1
            continue

        # 3차: 불일치
        fields_result[field] = {
            "label": "REFUTED",
            "reasoning": f"증거와 일치하지 않습니다. 정답: {correct_val}",
            "confidence": 90,
        }

    score = int(correct_count / 3 * 100)

    return {
        "fields": fields_result,
        "score": score,
        "correct_count": correct_count,
        "total_count": 3,
    }
