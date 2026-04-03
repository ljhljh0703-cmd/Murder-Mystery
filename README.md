# Fact Verification 기반 추리 게임

실제 범죄 사건을 기반으로 사용자가 범인(who), 수단(how), 장소(where)를 추리하면, 시스템이 multi-hop evidence retrieval과 fact verification을 통해 추리의 정오를 판정하는 CLI 프로젝트다.

이 프로젝트는 단순한 QA가 아니라, 사용자의 추리를 하나의 검증 대상 질의(query)로 보고, 외부 근거와의 일치 여부를 판단하는 Fact Verification task를 게임 형태로 재구성한 실험 시스템이다.

## Problem Definition

Fact Verification은 주어진 문장이 사실인지 여부를 단순 언어 이해가 아니라 외부 근거와의 일치 여부를 바탕으로 판단하는 작업이다.
특히 multi-hop verification은 하나의 문장만 확인하는 것이 아니라, 여러 문장과 문서에 흩어진 정보를 연결해 최종 결론을 내려야 한다.

본 프로젝트는 이 구조를 실제 사건 기반 추리 인터페이스에 적용한다.

사용 흐름은 다음과 같다.

1. 사용자는 사건 내러티브를 읽는다.
2. 범인, 수단, 장소를 추리한다.
3. 시스템은 query와 관련된 단서를 2-hop 구조로 탐색한다.
4. 탐색된 근거를 바탕으로 사용자의 추리를 항목별로 검증한다.
5. 최종적으로 정답 여부와 사건 브리핑을 제공한다.

## Dataset

본 프로젝트는 다음 데이터셋을 참고한다.

- **HOVER**
  - https://github.com/hover-nlp/hover/blob/main/data/hover/hover_train_release_v1.1.json
- **FEVER**
  - https://fever.ai/dataset/fever.html

추가로, 실제 게임 플레이용 사건은 수작업으로 구성한 5개 케이스를 사용한다.

## Methodology

### 1. Crime Example Extraction

HOVER와 FEVER는 범죄 전용 데이터셋이 아니므로, 전체 claim 중 범죄 관련 keyword가 포함된 항목만 별도로 추출한다.

**추출 결과**
- HOVER: 1,581개
- FEVER: 4,933개
- 총 6,514개 crime-related claim
이 결과를 통해 범죄 관련 fact verification 데이터가 실제로 데이터셋 내부에 존재함을 확인했다.

### 2. Case Representation

각 사건은 다음 구조를 가진다.

- `narrative`: 사건 개요
- `clues`: 증거 문서
- `entities`: 단서 내 named entity
- `claims`: 검증 가능한 참/거짓 문장
- `answer`: 정답
- `answer_aliases`: 정답 표기 변형
- `briefing`: 사건의 진실 설명

### 3. Multi-hop Retrieval

초기에는 유사도 검색 2회를 사용했지만, 문서 간 논리적 연결이 없다는 문제가 있었다.
이를 해결하기 위해 엔티티 기반 2-hop retrieval로 수정했다.

- **Hop 1**: query에서 엔티티 추출 후 관련 clue 검색
- **Hop 2**: hop1 문서의 새 엔티티를 bridge로 사용해 추가 clue 검색
즉, 단순한 semantic search가 아니라 공유 엔티티를 통한 evidence chain을 형성한다.

### 4. Verification Engine

초기 verifier는 전체 추리를 한 번에 채점했지만, 현재는 who / how / where를 개별 검증한다.

- 각 항목별 SUPPORTED / REFUTED / NOT_ENOUGH_INFO
- 부분 정답 허용
- 최종 점수 계산
LLM 기반 검증은 할루시네이션 문제가 있어, 현재는 규칙 기반 검증을 중심으로 안정성을 확보했다.

## Experimental Design

### Research Questions

1. 엔티티 기반 2-hop retrieval이 사건 단서 간 연결 구조를 형성하는가?
2. 사용자 추리를 항목별 claim으로 분리했을 때 검증 품질이 개선되는가?
3. 추리 게임 인터페이스가 fact verification 과제를 사용자 친화적으로 재구성할 수 있는가?

### Evaluation Targets

- **Retrieval**
  - hop1/hop2 결과 수
  - bridge entity 존재 여부
  - 정답 관련 엔티티 연결 성공 여부
- **Verification**
  - who/how/where 항목별 정확도
  - 부분 정답 처리 가능 여부
  - 점수 산출 일관성
- **Interaction**
  - 내러티브가 추리를 유도하는가
  - 힌트가 정답을 과도하게 노출하지 않는가
  - 브리핑이 판정 결과를 설명하는가

## Current Status

### Implemented

- 수작업 사건 5개 구성
- 엔티티 기반 2-hop retrieval
- 항목별 fact verification
- CLI 게임 루프
- HOVER/FEVER 범죄 claim 추출
- FAISS 기반 인덱싱

### Confirmed

- crime-related claims exist in HOVER/FEVER
- multi-hop retrieval structure works
- user guess can be treated as verification query
- per-field verification is more informative than one-shot grading

### Limitation

HOVER/FEVER claim을 자동으로 완성형 게임 케이스로 변환하는 것은 품질 문제로 인해 실패했다.
따라서 현재는:
- 게임 콘텐츠: 수작업 5개 사건
- 데이터셋 활용: crime claim 존재 검증 및 별도 추출
구조로 운영한다.

## Project Structure

```
.
├── data/
│   ├── cases.json
│   └── filtered_crime_claims.json
├── index/
├── src/
│   ├── data_loader.py
│   ├── dataset_loader.py
│   ├── extract_crime_claims.py
│   ├── game.py
│   ├── indexer.py
│   ├── retriever.py
│   └── verifier.py
├── build_index.py
├── download_datasets.py
├── main.py
└── requirements.txt
```

## Run

```bash
python3 build_index.py
python3 main.py
```

## Conclusion

본 프로젝트는 실제 사건 기반 추리 게임을 Fact Verification 실험 플랫폼으로 재구성한 시스템이다.

핵심은 다음과 같다.

1. 사용자 추리를 query로 본다.
2. query와 관련된 hop을 탐색한다.
3. 근거 기반으로 추리의 진위를 판정한다.
4. 정답 공개와 함께 사건 브리핑을 제공한다.

즉, 이 프로젝트는 단순한 추리 게임이 아니라, multi-hop retrieval + evidence-based verification 구조를 사용자 상호작용 형태로 구현한 응용 실험이다.
