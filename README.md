# 연금 Agent — RAG 데이터셋 구축

PDF·DOCX 문서를 **RAG 검색용 JSONL**로 변환하는 파이프라인입니다.

## 왜 txt/엑셀이 아니라 JSONL인가

| | 통합 txt | 엑셀 | **JSONL** |
|---|---|---|---|
| 출처(펀드코드·페이지) 추적 | ✗ | △ | ✓ |
| 수천 자 본문 안전 저장 | ✓ | ✗ (셀 제한·줄바꿈 깨짐) | ✓ |
| 표 구조 보존 | ✗ | △ | ✓ (마크다운) |
| 스트리밍 처리 / 증분 추가 | △ | ✗ | ✓ |
| 벡터 DB 적재 | 재가공 필요 | 재가공 필요 | 바로 가능 |

연금·펀드 도메인에서는 "이 수수료율이 **어느 펀드 몇 페이지**에서 나왔는지"를 답변에
달 수 있어야 합니다. 통합 txt는 그 정보를 구조적으로 잃어버립니다.

## 설치

```bash
python -m venv venv && source venv/bin/activate
pip install pdfplumber pypdfium2 python-docx python-dotenv requests
cp .env.example .env      # OCR 키 입력 (선택)
```

`pdf2image` / `poppler`는 필요 없습니다. 렌더링은 `pypdfium2`로 처리합니다.

## 실행

먼저 5개만 돌려 결과를 눈으로 확인합니다.

```bash
python build_dataset.py --input . --out ./dataset --limit 5
```

괜찮으면 전체를 돌립니다.

```bash
python build_dataset.py --input . --out ./dataset --workers 4
```

중단됐다면 같은 명령을 다시 실행하세요 — 끝난 문서는 자동으로 건너뜁니다.
추출 로직을 고쳤다면 `--force`를 붙입니다.

> zsh에서는 `#` 주석이 인식되지 않아 명령 인자로 넘어갑니다.
> 이 문서의 명령은 전부 주석 없이 적어두었습니다.

## 결과물

```
dataset/
├── chunks.jsonl      ← 최종 산출물. 1줄 = 청크 1개
├── manifest.csv      ← 문서별 요약. 품질 점검은 여기부터
└── docs/*.json       ← 문서별 중간 결과 (재개용 캐시)
```

### chunks.jsonl 스키마

```json
{
  "chunk_id":    "KR5157450090_p23_0071",
  "doc_id":      "KR5157450090",
  "doc_type":    "투자설명서",
  "fund_code":   "KR5157450090",
  "fund_name":   "마이다스 거북이 90 증권 자투자신탁 1호(주식)",
  "kofia_code":  null,
  "base_date":   "2025-10-01",
  "page":        23,
  "section":     "(2) 종류별 수수료 및 보수에 관한 사항",
  "content_type":"table",          // "text" | "table"
  "is_ocr":      false,
  "text":        "| 구분 | 선취 판매수수료 | ... |",
  "n_chars":     742,
  "source_path": "투자설명서/KR5157450090/R2_KR5157450090.pdf"
}
```

`content_type: "table"`인 청크는 마크다운 표입니다. 수수료율·보수율·세율처럼
**행과 열의 대응이 의미인 데이터**가 여기 들어갑니다. 임베딩 시 잘라내지 마세요.

## manifest.csv 읽는 법

`status` 열만 보면 됩니다.

| status | 의미 | 조치 |
|---|---|---|
| `OK` | 정상 | — |
| `NEEDS_OCR` | 텍스트 레이어 없음 = 스캔 PDF | `.env`에 OCR 키 넣고 재실행 |
| `LOW_YIELD` | 페이지 대비 추출량이 적음 | 해당 문서 육안 확인 |
| `ERROR` | 파싱 실패 | `error` 열 확인 |

`fund_name` / `base_date`가 빈 투자설명서가 있다면 운용사별 서식 차이입니다.
`build_dataset.py`의 `RE_FUND_NAME`, `RE_BASE_DATE`에 패턴을 추가하세요.

## 2단계: 임베딩 전 정리 (`finalize_dataset.py`)

```bash
python finalize_dataset.py --in ./dataset/chunks.jsonl \
                           --out ./dataset/chunks_final.jsonl \
                           --exclude ncp_data_test
```

투자설명서 100개는 자본시장법이 요구하는 **동일한 법정 문구**를 공유합니다
(수익자총회·손해배상책임·투자신탁 해지 등). 그대로 임베딩하면 —

- 같은 문장을 최대 24번 임베딩 → 비용 낭비
- "수익자총회가 뭐야?"에 **똑같은 청크 24개**가 상위를 채움 → 검색 품질 붕괴

완전히 동일한 청크는 하나만 남기고 `doc_ids`에 공유 문서를 기록합니다.

```json
{ "chunk_id":"COMMON_a3f9…", "scope":"공통", "n_docs":24,
  "doc_ids":["KR5109…","KR5113…", …],
  "fund_code":null, "fund_name":null }
```

`scope: "공통"` 청크는 `fund_code`가 `null`입니다. 특정 펀드의 것이 아니기 때문에
임의로 하나를 골라 붙이면 **잘못된 출처를 답변에 달게 됩니다.**
대신 검색 시 아래처럼 거르세요.

```
filter:  scope == "공통"  OR  doc_ids contains "KR5157450090"
```

> ⚠️ **유사 중복은 병합하지 않습니다.** 숫자만 다른 두 청크는
> "같은 서식의 다른 수수료율"일 수 있습니다. 0.72%와 1.30%를 하나로 합치는
> 순간 데이터셋이 거짓말을 시작합니다. 완전 일치만 병합합니다.

## 3단계: 임베딩 + Chroma 인덱싱 (`embed_and_index.py`)

```bash
pip install chromadb rank-bm25
```

**0단계 — API 없이 파이프라인만 점검**

```bash
python embed_and_index.py --provider dummy --limit 200
```

**1단계 — API 계약 확인.** 호출 1회로 응답 구조와 차원 수를 봅니다.

```bash
python embed_and_index.py --selftest
```

**2단계 — 소량 검증 후 전체**

```bash
python embed_and_index.py --limit 300
```

```bash
python embed_and_index.py --workers 8
```

`--selftest`를 **반드시 먼저** 돌리세요. 1만 건을 다 돌린 뒤에 응답 형식이
틀린 걸 아는 건 최악입니다. CLOVA Studio 임베딩 엔드포인트는 테스트 앱/서비스 앱에 따라
경로가 다르고 인증 헤더 방식도 두 가지라, `.env`에서 맞춰야 합니다.

```ini
CLOVA_API_KEY=nv-....................
CLOVA_EMBED_URL=https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2
CLOVA_EMBED_AUTH=bearer
CLOVA_APIGW_KEY=
```

`CLOVA_EMBED_AUTH`는 `bearer`로 안 되면 `ncp`로 바꾸세요.
`CLOVA_APIGW_KEY`는 `ncp` 방식일 때만 필요합니다.

임베딩 결과는 `dataset/emb_cache.sqlite`에 `text_hash`로 캐싱됩니다.
중간에 끊겨도 다시 실행하면 남은 것만 부릅니다 — **API 비용을 두 번 내지 않기 위한 장치**입니다.

## 4단계: 하이브리드 검색 (`search.py`)

```bash
python search.py "퇴직연금 중도인출 사유가 뭐야?"
python search.py "판매보수 얼마야" --fund KR5157450090
python search.py "위험등급" --type table -k 10
python search.py "중도인출" --json
```

`--json`은 Agent에 물릴 때 씁니다.

벡터 검색(Chroma)과 키워드 검색(BM25)을 RRF로 합칩니다. 연금 질의는 두 종류가 섞여 있어서요.

| 질의 | 강한 쪽 |
|---|---|
| "중도인출하면 세금 얼마나 떼?" | 벡터 (의미) |
| "종류C-P2 판매보수 알려줘" | BM25 (고유명사·코드) |

`종류C-P2`, `KR5157450090` 같은 코드는 임베딩 공간에서 서로 뭉쳐버려 벡터 검색만으로는
정확한 하나를 못 집습니다. BM25는 반대로 정확히 그걸 집습니다.

`--fund`로 거르면 해당 펀드의 개별 청크 + 그 펀드가 공유하는 공통 조항이 함께 나옵니다
(`doc_ids` 기반). 정보 손실 없이 다른 펀드 내용만 배제됩니다.

## 다음 단계

1. **답변 생성 붙이기** — `search.py --json` 결과를 HyperCLOVA X 프롬프트에 넣고,
   `fund_name` + `page` + `base_date`를 함께 전달해
   "2025-10-01 기준 투자설명서 23쪽"처럼 인용하게 만듭니다.
2. **평가셋 만들기** — 실제 상담 질문 30~50개와 정답 근거 청크를 짝지어두면
   청킹 크기나 검색 방식을 바꿀 때 좋아졌는지 나빠졌는지 판단할 수 있습니다.
   이거 없이 튜닝하면 감으로 하는 겁니다.
3. **파인튜닝은 나중에** — 문서 사실을 가중치에 넣지 말고, 말투·답변 형식을 잡을 때
   소량의 Q&A 쌍으로만. CLOVA Studio 규격은 CSV/JSONL, UTF-8,
   `C_ID / T_ID / Text / Completion`, 행당 8,000자 이하, 400행 이상 권장입니다.

## 파일 구성

```
build_dataset.py      1단계  문서 → chunks.jsonl
finalize_dataset.py   2단계  중복 제거 → chunks_final.jsonl
embed_and_index.py    3단계  임베딩 → Chroma
search.py             4단계  하이브리드 검색
```

## 튜닝 포인트

`build_dataset.py` 상단 상수만 바꾸면 됩니다.

```python
CHUNK_TARGET   = 900   # 청크 목표 길이. 검색 정밀도↑ 원하면 600
CHUNK_OVERLAP  = 150   # 겹침. 문맥 끊김이 보이면 250
TABLE_SETTINGS = {...} # 표가 안 잡히면 snap_tolerance를 3~8 사이로 조정
HEADING_PATTERNS = [...]  # 섹션이 안 잡히는 문서 유형이 있으면 패턴 추가
```
