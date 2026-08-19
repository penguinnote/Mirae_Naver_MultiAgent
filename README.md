# 연금 Agent

미래에셋증권 AI Festival 출품작. RAG + 멀티에이전트 연금 상담 시스템입니다.
문서 파싱부터 하이브리드 검색, Fee-SQL, 답변 생성까지 전 과정을 이 저장소에서 관리합니다.

## 평가용 엔드포인트

```
(배포 후 이 자리에 기입 — http://{공인 IP}/answer)
```

아직 배포 전입니다. 배포 완료 후 반드시 이 자리를 채워야 합니다(대회 필수 요구사항).

## 현재 상태 (2026-08-18)

| 계층 | 상태 | 수치 |
|---|---|---|
| 데이터 추출·정제 | ✅ | 157문서 → 14,745 청크 |
| 임베딩·인덱싱 | ✅ | 14,745/14,745, 실패 0 |
| 하이브리드 검색 | ✅ | 근거 충족률 88.9% (k=5) |
| `fund_fees` SQL | ✅ | 73/81 펀드, 363행 |
| 평가셋 | ✅ | v1 40문항 + v2 26문항 |
| **v0 에이전트** | ✅ | 정답률 90.0%(v1셋), 회귀 없음 확인 |
| Fee-SQL 노드 | ✅ | 상품 필터링 5/5 통과 |
| Calculator 노드 | ✅ | 연금수령한도 계산 오류 해결 |
| 안전성 가드(PII·인젝션) | ✅ | 표준 거절 응답, 오탐 0건 |
| FastAPI `/answer` | ⬜ 다음 작업 | |
| NCP 배포 | ⬜ 미착수 | |

세부 내역과 실패/성공 실험 기록은 프로젝트 문서(`claude/` 폴더 원본은 Claude 프로젝트에,
로컬에는 `CLAUDE.md`)에 있습니다.

---

## 1단계: 문서 → 청크 (`build_dataset.py`)

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

## 5단계: v0 에이전트 (`agent.py`)

검색·Fee-SQL·계산을 하나로 묶어 최종 답변을 만듭니다.

```
safety_check → route → retrieve → [fee_sql] → [calc] → compose → to_response
```

- **safety_check**: 개인정보(주민등록번호 등)·프롬프트 인젝션 패턴을 정규식으로 먼저 걸러
  HCX 호출 전에 표준 거절 응답을 준다.
- **route**: LLM 없이 규칙으로 문서유형(제도/투자설명서)을 판별하고, 수치 비교·정렬
  질문이면 `need_sql`, 연금수령한도 계산이면 `need_calc`를 켠다.
- **retrieve**: 하이브리드 검색(벡터+BM25+RRF). 같은 계열 펀드(단기/중장기/장기 등)를
  비교하는 질문이면 펀드별로 따로 검색해서 합친다(한 펀드가 상위를 독식해 나머지가
  근거에서 빠지는 문제 방지).
- **fee_sql**: 자연어를 HCX-007로 SQL로 바꿔 `fund_fees.sqlite`를 조회한다.
  SQL의 정렬을 신뢰하지 않고 "가장 싼/최저" 류 질문은 파이썬에서 재정렬한다.
- **calc**: 연금수령한도(`평가액/(11-연차)×120%`) 같은 정해진 공식은 LLM에게 시키지
  않고 코드로 직접 계산한다.
- **compose**: HCX-007로 최종 답변 생성. 근거 결론을 뒤집지 않기, 수치·조건을
  요약하지 않고 전부 옮기기, 사용자의 과장된 전제를 그대로 수긍하지 않기,
  정보가 부족해도 답변을 포기하지 않기 등을 프롬프트 규칙으로 강제한다.

```bash
python agent.py "퇴직연금 중도인출 사유가 뭐야?"
python agent.py "총보수 싼 연금저축 펀드 알려줘" --show-evidence
python agent.py --selftest      # HCX 연결·파라미터 조합 확인
```

## 6단계: 평가 (`build_evalset*.py`, `eval_answers.py`, `run_eval.py`)

```bash
# 답변까지 채점 (팀 공용, 표준 라이브러리만 사용)
python eval_answers.py --adapter agent:answer_for_eval --evalset evalset_v1.json --show

# 검색 계층만 따로 측정 (하이브리드/벡터/BM25 비교)
python run_eval.py --compare

# 배포된 엔드포인트 검증 (응답 규격까지 자동 점검)
python eval_answers.py --endpoint http://IP/answer --evalset evalset_v1.json
```

`eval_answers.py`는 팀원 모두가 자신의 시스템을 같은 기준으로 채점할 수 있도록
표준 라이브러리만 사용합니다. `eval_share/` 폴더에 배포용 사본이 있습니다.

- `evalset_v1.json` (40문항) — 우리가 고른 질문, 근거 청크까지 매핑
- `evalset_v2.json` (26문항) — 팀에서 받은 질문, 커버리지 점검용

## 다음 단계

1. **FastAPI `/answer` 서버** — 대회 규격(`GET /answer?question_id=&question=`,
   헤더 없음, 5개 문자열 필드)에 맞춰 `agent.py`를 감싼다. (다음 작업)
2. **NCP 배포** — 공인 IP, ACG 80포트 허용, nginx(`proxy_read_timeout 320s`),
   systemd(`Restart=always`).
3. **파인튜닝은 하지 않는다** — 문서 사실을 가중치에 넣지 않고, RAG로 근거를 준다.
   대회 규정상 답변 생성 LLM은 HyperCLOVA 계열이어야 한다.

## 파일 구성

```
build_dataset.py      1단계  문서 → chunks.jsonl
finalize_dataset.py   2단계  중복 제거 → chunks_final.jsonl
embed_and_index.py    3단계  임베딩 → Chroma
search.py             4단계  하이브리드 검색
build_fund_fees.py    -      표 청크 → fund_fees.sqlite (Fee-SQL용)
agent.py              5단계  v0 에이전트 (route→retrieve→fee_sql→calc→compose)
build_evalset.py      6단계  평가셋 v1 생성
build_evalset_v2.py   6단계  평가셋 v2 생성
eval_answers.py        6단계  답변 채점 (팀 공용)
run_eval.py            6단계  검색 계층만 측정
```

## 튜닝 포인트

`build_dataset.py` 상단 상수만 바꾸면 됩니다.

```python
CHUNK_TARGET   = 900   # 청크 목표 길이. 검색 정밀도↑ 원하면 600
CHUNK_OVERLAP  = 150   # 겹침. 문맥 끊김이 보이면 250
TABLE_SETTINGS = {...} # 표가 안 잡히면 snap_tolerance를 3~8 사이로 조정
HEADING_PATTERNS = [...]  # 섹션이 안 잡히는 문서 유형이 있으면 패턴 추가
```
