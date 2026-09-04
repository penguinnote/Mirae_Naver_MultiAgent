# 데이터 인터페이스 규격

이 에이전트(`agent.py`)를 **다른 방식으로 가공한 데이터셋**에 얹어 돌리려면
아래 세 가지가 규격에 맞아야 한다. 코드는 전혀 고칠 필요 없다.

> 왜 규격이 중요한가: 필드 이름이나 값이 어긋나면 **에러 없이 조용히 0건**이
> 나온다. `route()`가 `doc_type`으로 후보를 좁히는데, 그 값이 다르면 필터가
> 전부 걸러내고 "성능이 낮다"로 잘못 읽히기 쉽다.

---

## 1. `dataset/chunks_final.jsonl` — 청크 본문 + 메타데이터

JSON Lines. 한 줄이 청크 하나.

### 반드시 있어야 하는 필드

| 필드 | 타입 | 쓰이는 곳 | 비고 |
|---|---|---|---|
| `chunk_id` | str | 전 구간의 키. Chroma의 id와 **반드시 일치** | 예: `KR5110501016_p3_0005` |
| `text` | str | BM25 색인, 프롬프트 근거 | |
| `doc_type` | str | **`route()`의 필터** | 아래 값만 허용 |
| `content_type` | str | 표/본문 구분 | `text` \| `table` |
| `doc_id` | str | 출처 표기 | 예: `doc55` |
| `fund_code` | str\|null | 펀드 한정 검색, 계열 비교 감지 | 예: `KR5110501016` |
| `fund_name` | str\|null | 출처 표기, 계열 비교 감지 | |
| `page` | int\|null | 출처 표기 | 없으면 생략됨 |
| `section` | str\|null | 출처 표기(절·표 제목) | 쪽 개념 없는 문서에 중요 |
| `base_date` | str\|null | 출처 표기(기준일) | |
| `doc_ids` | list[str] | 공통 청크가 어느 문서에 속하는지 | 아래 설명 |

### `doc_type` — 이 세 값만 쓸 것

```
"투자설명서"   펀드 투자설명서 (현재 14,273청크)
"연금문서"     제도·업무 문서   (현재    760청크)
"기타"                          (현재    163청크)
```

`route()`가 질문을 보고 `doc_type="연금문서"` 또는 `"투자설명서"`로 좁힌다.
**다른 문자열을 쓰면 그 필터가 항상 0건**이 되고, 폴백이 돌긴 하지만
제도/상품 분리 검색(복합 질문 처리)이 무력화된다.

### `doc_ids` — 여러 문서가 공유하는 청크

투자설명서들이 똑같이 담고 있는 법정문구는 하나로 합쳐서 저장한다.
그 청크는 `fund_code=null`이고 `doc_ids`에 해당 문서들이 들어간다.
펀드 한정 검색이 이 값을 본다:

```python
if md["fund_code"] != fund and f",{fund}," not in md["doc_ids"]:
    제외
```

중복 제거를 안 했다면 `doc_ids`를 빈 리스트로 두고 `fund_code`만 채우면 된다.

---

## 2. `dataset/chroma` — 벡터 인덱스

- Chroma **PersistentClient**, 컬렉션 이름 **`pension`**
- id는 `chunks_final.jsonl`의 `chunk_id`와 **1:1로 일치**해야 한다
- metadata에 최소 `doc_type`, `content_type`, `fund_code`, `doc_ids`가 있어야 한다
  (벡터 검색 결과를 필터링할 때 Chroma metadata를 직접 본다)
- **임베딩 모델이 같아야 한다.** 현재 CLOVA 임베딩 v2 = **1024차원**.
  질의는 실행 시점에 `search.embed_query()`가 CLOVA로 임베딩하므로,
  인덱스를 다른 모델로 만들었으면 차원이 안 맞아 터지거나
  (같은 차원이면) 의미 없는 결과가 나온다.

경로·컬렉션명을 바꾸려면:
```python
from agent import Retriever
r = Retriever(db="./내dataset/chroma", collection="내컬렉션",
              chunks="./내dataset/chunks_final.jsonl",
              bm25_cache="./내dataset/bm25.pkl")
```

---

## 3. `dataset/fund_fees.sqlite` — 보수 조회용 (선택이지만 권장)

없으면 `fee_sql` 노드가 "DB를 찾지 못해 건너뜀"을 남기고 검색으로만 답한다.
**수수료 문항 정확도가 크게 떨어진다** (실측: 이 경로 복구로 상품 도메인
55% → 93%).

```sql
CREATE TABLE fund_fees (
    fund_code TEXT, fund_name TEXT, base_date TEXT,
    class_label TEXT,      -- '수수료미징구-오프라인-개인연금(C-P)' 같은 설명
    class_code TEXT,       -- 'C-P', 'A-e' 같은 코드  ← 조회는 이 컬럼으로
    account_type TEXT,     -- '연금저축' | '퇴직연금' | NULL(일반)
    channel TEXT,          -- '온라인' | '오프라인' | '온라인슈퍼' | NULL
    front_load_text TEXT,  -- 선취/환매수수료 설명
    fee_total REAL,        -- 총보수(%)
    fee_distribution REAL, -- 판매보수(%)
    fee_peer_avg REAL,     -- 유사펀드 평균(%)
    fee_total_cost REAL,   -- 총보수·비용(%)
    chunk_id TEXT, page INTEGER, source_path TEXT
);
```

`page`를 꼭 채울 것 — 답변의 출처 표기에 쓰인다.

`agent.py`의 `TEXT2SQL_PROMPT`에 **실제 `class_code` 목록이 하드코딩**돼 있다.
클래스 체계가 다르면 그 목록을 바꿔야 한다 (`agent.py`에서 `class_code 전체 목록` 검색).

---

## 4. 실행

```bash
git clone https://github.com/penguinnote/Mirae_Naver_MultiAgent.git
cd Mirae_Naver_MultiAgent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# .env 작성 — 필수 키는 다섯 개다
CLOVA_API_KEY=...
CLOVA_EMBED_URL=...
CLOVA_EMBED_AUTH=bearer
CLOVA_CHAT_REQUEST_ID=...
CLOVA_EMBED_REQUEST_ID=...

# 자기 데이터셋을 dataset/ 에 배치한 뒤
python agent.py "퇴직연금 중도인출 사유가 뭐야?"
```

### 규격이 맞는지 먼저 확인

```bash
python diag_retrieval.py     # 검색만 확인 (HCX 미사용, 토큰 거의 안 듦)
```
`후보 N개`가 0으로 나오면 `doc_type` 값이나 Chroma 컬렉션명을 확인할 것.

### 평가 실행

```bash
python make_raw_from_gold.py --gold {정답지경로} --out raw_run1.json --resume
python score_official.py --raw raw_run1.json
```

정답지 파일은 팀 blind 평가의 유효성을 지키기 위해 저장소에서 제외했다.

---

## 5. 비교할 때 주의할 점

**같은 조건을 맞춰야 의미가 있다.**

- `TOP_K`(현재 12), `POOL`(60), `EVIDENCE_CHARS`(1500) 등 상수는 `agent.py` 상단에 있다.
  데이터 구성이 다르면 최적값도 달라질 수 있으므로, **한쪽에만 유리한 설정이
  아닌지** 확인할 것.
- HCX는 `temperature=0.2`라 **같은 입력에도 실행마다 답이 조금씩 달라진다.**
  실측으로 검색 결과가 완전히 동일한데 점수가 33%p 차이 난 문항이 있었다.
  1회 실행 차이는 노이즈일 수 있으니 작은 차이에 의미를 두지 말 것.
- `score_official.py`는 문자열 매칭 기반 **근사치**다. 팩 정책상 최종 판정은
  사람이 하며, "표현이 다르다"는 오답 사유가 아니다.
  **팀 간 순위는 평가담당자가 동일 기준으로 매기는 것이 원칙이다.**

---

## 6. 참고 — 현재 데이터셋 규모

```
청크         15,196
  투자설명서  14,273 / 연금문서 760 / 기타 163
  본문 11,298 / 표 3,898
  개별 13,527 / 공통 1,669
원본 문서     316건 (PDF·DOCX, OK 314 / LOW_YIELD 2)
보수 DB       576행 / 92펀드
```
