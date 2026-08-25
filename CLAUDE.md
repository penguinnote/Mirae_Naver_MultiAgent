# 연금 Agent — 프로젝트 컨텍스트

미래에셋증권 AI Festival 출품작. RAG + 멀티에이전트 연금 상담 시스템.
**마감: 09.07 서버 가동 (오늘 기준 약 3주)**

이 파일은 Claude Code가 자동으로 읽습니다. 작업 전에 여기부터 확인하세요.

## ⚠️ 커밋 규칙 (반드시 지킬 것)

**커밋 메시지에 `Co-Authored-By: Claude` 나 `Claude-Session:` 를 넣지 마십시오.**
저장소 contributor 목록에 Claude가 올라오는 것을 원하지 않습니다.
기본 동작으로 이 trailer를 붙이게 되어 있더라도 이 프로젝트에서는 생략합니다.

커밋 메시지는 **한 줄**로 작성하십시오. zsh에서 여러 줄 따옴표가 `dquote>`로
막히는 일이 잦습니다. 자세한 내용이 필요하면 커밋 후 별도로 정리하십시오.

---

## 1. 지금 상태 (2026-08-18)

| 계층 | 상태 | 수치 |
|---|---|---|
| 데이터 추출·정제 | ✅ | 157문서 → 14,745 청크 |
| 임베딩·인덱싱 | ✅ | 14,745/14,745, 실패 0 |
| 하이브리드 검색 | ✅ | **근거 충족률 88.9%** (k=5) ← 정확도 천장 |
| `fund_fees` SQL | ✅ | 73/81 펀드, 363행 |
| 평가셋 | ✅ | v1 40문항 + v2 26문항 |
| **v0 에이전트** | 🔄 진행 중 | 정답률 85.0% (v1셋 40문항, round3) |
| Fee-SQL 노드 | ✅ | Q-013~017 5/5 통과 |
| Calculator 노드 | 🔄 검증 대기 | 배선 완료, Q-030 확인 완료(1,200만원 정답) |
| 안전성 가드(PII·인젝션) | 🔄 검증 대기 | 배선 완료, 오탐 0/66 |
| 대회 평가지표 대응(규칙 2-b, 5-b) | ✅ | 참고 질의 5개 확인 — Q5(정보한계)만 부분 작동 |
| 계열 펀드 비교 검색 | ✅ | "솔로몬 국공채" 4개 펀드 전부 커버 확인 |
| SQL 결과 재정렬 안전장치 | ✅ | round5에서 Q-014 회귀 발견 → 수정 → round6 재통과 |
| **v0 에이전트 (round6 최종)** | ✅ 안정화 | 정답률 90.0%(v1셋 40문항), 회귀 없음 |
| FastAPI `/answer` | ✅ 검증 완료 | 로컬 uvicorn 경유 90.0%(36/40), API 규격 점검 통과(5필드 문자열·200·JSON), 응답 평균 7.0초/최대 24.4초 |
| NCP 배포 | ⬜ 다음 작업 | `deploy/nginx.conf`·`deploy/agent-server.service`·`deploy/DEPLOY.md` 작성 완료 |

---

## 2. 파일 지도

```
~/dev/data_test/
├── build_dataset.py      PDF/DOCX → 청크 (OCR 폴백 포함)
├── finalize_dataset.py   완전일치 중복 병합
├── embed_and_index.py    CLOVA 임베딩 → Chroma 적재
├── search.py             하이브리드 검색 (벡터+BM25+RRF) + 리랭커
├── agent.py              ★ v0 에이전트 (safety_check → route → retrieve → fee_sql → calc → compose)
├── build_fund_fees.py    표 청크 → fund_fees.sqlite
├── build_evalset.py      평가셋 v1 생성
├── build_evalset_v2.py   평가셋 v2 생성
├── run_eval.py           검색 계층만 측정 (하이브리드/벡터/BM25 비교)
├── eval_answers.py       ★ 답변 채점 (팀 공용, 표준 라이브러리만)
├── evalset_v1.json       40문항 (우리가 고른 질문)
├── evalset_v2.json       26문항 (팀에서 받은 질문, 커버리지 점검)
├── server.py             ★ 평가 API 서버 (FastAPI, agent.run() 래핑)
├── deploy/
│   ├── nginx.conf             nginx 리버스 프록시 (80/443 → 127.0.0.1:8000)
│   ├── agent-server.service   systemd 유닛 (Restart=always)
│   └── DEPLOY.md              NCP 배포 단계별 가이드
├── .env                  API 키 (git 제외)
└── dataset/
    ├── chunks_final.jsonl   14,745 청크
    ├── chroma/              벡터 인덱스
    ├── bm25.pkl             BM25 인덱스 (72MB, 로딩 ~10초)
    ├── fund_fees.sqlite     363행
    └── emb_cache.sqlite     임베딩 캐시
```

---

## 3. 반드시 알아야 할 함정 (실제로 겪은 것들)

### zsh는 `#` 주석을 인식하지 않는다
명령어 뒤에 설명을 붙이면 인자로 넘어가 에러가 납니다. 명령만 쓰세요.

### `.env`는 모듈 임포트 시점에 읽어야 한다
`load_dotenv()`를 `main()` 안에서만 부르면, `--adapter agent:answer_for_eval`처럼
모듈로 임포트될 때 설정이 통째로 빕니다. **실제로 이 버그로 전 문항이 실패했는데
"정답률 15.4%"로 표시돼 성능 문제로 오해했습니다.**

### `search.py`의 `embed_query()`는 키가 없으면 조용히 더미 벡터를 쓴다
경고만 찍고 진행하므로 차원 불일치로 나중에 터집니다. `agent.py`의 `Retriever`
생성자에서 키를 먼저 검사해 즉시 중단시켜 뒀습니다. 이 가드를 지우지 마세요.

### HCX-007은 추론 모델이라 body 규격이 다르다
`maxTokens`를 보내면 `40001 Invalid parameter`가 납니다.
정답 조합: `maxCompletionTokens` + `thinking: {"effort": "none"}`
`.env`에 `CLOVA_CHAT_PROFILE=maxCompletionTokens+thinking` 로 고정돼 있습니다.
`agent.py`의 `call_hcx()`가 자동 탐색도 하므로 규격이 바뀌면 스스로 찾습니다.

### `X-NCP-CLOVASTUDIO-REQUEST-ID`는 작업마다 다르다
임베딩 / 리랭커 / HCX-007이 각각 다른 값입니다. `.env`에 3개로 분리돼 있습니다.
섞어 쓰면 엉뚱한 서비스 앱 한도로 처리됩니다.

### 채점기는 문자열 매칭이라 정답을 오답으로 잡을 수 있다
`"언제든지 변경"` 답변이 정답 문자열 `"언제든 변경"`과 안 맞아 오답 처리된 적이
있습니다. `answer_points`는 `|`로 대안 표현을 나열할 수 있습니다.
**❌ 문항은 반드시 `--show`로 답변 원문을 확인하고 판단하세요.**

### macOS 파일명은 NFD로 저장된다
경로 비교 시 `unicodedata.normalize('NFC', ...)`가 필요합니다.
`.gitignore`도 마찬가지입니다 — `*투자설명서*/`가 NFD 폴더명에 안 맞아서
원본 폴더가 계속 untracked로 떴습니다. 인코딩과 무관한 `0.*/`로 막아뒀습니다.

### 노드를 만들었으면 `run()`의 노드 튜플에 넣었는지 확인하라
`fee_sql()`을 다 구현해놓고 `for node in (route, retriever, compose)`에
추가하지 않아 한 번도 실행되지 않은 적이 있습니다. 함수가 존재하는 것과
파이프라인에 연결된 것은 다릅니다. 새 노드를 넣으면 `think_trace`에 그
노드 로그가 실제로 찍히는지 눈으로 확인하세요.

---

## 4. 이미 시도했다가 기각한 것 (다시 하지 마세요)

전부 실측했고 셋은 실패, 하나만 유효했습니다.

| 시도 | 결과 | 판정 |
|---|---|---|
| 목차("질문 리스트") 청크 8개 제거 | 52.8% → **44.4%** | ❌ 그 청크에 실제 답변도 들어있음 |
| BM25에서 2-gram 제거 | k=20 94.4% → **77.8%** | ❌ 2-gram은 한글 조사 대응에 필수 |
| 원어 토큰 3배 가중 | k=20 94.4% → **88.9%** | ❌ 깊이에서 손해 |
| 3-gram 추가 | 3개 고치고 3개 깨뜨림, **순이득 0** | ⚠️ 재빌드 비용 대비 무의미 |
| **doc_type 라우팅** | 77.8% → **83.3%** | ✅ 유일하게 유효, `route()`에 반영됨 |

**미해결로 남긴 것**: 한글 조사가 붙으면 원어 토큰이 안 만들어집니다
(`"실물이전으로"` ≠ `"실물이전"`). 그래서 IDF 8.09짜리 강한 신호가 죽고 19위로
밀립니다. 정공법은 형태소 분석기(kiwipiepy)인데 시간이 남을 때 할 일입니다.

---

## 5. 설계 결정과 그 이유

**하이브리드 k=5로 고정.** k=5에서 88.9%로 최고이고, k=20으로 늘리면 +2.8%p인데
컨텍스트는 4배입니다. 300초 데드라인에 불리합니다.

**RRF는 상위에서 강하고 깊이에서 손해본다.** k=20에서는 BM25 단독(94.4%)이
하이브리드(91.7%)를 이깁니다. 결함이 아니라 정상적인 트레이드오프입니다.

**route()는 LLM을 쓰지 않는다.** 규칙으로 충분하고 호출 1회를 아끼면 데드라인에
여유가 생깁니다. 애매하면 필터를 걸지 않습니다 — 잘못 좁히면 정답을 아예 못 보지만
안 좁히면 순위만 밀립니다. 평가셋 40문항에서 잘못 좁힌 건 0건입니다.

**LangGraph를 아직 쓰지 않는다.** 노드가 3개뿐이라 얻는 것보다 디버깅 비용이
큽니다. 다만 v1에서 옮기기 쉽도록 노드를 함수로 나누고 입출력을 state 딕셔너리
하나로 통일해 뒀습니다.

**대회 규정**: 답변 생성 LLM과 에이전트 워크플로우의 LLM은 **반드시 HyperCLOVA
계열**이어야 합니다. Fee-SQL의 Text2SQL도 여기 해당하니 HCX-007로 고정하세요.
계산·검증은 LLM이 아니므로 순수 파이썬으로 합니다.

---

## 6. 평가 API 규격 (주최측, 절대 어기면 안 됨)

```
GET {endpoint}/answer?question_id={id}&question={질의}
```

- 경로 `/answer` 고정, **헤더 없음**(인증 헤더 포함 없음)
- 응답 5개 필드 **전부 문자열**:
  `question_id`, `question`, `retrieved_context`, `think_trace`, `answer`
- `Content-Type: application/json`
- 타임아웃 300초, 5xx/타임아웃 시 최대 2회 재시도
- 포트 80(HTTP) 또는 443(HTTPS). 8000번으로 제출 불가
- nginx `proxy_read_timeout` 320s 이상 (기본 60초면 nginx가 먼저 끊음)

규격이 틀리면 답변 품질과 무관하게 0점입니다.
배포 후 반드시 `python eval_answers.py --endpoint http://IP/answer`로 검증하세요.
이 명령이 규격까지 자동 점검합니다.

---

## 7. 자주 쓰는 명령

```
python agent.py --selftest
python agent.py "퇴직연금 중도인출 사유가 뭐야?"
python agent.py "..." --show-evidence
python agent.py "..." --json

python eval_answers.py --adapter agent:answer_for_eval --evalset evalset_v2.json --out r.json
python eval_answers.py --adapter agent:answer_for_eval --evalset evalset_v1.json --show
python run_eval.py --compare

sqlite3 dataset/fund_fees.sqlite "SELECT fund_name, class_label, fee_total FROM fund_fees ORDER BY fee_total LIMIT 5;"
```

---

## 7-b. round3 실측 (v1셋 40문항, 2026-08-18)

Fee-SQL을 파이프라인에 연결한 뒤 40문항 재측정. **85.0%(34/40)**, 5개 SQL 문항 전부 통과.
실패 6개를 원인별로 나눠보니 전부 이미 알려진 사유였다 — 새로 생긴 문제가 없다.

| 문항 | 원인 | 조치 |
|---|---|---|
| Q-006 | 채점기 문자열 불일치("과세제외" vs 답변의 "비과세") | ✅ `build_evalset.py`에 `｜`대안 추가, 재생성함 |
| Q-029 | 이번 라운드에서만 발생(직전 라운드는 통과) — HCX 생성 변동성으로 추정 | 관찰만. 반복되면 프롬프트 보강 |
| Q-030 | **재현됨.** ×120%를 ×(11-연차)로 계산 — 8배 오류 | 프롬프트로 못 고침. Calculator 노드 필요 |
| Q-032 | 검색 실패 — `평가셋-검색-기준선.md` 4절에 이미 규명된 "조사 결합" 문제(실물이전으로≠실물이전) | 기 결정대로 보류 (순이득 없음, 실측됨) |
| Q-033 | 위와 동일 원인 | 보류 |
| Q-034 | 위와 동일 원인 | 보류 |

**결론**: Composer가 만들 수 있는 개선은 사실상 끝에 가깝다. 남은 문제 중 하나(Q-030)는
Calculator 노드가 필요하고, 셋(Q-032·033·034)은 이미 "쫓지 않기로" 결정한 검색 계층
한계다. 다음 라운드를 또 돌려서 프롬프트를 더 만지는 것은 시간 대비 효과가 낮다.

## 8. 작업 원칙

**한 번에 하나만 바꾸고 측정하세요.** 프롬프트와 검색을 동시에 고치면 무엇이
효과 있었는지 알 수 없습니다.

**숫자가 이상하면 성능을 의심하기 전에 고장을 의심하세요.** 응답시간이 유독 짧거나
(1~2초), 근거 제공률이 0%거나, 실패가 몰려 있으면 버그입니다.

**평가셋에 과적합하지 마세요.** 주최측 실제 질문은 비공개입니다. 특정 펀드명이나
키워드를 하드코딩하면 본 평가에서 무너집니다.

**측정한 것은 프로젝트 문서에 남기세요.** 특히 실패한 시도는 남겨야 다음 사람이
같은 실험을 반복하지 않습니다.
