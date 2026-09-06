# 평가용 API 명세

연금 Agent · 제10회 미래에셋증권 AI Festival

| | |
|---|---|
| Endpoint URL | `http://211.188.59.247/answer` |
| 프로토콜 | HTTP/1.1 |
| 포트 | 80 |
| 메서드 | GET |
| 인증 | 없음 |
| 운영 기간 | 2026-09-07 ~ 09-30 무중단 |

주최 측 헬스체크 안내(2026-09-04)에 따라 답변 수집은 HTTP·HTTPS 모두 허용되며, 본 엔드포인트는 HTTP(80)로 제공합니다.

---

## 요청

```
GET /answer?question_id={id}&question={질의}
```

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `question_id` | string | O | 주최 측 문항 ID. 응답에 그대로 반환 |
| `question` | string | O | 질의 원문. URL 인코딩 |

경로는 `/answer` 고정입니다. 인증 헤더나 쿼리 키가 필요하지 않습니다.

### cURL

```bash
curl -G "http://211.188.59.247/answer" \
  --data-urlencode "question_id=Q-001" \
  --data-urlencode "question=DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?"
```

### Python

```python
import requests

resp = requests.get(
    "http://211.188.59.247/answer",
    params={
        "question_id": "Q-001",
        "question": "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?",
    },
    timeout=300,
)
result = resp.json()
```

---

## 응답

`Content-Type: application/json`, UTF-8. 다섯 필드는 **모두 문자열**입니다.

| 필드 | 타입 | 내용 |
|---|---|---|
| `question_id` | string | 요청의 `question_id` 그대로 |
| `question` | string | 요청의 `question` 그대로 |
| `retrieved_context` | string | 답변 생성에 참고한 검색 문서. 청크 ID와 출처를 포함 |
| `think_trace` | string | 노드별 사고·추론·도구 사용 과정 |
| `answer` | string | 최종 생성 답변. 끝에 근거 문서 목록 부착 |

```json
{
  "question_id": "Q-001",
  "question": "DC와 DB, 퇴직금이 정해지는 방식이랑 운용 주체가 어떻게 다른가요?",
  "retrieved_context": "[doc22_p1_0003] 퇴직연금제도 기본 안내\n확정급여형(DB)은 ...",
  "think_trace": "[1] route: 제도 키워드 2개 / 상품 키워드 0개 → doc_type=연금문서\n[2] retrieve: 하이브리드 검색(벡터+BM25, RRF) → 상위 12개 [...]\n[3] compose: HCX-007로 답변 생성 (근거 12개, 5.1초)\n[4] 출처: 퇴직연금제도 기본 안내 1쪽 / ...",
  "answer": "DB와 DC의 가장 큰 차이는 퇴직급여가 정해지는 방식과 운용 주체입니다.\n...\n\n※ 근거: 퇴직연금제도 기본 안내 1쪽 · 퇴직연금 업무 체크포인트"
}
```

### `think_trace` 형식

노드별로 `[N] 노드명: 내용` 한 줄씩 누적됩니다. 실행되지 않은 노드는 나타나지 않습니다.

| 접두 | 의미 |
|---|---|
| `route:` | 키워드 판정 결과와 실행 계획 (`doc_type`, `hybrid`, `need_sql`, `need_calc`) |
| `retrieve:` | 검색 방식, 상위 청크 ID 목록, 분할 배분 여부 |
| `fee_sql:` | 생성된 SQL, 정규화 내역, 조회 행 수, 완화 재조회 |
| `calc:` | 계산식과 결과 |
| `compose:` | 근거 개수, 소요 시간 |
| `토큰:` | 입력·출력 토큰 수와 HCX 호출 횟수 |
| `출처:` | 답변에 부착한 근거 문서 목록 |

안전성 차단 시에는 `safety_check`에서 조기 반환하며 HCX를 호출하지 않습니다. 이 경우 `retrieved_context`가 빈 문자열이고 `think_trace`에 차단 사유가 기록됩니다.

---

## 동작 보장

**타임아웃** — 300초 이내에 응답합니다. 서버 내부 HCX 호출 상한은 110초, nginx `proxy_read_timeout`은 320초입니다.

**오류 처리** — `/answer` 처리 중 예외가 발생해도 5xx를 반환하지 않습니다. 규격에 맞는 5필드 JSON으로 응답하며, `answer`에 처리 실패를 명시하고 `think_trace`에 오류 내용을 남깁니다. 평가 API에서는 5xx보다 형식이 맞는 응답이 재시도 낭비가 적다고 판단했습니다.

**응답 시간** — 배포 API 실측 기준 평균 7.3초, 최대 21.2초입니다(evalset_v1 40문항 · 6회 · 09-03). 최종 코드 스모크에서는 7.2~9.3초였습니다. 안전성 차단 질의는 HCX를 호출하지 않아 0.1초 이내입니다.

**가용성** — systemd `Restart=always`와 nginx 리버스 프록시로 운영합니다. 기동 시 BM25 인덱스(72MB)와 Chroma를 미리 로딩해 첫 요청이 지연되지 않습니다. 심사 대상 코드 커밋은 `af91008`입니다.

**캐시** — 동일한 `(question_id, question)` 조합은 프로세스 메모리에 캐시됩니다. 같은 문항을 다시 호출하면 저장된 응답이 즉시 반환됩니다.

---

## 부가 엔드포인트

```
GET /health
→ {"status": "ok"}
```

생존 확인용이며 평가 규격에는 포함되지 않습니다.
