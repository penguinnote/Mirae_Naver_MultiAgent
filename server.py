"""
FastAPI 서버 — 대회 평가 API 규격(CLAUDE.md 6장) 준수.

    GET {endpoint}/answer?question_id={id}&question={질의}

응답은 5개 필드 전부 문자열: question_id, question, retrieved_context,
think_trace, answer. 헤더(인증 포함) 요구 없음.

로컬 기동 (개발용, 포트 8000):
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn server:app --host 127.0.0.1 --port 8000

로컬 검증:
    curl "http://127.0.0.1:8000/answer?question_id=t1&question=DC%EC%99%80%20DB%EC%9D%98%20%EC%B0%A8%EC%9D%B4%EA%B0%80%20%EB%AD%90%EC%95%BC"
    python eval_answers.py --adapter agent:answer_for_eval --evalset evalset_v1.json --show   (에이전트 자체 회귀 확인용)
    python eval_answers.py --endpoint http://127.0.0.1:8000/answer --evalset evalset_v1.json --show   (서버 경유 회귀 + 규격 확인용)

실제 배포에서는 uvicorn을 80/443에 직접 물리지 않는다 — 루트 권한 없이 80을 열 수
없고, nginx가 앞단에서 재시작 중 다운타임을 흡수해준다. nginx가 80/443 → 이
uvicorn(127.0.0.1:8000)으로 리버스 프록시한다. 설정은 deploy/nginx.conf,
deploy/agent-server.service 참고.
"""
from __future__ import annotations

import sys
import time

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import agent

app = FastAPI(title="연금 Agent 평가 API")


@app.on_event("startup")
def _startup() -> None:
    # BM25(72MB)+Chroma 인덱스 로딩을 첫 요청 전에 끝내둔다.
    # 이 시점에 실패하면(임베딩 인덱스 경로 오류 등) 프로세스가 뜨자마자
    # 콘솔에 스택트레이스가 보이므로 systemd 로그로 바로 확인 가능하다.
    t0 = time.monotonic()
    agent.warmup()
    print(f"[server] 기동 준비 완료 ({time.monotonic() - t0:.1f}초)", file=sys.stderr)


@app.get("/answer")
def answer(
    question: str = Query(..., description="상담 질의"),
    question_id: str = Query("", description="문항 ID"),
) -> JSONResponse:
    t0 = time.monotonic()
    try:
        resp = agent.run(question=question, question_id=question_id)
    except Exception as exc:  # noqa: BLE001
        # 평가 API는 5xx보다 "형식은 맞지만 내용이 빈 답"이 낫다 — 규격 위반은
        # 문항 자체가 0점 처리되지만, 예외가 그대로 새면 502가 나서 재시도
        # 횟수(최대 2회)까지 태울 위험이 있다.
        print(f"[server] /answer 처리 중 예외: {exc!r}", file=sys.stderr)
        resp = {
            "question_id": str(question_id or ""),
            "question": str(question),
            "retrieved_context": "",
            "think_trace": f"서버 오류: {exc}",
            "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        }
    print(f"[server] question_id={question_id or '-'} 처리 {time.monotonic() - t0:.1f}초",
          file=sys.stderr)
    return JSONResponse(content=resp, media_type="application/json")


@app.get("/health")
def health() -> dict:
    """평가 API 규격 외 — 우리가 서버 생존을 확인할 때만 쓰는 엔드포인트."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
