# 배포 가이드 — NCP 서버에 평가 API 올리기

목표: 주최측이 `GET http://{공인 IP}/answer?question_id=&question=` 로 호출했을 때
5개 필드(question_id/question/retrieved_context/think_trace/answer) 문자열
JSON을 돌려주는 서버를 09.07~09.20 사이 배정된 1주 동안 무중단으로 띄워두는 것.

## 0. 로컬(개발 머신)에서 먼저 검증

배포 전에 반드시 로컬에서 서버가 규격대로 동작하는지 확인합니다.

```bash
source venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

다른 터미널에서:

```bash
curl "http://127.0.0.1:8000/answer?question_id=t1&question=DC%EC%99%80%20DB%EC%9D%98%20%EC%B0%A8%EC%9D%B4%EA%B0%80%20%EB%AD%90%EC%95%BC"
python eval_answers.py --endpoint http://127.0.0.1:8000/answer --evalset evalset_v1.json --show
```

`eval_answers.py --endpoint`가 규격(5개 필드 전부 문자열인지 등)까지 같이 점검해줍니다.
로컬에서 40문항 정답률이 `agent:answer_for_eval` 어댑터로 돌린 결과와 같게 나오는지도
같이 확인하세요 — 다르면 서버 쪽 래핑에 버그가 있다는 뜻입니다.

## 1. NCP 서버 준비

- 서버 스펙: 최소 vCPU 2 / RAM 4GB 권장 (Chroma+BM25 인덱스를 메모리에 올림, 72MB+α)
- OS: Ubuntu 22.04 LTS 권장 (Python 3.10 이상 필요 — `agent.py`가 `X | None` 문법 사용)
- **ACG(Access Control Group)에서 인바운드 80번(HTTP) 포트를 0.0.0.0/0으로 열어야
  주최측이 접근 가능.** 443을 쓸 거면 443도 함께 연다.
- 공인 IP 확인해서 기억해두기 — README.md `## 평가용 엔드포인트`에 기입할 값.

## 2. 서버에 코드 배치

```bash
sudo apt update && sudo apt install -y python3.10-venv git nginx
sudo mkdir -p /opt/pension-agent && sudo chown $USER /opt/pension-agent
git clone https://github.com/penguinnote/Mirae_Naver_MultiAgent.git /opt/pension-agent
cd /opt/pension-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`.env`는 git에 올라가지 않으므로(`.gitignore`) **직접 서버에 옮겨야 합니다** — scp로
복사하거나 새로 만드세요. `CLAUDE_API_KEY` 등 CLOVA Studio 키 4종이 들어있어야
`agent.py`가 정상 동작합니다. `.env`는 `agent.py`와 **같은 폴더**에 두세요
(agent.py가 자기 파일 위치 기준으로 읽습니다).

### 옮겨야 할 데이터 — 4개뿐입니다

`dataset/`은 전부 합치면 585MB지만, 서버가 **실행 중에 실제로 여는 건 4개**입니다.
나머지는 인덱스를 만들 때만 쓰는 중간 산출물이라 서버에 없어도 됩니다.

| 옮긴다 | 크기 | | 안 옮겨도 된다 | 크기 |
|---|---|---|---|---|
| `dataset/chroma/` | 290M | | `dataset/docs/` | 58M |
| `dataset/bm25.pkl` | 70M | | `dataset/chunks.jsonl` | 69M |
| `dataset/chunks_final.jsonl` | 31M | | `dataset/emb_cache.sqlite` | 67M |
| `dataset/fund_fees.sqlite` | 124K | | | |

합계 **약 391MB**. `emb_cache.sqlite`는 임베딩을 다시 만들 때 쓰는 캐시라
`embed_and_index.py`만 봅니다 — 서버 코드에서는 열지 않습니다.

```bash
# 로컬에서
tar czf runtime_data.tgz dataset/chroma dataset/bm25.pkl \
    dataset/chunks_final.jsonl dataset/fund_fees.sqlite
scp runtime_data.tgz .env {서버}:/opt/pension-agent/
# 서버에서
cd /opt/pension-agent && tar xzf runtime_data.tgz && rm runtime_data.tgz
```

옮긴 뒤 확인:

```bash
cd /opt/pension-agent && source venv/bin/activate
python verify_deploy_fix.py      # 경로·캐시·호출간격을 HCX 없이 점검
```

## 3. 서비스 등록 (systemd)

`deploy/agent-server.service`의 `WorkingDirectory`/`ExecStart` 경로를 위에서 정한
실제 경로(`/opt/pension-agent`)에 맞게 확인한 뒤:

```bash
sudo cp deploy/agent-server.service /etc/systemd/system/agent-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now agent-server
sudo systemctl status agent-server
journalctl -u agent-server -f
```

`[server] 기동 준비 완료 (N초)` 로그가 뜨면 인덱스 로딩까지 끝난 것입니다 (`--startup`
에서 `agent.warmup()` 호출).

## 4. nginx 리버스 프록시

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/pension-agent
sudo ln -s /etc/nginx/sites-available/pension-agent /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

`proxy_read_timeout 320s`가 핵심입니다 — 기본 60초면 에이전트가 아직 답변 중인데
nginx가 먼저 연결을 끊어버립니다.

## 5. 외부에서 최종 검증

```bash
curl "http://{공인 IP}/answer?question_id=t1&question=DC%EC%99%80%20DB%EC%9D%98%20%EC%B0%A8%EC%9D%B4%EA%B0%80%20%EB%AD%90%EC%95%BC"
python eval_answers.py --endpoint http://{공인 IP}/answer --evalset evalset_v1.json --show
```

**포트가 80인지, 8000이 아닌지** 다시 한번 확인하세요 — 8000번은 제출 불가 규격입니다.

체크리스트:
- [ ] 응답 JSON이 5개 필드 전부 문자열
- [ ] 헤더(인증 등) 없이 호출해도 정상 응답
- [ ] `Content-Type: application/json`
- [ ] 40문항 돌려서 정답률이 로컬(`agent:answer_for_eval`)과 같음 — 다르면 배포 환경
      문제(`.env` 누락, 인덱스 경로 다름 등)
- [ ] 서버 프로세스를 강제로 죽여봐도(`sudo systemctl kill agent-server`) 몇 초 안에
      자동 재기동됨 (`Restart=always` 확인)

## 6. README.md 갱신

배포 성공 확인 후 README.md의 `## 평가용 엔드포인트` 자리에 실제 URL을 채워 넣습니다.

## 트러블슈팅

| 증상 | 원인 후보 |
|---|---|
| curl이 아예 안 붙음 (connection refused/timeout) | ACG 인바운드 80 안 열림, nginx 안 뜸, 공인 IP 오타 |
| 502 Bad Gateway | uvicorn(agent-server)이 안 떠 있음 — `systemctl status agent-server` |
| 504 Gateway Timeout | nginx `proxy_read_timeout` 설정이 실제로 반영 안 됨(`nginx -t` 후 reload 안 함) |
| 첫 요청만 유독 느림 | `agent.warmup()`이 startup에서 안 불렸거나 실패 — journalctl에서 "기동 준비 완료" 로그 확인 |
| 응답은 오는데 `answer`가 항상 오류 메시지 | `.env`가 서버에 없거나 CLOVA 키가 만료/틀림 |
