# 연금 Agent 평가 API — 제출 요건용 Dockerfile
#
# 이 이미지는 대회 제출 요건("Dockerfile 제공")을 충족하기 위한 것이다.
# 실제 NCP 운영 배포는 지금까지와 동일하게 deploy/DEPLOY.md의
# venv + systemd(agent-server.service) + nginx(80→8000, proxy_read_timeout
# 320s) 방식을 그대로 쓴다 — 이미 검증된 경로를 마감 직전에 건드리지
# 않기 위한 팀의 결정이다. 이 컨테이너는 로컬 재현·심사용으로 존재한다.
#
# 빌드에는 dataset/ 아래 4개 산출물이 이미 만들어져 있어야 한다
# (chroma/, bm25.pkl, chunks_final.jsonl, fund_fees.sqlite — 총 ~391MB).
# 이들은 .gitignore/.dockerignore 대상이라 저장소에는 없다. build_dataset.py
# → embed_and_index.py → build_fund_fees.py 파이프라인으로 먼저 만들거나,
# 이미 만들어 둔 dataset/ 폴더를 이 Dockerfile과 같은 곳에 두고 빌드한다.
#
# 빌드:  docker build -t pension-agent .
# 실행:  docker run --rm -p 8000:8000 --env-file .env pension-agent
#        (.env는 이미지에 담지 않는다 — 반드시 --env-file 또는 -e로 주입)
# 확인:  curl localhost:8000/health

# ⚠ 이 이미지와 운영 서버의 환경은 완전히 같지 않다. 심사자가 빌드한
# 컨테이너와 배포 API의 동작이 미세하게 다를 수 있어 아래에 밝혀 둔다.
#
#   · 의존성: 운영 서버는 requirements-lock.txt(pip freeze 정확 고정)로
#     설치해 chromadb 1.5.9·numpy 2.5.2 등을 못박는다. 이 이미지는
#     범위 지정인 requirements.txt를 쓰므로 빌드 시점에 따라 버전이
#     달라질 수 있다. chromadb는 버전 간 인덱스 호환성 문제가 있다.
#   · 인터프리터: 운영 서버는 Ubuntu 24.04의 Python 3.12.3이고
#     이 이미지는 3.11이다.
#
# lock 파일로 바꾸는 편이 정확하지만, 마감 전 실제 빌드로 검증할 수단이
# 없어(빌드 환경 부재) 적용하지 않았다. 운영과 동일한 환경을 재현하려면
# 베이스를 python:3.12-slim으로 올리고 아래 두 줄을 requirements-lock.txt로
# 바꾼 뒤 빌드·기동을 확인하면 된다.

FROM python:3.11-slim

# CLOVA API만 쓰고 poppler/tesseract 등 시스템 패키지는 필요 없다
# (requirements.txt 상단 주석 참고: pypdfium2가 PDF 렌더링을 자체 처리).
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 — agent.py → search.py → embed_and_index.py 순으로
# 로컬 모듈을 임포트한다(빌드 검증에서 확인: search.py를 빠뜨리면
# ModuleNotFoundError로 기동 자체가 실패한다).
COPY agent.py server.py search.py embed_and_index.py ./

# 런타임에 필요한 데이터 산출물 (빌드 시점에 존재해야 함 — 위 설명 참고)
COPY dataset/chroma/ dataset/chroma/
COPY dataset/bm25.pkl dataset/
COPY dataset/chunks_final.jsonl dataset/
COPY dataset/fund_fees.sqlite dataset/

# .env는 이미지에 넣지 않는다. agent.py는 .env 파일이 없으면 조용히
# 넘어가고(load_dotenv(None)) OS 환경변수를 그대로 쓰므로, CLOVA_* 값은
# `docker run --env-file .env` 로 주입한다.

EXPOSE 8000

# 컨테이너 안에서는 nginx가 없으므로 0.0.0.0으로 직접 바인딩한다
# (운영 중인 systemd 배포는 nginx 뒤에서 127.0.0.1로 바인딩 — deploy/DEPLOY.md 참고).
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
