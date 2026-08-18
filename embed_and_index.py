#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
연금 Agent — 임베딩 + Chroma 인덱싱
====================================
 
chunks_final.jsonl  →  CLOVA Studio 임베딩  →  Chroma 벡터 DB
 
설계 원칙
---------
1. **재개 가능**   임베딩 결과를 SQLite에 text_hash로 캐싱한다.
                  1만 건 돌리다 끊겨도 다시 실행하면 남은 것만 부른다.
                  API 비용을 두 번 내지 않기 위한 장치다.
2. **설정 분리**   엔드포인트/인증 헤더 방식이 바뀌어도 .env만 고치면 된다.
3. **먼저 검증**   `--selftest`로 API 1회만 호출해 응답 구조와 차원 수를 확인한 뒤
                  본 작업을 시작한다. 1만 건 돌린 뒤 형식이 틀린 걸 아는 건 최악이다.
4. **dummy 제공**  `--provider dummy`로 API 없이 Chroma 파이프라인만 점검할 수 있다.
 
사용법
------
    pip install chromadb requests python-dotenv tqdm
 
    # 0) API 없이 파이프라인만 점검
    python embed_and_index.py --provider dummy --limit 200
 
    # 1) API 응답 구조 확인 (호출 1회)
    python embed_and_index.py --selftest
 
    # 2) 소량으로 검증
    python embed_and_index.py --limit 300
 
    # 3) 전체
    python embed_and_index.py --workers 8
 
.env 설정
---------
    CLOVA_API_KEY=nv-....................
    # 아래는 콘솔의 '테스트 앱/서비스 앱' 정보에 맞춰 조정
    CLOVA_EMBED_URL=https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2
    CLOVA_EMBED_AUTH=bearer          # bearer | ncp
    # CLOVA_APIGW_KEY=...            # CLOVA_EMBED_AUTH=ncp 인 경우에만 필요
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_URL = "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2"

# --selftest가 훑어볼 후보들.
# CLOVA Studio는 앱 종류(테스트/서비스)와 모델에 따라 경로가 갈리고,
# 콘솔 개편으로 testapp 접두사가 있는 버전과 없는 버전이 공존한다.
# 문서를 뒤지는 것보다 한 번씩 찔러보고 되는 걸 쓰는 게 빠르다.
BASE = "https://clovastudio.stream.ntruss.com"
URL_CANDIDATES = [
    f"{BASE}/v1/api-tools/embedding/v2",
    f"{BASE}/testapp/v1/api-tools/embedding/v2",
    f"{BASE}/serviceapp/v1/api-tools/embedding/v2",
    f"{BASE}/v1/api-tools/embedding/clir-emb-dolphin",
    f"{BASE}/testapp/v1/api-tools/embedding/clir-emb-dolphin",
    f"{BASE}/v1/api-tools/embedding/bge-m3",
    f"{BASE}/testapp/v1/api-tools/embedding/bge-m3",
]
AUTH_CANDIDATES = ["bearer", "ncp"]


# ──────────────────────────────────────────────────────────────────────────
# 임베딩 캐시 (SQLite) — 재개의 핵심
# ──────────────────────────────────────────────────────────────────────────

class EmbedCache:
    """임베딩 캐시.

    ⚠️ 캐시 키에는 반드시 provider(모델)를 포함해야 한다.
       text_hash만으로 키를 잡으면 dummy로 만든 256차원 벡터와
       CLOVA의 1024차원 벡터가 같은 칸을 두고 뒤섞여
       "Inconsistent dimensions" 에러가 난다.
    """

    def __init__(self, path: Path, namespace: str) -> None:
        self.path = path
        self.ns = namespace
        self._lock = threading.Lock()
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS emb2 ("
            "  ns TEXT NOT NULL, h TEXT NOT NULL, dim INTEGER NOT NULL,"
            "  vec BLOB NOT NULL, PRIMARY KEY (ns, h))")
        self.con.commit()

    def get(self, h: str) -> list[float] | None:
        row = self.con.execute(
            "SELECT dim, vec FROM emb2 WHERE ns=? AND h=?", (self.ns, h)).fetchone()
        if not row:
            return None
        dim, blob = row
        return list(struct.unpack(f"<{dim}f", blob))

    def put(self, h: str, vec: list[float]) -> None:
        blob = struct.pack(f"<{len(vec)}f", *vec)
        with self._lock:
            self.con.execute("INSERT OR REPLACE INTO emb2 VALUES (?,?,?,?)",
                             (self.ns, h, len(vec), blob))
            self.con.commit()

    def count(self) -> int:
        return self.con.execute(
            "SELECT COUNT(*) FROM emb2 WHERE ns=?", (self.ns,)).fetchone()[0]

    def dims(self) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT DISTINCT dim FROM emb2 WHERE ns=?", (self.ns,))]


class RateLimiter:
    """전역 호출 간격 제한 + 429를 만나면 스스로 느려지는 적응형 리미터.

    재시도만으로는 부족하다. 8개 워커가 동시에 두드리면 재시도도 429를 맞는다.
    쿼터를 모르는 상태에서는 '맞으면 늦추고, 잘 되면 조금씩 당긴다'가 실용적이다.
    """

    def __init__(self, rps: float) -> None:
        self.min_interval = 1.0 / max(rps, 0.05)
        self.floor = self.min_interval
        self._next = 0.0
        self._lock = threading.Lock()
        self._ok_streak = 0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def penalize(self) -> None:
        with self._lock:
            self.min_interval = min(self.min_interval * 1.4, 5.0)
            self._ok_streak = 0

    def reward(self) -> None:
        # 회복이 너무 느리면 한 번 맞은 429 때문에 끝까지 기어간다.
        # 15회 연속 성공마다 20%씩 당겨 실제 한도를 다시 더듬는다.
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 15 and self.min_interval > self.floor:
                self.min_interval = max(self.min_interval * 0.8, self.floor)
                self._ok_streak = 0

    @property
    def rps(self) -> float:
        return 1.0 / self.min_interval


# ──────────────────────────────────────────────────────────────────────────
# 임베딩 제공자
# ──────────────────────────────────────────────────────────────────────────

class DummyEmbedder:
    """API 없이 파이프라인만 점검하기 위한 가짜 임베더. 검색 품질은 무의미하다."""
    name = "dummy"
    dim = 256

    def embed(self, text: str) -> list[float]:
        import math
        seed = hashlib.sha256(text.encode()).digest()
        v = [((seed[i % 32] * (i + 7)) % 251) /
             251.0 - 0.5 for i in range(self.dim)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


class ClovaEmbedder:
    name = "clova"

    def __init__(self) -> None:
        import requests
        self.requests = requests
        self.url = os.environ.get(
            "CLOVA_EMBED_URL", DEFAULT_URL).strip().rstrip("/")
        self.key = os.environ.get("CLOVA_API_KEY", "").strip()
        self.auth = os.environ.get(
            "CLOVA_EMBED_AUTH", "bearer").strip().lower()
        self.gw = os.environ.get("CLOVA_APIGW_KEY", "").strip()
        self.request_id = os.environ.get("CLOVA_EMBED_REQUEST_ID", "").strip()
        if not self.key:
            env = Path(".env")
            sys.exit(
                "CLOVA_API_KEY가 .env에 없습니다.\n\n"
                + (f"  .env는 있습니다 ({env.resolve()}).\n"
                   "  CLOVA_OCR_SECRET만 있고 CLOVA_API_KEY가 빠진 상태일 겁니다.\n"
                   "  OCR과 임베딩은 서로 다른 서비스라 키도 별개입니다.\n"
                   if env.exists() else
                   "  .env 파일 자체가 없습니다.  cp .env.example .env\n")
                + "\n  .env에 아래 줄을 추가하세요:\n"
                  "      CLOVA_API_KEY=발급받은_키\n\n"
                  "  키는 네이버 클라우드 콘솔 > CLOVA Studio 에서 발급합니다.\n"
                  "  (CLOVA OCR 키가 아닙니다)\n"
                  "  최신 .env.example에 항목 설명을 적어두었습니다.")
        self._local = threading.local()
        self.dim: int | None = None
        self.limiter: RateLimiter | None = None

    def _session(self):
        if not hasattr(self._local, "s"):
            self._local.s = self.requests.Session()
        return self._local.s

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.auth == "ncp":
            h["X-NCP-CLOVASTUDIO-API-KEY"] = self.key
            if self.gw:
                h["X-NCP-APIGW-API-KEY"] = self.gw
        else:
            h["Authorization"] = f"Bearer {self.key}"
        # 서비스 앱 식별자. 콘솔 '코드 보기'에 표시된 값을 .env에 넣으면
        # 테스트 앱이 아니라 서비스 앱 한도로 처리된다.
        if self.request_id:
            h["X-NCP-CLOVASTUDIO-REQUEST-ID"] = self.request_id
        return h

    @staticmethod
    def _pluck(js: dict) -> list[float] | None:
        """응답 구조가 버전마다 다르므로 알려진 위치를 차례로 시도한다."""
        for path in (("result", "embedding"), ("embedding",), ("data", 0, "embedding")):
            cur = js
            try:
                for k in path:
                    cur = cur[k]
                if isinstance(cur, list) and cur and isinstance(cur[0], (int, float)):
                    return [float(x) for x in cur]
            except (KeyError, IndexError, TypeError):
                continue
        return None

    def embed(self, text: str, _raw: bool = False):
        body = {"text": text}
        last = None
        for attempt in range(8):
            if self.limiter:
                self.limiter.wait()
            try:
                r = self._session().post(self.url, headers=self._headers(),
                                         json=body, timeout=60)
                if r.status_code == 429 or r.status_code >= 500:
                    if self.limiter:
                        self.limiter.penalize()
                    # Retry-After가 오면 그걸 따른다
                    ra = r.headers.get("Retry-After")
                    try:
                        wait = float(ra) if ra else min(2 ** attempt, 30)
                    except ValueError:
                        wait = min(2 ** attempt, 30)
                    time.sleep(wait)
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                if self.limiter:
                    self.limiter.reward()
                js = r.json()
                if _raw:
                    return js
                vec = self._pluck(js)
                if vec is None:
                    raise RuntimeError(
                        f"응답에서 embedding을 못 찾음: {str(js)[:300]}")
                if self.dim is None:
                    self.dim = len(vec)
                return vec
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                if attempt == 7:
                    raise RuntimeError(last)
                time.sleep(min(2 ** attempt, 20))
        raise RuntimeError(last or "unknown")


# ──────────────────────────────────────────────────────────────────────────
# Chroma 메타데이터 변환
# ──────────────────────────────────────────────────────────────────────────

def to_metadata(r: dict) -> dict:
    """Chroma 메타데이터는 str/int/float/bool만 허용한다. 리스트는 문자열로 만든다.

    doc_ids는 앞뒤에 쉼표를 붙여 저장한다(',A,B,'). 부분검색 시
    ',KR5157450090,' 로 찾으면 접두사가 같은 다른 코드에 걸리지 않는다.
    """
    return {
        "doc_id": r.get("doc_id") or "",
        "doc_type": r.get("doc_type") or "",
        "fund_code": r.get("fund_code") or "",
        "fund_name": r.get("fund_name") or "",
        "base_date": r.get("base_date") or "",
        "page": int(r.get("page") or 0),
        "section": (r.get("section") or "")[:200],
        "content_type": r.get("content_type") or "text",
        "scope": r.get("scope") or "개별",
        "n_docs": int(r.get("n_docs") or 1),
        "doc_ids": "," + ",".join(r.get("doc_ids") or [r.get("doc_id") or ""]) + ",",
        "is_ocr": bool(r.get("is_ocr")),
        "n_chars": int(r.get("n_chars") or len(r.get("text", ""))),
        "source_path": r.get("source_path") or "",
    }


# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="임베딩 + Chroma 인덱싱")
    ap.add_argument("--in", dest="inp", default="./dataset/chunks_final.jsonl")
    ap.add_argument("--db", default="./dataset/chroma")
    ap.add_argument("--cache", default="./dataset/emb_cache.sqlite")
    ap.add_argument("--collection", default="pension")
    ap.add_argument("--provider", choices=["clova", "dummy"], default="clova")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rps", type=float, default=2.0,
                    help="초당 최대 요청 수. 429가 계속 뜨면 낮추세요 (기본 2.0)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="API를 1회만 호출해 응답 구조와 차원 수를 확인하고 종료")
    ap.add_argument("--verify-cache", action="store_true",
                    help="이미 캐싱된 청크를 지금 엔드포인트로 다시 임베딩해 "
                         "같은 벡터가 나오는지 확인 (테스트앱→서비스앱 전환 후 필수)")
    ap.add_argument("--reset", action="store_true", help="컬렉션을 지우고 새로 만든다")
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # ── selftest: 되는 (URL, 인증) 조합을 찾아준다
    if a.selftest:
        import requests
        emb = ClovaEmbedder()
        print(f"API 키: {emb.key[:6]}…{emb.key[-4:]}  ({len(emb.key)}자)")
        print(
            f"앱 구분: {'서비스 앱 (REQUEST-ID 있음)' if emb.request_id else '⚠️ 테스트 앱 (REQUEST-ID 없음)'}\n")

        # .env에 적힌 URL을 맨 앞에 두고, 나머지 후보를 이어붙인다
        urls = [emb.url] + [u for u in URL_CANDIDATES if u != emb.url]
        auths = [emb.auth] + [x for x in AUTH_CANDIDATES if x != emb.auth]

        print(f"후보 {len(urls)}개 경로 × {len(auths)}개 인증 방식을 확인합니다…\n")
        found = None
        total = len(urls) * len(auths)
        i = 0
        for url in urls:
            for auth in auths:
                i += 1
                emb.url, emb.auth = url, auth
                short = url.replace(BASE, "")
                print(f"  [{i}/{total}] {auth:<7} {short}  시도 중…",
                      end=" ", flush=True)
                try:
                    r = emb._session().post(url, headers=emb._headers(),
                                            json={"text": "퇴직연금 중도인출 사유"},
                                            timeout=30)
                except requests.RequestException as e:
                    print(f"→ 연결 실패 ({type(e).__name__})", flush=True)
                    continue
                if r.status_code != 200:
                    print(f"→ HTTP {r.status_code}", flush=True)
                    continue
                try:
                    raw = r.json()
                except ValueError:
                    print("→ JSON 아님", flush=True)
                    continue
                vec = ClovaEmbedder._pluck(raw)
                if vec:
                    found = (url, auth, vec, raw)
                    print("→ ✅ 성공", flush=True)
                    break
                print(f"→ 200이지만 embedding 없음: "
                      f"{json.dumps(raw, ensure_ascii=False)[:120]}", flush=True)
            if found:
                break

        if not found:
            print("\n❌ 되는 조합을 찾지 못했습니다.")
            print("   위 응답 코드를 보고 판단하세요:")
            print("     401/403 → 키 문제. 테스트 API 키를 새로 발급받아 .env에 넣으세요.")
            print("     404     → 경로 문제. 콘솔의 API 가이드에서 임베딩 엔드포인트를 확인해")
            print("               CLOVA_EMBED_URL에 넣어주세요.")
            sys.exit(1)

        url, auth, vec, raw = found
        print(f"\n✅ 성공 — 차원 {len(vec)}")
        print(f"   앞 5개: {[round(x, 4) for x in vec[:5]]}")
        print(f"\n응답 원문(앞부분):\n  {json.dumps(raw, ensure_ascii=False)[:300]}")
        if url != os.environ.get("CLOVA_EMBED_URL", "") or auth != os.environ.get("CLOVA_EMBED_AUTH", ""):
            print("\n.env를 아래처럼 고정하세요:")
            print(f"  CLOVA_EMBED_URL={url}")
            print(f"  CLOVA_EMBED_AUTH={auth}")
        return

    # ── 캐시 호환성 검증
    #
    # 테스트 앱에서 만든 벡터와 서비스 앱에서 만든 벡터가 다른 공간이면,
    # 섞이는 순간 검색이 조용히 망가진다. 에러도 안 나고 결과만 이상해진다.
    # 같은 문장을 다시 임베딩해서 코사인 유사도를 재본다.
    if a.verify_cache:
        import math
        rows = [json.loads(l) for l in Path(
            a.inp).open(encoding="utf-8") if l.strip()]
        for r in rows:
            r.setdefault("text_hash", hashlib.md5(
                r["text"].encode()).hexdigest()[:16])
        emb = ClovaEmbedder()
        cache = EmbedCache(Path(a.cache), namespace=emb.name)
        cached = [r for r in rows if cache.get(r["text_hash"]) is not None][:5]
        if not cached:
            sys.exit("캐시에 비교할 벡터가 없습니다.")

        print(f"캐시된 청크 {len(cached)}개를 현재 엔드포인트로 재임베딩해 비교합니다.")
        print(f"  URL: {emb.url}\n")
        worst = 1.0
        for r in cached:
            old = cache.get(r["text_hash"])
            new = emb.embed(r["text"])
            if len(old) != len(new):
                sys.exit(f"❌ 차원이 다릅니다: 캐시 {len(old)} vs 현재 {len(new)}\n"
                         f"   캐시를 버리고 전부 다시 임베딩해야 합니다.")
            dot = sum(x * y for x, y in zip(old, new))
            n1 = math.sqrt(sum(x * x for x in old))
            n2 = math.sqrt(sum(y * y for y in new))
            cos = dot / (n1 * n2) if n1 and n2 else 0.0
            worst = min(worst, cos)
            print(f"  {r['chunk_id'][:34]:<34} 유사도 {cos:.6f}")

        print()
        if worst > 0.9999:
            print("✅ 동일한 모델입니다. 기존 캐시를 그대로 이어서 쓰세요.")
        elif worst > 0.98:
            print("⚠️ 거의 같지만 미세하게 다릅니다. 실무상 문제없을 가능성이 높지만,")
            print("   시간 여유가 있다면 캐시를 새로 만드는 편이 안전합니다.")
        else:
            print(f"❌ 다른 벡터 공간입니다 (최저 유사도 {worst:.4f}).")
            print("   기존 캐시를 지우고 전부 다시 임베딩하세요:")
            print(f"      rm -f {a.cache}*")
        return

    rows = [json.loads(l)
            for l in Path(a.inp).open(encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print(f"대상 청크 {len(rows):,}개 | provider={a.provider}")

    embedder = DummyEmbedder() if a.provider == "dummy" else ClovaEmbedder()
    limiter = RateLimiter(a.rps)
    if isinstance(embedder, ClovaEmbedder):
        embedder.limiter = limiter
        print(f"  요청 속도 상한 {a.rps}/s (429가 뜨면 자동으로 느려집니다)")
    # 캐시는 provider별로 분리한다 — dummy 벡터와 섞이면 차원이 어긋난다
    cache = EmbedCache(Path(a.cache), namespace=embedder.name)

    # ── 캐시에 없는 것만 골라 임베딩
    for r in rows:
        r.setdefault("text_hash", hashlib.md5(
            r["text"].encode()).hexdigest()[:16])
    todo = [r for r in rows if cache.get(r["text_hash"]) is None]
    print(f"  캐시 보유 {len(rows)-len(todo):,} / 신규 임베딩 {len(todo):,}")
    if todo and isinstance(embedder, ClovaEmbedder):
        print(f"  예상 소요 약 {len(todo)/max(a.rps, 0.01)/60:.0f}분 "
              f"(429가 나면 더 길어집니다)\n", flush=True)

    if todo:
        done = 0
        failed = []
        t0 = time.time()
        last_print = [0.0]
        lock = threading.Lock()
        # 속도가 느리면(2/s) 200건마다 찍는 건 100초 침묵이다.
        # 건수 또는 시간 중 먼저 오는 쪽으로 알린다.
        step = max(10, min(200, len(todo) // 40 or 10))

        def work(r: dict) -> None:
            nonlocal done
            try:
                cache.put(r["text_hash"], embedder.embed(r["text"]))
            except Exception as e:  # noqa: BLE001
                with lock:
                    failed.append((r.get("chunk_id"), str(e)[:120]))
            with lock:
                done += 1
                now = time.time()
                if done % step == 0 or done == len(todo) or now - last_print[0] > 10:
                    last_print[0] = now
                    el = now - t0
                    rate = done / el if el else 0
                    eta = (len(todo) - done) / rate if rate else 0
                    bar_n = int(24 * done / len(todo))
                    bar = "█" * bar_n + "·" * (24 - bar_n)
                    extra = f", 실패 {len(failed)}" if failed else ""
                    cap = (f"  상한 {limiter.rps:.2f}/s"
                           if isinstance(embedder, ClovaEmbedder) else "")
                    print(f"  [{bar}] {done:,}/{len(todo):,}  "
                          f"{rate:.2f}/s{cap}  남은 {eta/3600:.1f}시간{extra}"
                          if eta > 3600 else
                          f"  [{bar}] {done:,}/{len(todo):,}  "
                          f"{rate:.2f}/s{cap}  남은 {eta/60:.0f}분{extra}", flush=True)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, todo))

        if failed:
            n429 = sum(1 for _, e in failed if "429" in e)
            print(f"\n⚠️ 임베딩 실패 {len(failed)}건 (다시 실행하면 재시도됩니다)")
            for cid, err in failed[:3]:
                print(f"   {cid}: {err}")
            if n429:
                print(f"\n   {n429}건이 요청 한도(429) 초과입니다.")
                print(f"   현재 상한 {limiter.rps:.2f}/s. 다시 실행할 때 낮춰보세요:")
                print(f"      python embed_and_index.py --rps 1 --workers 2")

    # ── Chroma 적재
    dims = cache.dims()
    if len(dims) > 1:
        sys.exit(f"캐시에 차원이 뒤섞여 있습니다: {dims}\n"
                 f"  --cache 파일을 지우고 다시 실행하세요.")

    import chromadb
    client = chromadb.PersistentClient(path=a.db)
    if a.reset:
        try:
            client.delete_collection(a.collection)
            print("기존 컬렉션 삭제됨")
        except Exception:  # noqa: BLE001
            pass

    # 기존 컬렉션의 차원이 지금 벡터와 다르면 upsert가 통째로 실패한다.
    # (예: dummy 256차원으로 만든 컬렉션에 CLOVA 1024차원을 넣는 경우)
    # 애매한 에러 대신 먼저 잡아서 해결책을 알려준다.
    if dims:
        try:
            existing = client.get_collection(a.collection)
            if existing.count():
                got = existing.peek(limit=1).get("embeddings")
                if got is not None and len(got) and len(got[0]) != dims[0]:
                    sys.exit(
                        f"\n❌ 컬렉션 '{a.collection}'은 {len(got[0])}차원으로 만들어졌는데 "
                        f"지금 벡터는 {dims[0]}차원입니다.\n"
                        f"   (dummy로 테스트한 컬렉션이 남아있을 때 생깁니다)\n\n"
                        f"   아래처럼 컬렉션을 새로 만드세요:\n"
                        f"      python embed_and_index.py --reset")
        except Exception as e:  # noqa: BLE001
            if "does not exist" not in str(e) and "NotFound" not in type(e).__name__:
                raise

    col = client.get_or_create_collection(
        name=a.collection, metadata={"hnsw:space": "cosine"})

    ready = [r for r in rows if cache.get(r["text_hash"]) is not None]
    print(
        f"\nChroma 적재: {len(ready):,}개 → {a.db} / {a.collection}  ({dims[0] if dims else '?'}차원)")

    B = 500
    for i in range(0, len(ready), B):
        batch = ready[i:i + B]
        col.upsert(
            ids=[r["chunk_id"] for r in batch],
            embeddings=[cache.get(r["text_hash"]) for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[to_metadata(r) for r in batch],
        )
        print(f"  {min(i+B, len(ready)):,}/{len(ready):,}")

    print(f"\n완료. 컬렉션 크기: {col.count():,}")
    print(f"  캐시: {cache.count():,}건 ({embedder.name} / {a.cache})")
    missing = len(rows) - len(ready)
    if missing:
        print(f"  ⚠️ 아직 임베딩 안 된 청크 {missing:,}건 — 같은 명령을 다시 실행하세요")
    print("\n다음: python search.py \"퇴직연금 중도인출 사유가 뭐야?\"")


if __name__ == "__main__":
    main()
