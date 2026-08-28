#!/usr/bin/env python3
"""평가셋 한 번 실행에 HCX 토큰을 얼마나 썼는지 집계한다.

agent.py가 문항마다 think_trace에 남긴 토큰 기록을 읽어 합산한다.
실행이 끝난 뒤에도, 여러 실행을 나란히 비교할 때도 쓸 수 있다.

    python token_report.py raw_penguinnote_integrated_v4.json
    python token_report.py raw_v3.json raw_v4.json      # 여러 개 비교
"""
import json
import re
import sys
from pathlib import Path

# "토큰: 입력 7,123 + 출력 412 = 7,535 (HCX 호출 2회)"
PAT = re.compile(
    r"토큰: 입력 ([\d,]+) \+ 출력 ([\d,]+) = ([\d,]+) \(HCX 호출 (\d+)회\)")

# NCP 문서 기준 (guide.ncloud-docs.com/docs/en/clovastudio-ratelimiting)
LIMITS = {"테스트/플레이그라운드": (60, 60_000), "서비스 앱": (180, 300_000)}


def num(s):
    return int(s.replace(",", ""))


def report(path: Path):
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        cands = sorted(Path(".").glob("raw_*.json"))
        if cands:
            print("\n이 폴더에 있는 실행 결과 파일:")
            for c in cands:
                print(f"  {c.name}")
        print("\n토큰 기록은 집계 기능이 들어간 뒤의 실행부터 남습니다.")
        print("한 문항만 먼저 재보려면:")
        print('  python agent.py "퇴직연금 중도인출 사유가 뭐야?"')
        return None

    d = json.loads(path.read_text(encoding="utf-8"))
    results = d.get("results", d if isinstance(d, list) else [])

    rows, missing = [], 0
    for r in results:
        m = PAT.search(r.get("think_trace") or "")
        if not m:
            missing += 1
            continue
        rows.append({
            "qid": r.get("question_id"),
            "prompt": num(m.group(1)), "completion": num(m.group(2)),
            "total": num(m.group(3)), "calls": int(m.group(4)),
            "sec": r.get("elapsed_sec") or 0,
        })

    print("=" * 68)
    print(f"{path.name}")
    print("=" * 68)
    if not rows:
        print("  토큰 기록이 없습니다.")
        print("  이 기록은 agent.py에 집계가 들어간 뒤의 실행부터 남습니다.")
        print("  (예전 결과 파일에는 없는 것이 정상)")
        return None

    n = len(rows)
    P = sum(r["prompt"] for r in rows)
    C = sum(r["completion"] for r in rows)
    T = sum(r["total"] for r in rows)
    K = sum(r["calls"] for r in rows)
    S = sum(r["sec"] for r in rows)

    if missing:
        print(f"  ⚠ 토큰 기록이 없는 문항 {missing}개는 제외했습니다")
    print(f"  문항 {n}개 | HCX 호출 {K}회 | 총 소요 {S/60:.1f}분")
    print()
    print(f"  입력(프롬프트) {P:>9,}  ({P/T*100:.0f}%)")
    print(f"  출력(생성)     {C:>9,}  ({C/T*100:.0f}%)")
    print(f"  ─ 합계         {T:>9,}")
    print()
    print(f"  문항당 평균    {T/n:>9,.0f}")
    print(f"  호출당 평균    {T/K:>9,.0f}")
    print()

    # 무거운 문항
    top = sorted(rows, key=lambda r: -r["total"])[:5]
    print("  가장 많이 쓴 문항")
    for r in top:
        print(f"    {r['qid']}  {r['total']:>7,} "
              f"(입력 {r['prompt']:,} / 출력 {r['completion']:,}, "
              f"호출 {r['calls']}회)")
    print()

    # 분당 한도 대비
    if S > 0:
        tpm = T / (S / 60)
        print(f"  실측 분당 토큰(TPM) 약 {tpm:,.0f}")
        for name, (qpm, lim) in LIMITS.items():
            pct = tpm / lim * 100
            mark = "⚠️ 한도 근접" if pct >= 70 else "여유"
            print(f"    {name:16} 한도 {lim:,} → {pct:5.1f}% 사용  {mark}")
        print()
        print("  ※ 실제 TPM은 이보다 큽니다. NCP는 출력 실측치가 아니라")
        print("    maxCompletionTokens(예약분)를 TPM에 반영합니다.")
    return {"n": n, "total": T, "prompt": P, "completion": C}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    outs = []
    for a in sys.argv[1:]:
        s = report(Path(a))
        if s:
            outs.append((Path(a).name, s))
        print()
    if len(outs) > 1:
        print("=" * 68)
        print("비교")
        print("=" * 68)
        base = outs[0][1]["total"]
        for name, s in outs:
            d = (s["total"] - base) / base * 100 if base else 0
            print(f"  {name:44} {s['total']:>9,}  ({d:+.1f}%)")


if __name__ == "__main__":
    main()
