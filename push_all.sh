#!/usr/bin/env bash
#
# 원격 3곳에 올린다.  사용법:  bash push_all.sh
#
#   origin/main         penguinnote/Mirae_Naver_MultiAgent   (개인 저장소)
#   pen/penguinnote     miraeasset-.../pen-056               (작업 브랜치)
#   pen/main            miraeasset-.../pen-056               (심사자가 보는 브랜치)
#
# 이 폴더의 .git은 잠금 파일을 지울 수 없다(mv만 된다). 그래서 git을 부르기
# 전마다 잠금을 _to_delete/로 치운다. 이게 없으면 "unable to unlink"로 죽는다.
#
# backup-20260828 브랜치는 **어떤 경우에도 올리지 않는다.**
# 그 브랜치에는 이력에서 걷어낸 Claude 트레일러가 그대로 남아 있다.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

unlock() { mkdir -p _to_delete; mv .git/*.lock _to_delete/ 2>/dev/null; true; }
die() { printf '\n중단: %s\n' "$1" >&2; exit 1; }

unlock
BR=$(git rev-parse --abbrev-ref HEAD)
[ "$BR" = "main" ] || die "현재 브랜치가 main이 아니라 '$BR' 입니다."

echo "── 사전 점검 ─────────────────────────────────────────────"

unlock
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  git status --short --untracked-files=no
  die "커밋 안 된 변경이 있습니다. 먼저 커밋하거나 되돌리세요."
fi
echo "  ✅ 작업 트리 깨끗함"

unlock
N=$(git log origin/main..HEAD --format='%H%n%B' \
      | grep -icE 'co-authored-by: *claude|claude-session|claude\.ai/code')
[ "$N" = "0" ] || die "푸시 대상 커밋에 Claude 트레일러가 ${N}건 있습니다."
echo "  ✅ 커밋 메시지에 Claude 흔적 없음"

unlock
LEAK=$(git ls-files \
        | grep -E '^(gold_holdout_|gold_private_|killing_camp_|raw_|scored_|공유_|분석_|questions_only_)')
[ -z "$LEAK" ] || { printf '%s\n' "$LEAK"; die "정답지·실행결과가 추적되고 있습니다."; }
echo "  ✅ 정답지·실행결과 추적 안 됨"

# 위 검사는 이름 "앞"만 본다. base_with_gold_*.json / result_with_gold_*.csv 처럼
# gold가 중간에 낀 파생물이 전부 빠져나갔다. 위치를 가리지 않고 다시 본다.
unlock
# 이 3개는 2026-09-01에 내용을 확인했다. gold_answer/answer_points가
# 나오는 자리는 전부 딕셔너리 키이고, 정답 문자열은 파일에 없다.
# 스크립트를 수정하면 이 확인은 무효다. 다시 열어보고 판단해라.
# 마감 후에는 파일명에서 gold를 빼는 개명으로 이 예외를 없앨 것.
GOLD_OK='make_raw_from_gold.py
merge_eval_result_with_gold_v4.py
merge_v5_result_with_gold.py'
LEAK2=$(git ls-files | grep -i 'gold' | grep -vxF "$GOLD_OK")
[ -z "$LEAK2" ] || { printf '%s\n' "$LEAK2"; die "이름에 gold가 든 파일이 추적되고 있습니다."; }
echo "  ✅ gold 파생 파일 추적 안 됨"

# 실행 결과는 전부 정답지에서 파생된다. result_ 로 시작하면 무조건 막는다.
unlock
LEAK3=$(git ls-files | grep -E '^result_')
[ -z "$LEAK3" ] || { printf '%s\n' "$LEAK3"; die "실행 결과(result_*)가 추적되고 있습니다."; }
echo "  ✅ 실행 결과(result_*) 추적 안 됨"

# git ls-files 는 한글 경로를 \353\266\204 같은 8진 이스케이프로 내보내고(quotepath),
# macOS는 파일명을 NFD로 저장한다. 그래서 위 grep의 '공유_|분석_' 한글 패턴은
# 사실 한 번도 걸린 적이 없다. -z 로 원문을 받아 NFC로 되돌린 뒤 다시 본다.
unlock
LEAK4=$(git ls-files -z | python3 -c '
import re, sys, unicodedata
pat = re.compile(r"CLAUDE|작업지시|지시문|공유_|분석_", re.I)
for f in sys.stdin.buffer.read().decode("utf-8").split("\0"):
    if f and pat.search(unicodedata.normalize("NFC", f)):
        print(f)
')
[ -z "$LEAK4" ] || { printf '%s\n' "$LEAK4"; die "Claude 작업 문서·공유 리포트가 추적되고 있습니다."; }
echo "  ✅ Claude 작업 문서 추적 안 됨"

echo
echo "── 올라갈 커밋 ───────────────────────────────────────────"
unlock
git log --oneline origin/main..HEAD
echo
printf '  origin/main      %s → %s\n' "$(git rev-parse --short origin/main)" "$(git rev-parse --short HEAD)"
printf '  pen/penguinnote  %s → %s\n' "$(git rev-parse --short pen/penguinnote)" "$(git rev-parse --short HEAD)"
printf '  pen/main         %s → %s  (덮어씀: 원격에 Initial commit만 있음)\n' \
       "$(git rev-parse --short pen/main)" "$(git rev-parse --short HEAD)"
echo
printf '진행할까요? [y/N] '
read -r yn
case "$yn" in [yY]*) ;; *) echo "취소했습니다."; exit 0 ;; esac

echo
echo "── 푸시 ──────────────────────────────────────────────────"

unlock
echo "[1/3] origin main"
git push origin main || die "origin main 실패"

unlock
echo "[2/3] pen penguinnote"
git push pen main:penguinnote || die "pen penguinnote 실패"

# pen/main에는 조직이 만든 'Initial commit'(README 2줄)만 있고 우리 이력과
# 조상이 겹치지 않는다. 그래서 일반 푸시로는 안 올라간다.
# --force-with-lease는 방금 fetch한 상태와 원격이 같을 때만 덮어쓴다.
# 팀원이 그 사이에 뭔가 올렸다면 실패하면서 알려준다. 맹목적 --force와 다르다.
unlock
echo "[3/3] pen main  (force-with-lease)"
git push --force-with-lease=main:"$(git rev-parse pen/main)" pen main:main \
  || die "pen main 실패 — 원격이 바뀌었을 수 있습니다. git fetch pen 후 다시 확인하세요."

unlock
echo
echo "── 결과 ──────────────────────────────────────────────────"
git fetch --quiet origin pen 2>/dev/null
unlock
for r in origin/main pen/main pen/penguinnote; do
  printf '  %-18s %s\n' "$r" "$(git rev-parse --short "$r" 2>/dev/null)"
done
echo "  HEAD               $(git rev-parse --short HEAD)"
echo
echo "✅ 완료"
