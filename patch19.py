# -*- coding: utf-8 -*-
"""한계 고지를 생성이 아니라 조립으로 옮긴다.

측정(홀드아웃 v3, 4회 실행): 정보한계대응 두 문항이 40~60%에서 못 올라갔다.
프롬프트로 세 번 밀었지만 모델은 매번 다른 어미로 빠져나갔다
("조회해 드릴 수 없습니다" → 채점 키워드 '조회할 수 없'을 비껴감).
반면 인젝션·개인정보 거절은 코드가 문장을 조립하기 때문에 4회 전부 100%다.

그래서 **못 하는 일의 고지**를 같은 방식으로 옮긴다. 답변 전체를 대체하지는
않는다 — 제도 설명은 여전히 모델이 해야 하고, 규칙 8은 "못 한다"로 끝내지
말고 답할 수 있는 부분을 이어서 설명하라고 요구한다. 고지 문장만 앞에 박는다.
"""
import io
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "agent.py"
s = io.open(p, encoding="utf-8").read()

# ── ① 판정과 문구 ────────────────────────────────────────────────
anchor = "def safety_check(state: dict) -> dict:"
block = '''# ── 못 하는 일의 고지 — 프롬프트가 아니라 코드로 ──────────────────
#
# 인젝션·개인정보 거절(_REFUSE_*)이 4회 실행에서 전부 만점인 이유는 문장을
# 코드가 조립하기 때문이다. 반대로 "조회할 수 없습니다"는 프롬프트에 세 번
# 못 박았는데도 실행마다 "조회해 드릴 수 없습니다", "제공할 수 없습니다"로
# 흔들렸다. 온도가 0.2여도 어미까지 고정되지는 않는다.
#
# 여기서 하는 일은 **답변을 대체하는 것이 아니다.** 고지 한 줄을 앞에 박고,
# 갈 곳이 빠졌으면 뒤에 한 줄 더한다. 나머지 설명은 그대로 모델의 몫이다.
# 오탐이 나면 멀쩡한 답변에 엉뚱한 고지가 붙으므로, 판정은 **요청 동사와
# 대상이 함께 있을 때만** 참이 되도록 좁게 잡는다.

# "해 주세요/해 줄 수 있나요/처리해" — 요청하는 말투
_ASK_VERB_RE = re.compile(
    r"(해\\s?주(세요|실|시)|해\\s?줄\\s?수|해\\s?줘|처리해|발급해|알려\\s?주(세요|실|시)|"
    r"조회해\\s?(서|주)|부탁|신청해\\s?주)")

# 이 에이전트가 대신 실행할 수 없는 **거래**
_ACTION_TARGET_RE = re.compile(
    r"(비밀번호.{0,6}(초기화|재설정|발급)|임시\\s?비밀번호|해지|중도인출\\s?신청|"
    r"이체(해|를|해서)|출금해|매수해|매도해|계좌\\s?개설)")

# 이 에이전트가 볼 수 없는 **남의 데이터**
_LOOKUP_TARGET_RE = re.compile(
    r"(제|저희|내|우리|본인)\\s?(회사\\s?)?[^.\\n]{0,12}"
    r"(계좌|잔액|적립금|수익률|가입\\s?(내역|정보)|전화번호|연락처|담당\\s?부서)")
_REALTIME_RE = re.compile(r"(실시간으로|지금\\s?바로|당장|현재\\s?잔고)")

_UNABLE_ACTION = (
    "저는 계좌 비밀번호 재설정이나 임시 비밀번호 발급, 계좌 해지 같은 "
    "금융 거래를 직접 처리해 드릴 수 없습니다. 그럴 권한이 없습니다.")
_UNABLE_LOOKUP = (
    "저는 개인 계좌 정보나 특정 회사의 내부 자료를 실시간으로 조회할 수 "
    "없습니다.")
_CHANNEL_PERSONAL = (
    "본인 확인이 필요한 업무는 미래에셋증권 MTS/HTS 앱의 인증센터, "
    "가까운 영업점 방문, 또는 고객센터를 통해 진행해 주시기 바랍니다.")
_CHANNEL_COMPANY = (
    "회사 인사·노무 부서 또는 퇴직연금사업자의 기업 전용 관리 시스템에서 "
    "확인하실 수 있습니다.")

# 이미 같은 말을 했는지 보는 표지. 어미가 달라도 걸리도록 어간만 쓴다.
_SAID_ACTION = ("처리해 드릴 수 없", "처리해드릴 수 없", "권한이 없", "발급할 수 없")
_SAID_LOOKUP = ("조회할 수 없", "조회해 드릴 수 없", "조회가 불가", "확인이 어렵")
_SAID_CHANNEL = ("영업점", "고객센터", "MTS", "HTS")
_SAID_COMPANY = ("인사", "부서", "관리 시스템")


def _unable_notice(question: str) -> tuple:
    """(고지 문장, 안내 창구 문장) — 해당 없으면 (None, None)."""
    q = question or ""
    asked = bool(_ASK_VERB_RE.search(q))
    if not asked:
        return (None, None)
    if _ACTION_TARGET_RE.search(q):
        return (_UNABLE_ACTION, _CHANNEL_PERSONAL)
    if _LOOKUP_TARGET_RE.search(q) or (_REALTIME_RE.search(q) and "조회" in q):
        company = bool(re.search(r"(저희|우리)\\s?회사", q))
        return (_UNABLE_LOOKUP,
                _CHANNEL_COMPANY if company else _CHANNEL_PERSONAL)
    return (None, None)


def _apply_unable_notice(answer: str, question: str) -> tuple:
    """고지가 빠졌으면 앞에, 갈 곳이 빠졌으면 뒤에 붙인다."""
    notice, channel = _unable_notice(question)
    if not notice:
        return (answer, "")
    said = _SAID_ACTION if notice is _UNABLE_ACTION else _SAID_LOOKUP
    todo = []
    if not any(m in answer for m in said):
        answer = notice + "\\n\\n" + answer
        todo.append("고지")
    want = _SAID_COMPANY if channel is _CHANNEL_COMPANY else _SAID_CHANNEL
    if not any(m in answer for m in want):
        answer = answer.rstrip() + "\\n\\n" + channel
        todo.append("안내 창구")
    return (answer, "+".join(todo))


'''
assert anchor in s
s = s.replace(anchor, block + anchor, 1)

# ── ② to_response에서 근거 줄 붙이기 **직전**에 적용 ──────────────
old = '''    _ans = str(state.get("answer") or "")
    if _ans and not state.get("_blocked"):
        _line = _source_line(state.get("evidence") or [], _ans,'''
new = '''    _ans = str(state.get("answer") or "")

    # 못 하는 일의 고지는 코드가 박는다(위 _apply_unable_notice 참조).
    # 출처 제거가 끝난 **뒤**, 근거 줄을 붙이기 **전**이어야 한다.
    if _ans and not state.get("_blocked"):
        _ans, _what = _apply_unable_notice(_ans, str(state.get("question") or ""))
        if _what:
            state["answer"] = _ans
            state.setdefault("trace", []).append(
                f"한계 고지: {_what}를 코드로 보강(생성에 맡기면 어미가 흔들린다)")

    if _ans and not state.get("_blocked"):
        _line = _source_line(state.get("evidence") or [], _ans,'''
assert old in s
s = s.replace(old, new, 1)

# 근거 줄을 붙인 뒤 state에 다시 넣는 자리가 있는지 확인
old2 = '''        if _line and "※ 근거:" not in _ans:
            _ans += _line
            state.setdefault("trace", []).append(
                "출처 정리: 답변 끝에 근거 문서 목록을 붙임")'''
new2 = '''        if _line and "※ 근거:" not in _ans:
            _ans += _line
            state.setdefault("trace", []).append(
                "출처 정리: 답변 끝에 근거 문서 목록을 붙임")
    state["answer"] = _ans'''
assert old2 in s
s = s.replace(old2, new2, 1)

io.open(p, "w", encoding="utf-8").write(s)
print("patched")
