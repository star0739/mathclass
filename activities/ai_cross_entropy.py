# activities/ai_cross_entropy.py
from __future__ import annotations

import math
import pandas as pd
import streamlit as st

TITLE = "교차 엔트로피(Cross Entropy)"


# -----------------------------
# 고정: 시그모이드 모델 & 데이터
# -----------------------------
A = 8
B = -4  # ax+b=0 => x=0.5에서 f(x)=0.5

DATA = [
    ("A", 0.20, 0),
    ("B", 0.40, 0),
    ("C", 0.55, 1),
    ("D", 0.70, 1),
    ("E", 0.90, 1),
]


def sigmoid(z: float) -> float:
    # 오버/언더플로 방지(간단 클램프)
    if z > 50:
        return 1.0
    if z < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def f(x: float) -> float:
    return sigmoid(A * x + B)


def ce_loss(y: int, p: float) -> float:
    # log(0) 방지
    eps = 1e-15
    p = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _parse_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # NaN 방지
        if isinstance(val, float) and val != val:
            return None
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _default_student_table():
    # 학생 입력: f(x_i)와 손실 L_i
    rows = []
    for case, x, y in DATA:
        rows.append(
            {
                "장면": case,
                "특징값 x": x,
                "실제 y": y,
                "예측확률 f(x)": "",
                "손실 L_i": "",
            }
        )
    return pd.DataFrame(rows)


def _answer_table():
    rows = []
    for case, x, y in DATA:
        p = f(x)
        L = ce_loss(y, p)
        rows.append({"장면": case, "p": p, "L": L})
    return pd.DataFrame(rows)


def render(show_title: bool = True, key_prefix: str = "ai_ce") -> None:
    if show_title:
        st.subheader(TITLE)

    ss_tbl = f"{key_prefix}_tbl"
    ss_choice_focus = f"{key_prefix}_focus_choice"

    if ss_tbl not in st.session_state:
        st.session_state[ss_tbl] = _default_student_table()
    if ss_choice_focus not in st.session_state:
        ss_choice_focus = f"{key_prefix}_focus_choice"
        st.session_state[ss_choice_focus] = "boundary"

    with st.sidebar:
        if st.button("초기화", key=f"{key_prefix}_reset"):
            st.session_state[ss_tbl] = _default_student_table()
            st.session_state[ss_choice_focus] = "boundary"
            st.rerun()

        st.markdown(
            r"""
- **계산기 사용 가능** (자연로그 \(\ln\) 사용)
- 입력은 소수로 해도 됩니다. (예: 0.73)
- 정답 확인은 “맞/틀”만 표시됩니다.
"""
        )

    with st.expander("상황", expanded=True):
        st.markdown(
            rf"""
어린이 보호 구역 표지판을 인식하는 인공지능이 다음 **예측함수**로 “어린이 보호 구역(\(y=1\)) / 일반 도로(\(y=0\))”를 판단한다고 하자.

예측함수(시그모이드):

$$
f(x)=\frac{{1}}{{1+e^{{-(ax+b)}}}}
$$

여기서

$$
a={A},\quad b={B}
$$

다음 5개의 장면에 대해 실제 정답 \(y\)와 특징값 \(x\)는 아래 표와 같다.
"""
        )

    # 데이터(정답) 제시 표
    st.markdown("### 주어진 데이터")
    st.dataframe(_true_df(), use_container_width=True, hide_index=True)

    # 손실함수 제시
    st.markdown(
        r"""
교차 엔트로피 손실(이진 분류)은 다음과 같이 계산한다.

$$
L_i = -\left[y_i\ln(f(x_i))+(1-y_i)\ln(1-f(x_i))\right]
$$

$$
E(a,b)=\frac{1}{n}\sum_{i=1}^{n} L_i
$$
"""
    )

    st.markdown("### 활동 1: 예측확률과 손실을 직접 계산해 표를 완성하시오")
    st.markdown(
        r"""
- 각 장면에 대해 \(f(x_i)\)를 계산해 **예측확률** 칸에 입력한다.
- 이어서 각 장면의 손실 \(L_i\)를 계산해 입력한다.
"""
    )

    df_edit = st.data_editor(
        st.session_state[ss_tbl],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "장면": st.column_config.TextColumn(disabled=True),
            "특징값 x": st.column_config.NumberColumn(disabled=True),
            "실제 y": st.column_config.NumberColumn(disabled=True),
            "예측확률 f(x)": st.column_config.TextColumn(),
            "손실 L_i": st.column_config.TextColumn(),
        },
        key=f"{key_prefix}_editor",
    )
    st.session_state[ss_tbl] = df_edit

    # 정답 확인(간단)
    ans = _answer_table()

    def _check_student():
        # 학생 입력 파싱
        p_s = [_parse_float(v) for v in df_edit["예측확률 f(x)"].tolist()]
        L_s = [_parse_float(v) for v in df_edit["손실 L_i"].tolist()]

        # 모두 입력했는지
        if any(v is None for v in p_s) or any(v is None for v in L_s):
            return False, "빈칸이 있습니다. 모든 칸을 채워주세요."

        # 오차 허용(수치 계산 오차 고려)
        tol_p = 0.02      # 확률 ±0.02
        tol_L = 0.05      # 손실 ±0.05

        ok = True
        for i in range(len(DATA)):
            if abs(p_s[i] - float(ans.loc[i, "p"])) > tol_p:
                ok = False
            if abs(L_s[i] - float(ans.loc[i, "L"])) > tol_L:
                ok = False

        return ok, ""

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        if st.button("정답 확인", key=f"{key_prefix}_check"):
            ok, msg = _check_student()
            if msg:
                st.warning(msg)
            else:
                if ok:
                    st.success("정답입니다!")
                else:
                    st.error("오답입니다. 다시 생각해보세요.")

    # 활동 2: 해석(핵심)
    st.divider()
    st.markdown("### 활동 2: 현실적으로 해석하기")

    # 내부 정답 기준으로 "손실이 큰 장면"을 정답으로 삼아 해석 유도
    max_idx = int(ans["L"].idxmax())
    max_case = ans.loc[max_idx, "장면"]

    st.markdown(
        r"""
교차 엔트로피는 **“확신을 가지고 틀린 예측”**에 큰 벌점을 준다.
이제 다음 질문에 답해보자.
"""
    )

    st.markdown(
        f"""
- (1) 손실 \(L_i\)가 가장 크게 나오는 장면은 무엇이라고 생각하는가?  
  → 위 표에서 계산한 값으로 판단하자.
"""
    )
    st.markdown(
        f"""
- (2) 만약 이 인공지능의 오류를 줄이려면, **어떤 종류의 데이터를 더 많이 학습**시키는 것이 좋을까?
"""
    )

    st.radio(
        "집중 학습 데이터 선택",
        options=["boundary", "easy"],
        format_func=lambda v: "판단이 헷갈리는(경계/오판) 데이터에 집중" if v == "boundary" else "이미 잘 맞는(쉬운) 데이터에 집중",
        key=ss_choice_focus,
    )

    if st.button("선택 정답 확인", key=f"{key_prefix}_check_focus"):
        # 정답 논리: 손실이 큰(헷갈리거나 확신 오판) 사례를 줄이려면 그런 데이터에 집중
        if st.session_state[ss_choice_focus] == "boundary":
            st.success("정답입니다!")
        else:
            st.error("오답입니다. 다시 생각해보세요.")

        st.markdown(
            f"""
참고: 이 예측함수에서는 손실이 가장 크게 나오는 장면이 **{max_case}**입니다.  
따라서 모델을 개선하려면 **손실이 큰 장면과 비슷한 데이터(오판/경계 데이터)**를 더 많이 학습시키는 것이 합리적입니다.
"""
        )


def _true_df() -> pd.DataFrame:
    # 주어진 데이터(고정)
    return pd.DataFrame({"장면": [d[0] for d in DATA], "특징값 x": [d[1] for d in DATA], "실제 y": [d[2] for d in DATA]})


if __name__ == "__main__":
    try:
        st.set_page_config(page_title=TITLE, layout="wide")
    except Exception:
        pass
    render(show_title=True, key_prefix="ai_ce_debug")
