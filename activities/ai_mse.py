# activities/ai_mse.py
from __future__ import annotations
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st

TITLE = "평균제곱오차(MSE)로 예측함수 비교"


# -----------------------------
# 예측함수
# -----------------------------
def f1(x: float) -> float:
    return x - 0.5


def f2(x: float) -> float:
    return 0.5 * x + 0.5


# -----------------------------
# 유틸
# -----------------------------
def _to_float(v) -> float:
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _fmt_num(v: float) -> str:
    """LaTeX에 넣기 좋은 숫자 문자열."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{?}"
    v2 = float(np.round(float(v), 10))
    if abs(v2 - int(v2)) < 1e-10:
        return str(int(v2))
    s = f"{v2:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "-0.0") else s


def _latex_paren(v: float) -> str:
    """음수일 때 ( -0.5 )처럼 괄호로 감싸기."""
    s = _fmt_num(v)
    if s.startswith("-"):
        return rf"\left({s}\right)"
    return s


# -----------------------------
# 기본 입력(빈칸)
# -----------------------------
def _default_inputs_blank() -> pd.DataFrame:
    # 학생이 문제를 보고 스스로 채우도록 빈칸(np.nan)
    return pd.DataFrame(
        {
            "사례": ["P", "Q", "R"],
            "입력값 x": [np.nan, np.nan, np.nan],
            "출력값 y": [np.nan, np.nan, np.nan],
        }
    )


# -----------------------------
# 계산/검증
# -----------------------------
def _compute_view(df_in: pd.DataFrame, f) -> pd.DataFrame:
    df = df_in.copy()

    # 컬럼 보정(혹시 깨졌을 때)
    for col in ["사례", "입력값 x", "출력값 y"]:
        if col not in df.columns:
            df[col] = np.nan

    df["입력값 x"] = df["입력값 x"].map(_to_float)
    df["출력값 y"] = df["출력값 y"].map(_to_float)

    df["예측값 f(x)"] = df["입력값 x"].map(lambda x: f(x) if not np.isnan(x) else np.nan)
    df["오차 y-f(x)"] = df["출력값 y"] - df["예측값 f(x)"]

    return df[["사례", "입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]]


def _valid_for_mse(df_view: pd.DataFrame) -> bool:
    need = ["입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]
    return not df_view[need].isna().any().any()


def _mse_fraction(df_view: pd.DataFrame) -> Fraction:
    """
    MSE를 분수 형태(Fraction)로 계산.
    """
    errors = df_view["오차 y-f(x)"].to_numpy()

    # 오차^2를 분수로 계산
    sq_sum = Fraction(0, 1)
    for e in errors:
        if np.isnan(e):
            return None
        fe = Fraction(str(e))  # 소수 -> 분수 변환
        sq_sum += fe * fe

    n = len(errors)
    return sq_sum / n



# -----------------------------
# MSE 식(숫자 대입 형태)
# -----------------------------
def _latex_mse_substitution(df_view: pd.DataFrame) -> str:
    n = len(df_view)
    terms = []
    for _, r in df_view.iterrows():
        y = r["출력값 y"]
        yhat = r["예측값 f(x)"]
        terms.append(rf"\left({_fmt_num(y)} - {_latex_paren(yhat)}\right)^2")
    joined = " + ".join(terms)
    return rf"\text{{MSE}}=\frac{{1}}{{{n}}}\left\{{{joined}\right\}}"


# -----------------------------
# 렌더
# -----------------------------
def render(show_title: bool = True, key_prefix: str = "ai_mse") -> None:
    if show_title:
        st.subheader(TITLE)

    # 공통 입력표(한 번 입력)
    ss_in = f"{key_prefix}_inputs_shared"
    if ss_in not in st.session_state:
        st.session_state[ss_in] = _default_inputs_blank()

    # -----------------------------
    # 문제(글+수식 혼합: calculus 스타일)
    # -----------------------------
    with st.expander("문제", expanded=True):
        st.markdown(
            r"""
어느 대나무 세 그루가 각각 $x$일 동안 자라는 길이 $y\text{m}$를 조사한 결과의 순서쌍 $(x,y)$가 각각 다음과 같다.

$$
P(1,1),\quad Q(2,2),\quad R(3,2)
$$

이 대나무가 어떤 기간 $x$에 대하여 자란 길이 $y$를 예측하는 두 함수가 각각 다음과 같다고 하자.

$$
f_1(x)=x-0.5,\qquad f_2(x)=0.5x+0.5
$$

1) 예측함수 $f_1(x)$에 대한 평균제곱오차 $E(1,-0.5)$의 값을 구하시오.  
2) 예측함수 $f_2(x)$에 대한 평균제곱오차 $E(0.5,0.5)$의 값을 구하시오.  
3) 두 예측함수 $f_1, f_2$ 중에서 자료의 경향성을 더 잘 나타내는 것을 고르시오.
"""
        )

    with st.sidebar:
        if st.button("입력표 초기화", key=f"{key_prefix}_reset_shared"):
            st.session_state[ss_in] = _default_inputs_blank()
            st.rerun()

        st.markdown(
            r"""
- 아래 표에서 **입력값 $x$**, **출력값 $y$**를 한 번만 입력하세요.  
- 입력값을 바꾸면 $f_1, f_2$의 예측값/오차, MSE가 자동으로 갱신됩니다.
"""
        )

    # -----------------------------
    # 1) 공통 입력
    # -----------------------------
    st.markdown("### 1) 데이터 입력")
    df_in = st.data_editor(
        st.session_state[ss_in],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "사례": st.column_config.TextColumn(disabled=True),
            "입력값 x": st.column_config.NumberColumn(step=1.0),
            "출력값 y": st.column_config.NumberColumn(step=1.0),
        },
        key=f"{key_prefix}_editor_shared",
    )
    st.session_state[ss_in] = df_in

    # 공통 입력 -> 두 모델 계산
    df_view1 = _compute_view(df_in, f1)
    df_view2 = _compute_view(df_in, f2)

    left, right = st.columns([1, 1], gap="large")

    # -----------------------------
    # f1
    # -----------------------------
    with left:
        st.markdown(r"### $f_1(x)=x-0.5$")
        st.dataframe(df_view1, use_container_width=True, hide_index=True)

        st.markdown("### $f_1$의 평균제곱오차(MSE)")
        st.markdown(f"$$\n{_latex_mse_substitution(df_view1)}\n$$")

        if _valid_for_mse(df_view1):
            mse1 = _mse_fraction(df_view1)
            if mse1 is not None:
                st.markdown(f"$$E(1,-0.5)=\\frac{{{mse1.numerator}}}{{{mse1.denominator}}}$$")
        else:
            st.info("P, Q, R의 $x$와 $y$를 모두 입력하면 $E(1,-0.5)$를 계산할 수 있습니다.")

    # -----------------------------
    # f2
    # -----------------------------
    with right:
        st.markdown(r"### $f_2(x)=0.5x+0.5$")
        st.dataframe(df_view2, use_container_width=True, hide_index=True)

        st.markdown("### $f_2$의 평균제곱오차(MSE)")
        st.markdown(f"$$\n{_latex_mse_substitution(df_view2)}\n$$")

        if _valid_for_mse(df_view2):
            mse2 = _mse_fraction(df_view2)
            st.markdown(f"$$E(0.5,0.5)=\\frac{{{mse2.numerator}}}{{{mse2.denominator}}}$$")
        else:
            st.info("P, Q, R의 $x$와 $y$를 모두 입력하면 $E(0.5,0.5)$를 계산할 수 있습니다.")

    # -----------------------------
    # 3) 비교
    # -----------------------------
st.divider()
st.markdown("## 2) 어떤 함수가 더 적절한가?")

mse1_fr = _mse_fraction(df_view1)
mse2_fr = _mse_fraction(df_view2)

if mse1_fr is None or mse2_fr is None:
    st.warning("P, Q, R의 $x$와 $y$를 모두 입력하면 선택 문제를 풀 수 있습니다.")
else:
    # 정답 결정(더 작은 MSE)
    if mse1_fr < mse2_fr:
        correct = "f_1"
        correct_label = r"$f_1$"
    elif mse2_fr < mse1_fr:
        correct = "f_2"
        correct_label = r"$f_2$"
    else:
        correct = "same"
        correct_label = r"$f_1, f_2$ (동일)"

    st.markdown("다음 중 평균제곱오차(MSE)가 더 작아 자료의 경향성을 더 잘 나타내는 함수를 고르시오.")

    choice = st.radio(
        "선택",
        options=["f_1", "f_2"],
        format_func=lambda v: r"$f_1$" if v == "f_1" else r"$f_2$",
        key=f"{key_prefix}_choice",
        horizontal=True,
    )

    # 제출 버튼(즉시 채점)
    if st.button("정답 확인", key=f"{key_prefix}_check"):
        if correct == "same":
            st.info("두 함수의 MSE가 같아서 어느 쪽을 골라도 동일한 적합도입니다.")
        elif choice == correct:
            st.success("정답입니다!")
        else:
            st.error("오답입니다.")

        # 근거(분수 형태)도 함께 제시
        st.markdown("근거(MSE):")
        st.markdown(
            rf"$$E(1,-0.5) = {_latex_frac(mse1_fr)},\qquad E(0.5,0.5) = {_latex_frac(mse2_fr)}$$"
        )
        st.markdown(rf"따라서 정답은 {correct_label} 입니다.")
