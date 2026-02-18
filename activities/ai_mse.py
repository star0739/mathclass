# activities/ai_mse.py
from __future__ import annotations

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
# 표현 유틸 (LaTeX에 들어갈 숫자/괄호)
# -----------------------------
def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def _fmt_num(v: float) -> str:
    """LaTeX에 넣기 좋은 숫자 문자열(정수는 정수로, 그 외는 적당한 소수)."""
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
# 기본 데이터
# -----------------------------
def _default_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "사례": ["P", "Q", "R"],
            "입력값 x": [1.0, 2.0, 3.0],
            "출력값 y": [1.0, 2.0, 2.0],
        }
    )


# -----------------------------
# 계산
# -----------------------------
def _compute(df_in: pd.DataFrame, f) -> pd.DataFrame:
    df = df_in.copy()
    df["x"] = df["입력값 x"].map(_to_float)
    df["y"] = df["출력값 y"].map(_to_float)

    df["yhat"] = df["x"].map(lambda t: f(t) if not np.isnan(t) else np.nan)
    df["err"] = df["y"] - df["yhat"]
    df["sqerr"] = df["err"] ** 2

    return pd.DataFrame(
        {
            "사례": df["사례"],
            "입력값 x": df["x"],
            "출력값 y": df["y"],
            "예측값 f(x)": df["yhat"],
            "오차 y-f(x)": df["err"],
            "제곱오차 (y-f(x))^2": df["sqerr"],
        }
    )


def _is_valid(df_calc: pd.DataFrame) -> bool:
    # x,y가 모두 숫자이고 결측이 없어야 MSE 계산/표현 가능
    return not df_calc[["입력값 x", "출력값 y", "예측값 f(x)"]].isna().any().any()


def _mse_value(df_calc: pd.DataFrame) -> float:
    return float(np.mean(df_calc["오차 y-f(x)"].to_numpy() ** 2))


# -----------------------------
# LaTeX: "식 형태"를 보여주기
# -----------------------------
def _latex_mse_substitution(df_calc: pd.DataFrame) -> str:
    """
    MSE = 1/3{(y1-yhat1)^2 + (y2-yhat2)^2 + (y3-yhat3)^2}
    형태로, 실제 숫자를 대입해 보여줌.
    """
    n = len(df_calc)

    terms = []
    for _, r in df_calc.iterrows():
        y = r["출력값 y"]
        yhat = r["예측값 f(x)"]
        terms.append(rf"\left({_fmt_num(y)} - {_latex_paren(yhat)}\right)^2")

    joined = " + ".join(terms) if terms else r"\text{(데이터 없음)}"
    return rf"""
$$
\text{{MSE}}=\frac{{1}}{{{n}}}\left\{{ {joined} \right\}}
$$
"""


def _latex_mse_numbers(df_calc: pd.DataFrame) -> str:
    """
    제곱오차 값까지 대입된 형태:
    MSE = 1/3{a + b + c}
    를 보여줌(계산값은 보여주되, '형태' 유지).
    """
    n = len(df_calc)
    vals = [r["제곱오차 (y-f(x))^2"] for _, r in df_calc.iterrows()]
    joined = " + ".join(_latex_paren(v) for v in vals) if vals else r"\text{(데이터 없음)}"
    return rf"""
$$
\text{{MSE}}=\frac{{1}}{{{n}}}\left\{{ {joined} \right\}}
$$
"""


# -----------------------------
# 렌더
# -----------------------------
def render(show_title: bool = True, key_prefix: str = "ai_mse") -> None:
    if show_title:
        st.subheader(TITLE)

    # 세션 초기화
    ss_key = f"{key_prefix}_df"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = _default_input_df()

    with st.expander("문제", expanded=True):
        st.markdown(
            r"""
어느 대나무 세 그루가 각각 \(x\)일 동안 자라는 길이 \(y\) m를 조사한 결과의 순서쌍 \((x, y)\)가 각각
\(P(1,1),\;Q(2,2),\;R(3,2)\) 라 하자.

이 대나무가 어떤 기간 \(x\)에 대하여 자란 길이 \(y\)를 예측하는 두 함수가 각각
\[
f_1(x)=x-0.5,\qquad f_2(x)=0.5x+0.5
\]
라 하자.

(1) \(f_1\)에 대한 평균제곱오차 \(E(1,-0.5)\)의 값을 구하시오.  
(2) \(f_2\)에 대한 평균제곱오차 \(E(0.5,0.5)\)의 값을 구하시오.  
(3) 두 예측함수 \(f_1, f_2\) 중 자료의 경향성을 더 잘 나타내는 것을 고르시오.
"""
        )

    # 입력
    left, right = st.columns([1, 1], gap="large")

    with st.sidebar:
        if st.button("기본값으로 초기화", key=f"{key_prefix}_reset"):
            st.session_state[ss_key] = _default_input_df()
            st.rerun()

    st.markdown("### 1) 데이터 입력 (P, Q, R의 x, y)")
    df_edit = st.data_editor(
        st.session_state[ss_key],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "사례": st.column_config.TextColumn(disabled=True),
            "입력값 x": st.column_config.NumberColumn(step=1.0),
            "출력값 y": st.column_config.NumberColumn(step=1.0),
        },
        key=f"{key_prefix}_editor",
    )
    st.session_state[ss_key] = df_edit

    # 계산
    df1 = _compute(df_edit, f1)
    df2 = _compute(df_edit, f2)

    # 표시
    with left:
        st.markdown(r"## \(f_1(x)=x-0.5\)")
        st.dataframe(df1, use_container_width=True, hide_index=True)

        st.markdown("### (1) MSE 식 (예측값 대입 형태)")
        st.markdown(_latex_mse_substitution(df1))

        st.markdown("### 제곱오차까지 계산해 대입한 형태")
        st.markdown(_latex_mse_numbers(df1))

        if _is_valid(df1):
            mse1 = _mse_value(df1)
            st.markdown(rf"$$E(1,-0.5) = \text{{MSE}}_1 = {_fmt_num(mse1)}$$")
        else:
            st.info("x, y에 빈칸(또는 숫자가 아닌 값)이 있으면 계산할 수 없습니다.")

    with right:
        st.markdown(r"## \(f_2(x)=0.5x+0.5\)")
        st.dataframe(df2, use_container_width=True, hide_index=True)

        st.markdown("### (2) MSE 식 (예측값 대입 형태)")
        st.markdown(_latex_mse_substitution(df2))

        st.markdown("### 제곱오차까지 계산해 대입한 형태")
        st.markdown(_latex_mse_numbers(df2))

        if _is_valid(df2):
            mse2 = _mse_value(df2)
            st.markdown(rf"$$E(0.5,0.5) = \text{{MSE}}_2 = {_fmt_num(mse2)}$$")
        else:
            st.info("x, y에 빈칸(또는 숫자가 아닌 값)이 있으면 계산할 수 없습니다.")

    st.divider()
    st.markdown("### (3) 어떤 함수가 자료의 경향성을 더 잘 나타내는가?")

    if _is_valid(df1) and _is_valid(df2):
        mse1 = _mse_value(df1)
        mse2 = _mse_value(df2)

        if abs(mse1 - mse2) < 1e-12:
            st.success(r"두 예측함수의 평균제곱오차가 같습니다. (동일한 적합도)")
        elif mse1 < mse2:
            st.success(rf"\(\text{{MSE}}_1 < \text{{MSE}}_2\) 이므로 \(f_1\)이 더 적절합니다.")
        else:
            st.success(rf"\(\text{{MSE}}_2 < \text{{MSE}}_1\) 이므로 \(f_2\)가 더 적절합니다.")
    else:
        st.warning("먼저 P, Q, R의 x와 y를 모두 숫자로 입력해 주세요.")


if __name__ == "__main__":
    # 단독 실행(디버그)용
    try:
        st.set_page_config(page_title=TITLE, layout="wide")
    except Exception:
        pass
    render(show_title=True, key_prefix="ai_mse_debug")
