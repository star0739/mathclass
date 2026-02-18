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
# 숫자/LaTeX 유틸
# -----------------------------
def _to_float(v) -> float:
    try:
        if v is None:
            return np.nan
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
# 기본 입력표(빈칸)
# -----------------------------
def _default_table_blank() -> pd.DataFrame:
    # 학생이 문제를 보고 스스로 채우도록 x,y는 빈칸(np.nan)
    return pd.DataFrame(
        {
            "사례": ["P", "Q", "R"],
            "입력값 x": [np.nan, np.nan, np.nan],
            "출력값 y": [np.nan, np.nan, np.nan],
            "예측값 f(x)": [np.nan, np.nan, np.nan],
            "오차 y-f(x)": [np.nan, np.nan, np.nan],
        }
    )


# -----------------------------
# 표 계산(예측값/오차 자동 갱신)
# -----------------------------
def _recompute_table(df_edit: pd.DataFrame, f) -> pd.DataFrame:
    df = df_edit.copy()

    # 필요한 컬럼이 없다면 보정
    for col in ["사례", "입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]:
        if col not in df.columns:
            df[col] = np.nan

    df["입력값 x"] = df["입력값 x"].map(_to_float)
    df["출력값 y"] = df["출력값 y"].map(_to_float)

    def _yhat(x):
        return f(x) if not np.isnan(x) else np.nan

    df["예측값 f(x)"] = df["입력값 x"].map(_yhat)
    df["오차 y-f(x)"] = df["출력값 y"] - df["예측값 f(x)"]
    return df[["사례", "입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]]


def _is_valid_for_mse(df: pd.DataFrame) -> bool:
    need = ["입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]
    return not df[need].isna().any().any()


def _mse_value(df: pd.DataFrame) -> float:
    e = df["오차 y-f(x)"].to_numpy(dtype=float)
    return float(np.mean(e**2))


# -----------------------------
# MSE LaTeX(식 형태: 대입된 형태를 보여줌)
# -----------------------------
def _latex_mse_substitution(df: pd.DataFrame) -> str:
    """
    $$\\text{MSE}=\\frac{1}{3}\\{(y_1-\\hat y_1)^2+(y_2-\\hat y_2)^2+(y_3-\\hat y_3)^2\\}$$
    에 숫자를 대입한 형태로 출력.
    """
    n = len(df)
    terms = []
    for _, r in df.iterrows():
        y = r["출력값 y"]
        yhat = r["예측값 f(x)"]
        terms.append(rf"\left({_fmt_num(y)} - {_latex_paren(yhat)}\right)^2")
    joined = " + ".join(terms) if terms else r"\text{(데이터 없음)}"
    return rf"""
$$
\text{{MSE}}
=
\frac{{1}}{{{n}}}
\left\{{ {joined} \right\}}
$$
"""


# -----------------------------
# 렌더
# -----------------------------
def render(show_title: bool = True, key_prefix: str = "ai_mse") -> None:
    if show_title:
        st.subheader(TITLE)

    # 세션(표 2개: f1용 / f2용)
    ss_f1 = f"{key_prefix}_df_f1"
    ss_f2 = f"{key_prefix}_df_f2"
    if ss_f1 not in st.session_state:
        st.session_state[ss_f1] = _default_table_blank()
    if ss_f2 not in st.session_state:
        st.session_state[ss_f2] = _default_table_blank()

    with st.expander("문제", expanded=True):
        st.markdown(
            r"""
$$
P(1,1),\; Q(2,2),\; R(3,2)
$$

$$
f_1(x)=x-0.5,\qquad f_2(x)=0.5x+0.5
$$

$$
(1)\;\; f_1 \text{에 대한 평균제곱오차 } E(1,-0.5)\text{의 값을 구하시오.}
$$
$$
(2)\;\; f_2 \text{에 대한 평균제곱오차 } E(0.5,0.5)\text{의 값을 구하시오.}
$$
$$
(3)\;\; f_1,\;f_2 \text{ 중 자료의 경향성을 더 잘 나타내는 것을 고르시오.}
$$
"""
        )

    with st.sidebar:
        if st.button("두 표 모두 초기화", key=f"{key_prefix}_reset_all"):
            st.session_state[ss_f1] = _default_table_blank()
            st.session_state[ss_f2] = _default_table_blank()
            st.rerun()

        st.markdown(
            r"""
- 각 표에서 **입력값 \(x\)**, **출력값 \(y\)** 를 직접 입력하세요.  
- **예측값**, **오차**는 자동으로 갱신됩니다.
"""
        )

    left, right = st.columns([1, 1], gap="large")

    # -----------------------------
    # f1 테이블
    # -----------------------------
    with left:
        st.markdown(r"## $$f_1(x)=x-0.5$$")

        # 현재 편집값을 기반으로 (예측값/오차) 갱신된 df를 에디터에 공급
        df1_current = _recompute_table(st.session_state[ss_f1], f1)

        df1_edit = st.data_editor(
            df1_current,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "사례": st.column_config.TextColumn(disabled=True),
                "입력값 x": st.column_config.NumberColumn(step=1.0),
                "출력값 y": st.column_config.NumberColumn(step=1.0),
                "예측값 f(x)": st.column_config.NumberColumn(disabled=True),
                "오차 y-f(x)": st.column_config.NumberColumn(disabled=True),
            },
            key=f"{key_prefix}_editor_f1",
        )

        # 학생 편집 결과를 세션에 저장 후, 다시 자동 계산값으로 덮어씀(예측/오차는 항상 최신)
        st.session_state[ss_f1] = _recompute_table(df1_edit, f1)

        st.markdown(r"### $$\text{MSE 식(숫자 대입 형태)}$$")
        st.markdown(_latex_mse_substitution(st.session_state[ss_f1]))

        if _is_valid_for_mse(st.session_state[ss_f1]):
            mse1 = _mse_value(st.session_state[ss_f1])
            st.markdown(rf"$$E(1,-0.5)=\text{{MSE}}_1={_fmt_num(mse1)}$$")
        else:
            st.info(r"$$x,\;y \text{ 를 모두 입력하면 } E(1,-0.5)\text{ 를 계산할 수 있습니다.}$$")

    # -----------------------------
    # f2 테이블
    # -----------------------------
    with right:
        st.markdown(r"## $$f_2(x)=0.5x+0.5$$")

        df2_current = _recompute_table(st.session_state[ss_f2], f2)

        df2_edit = st.data_editor(
            df2_current,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "사례": st.column_config.TextColumn(disabled=True),
                "입력값 x": st.column_config.NumberColumn(step=1.0),
                "출력값 y": st.column_config.NumberColumn(step=1.0),
                "예측값 f(x)": st.column_config.NumberColumn(disabled=True),
                "오차 y-f(x)": st.column_config.NumberColumn(disabled=True),
            },
            key=f"{key_prefix}_editor_f2",
        )

        st.session_state[ss_f2] = _recompute_table(df2_edit, f2)

        st.markdown(r"### $$\text{MSE 식(숫자 대입 형태)}$$")
        st.markdown(_latex_mse_substitution(st.session_state[ss_f2]))

        if _is_valid_for_mse(st.session_state[ss_f2]):
            mse2 = _mse_value(st.session_state[ss_f2])
            st.markdown(rf"$$E(0.5,0.5)=\text{{MSE}}_2={_fmt_num(mse2)}$$")
        else:
            st.info(r"$$x,\;y \text{ 를 모두 입력하면 } E(0.5,0.5)\text{ 를 계산할 수 있습니다.}$$")

    # -----------------------------
    # 비교 결론
    # -----------------------------
    st.divider()
    st.markdown(r"## $$(3)\;\;\text{어떤 함수가 더 적절한가?}$$")

    df1 = st.session_state[ss_f1]
    df2 = st.session_state[ss_f2]

    if _is_valid_for_mse(df1) and _is_valid_for_mse(df2):
        mse1 = _mse_value(df1)
        mse2 = _mse_value(df2)

        if abs(mse1 - mse2) < 1e-12:
            st.success(r"$$\text{두 예측함수의 평균제곱오차가 같습니다.}$$")
        elif mse1 < mse2:
            st.success(r"$$\text{MSE}_1<\text{MSE}_2 \;\Rightarrow\; f_1 \text{ 이 더 적절합니다.}$$")
        else:
            st.success(r"$$\text{MSE}_2<\text{MSE}_1 \;\Rightarrow\; f_2 \text{ 가 더 적절합니다.}$$")
    else:
        st.warning(r"$$f_1,\;f_2 \text{ 표의 } x,\;y \text{ 를 모두 입력하면 비교가 가능합니다.}$$")


if __name__ == "__main__":
    try:
        st.set_page_config(page_title=TITLE, layout="wide")
    except Exception:
        pass
    render(show_title=True, key_prefix="ai_mse_debug")
