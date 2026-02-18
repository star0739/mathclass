# activities/ai_mse.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

TITLE = "평균제곱오차(MSE)로 예측함수 비교"


def f1(x: float) -> float:
    return x - 0.5


def f2(x: float) -> float:
    return 0.5 * x + 0.5


def _to_float(v) -> float:
    try:
        if v is None or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _fmt_num(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\text{?}"
    v2 = float(np.round(float(v), 10))
    if abs(v2 - int(v2)) < 1e-10:
        return str(int(v2))
    s = f"{v2:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "-0.0") else s


def _latex_paren(v: float) -> str:
    s = _fmt_num(v)
    if s.startswith("-"):
        return rf"\left({s}\right)"
    return s


def _default_inputs_blank() -> pd.DataFrame:
    # 학생이 문제 보고 스스로 채우도록 빈칸
    return pd.DataFrame(
        {"사례": ["P", "Q", "R"], "입력값 x": [np.nan, np.nan, np.nan], "출력값 y": [np.nan, np.nan, np.nan]}
    )


def _compute_view(df_in: pd.DataFrame, f) -> pd.DataFrame:
    df = df_in.copy()
    df["입력값 x"] = df["입력값 x"].map(_to_float)
    df["출력값 y"] = df["출력값 y"].map(_to_float)

    df["예측값 f(x)"] = df["입력값 x"].map(lambda x: f(x) if not np.isnan(x) else np.nan)
    df["오차 y-f(x)"] = df["출력값 y"] - df["예측값 f(x)"]

    return df[["사례", "입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]]


def _valid_for_mse(df_view: pd.DataFrame) -> bool:
    return not df_view[["입력값 x", "출력값 y", "예측값 f(x)", "오차 y-f(x)"]].isna().any().any()


def _mse_value(df_view: pd.DataFrame) -> float:
    e = df_view["오차 y-f(x)"].to_numpy(dtype=float)
    return float(np.mean(e**2))


def _latex_mse_substitution(df_view: pd.DataFrame) -> str:
    n = len(df_view)
    terms = []
    for _, r in df_view.iterrows():
        y = r["출력값 y"]
        yhat = r["예측값 f(x)"]
        terms.append(rf"\left({_fmt_num(y)} - {_latex_paren(yhat)}\right)^2")
    joined = " + ".join(terms)
    return rf"\text{{MSE}}=\frac{{1}}{{{n}}}\left\{{{joined}\right\}}"


def render(show_title: bool = True, key_prefix: str = "ai_mse") -> None:
    if show_title:
        st.subheader(TITLE)

    # 편집용 입력표(각 모델별로 따로)만 세션에 저장: 사례,x,y
    ss_in1 = f"{key_prefix}_inputs_f1"
    ss_in2 = f"{key_prefix}_inputs_f2"
    if ss_in1 not in st.session_state:
        st.session_state[ss_in1] = _default_inputs_blank()
    if ss_in2 not in st.session_state:
        st.session_state[ss_in2] = _default_inputs_blank()

    with st.expander("문제", expanded=True):
        st.write("세 그루의 대나무에 대해 (x일, y m) 데이터를 관측하였다.")
        st.latex(r"P(1,1),\quad Q(2,2),\quad R(3,2)")
        st.write("두 예측함수는 다음과 같다.")
        st.latex(r"f_1(x)=x-0.5,\qquad f_2(x)=0.5x+0.5")
        st.write("(1) f₁에 대한 평균제곱오차 E(1,-0.5)를 구하시오.")
        st.write("(2) f₂에 대한 평균제곱오차 E(0.5,0.5)를 구하시오.")
        st.write("(3) 두 함수 중 자료의 경향성을 더 잘 나타내는 것을 고르시오.")

    with st.sidebar:
        if st.button("두 표 모두 초기화", key=f"{key_prefix}_reset_all"):
            st.session_state[ss_in1] = _default_inputs_blank()
            st.session_state[ss_in2] = _default_inputs_blank()
            st.rerun()
        st.write("각 표에서 x, y만 입력하세요. 예측값과 오차는 아래에서 자동으로 계산됩니다.")

    left, right = st.columns([1, 1], gap="large")

    # ---------------- f1 ----------------
    with left:
        st.write("f₁(x)=x-0.5")
        st.latex(r"f_1(x)=x-0.5")

        st.write("입력(학생이 직접 채우기):")
        df_in1 = st.data_editor(
            st.session_state[ss_in1],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "사례": st.column_config.TextColumn(disabled=True),
                "입력값 x": st.column_config.NumberColumn(step=1.0),
                "출력값 y": st.column_config.NumberColumn(step=1.0),
            },
            key=f"{key_prefix}_editor_inputs_f1",
        )
        st.session_state[ss_in1] = df_in1

        df_view1 = _compute_view(df_in1, f1)
        st.write("자동 계산(예측값, 오차):")
        st.dataframe(df_view1, use_container_width=True, hide_index=True)

        st.write("MSE 식(숫자 대입 형태):")
        st.latex(_latex_mse_substitution(df_view1))

        if _valid_for_mse(df_view1):
            mse1 = _mse_value(df_view1)
            st.write("계산 결과:")
            st.latex(rf"E(1,-0.5)=\text{{MSE}}_1={_fmt_num(mse1)}")
        else:
            st.info("P,Q,R의 x와 y를 모두 입력하면 E(1,-0.5)가 계산됩니다.")

    # ---------------- f2 ----------------
    with right:
        st.write("f₂(x)=0.5x+0.5")
        st.latex(r"f_2(x)=0.5x+0.5")

        st.write("입력(학생이 직접 채우기):")
        df_in2 = st.data_editor(
            st.session_state[ss_in2],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "사례": st.column_config.TextColumn(disabled=True),
                "입력값 x": st.column_config.NumberColumn(step=1.0),
                "출력값 y": st.column_config.NumberColumn(step=1.0),
            },
            key=f"{key_prefix}_editor_inputs_f2",
        )
        st.session_state[ss_in2] = df_in2

        df_view2 = _compute_view(df_in2, f2)
        st.write("자동 계산(예측값, 오차):")
        st.dataframe(df_view2, use_container_width=True, hide_index=True)

        st.write("MSE 식(숫자 대입 형태):")
        st.latex(_latex_mse_substitution(df_view2))

        if _valid_for_mse(df_view2):
            mse2 = _mse_value(df_view2)
            st.write("계산 결과:")
            st.latex(rf"E(0.5,0.5)=\text{{MSE}}_2={_fmt_num(mse2)}")
        else:
            st.info("P,Q,R의 x와 y를 모두 입력하면 E(0.5,0.5)가 계산됩니다.")

    # ---------------- 비교 ----------------
    st.divider()
    st.write("(3) 어떤 함수가 더 적절한가?")

    df_view1 = _compute_view(st.session_state[ss_in1], f1)
    df_view2 = _compute_view(st.session_state[ss_in2], f2)

    if _valid_for_mse(df_view1) and _valid_for_mse(df_view2):
        mse1 = _mse_value(df_view1)
        mse2 = _mse_value(df_view2)
        if abs(mse1 - mse2) < 1e-12:
            st.success("두 함수의 평균제곱오차가 같습니다.")
        elif mse1 < mse2:
            st.success("MSE가 더 작은 f₁이 자료의 경향성을 더 잘 나타냅니다.")
        else:
            st.success("MSE가 더 작은 f₂가 자료의 경향성을 더 잘 나타냅니다.")
    else:
        st.warning("두 표 모두에서 P,Q,R의 x와 y를 입력하면 비교할 수 있습니다.")


if __name__ == "__main__":
    try:
        st.set_page_config(page_title=TITLE, layout="wide")
    except Exception:
        pass
    render(show_title=True, key_prefix="ai_mse_debug")
