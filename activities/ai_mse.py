# activities/ai_mse.py
from __future__ import annotations

from fractions import Fraction
import numpy as np
import pandas as pd
import streamlit as st

TITLE = "평균제곱오차(MSE)로 예측함수 비교"


# -----------------------------
# 데이터(고정)
# -----------------------------
CASES = ["P", "Q", "R"]
X_TRUE = [1, 2, 3]
Y_TRUE = [1, 2, 2]


def _true_df() -> pd.DataFrame:
    return pd.DataFrame({"사례": CASES, "입력값 x": X_TRUE, "출력값 y": Y_TRUE})


# -----------------------------
# 예측함수(정답용)
# -----------------------------
def f1(x: float) -> float:
    return x - 0.5


def f2(x: float) -> float:
    return 0.5 * x + 0.5


# -----------------------------
# 파싱/표현 유틸
# -----------------------------
def _parse_fraction(s) -> Fraction | None:
    """
    학생 입력을 Fraction으로 파싱.
    - 1/4, -3/2 같은 분수
    - 0.25 같은 소수도 허용
    - 빈칸/파싱 실패는 None
    """
    if s is None:
        return None
    if isinstance(s, (int, np.integer)):
        return Fraction(int(s), 1)
    if isinstance(s, (float, np.floating)):
        if np.isnan(s):
            return None
        return Fraction(str(float(s)))
    if isinstance(s, str):
        t = s.strip()
        if t == "":
            return None
        try:
            return Fraction(t)
        except Exception:
            try:
                return Fraction(str(float(t)))
            except Exception:
                return None
    return None


def _frac_latex(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    return rf"\frac{{{fr.numerator}}}{{{fr.denominator}}}"


def _num_to_frac_exact(x: float) -> Fraction:
    """0.5 같은 값을 정확 분수로(정답용)."""
    return Fraction(str(float(x)))


def _mse_from_errors(errors: list[Fraction]) -> Fraction:
    n = len(errors)
    s = Fraction(0, 1)
    for e in errors:
        s += e * e
    return s / n


# -----------------------------
# 정답 계산(분수)
# -----------------------------
def _answer_for_model(f) -> dict:
    # 예측값(분수)
    yhat = [_num_to_frac_exact(f(x)) for x in X_TRUE]
    # 오차(분수) = y - yhat
    err = [Fraction(y, 1) - yh for y, yh in zip(Y_TRUE, yhat)]
    # mse
    mse = _mse_from_errors(err)
    return {"yhat": yhat, "err": err, "mse": mse}


ANS1 = _answer_for_model(f1)
ANS2 = _answer_for_model(f2)


# -----------------------------
# 학생 입력 표(모델별)
# -----------------------------
def _blank_student_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "사례": CASES,
            "입력값 x": X_TRUE,
            "출력값 y": Y_TRUE,
            "예측값 f(x)": ["", "", ""],   # 학생 입력(문자열/분수)
            "오차 y-f(x)": ["", "", ""],   # 학생 입력(문자열/분수)
        }
    )


def _table_to_student_fracs(df: pd.DataFrame) -> tuple[list[Fraction | None], list[Fraction | None]]:
    """표에서 예측값/오차를 Fraction 리스트로 변환."""
    yhat_s = [_parse_fraction(v) for v in df["예측값 f(x)"].tolist()]
    err_s = [_parse_fraction(v) for v in df["오차 y-f(x)"].tolist()]
    return yhat_s, err_s


def _all_filled(vals: list[Fraction | None]) -> bool:
    return all(v is not None for v in vals)


def _check_list_eq(student: list[Fraction | None], answer: list[Fraction]) -> bool:
    if not _all_filled(student):
        return False
    return all(s == a for s, a in zip(student, answer))


# -----------------------------
# 렌더
# -----------------------------
def render(show_title: bool = True, key_prefix: str = "ai_mse") -> None:
    if show_title:
        st.subheader(TITLE)

    # 세션 상태
    ss_t1 = f"{key_prefix}_tbl_f1"
    ss_t2 = f"{key_prefix}_tbl_f2"
    ss_mse1 = f"{key_prefix}_mse1"
    ss_mse2 = f"{key_prefix}_mse2"
    ss_choice = f"{key_prefix}_choice"

    if ss_t1 not in st.session_state:
        st.session_state[ss_t1] = _blank_student_table()
    if ss_t2 not in st.session_state:
        st.session_state[ss_t2] = _blank_student_table()
    if ss_mse1 not in st.session_state:
        st.session_state[ss_mse1] = ""
    if ss_mse2 not in st.session_state:
        st.session_state[ss_mse2] = ""
    if ss_choice not in st.session_state:
        st.session_state[ss_choice] = "f_1"

    with st.sidebar:
        if st.button("전체 초기화", key=f"{key_prefix}_reset_all"):
            st.session_state[ss_t1] = _blank_student_table()
            st.session_state[ss_t2] = _blank_student_table()
            st.session_state[ss_mse1] = ""
            st.session_state[ss_mse2] = ""
            st.session_state[ss_choice] = "f_1"
            st.rerun()

        st.markdown(
            r"""
- 데이터는 이미 주어져 있습니다: $P(1,1),Q(2,2),R(3,2)$  
- 각 모델에서 **예측값**과 **오차**를 직접 계산해 입력하세요.  
- 분수 입력 예: `1/2`, `-3/2`  (소수 입력도 가능)
"""
        )

    with st.expander("문제", expanded=True):
        st.markdown(
            r"""
관측 데이터는 다음과 같다.

$$
P(1,1),\quad Q(2,2),\quad R(3,2)
$$

두 예측함수는 다음과 같다.

$$
f_1(x)=x-0.5,\qquad f_2(x)=0.5x+0.5
$$

(1) $f_1$에 대한 평균제곱오차 $E(1,-0.5)$를 구하시오.  
(2) $f_2$에 대한 평균제곱오차 $E(0.5,0.5)$를 구하시오.  
(3) $f_1, f_2$ 중 자료의 경향성을 더 잘 나타내는 것을 고르시오.
"""
        )

    # 공통 안내: MSE 형태(빈칸 형태)
    st.markdown(
        r"""
평균제곱오차는 다음과 같이 계산한다.

$$
\text{MSE}=\frac{1}{3}\left\{(y_1-\hat y_1)^2+(y_2-\hat y_2)^2+(y_3-\hat y_3)^2\right\}
$$
"""
    )

    left, right = st.columns([1, 1], gap="large")

    # -----------------------------
    # f1
    # -----------------------------
    with left:
        st.markdown(r"## $f_1(x)=x-0.5$")

        st.markdown("### 표를 완성하시오 (예측값과 오차를 직접 계산)")
        df1 = st.data_editor(
            st.session_state[ss_t1],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "사례": st.column_config.TextColumn(disabled=True),
                "입력값 x": st.column_config.NumberColumn(disabled=True),
                "출력값 y": st.column_config.NumberColumn(disabled=True),
                "예측값 f(x)": st.column_config.TextColumn(),
                "오차 y-f(x)": st.column_config.TextColumn(),
            },
            key=f"{key_prefix}_editor_f1",
        )
        st.session_state[ss_t1] = df1

        # 학생 입력 -> 분수
        yhat_s, err_s = _table_to_student_fracs(df1)

        # 학생이 입력한 오차로 "식 형태"를 보여줌(완성 느낌)
        eP, eQ, eR = [(_frac_latex(v) if v is not None else r"\square") for v in err_s]
        st.markdown("학생이 입력한 오차로 MSE 식을 채우면:")
        st.markdown(
            rf"""
$$
\text{{MSE}}=\frac{{1}}{{3}}\left\{\left({eP}\right)^2+\left({eQ}\right)^2+\left({eR}\right)^2\right\}
$$
"""
        )

        st.markdown("### 최종 MSE를 입력하시오")
        st.text_input("E(1,-0.5) =", key=ss_mse1, placeholder="예: 1/4 또는 0.25")

        if st.button("f₁ 정답 확인", key=f"{key_prefix}_check_f1"):
            ok_yhat = _check_list_eq(yhat_s, ANS1["yhat"])
            ok_err = _check_list_eq(err_s, ANS1["err"])
            mse_in = _parse_fraction(st.session_state[ss_mse1])
            ok_mse = (mse_in is not None) and (mse_in == ANS1["mse"])

            if ok_yhat and ok_err and ok_mse:
                st.success("정답입니다!")
            else:
                st.error("오답입니다. 다시 생각해보세요.")

    # -----------------------------
    # f2
    # -----------------------------
    with right:
        st.markdown(r"## $f_2(x)=0.5x+0.5$")

        st.markdown("### 표를 완성하시오 (예측값과 오차를 직접 계산)")
        df2 = st.data_editor(
            st.session_state[ss_t2],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "사례": st.column_config.TextColumn(disabled=True),
                "입력값 x": st.column_config.NumberColumn(disabled=True),
                "출력값 y": st.column_config.NumberColumn(disabled=True),
                "예측값 f(x)": st.column_config.TextColumn(),
                "오차 y-f(x)": st.column_config.TextColumn(),
            },
            key=f"{key_prefix}_editor_f2",
        )
        st.session_state[ss_t2] = df2

        yhat_s, err_s = _table_to_student_fracs(df2)

        eP, eQ, eR = [(_frac_latex(v) if v is not None else r"\square") for v in err_s]
        st.markdown("학생이 입력한 오차로 MSE 식을 채우면:")
        st.markdown(
            rf"""
$$
\text{{MSE}}=\frac{{1}}{{3}}\left\{\left({eP}\right)^2+\left({eQ}\right)^2+\left({eR}\right)^2\right\}
$$
"""
        )

        st.markdown("### 최종 MSE를 입력하시오")
        st.text_input("E(0.5,0.5) =", key=ss_mse2, placeholder="예: 5/6 또는 0.8333")

        if st.button("f₂ 정답 확인", key=f"{key_prefix}_check_f2"):
            ok_yhat = _check_list_eq(yhat_s, ANS2["yhat"])
            ok_err = _check_list_eq(err_s, ANS2["err"])
            mse_in = _parse_fraction(st.session_state[ss_mse2])
            ok_mse = (mse_in is not None) and (mse_in == ANS2["mse"])

            if ok_yhat and ok_err and ok_mse:
                st.success("정답입니다!")
            else:
                st.error("오답입니다. 다시 생각해보세요.")

    # -----------------------------
    # (3) 선택형
    # -----------------------------
    st.divider()
    st.markdown("## (3) 더 적절한 함수를 고르시오")

    # 정답(더 작은 MSE)
    if ANS1["mse"] < ANS2["mse"]:
        correct = "f_1"
    elif ANS2["mse"] < ANS1["mse"]:
        correct = "f_2"
    else:
        correct = "same"

    st.radio(
        "선택",
        options=["f_1", "f_2"],
        format_func=lambda v: r"$f_1$" if v == "f_1" else r"$f_2$",
        key=ss_choice,
        horizontal=True,
    )

    if st.button("선택 정답 확인", key=f"{key_prefix}_check_choice"):
        if correct == "same":
            st.info("두 함수의 평균제곱오차가 같습니다.")
        else:
            if st.session_state[ss_choice] == correct:
                st.success("정답입니다!")
            else:
                st.error("오답입니다. 다시 생각해보세요.")


if __name__ == "__main__":
    try:
        st.set_page_config(page_title=TITLE, layout="wide")
    except Exception:
        pass
    render(show_title=True, key_prefix="ai_mse_debug")
