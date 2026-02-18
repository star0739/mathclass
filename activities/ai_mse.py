# activities/ai_mse.py
from __future__ import annotations

from fractions import Fraction
import pandas as pd
import streamlit as st

TITLE = "평균제곱오차(MSE)로 예측함수 비교"

# 고정 데이터
CASES = ["P", "Q", "R"]
X_TRUE = [1, 2, 3]
Y_TRUE = [1, 2, 2]


def f1(x: float) -> float:
    return x - 0.5


def f2(x: float) -> float:
    return 0.5 * x + 0.5


def _true_df() -> pd.DataFrame:
    return pd.DataFrame({"사례": CASES, "입력값 x": X_TRUE, "출력값 y": Y_TRUE})


def _blank_student_table() -> pd.DataFrame:
    # 학생이 예측값/오차를 직접 입력
    return pd.DataFrame(
        {
            "사례": CASES,
            "입력값 x": X_TRUE,
            "출력값 y": Y_TRUE,
            "예측값 f(x)": ["", "", ""],
            "오차 y-f(x)": ["", "", ""],
        }
    )


def _parse_fraction(val):
    """
    학생 입력(문자열/숫자)을 Fraction으로 변환.
    허용: '1/4', '-3/2', '0.25', 2, -1 등
    실패/빈칸: None
    """
    if val is None:
        return None
    if isinstance(val, Fraction):
        return val
    if isinstance(val, int):
        return Fraction(val, 1)
    if isinstance(val, float):
        # NaN 방지
        if val != val:
            return None
        return Fraction(str(val))

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        # 1) 분수 시도
        try:
            return Fraction(s)
        except Exception:
            pass
        # 2) 소수 시도
        try:
            return Fraction(str(float(s)))
        except Exception:
            return None

    # 기타 타입
    try:
        return Fraction(str(val))
    except Exception:
        return None


def _frac_latex(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    return r"\frac{" + str(fr.numerator) + "}{" + str(fr.denominator) + "}"


def _num_to_frac_exact(x: float) -> Fraction:
    return Fraction(str(float(x)))


def _answer_for_model(f):
    yhat = [_num_to_frac_exact(f(x)) for x in X_TRUE]
    err = [Fraction(y, 1) - yh for y, yh in zip(Y_TRUE, yhat)]
    mse = (err[0] * err[0] + err[1] * err[1] + err[2] * err[2]) / 3
    return {"yhat": yhat, "err": err, "mse": mse}


ANS1 = _answer_for_model(f1)
ANS2 = _answer_for_model(f2)


def _table_to_fracs(df: pd.DataFrame):
    yhat_s = [_parse_fraction(v) for v in df["예측값 f(x)"].tolist()]
    err_s = [_parse_fraction(v) for v in df["오차 y-f(x)"].tolist()]
    return yhat_s, err_s


def _all_filled(vals) -> bool:
    return all(v is not None for v in vals)


def _check_list(student, answer) -> bool:
    if not _all_filled(student):
        return False
    return all(s == a for s, a in zip(student, answer))


def render(show_title: bool = True, key_prefix: str = "ai_mse") -> None:
    if show_title:
        st.subheader(TITLE)

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
어느 대나무 세 그루가 각각 $x$일 동안 자라는 길이 $y\text{ m}$를 조사한 결과의 순서쌍 $(x,y)$가 각각 다음과 같다.

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

    st.markdown(
        r"""
예측함수 $f(x)=ax+b$에 대한 평균제곱오차 $E(a,b)$는 다음과 같이 계산한다.

$$
\text{MSE}=\frac{1}{n}\left\{(y_1-f(x_1))^2+(y_2-f(x_2))^2+\cdots+(y_n-f(x_n))^2\right\}
$$
"""
    )

    st.markdown("")
    
    left, right = st.columns([1, 1], gap="large")

    # -------- f1 --------
    with left:
        st.markdown(r"### $f_1(x)=x-0.5$")

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

        yhat_s, err_s = _table_to_fracs(df1)
        boxes = []
        for v in err_s:
            boxes.append(_frac_latex(v) if v is not None else r"\square")

        st.markdown("입력한 오차로 MSE 식을 채우면:")
        st.markdown(
            r"$$\text{E(1,-0.5)}=\frac{1}{3}\left\{\left("
            + boxes[0]
            + r"\right)^2+\left("
            + boxes[1]
            + r"\right)^2+\left("
            + boxes[2]
            + r"\right)^2\right\}$$"
        )


        st.text_input("", key=ss_mse1, placeholder="예: 1/4 또는 0.25")

        if st.button("$f_1$ 정답 확인", key=f"{key_prefix}_check_f1"):
            ok_yhat = _check_list(yhat_s, ANS1["yhat"])
            ok_err = _check_list(err_s, ANS1["err"])
            mse_in = _parse_fraction(st.session_state[ss_mse1])
            ok_mse = (mse_in is not None) and (mse_in == ANS1["mse"])

            if ok_yhat and ok_err and ok_mse:
                st.success("정답입니다!")
            else:
                st.error("오답입니다. 다시 생각해보세요.")

    # -------- f2 --------
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

        yhat_s, err_s = _table_to_fracs(df2)
        boxes = []
        for v in err_s:
            boxes.append(_frac_latex(v) if v is not None else r"\square")

        st.markdown("학생이 입력한 오차로 MSE 식을 채우면:")
        st.markdown(
            r"$$\text{MSE}=\frac{1}{3}\left\{\left("
            + boxes[0]
            + r"\right)^2+\left("
            + boxes[1]
            + r"\right)^2+\left("
            + boxes[2]
            + r"\right)^2\right\}$$"
        )

        st.markdown("### 최종 MSE를 입력하시오")
        st.text_input("E(0.5,0.5) =", key=ss_mse2, placeholder="예: 5/6 또는 0.8333")

        if st.button("f₂ 정답 확인", key=f"{key_prefix}_check_f2"):
            ok_yhat = _check_list(yhat_s, ANS2["yhat"])
            ok_err = _check_list(err_s, ANS2["err"])
            mse_in = _parse_fraction(st.session_state[ss_mse2])
            ok_mse = (mse_in is not None) and (mse_in == ANS2["mse"])

            if ok_yhat and ok_err and ok_mse:
                st.success("정답입니다!")
            else:
                st.error("오답입니다. 다시 생각해보세요.")

    # -------- (3) 선택 --------
    st.divider()
    st.markdown("## (3) 더 적절한 함수를 고르시오")

    correct = "same"
    if ANS1["mse"] < ANS2["mse"]:
        correct = "f_1"
    elif ANS2["mse"] < ANS1["mse"]:
        correct = "f_2"

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
