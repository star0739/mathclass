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
    rows = []
    for idx, (case, x, y) in enumerate(DATA, start=1):
        rows.append(
            {
                "장면(i)": idx,
                "입력값 x_i": x,
                "실제 정답 y_i": y,
                "AI 예측값 f(x_i)": round(f(x), 3),   # 자동 계산
                "오차 크기 |e_i|": "",
                "맞은 정도 (1-|e_i|)": "",
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

    with st.expander("문제", expanded=True):
        st.markdown(
            rf"""
어린이 보호 구역 표지판을 인식하는 인공지능은

각 장면(번호: $i$)에서 표지판의 형태, 색상, 주변 환경 등을 수치화하여 하나의 입력값($x_i$)으로 변환한다.

그리고 다음 **예측함수**를 사용하여 각 장면이 “어린이 보호 구역일 확률"을 0과 1 사이의 값으로 제시한다.

예측함수(시그모이드): 

$$
f(x)=\frac{{1}}{{1+e^{{-(ax+b)}}}} (a={A}, b={B})
$$

여기서 $y=1$는 어린이 보호 구역, $y=0$은 일반 도로를 의미한다.

다음 5개 장면에 대해 입력값($x_i$)과 실제 정답($y_i$)은 아래 표와 같다.
"""
        )

    # 인공지능의 장면 분석 데이터
    st.markdown("### 인공지능의 장면 분석 데이터")
    st.dataframe(_true_df(), use_container_width=True, hide_index=True)

    st.markdown("")
    
    st.markdown("### 1) 교차 엔트로피 손실함수")

    # 손실함수 제시
    st.markdown(
        r"""
    교차 엔트로피 손실함수는 다음과 같이 계산한다.

    $$
    E(a,b)=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i \ln f(x_i)+(1-y_i)\ln\left(1-f(x_i)\right)\right]
    $$

    $$
    =-\frac{1}{n}\sum_{i=1}^{n}\ln\left(1-|e_i|\right)
    $$

    """
    )

    st.markdown("")    
    
    st.markdown(
        r"""
오차 크기 $|e_i|$와 맞은 정도($1-|e_i|$)를 직접 계산하여 입력하세요.
"""
    )

    
    df_edit = st.data_editor(
        st.session_state[ss_tbl],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "장면(i)": st.column_config.NumberColumn(disabled=True),
            "입력값 x_i": st.column_config.NumberColumn(disabled=True),
            "실제 정답 y_i": st.column_config.NumberColumn(disabled=True),
            "AI 예측값 f(x_i)": st.column_config.NumberColumn(disabled=True),
            "오차 크기 |e_i|": st.column_config.TextColumn(),
            "맞은 정도 (1-|e_i|)": st.column_config.TextColumn(),
        },
        key=f"{key_prefix}_editor",
    )
    st.session_state[ss_tbl] = df_edit

    st.markdown("### 손실함수 E(8, -4)")

    match_vals = []
    for val in df_edit["맞은 정도 (1-|e_i|)"].tolist():
        parsed = _parse_float(val)
        if parsed is not None:
            match_vals.append(parsed)
        else:
            match_vals.append(None)

    # 식에 들어갈 LaTeX 구성
    terms = []
    for v in match_vals:
        if v is None:
            terms.append(r"\ln(\square)")
        else:
            terms.append(rf"\ln({v})")

    latex_expr = r"E(8,-4)=-\frac{1}{5}\left\{" + " + ".join(terms) + r"\right\}"

    st.markdown(f"$$ {latex_expr} $$")

    if all(v is not None for v in match_vals):
        try:
            total = -sum(math.log(v) for v in match_vals) / len(match_vals)
            st.markdown(f"$$E(8,-4) \\approx {round(total,3)}$$")
        except Exception:
            pass

    # 정답 확인(간단)
    ans = _answer_table()

    def _check_student():
        # 학생 입력 파싱
        err_s = [_parse_float(v) for v in df_edit["오차 크기 |e_i|"].tolist()]
        match_s = [_parse_float(v) for v in df_edit["맞은 정도 (1-|e_i|)"].tolist()]

        if any(v is None for v in err_s) or any(v is None for v in match_s):
            return False, "빈칸이 있습니다. 모든 칸을 채워주세요."

        # 표에 표시된 f(x_i)를 기준으로 '정답' 계산 (반올림 차이 문제 해결)
        p_list = [float(v) for v in df_edit["AI 예측값 f(x_i)"].tolist()]
        y_list = [int(v) for v in df_edit["실제 정답 y_i"].tolist()]

        err_ans = [abs(y - p) for y, p in zip(y_list, p_list)]
        match_ans = [1 - e for e in err_ans]

        # 허용 오차(반올림/계산기 입력 오차 고려)
        tol = 0.002  # 필요하면 0.005까지 늘려도 됨

        ok = True
        for i in range(len(err_s)):
            if abs(err_s[i] - err_ans[i]) > tol:
                ok = False
            if abs(match_s[i] - match_ans[i]) > tol:
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
    st.markdown("### 2) 예측 결과 해석하기")

    # 내부 정답 기준으로 "손실이 큰 장면"을 정답으로 삼아 해석 유도
    max_idx = int(ans["L"].idxmax())
    max_case = ans.loc[max_idx, "장면"]

    st.markdown(
        r"""
교차 엔트로피는 "확신을 가지고 틀린 예측"에 매우 큰 손실을 부여합니다. 특히 정답이 아닌 경우에 높은 확률로 예측했다가 실제로 틀리면 손실함수 값이 급격히 커집니다.

앞서 계산한 결과를 바탕으로 다음 질문에 답해 봅시다.

"""
    )

    st.markdown(
        f"""
- 가장 큰 오차($|e_i|$)가 발생한 장면은 무엇인가요?

→ 위 표에서 계산한 값을 근거로 판단해 보세요.
"""
    )
    st.markdown(
        f"""
- 이 인공지능의 실력을 더 키우려면, 어떤 데이터를 더 집중적으로 학습시켜야 할까요?
"""
    )

    st.radio(
        "집중 학습 데이터 선택",
        options=["boundary", "easy"],
        format_func=lambda v: "판단이 모호하거나 오답을 냈던 데이터에 집중하기" if v == "boundary" else "이미 정답을 잘 맞히고 있는 데이터에 집중하기",
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
참고: 이 예측함수는 **{max_case}**장면에서 가장 큰 오차를 보였습니다.  
인공지능이 더 똑똑해지려면 이미 잘 아는 데이터보다는, **틀리거나 판단이 모호한(경계/오판) 데이터**를 더 집중적으로 학습해야 합니다.
"""
        )


def _true_df() -> pd.DataFrame:
    # 주어진 데이터(고정)
    return pd.DataFrame({"장면": [d[0] for d in DATA], "입력값 x_i": [d[1] for d in DATA], "실제 정답 y_i": [d[2] for d in DATA]})


if __name__ == "__main__":
    try:
        st.set_page_config(page_title=TITLE, layout="wide")
    except Exception:
        pass
    render(show_title=True, key_prefix="ai_ce_debug")
