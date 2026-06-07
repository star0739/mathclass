# data_integral.py
# ------------------------------------------------------------
# 수치적분과 정적분 기반 분석
# - CSV 업로드
# - X/Y 데이터 시각화
# - AI가 제안한 모델식 입력
# - 전체 유효 데이터 구간을 기준으로 적분
# - 데이터 기반 수치적분(직사각형/사다리꼴)과 모델 정적분 비교
# ------------------------------------------------------------

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt


# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 5


# -----------------------------
# CSV 로더
# -----------------------------
def read_csv_kosis(file) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None

    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding=enc,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e

    raise last_err if last_err else ValueError("CSV를 읽을 수 없습니다.")


# -----------------------------
# 년/월/년월 파서
# 예: 2015.01, 2015-01, 2015/01, 201501, 2015
# -----------------------------
def parse_year_month(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.+$", "", regex=True)
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)

    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m")

    mask = dt.isna()
    if mask.any():
        digits = s[mask].str.replace(r"\D", "", regex=True)
        m6 = digits.str.fullmatch(r"\d{6}")
        if m6.any():
            dt2 = pd.to_datetime(digits[m6], errors="coerce", format="%Y%m")
            dt.loc[digits[m6].index] = dt2

    mask = dt.isna()
    if mask.any():
        y4 = s[mask].str.fullmatch(r"\d{4}")
        if y4.any():
            years = s[mask][y4].astype(int)
            dt2 = pd.to_datetime(dict(year=years, month=1, day=1), errors="coerce")
            dt.loc[years.index] = dt2

    return dt


# -----------------------------
# 데이터 전처리
# - X/Y 열 선택
# - 날짜형 X축이면 첫 시점을 0으로 둔 월 단위 t로 변환
# - 숫자형 X축이면 해당 숫자값을 t로 사용
# -----------------------------
def prepare_xy_data(df: pd.DataFrame, x_col: str, y_col: str, x_mode: str):
    y_series = pd.to_numeric(df[y_col], errors="coerce")

    if x_mode == "숫자":
        x_series = pd.to_numeric(df[x_col], errors="coerce")
        x_type = "numeric"
    else:
        x_dt = parse_year_month(df[x_col])

        if x_mode == "자동(권장)" and x_dt.notna().mean() < 0.6:
            x_series = pd.to_numeric(df[x_col], errors="coerce")
            x_type = "numeric"
        else:
            x_series = x_dt
            x_type = "datetime"

    valid = x_series.notna() & y_series.notna()
    xv = x_series[valid]
    yv = y_series[valid]

    if len(xv) >= 2:
        if x_type == "datetime":
            order = np.argsort(xv.values)
        else:
            order = np.argsort(xv.to_numpy())

        xv = xv.iloc[order]
        yv = yv.iloc[order]

    y_all = yv.to_numpy(dtype=float)

    if x_type == "datetime":
        base = xv.iloc[0]
        t_all = ((xv.dt.year - base.year) * 12 + (xv.dt.month - base.month)).to_numpy(dtype=float)
        t_description = "t는 첫 번째 날짜를 0으로 두고 계산한 월 단위 시간 인덱스입니다."
    else:
        t_all = xv.to_numpy(dtype=float)
        t_description = "t는 선택한 X축 숫자값입니다."

    if len(t_all) >= 2 and len(np.unique(t_all)) < len(t_all):
        raise ValueError(
            "X축 값 또는 t 값에 중복이 있습니다. "
            "적분 계산을 위해 X축 값이 중복되지 않아야 합니다."
        )

    return xv, yv, t_all, y_all, x_type, t_description


# -----------------------------
# 수학식 전처리 및 해석
# 학생 입력 예:
# y = 23.5 - 0.01t - 0.0003t^2 + 4.5cos((2*pi/12)t) - 3.5sin((2*pi/12)t)
# -----------------------------
def normalize_math_text(text: str) -> str:
    if text is None:
        return ""

    expr = str(text).strip()

    expr = expr.replace("$$", "")
    expr = expr.replace("$", "")
    expr = expr.replace("π", "pi")
    expr = expr.replace("Π", "pi")
    expr = expr.replace("−", "-")
    expr = expr.replace("–", "-")
    expr = expr.replace("—", "-")
    expr = expr.replace("×", "*")
    expr = expr.replace("·", "*")
    expr = expr.replace("÷", "/")
    expr = expr.replace("²", "^2")
    expr = expr.replace("³", "^3")

    # y = ..., f(t) = ... 형태면 오른쪽만 사용
    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()

    # 흔한 함수 표기 변환
    expr = re.sub(r"\bln\s*\(", "log(", expr)
    expr = re.sub(r"\bLog\s*\(", "log(", expr)
    expr = re.sub(r"\bSin\s*\(", "sin(", expr)
    expr = re.sub(r"\bCos\s*\(", "cos(", expr)
    expr = re.sub(r"\bTan\s*\(", "tan(", expr)
    expr = re.sub(r"\bExp\s*\(", "exp(", expr)

    # e^(...) → exp(...)
    expr = re.sub(r"\be\s*\^\s*\(", "exp(", expr)

    return expr.strip()


def parse_math_expression(text: str):
    t = sp.symbols("t")
    expr_text = normalize_math_text(text)

    if not expr_text:
        raise ValueError("수식이 비어 있습니다.")

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
        function_exponentiation,
    )

    local_dict = {
        "t": t,
        "pi": sp.pi,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "ln": sp.log,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "abs": sp.Abs,
    }

    try:
        expr = parse_expr(
            expr_text,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as e:
        raise ValueError(
            "수식을 해석하지 못했습니다. "
            "예시처럼 t, sin(...), cos(...), log(...), exp(...) 형태로 입력해 주세요."
        ) from e

    unknown_symbols = expr.free_symbols - {t}
    if unknown_symbols:
        unknown_str = ", ".join(str(s) for s in sorted(unknown_symbols, key=lambda x: str(x)))
        raise ValueError(f"수식에는 변수 t만 사용할 수 있습니다. 확인이 필요한 기호: {unknown_str}")

    return expr, t, expr_text


def evaluate_sympy_expression(expr, t_symbol, t_values: np.ndarray) -> np.ndarray:
    try:
        func = sp.lambdify(t_symbol, expr, modules=["numpy"])
        result = func(t_values)
    except Exception as e:
        raise ValueError(f"수식을 수치 계산하는 중 오류가 발생했습니다: {e}") from e

    if np.isscalar(result):
        result = np.full_like(t_values, float(result), dtype=float)
    else:
        result = np.asarray(result, dtype=float)

    if result.shape != t_values.shape:
        try:
            result = np.broadcast_to(result, t_values.shape).astype(float)
        except Exception as e:
            raise ValueError(
                f"계산 결과의 길이가 데이터 길이와 다릅니다. "
                f"결과 shape={result.shape}, t shape={t_values.shape}"
            ) from e

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "계산 결과에 NaN 또는 무한대 값이 포함되어 있습니다. "
            "log(t), 1/t처럼 특정 t에서 정의되지 않는 식이 있는지 확인하세요."
        )

    return result


# -----------------------------
# 수치적분
# -----------------------------
def data_rect_left(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return 0.0

    dt = t[1:] - t[:-1]
    return float(np.sum(y[:-1] * dt))


def data_trap(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return 0.0

    dt = t[1:] - t[:-1]
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dt))


def trapz(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) != len(t):
        raise ValueError("y와 t의 길이가 다릅니다.")
    if len(y) < 2:
        return 0.0

    dt = t[1:] - t[:-1]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * dt))


def rect_y0y1(height: float) -> tuple[float, float]:
    return (0.0, float(height)) if height >= 0 else (float(height), 0.0)


# -----------------------------
# 모델 정적분
# -----------------------------
def model_definite_integral(expr, t_symbol, a: float, b: float) -> tuple[float, str]:
    """
    1차 시도: SymPy 정적분
    실패 시: 고해상도 사다리꼴 수치적분
    """
    try:
        I = sp.integrate(expr, (t_symbol, a, b))

        if isinstance(I, sp.Integral) or I.has(sp.Integral):
            raise ValueError("sympy가 정적분을 기호적으로 계산하지 못했습니다.")

        return float(sp.N(I)), "symbolic"

    except Exception as e:
        tt_ref = np.linspace(a, b, 20001, dtype=float)
        ff_ref = evaluate_sympy_expression(expr, t_symbol, tt_ref)
        I_num = trapz(ff_ref, tt_ref)
        return I_num, f"numeric_fallback: {e}"


# -----------------------------
# 그래프
# -----------------------------
def plot_raw_data(xv, y_all, x_label: str, y_label: str):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=y_all,
                mode="lines+markers",
                name="Data",
            )
        )
        fig.update_layout(
            height=480,
            title="원자료 그래프",
            xaxis_title=str(x_label),
            yaxis_title=str(y_label),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(xv, y_all, marker="o")
        ax.set_title("원자료 그래프")
        ax.set_xlabel(str(x_label))
        ax.set_ylabel(str(y_label))
        st.pyplot(fig, use_container_width=True)


def plot_integral_compare(
    t: np.ndarray,
    y: np.ndarray,
    expr,
    t_symbol,
    y_col: str,
    vis_mode: str,
):
    if PLOTLY_AVAILABLE:
        a = float(t[0])
        b = float(t[-1])

        tt = np.linspace(a, b, 600, dtype=float)
        ff = evaluate_sympy_expression(expr, t_symbol, tt)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=tt,
                y=ff,
                mode="lines",
                name="모델 f(t)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="markers+lines",
                name="데이터",
            )
        )

        if vis_mode == "직사각형(좌측)":
            for i in range(len(t) - 1):
                x0 = float(t[i])
                x1 = float(t[i + 1])
                h = float(y[i])
                y0, y1 = rect_y0y1(h)

                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    line=dict(width=1),
                    fillcolor="rgba(0,0,0,0.08)",
                )

        else:
            for i in range(len(t) - 1):
                x0 = float(t[i])
                x1 = float(t[i + 1])
                y_l = float(y[i])
                y_r = float(y[i + 1])

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x0, x1, x1, x0],
                        y=[0, y_l, y_r, 0, 0],
                        mode="lines",
                        fill="toself",
                        name="사다리꼴",
                        showlegend=(i == 0),
                        opacity=0.25,
                    )
                )

        fig.update_layout(
            height=500,
            title="수치적분 도형과 모델 f(t)",
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis_title="t (개월 인덱스 또는 수치)",
            yaxis_title=str(y_col),
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Plotly가 없어 도형 시각화는 생략됩니다. 계산값은 위에서 확인할 수 있습니다.")


# ============================================================
# UI 시작
# ============================================================
st.title("수치적분과 정적분 기반 분석")
st.caption("데이터로 만든 직사각형/사다리꼴 면적과, 모델 f(t)의 정적분 값을 비교합니다.")
st.divider()


# ============================================================
# 1) CSV 업로드
# ============================================================
st.subheader("1) CSV 업로드")

csv_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key="data_integral_csv_upload")

if csv_file is None:
    st.info("CSV를 업로드하면 다음 단계로 진행할 수 있습니다.")
    st.stop()

try:
    df = read_csv_kosis(csv_file)
    st.success(f"CSV 업로드 완료 ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
except Exception as e:
    st.error("CSV를 읽지 못했습니다.")
    st.exception(e)
    st.stop()

st.markdown("#### 참고: 데이터 미리보기")
st.dataframe(df.head(30), use_container_width=True)

st.divider()


# ============================================================
# 2) 시각화 (X/Y 선택)
# ============================================================
st.subheader("2) 시각화 (X/Y 선택)")

cols = list(df.columns)

if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다.")
    st.stop()

col_x, col_y = st.columns(2)

with col_x:
    x_col = st.selectbox(
        "X축(시간/연도/년월)",
        cols,
        index=0,
        key="data_integral_x_col",
    )

with col_y:
    y_default_idx = 1 if len(cols) > 1 else 0
    if cols[y_default_idx] == x_col:
        y_default_idx = 0

    y_col = st.selectbox(
        "Y축(수치)",
        cols,
        index=y_default_idx,
        key="data_integral_y_col",
    )

x_mode = st.radio(
    "X축 해석 방식",
    ["자동(권장)", "날짜(년월)", "숫자"],
    horizontal=True,
    key="data_integral_x_mode",
)

try:
    xv, yv, t_all, y_all, x_type, t_description = prepare_xy_data(
        df=df,
        x_col=x_col,
        y_col=y_col,
        x_mode=x_mode,
    )
except Exception as e:
    st.error("X/Y 데이터를 처리하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

if len(t_all) < 2:
    st.warning("유효한 데이터가 부족하여 적분을 계산할 수 없습니다. X/Y 열 값을 확인하세요.")
    st.stop()

plot_raw_data(xv, y_all, x_col, y_col)

st.caption(f"※ {t_description}")

st.divider()


# ============================================================
# 데이터 개수 점검
# ============================================================
st.subheader("✅ 데이터 개수 점검")

valid_n = int(len(t_all))
st.metric("유효 데이터 점(숫자쌍) 개수", valid_n)

if valid_n < MIN_VALID_POINTS:
    st.warning(
        f"유효 데이터 점이 {MIN_VALID_POINTS}개 미만입니다. "
        "적분 계산은 가능하지만, 수치적분 비교가 불안정할 수 있습니다."
    )

st.divider()


# ============================================================
# 3) 모델식 확인 & 적분 구간 확인
# ============================================================
st.subheader("3) 모델식 확인 & 적분 구간 확인")

st.info(
    "AI가 제안한 모델식을 지오지브라에 입력할 수 있는 수학식 형태로 입력하세요.(지난 시간에 사용한 식) "
    "변수는 반드시 t를 사용합니다."
)

with st.expander("모델식 입력 예시 보기", expanded=True):
    st.markdown(
        """
아래와 같은 형태로 입력할 수 있습니다.

예시 1

    y = 23.5 - 0.01t - 0.0003t^2 + 4.5cos((2*pi/12)t) - 3.5sin((2*pi/12)t)

예시 2

    23.5 - 0.01*t - 0.0003*t^2 + 4.5*cos((2*pi/12)*t) - 3.5*sin((2*pi/12)*t)

지원되는 표현 예시는 다음과 같습니다.

- 거듭제곱: `t^2`, `t**2`
- 삼각함수: `sin(t)`, `cos(t)`, `tan(t)`
- 지수함수: `exp(0.03t)`, `e^(0.03t)`
- 로그함수: `log(t+1)`, `ln(t+1)`
- 원주율: `pi`, `π`
- 생략 곱셈: `0.01t`, `2(t+1)`, `cos((2*pi/12)t)`
"""
    )

model_expr_text = st.text_area(
    "모델식 f(t)",
    height=120,
    placeholder="예: y = 23.5 - 0.01t - 0.0003t^2 + 4.5cos((2*pi/12)t) - 3.5sin((2*pi/12)t)",
    key="data_integral_model_expr_text",
)

if not model_expr_text.strip():
    st.info("모델식을 입력하면 전체 유효 데이터 구간에 대한 적분 비교 결과가 표시됩니다.")
    st.stop()

try:
    expr, t_symbol, normalized_expr_text = parse_math_expression(model_expr_text)
except Exception as e:
    st.error(f"모델식 처리 오류: {e}")
    st.stop()

st.markdown("#### 입력한 모델식 확인")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**앱이 해석한 입력식**")
    st.code(normalized_expr_text, language="text")

with c2:
    st.markdown("**수식 미리보기**")
    st.latex(r"f(t)=" + sp.latex(expr))

# 전체 유효 데이터 구간 사용
t = t_all
y = y_all
xv_selected = xv

a = float(t[0])
b = float(t[-1])

st.markdown("#### 적분 구간 확인")
st.info("이 활동에서는 전체 유효 데이터 구간을 기준으로 수치적분값과 모델 정적분값을 비교합니다.")

st.caption(
    f"적분 구간: **{xv_selected.iloc[0]} ~ {xv_selected.iloc[-1]}** "
    f"| t = **{a:.6g} ~ {b:.6g}**"
)

st.divider()


# ============================================================
# 4) 수치적분(데이터) vs 정적분(모델)
# ============================================================
st.subheader("4) 수치적분(데이터) vs 정적분(모델)")

try:
    # 데이터 기반 수치적분
    A_rect = data_rect_left(y, t)
    A_trap = data_trap(y, t)

    # 모델 정적분
    I_model, integral_method = model_definite_integral(expr, t_symbol, a, b)

except Exception as e:
    st.error(f"적분 계산 오류: {e}")
    st.stop()

if integral_method != "symbolic":
    st.caption(f"참고: SymPy 정적분이 어려워 고해상도 수치적분으로 근사했습니다. ({integral_method})")

err_rect = abs(A_rect - I_model)
err_trap = abs(A_trap - I_model)

rel_rect = err_rect / (abs(I_model) + 1e-12)
rel_trap = err_trap / (abs(I_model) + 1e-12)

c1, c2, c3 = st.columns(3)
c1.metric("직사각형 값(데이터, 좌측)", f"{A_rect:,.6g}")
c2.metric("사다리꼴 값(데이터)", f"{A_trap:,.6g}")
c3.metric("정적분 값(모델)", f"{I_model:,.6g}")

d1, d2, d3 = st.columns(3)
d1.metric("직사각형 오차 |A-I|", f"{err_rect:,.6g}")
d2.metric("사다리꼴 오차 |A-I|", f"{err_trap:,.6g}")
d3.metric("사다리꼴 상대오차", f"{rel_trap:.3%}")

if err_rect < err_trap:
    st.success("전체 구간에서는 직사각형 방법이 모델 정적분값에 더 가깝습니다.")
elif err_trap < err_rect:
    st.success("전체 구간에서는 사다리꼴 방법이 모델 정적분값에 더 가깝습니다.")
else:
    st.info("전체 구간에서는 직사각형 방법과 사다리꼴 방법의 오차가 같습니다.")

st.divider()


# ============================================================
# 5) 직사각형/사다리꼴 시각화
# ============================================================
st.subheader("5) 직사각형/사다리꼴 시각화")

vis_mode = st.radio(
    "도형 표시",
    ["직사각형(좌측)", "사다리꼴"],
    horizontal=True,
    key="data_integral_vis_mode",
)

plot_integral_compare(
    t=t,
    y=y,
    expr=expr,
    t_symbol=t_symbol,
    y_col=y_col,
    vis_mode=vis_mode,
)

st.divider()


# ============================================================
# 6) 계산 결과 표
# ============================================================
st.subheader("6) 계산 결과 표")

result_summary = pd.DataFrame(
    {
        "항목": [
            "직사각형 값(데이터, 좌측)",
            "사다리꼴 값(데이터)",
            "정적분 값(모델)",
            "직사각형 오차 |A-I|",
            "사다리꼴 오차 |A-I|",
            "직사각형 상대오차",
            "사다리꼴 상대오차",
        ],
        "값": [
            A_rect,
            A_trap,
            I_model,
            err_rect,
            err_trap,
            rel_rect,
            rel_trap,
        ],
    }
)

st.dataframe(result_summary, use_container_width=True)

with st.expander("전체 유효 데이터 확인", expanded=False):
    selected_df = pd.DataFrame(
        {
            "X축 값": xv_selected.astype(str).to_numpy(),
            "t": t,
            "실제 데이터 y": y,
        }
    )
    st.dataframe(selected_df, use_container_width=True)
