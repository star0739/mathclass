
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False

from assessment.common import (
    init_assessment_session,
    require_student_id,
    set_df,
    get_df,
    get_xy,
    get_step1_summary,
)

from assessment.google_sheets import append_step3_row


# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 5


# -----------------------------
# 세션용 step2/step3 저장
# -----------------------------
def _get_step2_state() -> dict:
    return st.session_state.get("assessment_step2", {})


def _set_step2_state(d: dict) -> None:
    st.session_state["assessment_step2"] = d


def _get_step3_state() -> dict:
    return st.session_state.get("assessment_step3", {})


def _set_step3_state(d: dict) -> None:
    st.session_state["assessment_step3"] = d


# -----------------------------
# CSV 로더 (관대)
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
# Step2 백업 TXT 파서(최소)
# -----------------------------
def parse_step2_backup_txt(text: str) -> dict:
    out: dict[str, str] = {}
    lines = [ln.rstrip("\n") for ln in (text or "").splitlines()]
    stripped = [ln.strip() for ln in lines]

    def find_value(prefix: str) -> str:
        for ln in stripped:
            if ln.startswith(prefix):
                return ln.replace(prefix, "", 1).strip()
        return ""

    out["student_id"] = find_value("학번:")
    out["data_source"] = find_value("- 데이터 출처:")
    out["x_col"] = ""
    out["y_col"] = ""
    for ln in stripped:
        if ln.startswith("- X축:"):
            m = re.search(r"- X축:\s*(.*?)\s*\|\s*Y축:\s*(.*)$", ln)
            if m:
                out["x_col"] = m.group(1).strip()
                out["y_col"] = m.group(2).strip()
    out["valid_n"] = find_value("- 유효 데이터 점:")
    return out


# -----------------------------
# np.trapz 의존 제거 사다리꼴 적분
# -----------------------------
def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) != len(t):
        raise ValueError("y와 t의 길이가 다릅니다.")
    if len(y) < 2:
        return 0.0

    dt = t[1:] - t[:-1]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * dt))


# -----------------------------
# 데이터 기반 수치적분 (좌측 직사각형 / 사다리꼴)
# -----------------------------
def _data_rect_left(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 2:
        return 0.0
    dt = t[1:] - t[:-1]
    return float(np.sum(y[:-1] * dt))


def _data_trap(y: np.ndarray, t: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 2:
        return 0.0
    dt = t[1:] - t[:-1]
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dt))


def _rect_y0y1(height: float) -> tuple[float, float]:
    return (0.0, float(height)) if height >= 0 else (float(height), 0.0)


# -----------------------------
# 모델식 eval (plot용)
# -----------------------------
def _eval_model_expr(expr: str, t: np.ndarray) -> np.ndarray:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("py_model이 비어 있습니다.")
    if expr.startswith("="):
        raise ValueError("py_model이 '='로 시작합니다. 수식이 아니라 '표현식'만 입력하세요.")

    blocked = ["__", "import", "open(", "exec(", "eval(", "os.", "sys.", "subprocess", "pickle", "globals", "locals"]
    if any(tok in expr for tok in blocked):
        raise ValueError("허용되지 않는 토큰이 포함되어 py_model을 계산할 수 없습니다.")

    env = {
        "np": np,
        "t": t,
        "sin": np.sin,
        "cos": np.cos,
        "exp": np.exp,
        "log": np.log,
        "pi": np.pi,
    }
    y_hat = eval(expr, {"__builtins__": {}}, env)
    y_hat = np.asarray(y_hat, dtype=float)
    if y_hat.shape == ():
        y_hat = np.full_like(t, float(y_hat), dtype=float)
    if len(y_hat) != len(t):
        raise ValueError("모델 결과 길이가 t와 일치하지 않습니다.")
    return y_hat


# -----------------------------
# sympy 정적분
# -----------------------------
def _sympy_definite_integral(py_expr: str, a: float, b: float) -> float:
    expr = (py_expr or "").strip()
    if not expr:
        raise ValueError("py_model이 비어 있습니다.")
    if expr.startswith("="):
        raise ValueError("py_model이 '='로 시작합니다. '=' 없이 표현식만 입력하세요.")

    expr = expr.replace("np.", "").replace("numpy.", "")

    t = sp.Symbol("t", real=True)
    locals_map = {
        "t": t,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "pi": sp.pi,
        "E": sp.E,
        "Abs": sp.Abs,
    }

    sym = sp.sympify(expr, locals=locals_map)
    I = sp.integrate(sym, (t, a, b))

    if isinstance(I, sp.Integral) or I.has(sp.Integral):
        raise ValueError("sympy가 정적분을 기호적으로 계산하지 못했습니다.")
    return float(sp.N(I))


# -----------------------------
# Step3 백업 생성
# -----------------------------
def build_step3_backup(payload: dict) -> bytes:
    """
    3차시 백업 TXT(학생 복구/증빙용)
    - 구글시트 저장 필드와 동일한 핵심 값들을 사람이 읽기 좋게 출력
    - payload 키:
      student_id, data_source, x_col, y_col, valid_n, i0, i1,
      py_model, A_rect, A_trap, I_model, err_rect, err_trap, rel_trap,
      student_critical_review2
    """
    def _s(v) -> str:
        return "" if v is None else str(v).strip()

    def _num(v) -> str:
        if v is None or v == "":
            return ""
        try:
            x = float(v)
        except Exception:
            return str(v)
        # 너무 긴 소수는 줄이고, 지수표기/일반표기 모두 자연스럽게
        return f"{x:,.12g}"

    def _pct(v) -> str:
        if v is None or v == "":
            return ""
        try:
            x = float(v)
            return f"{x:.3%}"
        except Exception:
            return str(v)

    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    student_id = _s(payload.get("student_id"))
    data_source = _s(payload.get("data_source"))
    x_col = _s(payload.get("x_col"))
    y_col = _s(payload.get("y_col"))
    valid_n = _s(payload.get("valid_n"))
    i0 = _s(payload.get("i0"))
    i1 = _s(payload.get("i1"))

    py_model = _s(payload.get("py_model"))

    A_rect = _num(payload.get("A_rect"))
    A_trap = _num(payload.get("A_trap"))
    I_model = _num(payload.get("I_model"))
    err_rect = _num(payload.get("err_rect"))
    err_trap = _num(payload.get("err_trap"))

    # rel_trap은 payload에 없을 수도 있으니 계산 시도
    rel_trap_val = payload.get("rel_trap", "")
    if rel_trap_val in ("", None):
        try:
            if payload.get("err_trap") not in ("", None) and payload.get("I_model") not in ("", None):
                rel_trap_val = float(payload["err_trap"]) / (abs(float(payload["I_model"])) + 1e-12)
        except Exception:
            rel_trap_val = ""
    rel_trap = _pct(rel_trap_val)

    review = _s(payload.get("student_critical_review2"))
    if not review:
        review = "(미입력)"

    lines: list[str] = []
    lines.append("공공데이터 분석 수행 (3차시) 백업")
    lines.append("=" * 56)
    lines.append(f"저장시각: {ts}")
    lines.append(f"학번: {student_id}")
    lines.append("")

    lines.append("[데이터 정보]")
    lines.append(f"- 데이터 출처: {data_source}")
    lines.append(f"- X축(시간): {x_col}")
    lines.append(f"- Y축(수치): {y_col}")
    lines.append(f"- 유효 데이터 점: {valid_n}")
    lines.append(f"- 적분 구간(인덱스): {i0} ~ {i1}")
    lines.append("")

    lines.append("[모델식 f(t) (py_model)]")
    lines.append(py_model if py_model else "(미입력)")
    lines.append("")

    lines.append("[적분 비교 결과]")
    lines.append("※ 데이터 기반 수치적분(직사각형/사다리꼴) vs 모델 정적분")
    lines.append(f"- 직사각형 값(데이터, 좌측): {A_rect}")
    lines.append(f"- 사다리꼴 값(데이터): {A_trap}")
    lines.append(f"- 정적분 값(모델): {I_model}")
    lines.append("")
    lines.append("[오차]")
    lines.append(f"- 직사각형 오차 |A_rect - I_model|: {err_rect}")
    lines.append(f"- 사다리꼴 오차 |A_trap - I_model|: {err_trap}")
    lines.append(f"- 사다리꼴 상대오차: {rel_trap}")
    lines.append("")

    lines.append("[4) 적분 관점의 모델 분석(학생 서술)]")
    lines.append(review)
    lines.append("")

    lines.append("※ 이 파일은 학생 개인 백업용입니다. 필요 시 앱에 값을 다시 입력하는 데 활용하세요.")
    return "\n".join(lines).encode("utf-8-sig")



# ============================================================
# UI
# ============================================================
init_assessment_session()
student_id = require_student_id("학번을 입력하세요.")

st.title("3차시: 수치적분과 정적분 기반 분석")
st.caption("데이터로 만든 직사각형/사다리꼴 면적과, 모델 f(t)의 정적분 값을 비교합니다.")
st.divider()

# 0) Step2 TXT + CSV
st.subheader("0) 2차시 기록 불러오기 & CSV 업로드")

step1 = get_step1_summary() or {}
step2 = _get_step2_state() or {}
step3_prev = _get_step3_state() or {}

colA, colB = st.columns([1.2, 1])
with colA:
    st.markdown("**2차시 TXT 업로드로 복구(선택)**")
    step2_txt = st.file_uploader("2차시 백업 TXT 업로드", type=["txt"], key="step3_step2_txt_upload")
with colB:
    st.markdown("**CSV 업로드(필수)**")
    csv_file = st.file_uploader("CSV 업로드", type=["csv"], key="step3_csv_upload")

if step2_txt is not None:
    try:
        raw = step2_txt.getvalue().decode("utf-8", errors="replace")
        parsed2 = parse_step2_backup_txt(raw)
        step2 = {
            **step2,
            "student_id": parsed2.get("student_id") or step2.get("student_id") or student_id,
            "data_source": parsed2.get("data_source") or step2.get("data_source") or step1.get("data_source", ""),
            "x_col": parsed2.get("x_col") or step2.get("x_col") or step1.get("x_col", ""),
            "y_col": parsed2.get("y_col") or step2.get("y_col") or step1.get("y_col", ""),
            "valid_n": parsed2.get("valid_n") or step2.get("valid_n") or step1.get("valid_n", ""),
        }
        _set_step2_state(step2)
        st.success("TXT에서 2차시 정보를(부분적으로) 불러왔습니다.")
    except Exception as e:
        st.error("2차시 TXT를 읽는 중 오류가 발생했습니다.")
        st.exception(e)

if csv_file is not None:
    try:
        df_up = read_csv_kosis(csv_file)
        set_df(df_up)
        st.success(f"CSV 업로드 완료 ({df_up.shape[0]:,}행 × {df_up.shape[1]:,}열)")
    except Exception as e:
        st.error("CSV를 읽지 못했습니다.")
        st.exception(e)

df = get_df()
if df is None:
    st.info("CSV를 업로드하면 다음 단계로 진행할 수 있습니다.")
    st.stop()

st.divider()

# 1) 자동 열 설정 + t 변환
st.subheader("1) 데이터 열 자동 설정")

cols = list(df.columns)
if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다.")
    st.stop()

x_prev, y_prev = get_xy()
x_col = (step2.get("x_col") or step1.get("x_col") or (x_prev if x_prev in cols else "")).strip()
y_col = (step2.get("y_col") or step1.get("y_col") or (y_prev if y_prev in cols else "")).strip()

if x_col not in cols:
    x_col = cols[0]
if y_col not in cols:
    y_col = cols[1]
if y_col == x_col and len(cols) > 1:
    y_col = cols[1] if cols[1] != x_col else cols[0]

st.caption(f"X축(시간): **{x_col}**  |  Y축(수치): **{y_col}**")

y_series = pd.to_numeric(df[y_col], errors="coerce")
x_dt = parse_year_month(df[x_col])
if x_dt.notna().mean() >= 0.6:
    x_series = x_dt
    x_type = "datetime"
else:
    x_series = pd.to_numeric(df[x_col], errors="coerce")
    x_type = "numeric"

valid = x_series.notna() & y_series.notna()
xv = x_series[valid]
yv = y_series[valid]

if len(xv) < MIN_VALID_POINTS:
    st.warning(f"유효 데이터 점이 {len(xv)}개입니다.")
    if len(xv) < 2:
        st.stop()

order = np.argsort(xv.values) if x_type == "datetime" else np.argsort(xv.to_numpy())
xv = xv.iloc[order]
yv = yv.iloc[order]

if x_type == "datetime":
    base = xv.iloc[0]
    t_all = ((xv.dt.year - base.year) * 12 + (xv.dt.month - base.month)).to_numpy(dtype=float)

    raw = df.loc[valid, x_col].astype(str).str.strip().iloc[order]
    if (raw.str.fullmatch(r"\d{4}")).mean() >= 0.8:
        st.warning("시간 데이터가 '연도(YYYY)' 중심으로 보입니다. 월 단위(1월 가정)로 변환되어 해석이 거칠 수 있습니다.")
else:
    t_all = xv.to_numpy(dtype=float)

y_all = yv.to_numpy(dtype=float)
st.metric("유효 데이터 점(숫자쌍) 개수", int(len(t_all)))
st.divider()

# 2) 모델식 + 구간 선택
st.subheader("2) 모델식 확인 & 적분 구간 선택")

py_model_default = (step2.get("py_model") or "").strip()
py_model = st.text_input(
    "모델식 f(t) (numpy 사용, 변수는 t)",
    value=py_model_default,
    placeholder="예: 22 - 0.017*t + 6*np.cos(2*np.pi*t/12) + 4*np.sin(2*np.pi*t/12)",
)

n = len(t_all)
i0_default = int(step3_prev.get("i0", 0) or 0)
i1_default = int(step3_prev.get("i1", n - 1) or (n - 1))
i0_default = max(0, min(n - 2, i0_default))
i1_default = max(i0_default + 1, min(n - 1, i1_default))

i0, i1 = st.slider(
    "적분 구간(인덱스)",
    min_value=0,
    max_value=n - 1,
    value=(i0_default, i1_default),
    step=1,
)

t = t_all[i0 : i1 + 1]
y = y_all[i0 : i1 + 1]

st.divider()

# 3) 비교 + 시각화
st.subheader("3) 수치적분(데이터) vs 정적분(모델)")

if not py_model.strip():
    st.warning("모델식(py_model)이 비어 있습니다.")
    st.stop()

# 데이터 기반 값
A_rect = _data_rect_left(y, t)
A_trap = _data_trap(y, t)

# 모델 정적분(우선 sympy)
a, b = float(t[0]), float(t[-1])
I_model = None
I_msg = ""

try:
    I_model = _sympy_definite_integral(py_model, a, b)
except Exception as e:
    # sympy 실패 시 고해상도 수치적분으로 기준값 생성
    I_msg = str(e)
    tt_ref = np.linspace(a, b, 20001, dtype=float)
    ff_ref = _eval_model_expr(py_model, tt_ref)
    I_model = _trapz(ff_ref, tt_ref)

if I_msg:
    st.caption(f"참고: sympy 정적분이 어려워 수치적으로 근사했습니다. ({I_msg})")

# 값 표시(윗줄 3칸)
c1, c2, c3 = st.columns(3)
c1.metric("직사각형 값(데이터, 좌측)", f"{A_rect:,.6g}")
c2.metric("사다리꼴 값(데이터)", f"{A_trap:,.6g}")
c3.metric("정적분 값(모델)", f"{I_model:,.6g}")

# 오차(아랫줄 3칸: 마지막 빈칸)
err_rect = abs(A_rect - I_model)
err_trap = abs(A_trap - I_model)
d1, d2, d3 = st.columns(3)
d1.metric("직사각형 오차 |A-I|", f"{err_rect:,.6g}")
d2.metric("사다리꼴 오차 |A-I|", f"{err_trap:,.6g}")
d3.metric("", "")

# 시각화
if PLOTLY_AVAILABLE:
    vis_mode = st.radio("도형 표시", ["직사각형(좌측)", "사다리꼴"], horizontal=True)

    # 모델 곡선 샘플
    tt = np.linspace(a, b, 600, dtype=float)
    ff = _eval_model_expr(py_model, tt)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tt, y=ff, mode="lines", name="모델 f(t)"))
    fig.add_trace(go.Scatter(x=t, y=y, mode="markers+lines", name="데이터(구간)"))

    if vis_mode == "직사각형(좌측)":
        for i in range(len(t) - 1):
            x0 = float(t[i])
            x1 = float(t[i + 1])
            h = float(y[i])
            y0, y1 = _rect_y0y1(h)
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(width=1),
                fillcolor="rgba(0,0,0,0.08)",
            )
    else:
        for i in range(len(t) - 1):
            x0 = float(t[i])
            x1 = float(t[i + 1])
            yL = float(y[i])
            yR = float(y[i + 1])
            fig.add_trace(go.Scatter(
                x=[x0, x0, x1, x1, x0],
                y=[0,  yL, yR, 0,  0],
                mode="lines",
                fill="toself",
                name="사다리꼴",
                showlegend=(i == 0),
                opacity=0.25,
            ))

    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="t (개월 인덱스 또는 수치)",
        yaxis_title=str(y_col),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Plotly가 없어 그래프 시각화는 생략됩니다. (값/오차는 위에서 확인 가능)")

st.divider()

# 4) 결론
st.subheader("4) 적분 관점의 모델 분석")

st.info(
    "🔹 누적량 비교\n\n"
    "데이터를 이용하여 계산한 수치적분 값(직사각형·사다리꼴)과 "
    "모델 식으로부터 구한 정적분 값을 비교하시오. \n\n"
    "어느 방법이 정적분 값에 더 가까웠는지 수치적 근거를 들어 설명하시오.\n\n"

    "🔹 근사 방법의 특성 분석\n\n"
    "직사각형 방법과 사다리꼴 방법의 오차 차이를 분석하시오. \n\n"
    "구간 내 함수의 증가·감소, 오목·볼록 성질과 연결하여 "
    "왜 그러한 차이가 발생하는지 설명하시오.\n\n"

    "🔹 적분값의 의미 해석\n\n"
    "계산된 정적분 값이 현실에서 무엇을 의미하는지 서술하시오. \n\n"
    "이 값이 나타내는 전체 변화량 또는 누적 효과를 구체적으로 설명하시오.\n\n"
)

student_critical_review2 = st.text_area(
    "적분 분석 내용(필수)",
    value=step3_prev.get("student_critical_review2", ""),
    height=220,
    placeholder=(
        "본문에는 수식 대신 일반 텍스트와 기호를 사용해 주세요. 수식 편집기를 사용하면 추후 보고서 취합 시 글자 깨짐 현상이 발생할 수 있습니다."
    ),
)


st.divider()

# 5) 저장 및 백업
st.subheader("5) 저장 및 백업")

data_source = (step2.get("data_source") or step1.get("data_source") or "").strip()
valid_n_now = int(len(t_all))

payload = {
    "student_id": student_id,
    "data_source": data_source,
    "x_col": x_col,
    "y_col": y_col,
    "valid_n": valid_n_now,
    "i0": int(i0),
    "i1": int(i1),
    "py_model": py_model.strip(),
    "A_rect": float(A_rect),
    "A_trap": float(A_trap),
    "I_model": float(I_model),
    "err_rect": float(err_rect),
    "err_trap": float(err_trap),
    "student_critical_review2": student_critical_review2.strip(),
}

rel_trap = float(err_trap / (abs(I_model) + 1e-12))
payload["rel_trap"] = rel_trap

col1, col2, col3 = st.columns([1, 1, 1.2])
save_clicked = col1.button("💾 저장(구글시트)", use_container_width=True)
download_clicked = col2.button("⬇️ TXT 백업 만들기", use_container_width=True)
go_next = col3.button("➡️ 최종 보고서 작성", use_container_width=True)

backup_bytes = build_step3_backup(payload)
st.download_button(
    label="📄 (다운로드) 3차시 백업 TXT",
    data=backup_bytes,
    file_name=f"미적분_수행평가_3차시_{student_id}.txt",
    mime="text/plain; charset=utf-8",
)

def _validate_step3() -> bool:
    if not payload["student_critical_review2"]:
        st.warning("적분 분석 내용을 입력하세요.")
        return False
    if not payload["py_model"]:
        st.warning("모델식(py_model)을 입력/확인하세요.")
        return False
    return True

if save_clicked or download_clicked or go_next:
    if not _validate_step3():
        st.stop()

    _set_step3_state({**payload, "saved_at": pd.Timestamp.now().isoformat()})

    if download_clicked:
        st.success("✅ 백업 데이터가 준비되었습니다. 위 '다운로드' 버튼을 눌러주세요.")

    if save_clicked or go_next:
        try:
            append_step3_row(
                student_id=payload["student_id"],
                data_source=payload["data_source"],
                x_col=payload["x_col"],
                y_col=payload["y_col"],
                valid_n=payload["valid_n"],
                i0=payload["i0"],
                i1=payload["i1"],
                py_model=payload["py_model"],
                A_rect=payload["A_rect"],
                A_trap=payload["A_trap"],
                I_model=payload["I_model"],
                err_rect=payload["err_rect"],
                err_trap=payload["err_trap"],
                rel_trap=payload["rel_trap"],
                student_critical_review2=payload["student_critical_review2"],
            )
            st.success("✅ 구글 시트에 성공적으로 저장되었습니다.")
        except Exception as e:
            st.error(f"⚠️ 구글 시트 저장 오류: {e}")
            st.stop()

    if go_next:
        st.success("최종 보고서 작성 페이지로 이동합니다.")
        st.switch_page("assessment/final_report.py")
