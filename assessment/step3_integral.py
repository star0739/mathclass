# assessment/step3_integral.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 3차시: 정적분(면적)과 수치적분(구간합)의 관계 이해
#
# UX(1/2차시와 유사):
# 0) 2차시 기록 불러오기(TXT 업로드) + CSV 업로드
# 1) 데이터 열 자동 설정(X/Y) + 시간축(t) 자동 변환
# 2) 모델식 확인 + 적분 구간 선택
# 3) f(t) 그래프 위에 직사각형/사다리꼴 도형 시각화 + 근사값/오차 비교
# 4) 종합 결론(장점/한계/개선)
# 5) 저장 및 백업(구글시트 + TXT)
# ------------------------------------------------------------

from __future__ import annotations

import re
import sympy as sp
import numpy as np
import pandas as pd
import streamlit as st

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
# CSV 로더 (1/2차시와 동일하게 관대)
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
# 년/월/년월 파서 (Step2 계열)
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

    # (보조) YYYY만 있는 경우: 1월로 간주 (해석 주의)
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
    out = {}
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
    out["py_model"] = ""  # Step2 백업 포맷에 따라 안정적 추출이 어려움(세션 값 우선)
    return out


# -----------------------------
# 수치적분(사다리꼴) - np.trapz 의존 제거
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
# 모델 평가: Step2 py_model(표현식) eval
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

def _sympy_definite_integral(py_expr: str, a: float, b: float) -> float:
    """
    py_model 표현식(예: 22 - 0.017*t + 6*np.cos(...) + ...)을 sympy로 정적분.
    실패하면 예외 발생(호출부에서 수치적분 fallback 가능).
    """
    expr = (py_expr or "").strip()
    if not expr:
        raise ValueError("py_model이 비어 있습니다.")

    # numpy 접두어 제거
    expr = expr.replace("np.", "")
    expr = expr.replace("numpy.", "")

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

    # 적분이 미해결(Integral 형태)인 경우 방지
    if isinstance(I, sp.Integral) or I.has(sp.Integral):
        raise ValueError("sympy가 정적분을 기호적으로 계산하지 못했습니다.")

    return float(sp.N(I))


# -----------------------------
# 리만합 분할
# -----------------------------
def _riemann_partitions(a: float, b: float, n: int) -> tuple[np.ndarray, float]:
    n = int(n)
    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다.")
    nodes = np.linspace(a, b, n + 1, dtype=float)
    dt = (b - a) / n
    return nodes, dt


def _rect_y0y1(height: float) -> tuple[float, float]:
    return (0.0, float(height)) if height >= 0 else (float(height), 0.0)

# -----------------------------
# 데이터 기반 수치적분 (직사각형/사다리꼴)
# -----------------------------
def _data_rect_left(y: np.ndarray, t: np.ndarray) -> float:
    """
    좌측 직사각형 합:
    각 구간 [t_i, t_{i+1}]에서 높이를 y_i로 사용
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return 0.0

    dt = t[1:] - t[:-1]
    return float(np.sum(y[:-1] * dt))


def _data_trap(y: np.ndarray, t: np.ndarray) -> float:
    """
    사다리꼴 합:
    각 구간에서 0.5*(y_i + y_{i+1}) * dt
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(y) < 2:
        return 0.0

    dt = t[1:] - t[:-1]
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * dt))

# -----------------------------
# Step3 백업 생성 (Step2 UX와 유사)
# -----------------------------
def build_step3_backup(payload: dict) -> bytes:
    lines: list[str] = []
    lines.append("공공데이터 분석 수행 (3차시) 백업")
    lines.append("=" * 40)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")

    lines.append("[데이터 정보]")
    lines.append(f"- 데이터 출처: {payload.get('data_source','')}")
    lines.append(f"- X축: {payload.get('x_col','')} | Y축: {payload.get('y_col','')}")
    lines.append(f"- 유효 데이터 점: {payload.get('valid_n','')}")
    lines.append(f"- 적분 구간 인덱스: {payload.get('i0','')} ~ {payload.get('i1','')}")
    lines.append("")

    lines.append("[모델식(py_model)]")
    lines.append(payload.get("py_model", "").strip() or "(미입력)")
    lines.append("")

    lines.append("[정적분(기준값)과 수치적분 비교]")
    lines.append(f"- 기준 정적분값 I(근사): {payload.get('I_ref','')}")
    lines.append(f"- 분할 수 n: {payload.get('n_div','')}")
    lines.append(f"- 좌/중/우 직사각형 합: {payload.get('S_left','')}, {payload.get('S_mid','')}, {payload.get('S_right','')}")
    lines.append(f"- 사다리꼴 합: {payload.get('S_trap','')}")
    lines.append(f"- 오차 |S-I| (좌/중/우/사다리꼴): {payload.get('err_left','')}, {payload.get('err_mid','')}, {payload.get('err_right','')}, {payload.get('err_trap','')}")
    lines.append("")

    lines.append("[종합 결론(학생 작성)]")
    lines.append(payload.get("conclusion", "").strip() or "(미입력)")
    lines.append("")
    lines.append("[추가 메모]")
    lines.append(payload.get("note", "").strip() or "(없음)")
    lines.append("")
    lines.append("※ 이 파일은 학생 개인 백업용입니다.")
    return "\n".join(lines).encode("utf-8-sig")


# ============================================================
# UI 시작
# ============================================================
init_assessment_session()
student_id = require_student_id("학번을 입력하세요.")

st.title("(3차시) 정적분과 수치적분(구간합) 비교")
st.caption("모델 f(t)의 정적분(면적) 값을 기준으로, 직사각형 합/사다리꼴 합의 오차가 n에 따라 어떻게 줄어드는지 관찰합니다.")
st.divider()

# ============================================================
# 0) 2차시 기록 불러오기/복구 + CSV 업로드
# ============================================================
st.subheader("0) 2차시 기록 불러오기")

step1 = get_step1_summary() or {}
step2 = _get_step2_state() or {}
step3_prev = _get_step3_state() or {}

colA, colB = st.columns([1.2, 1])
with colA:
    st.markdown("**2차시 TXT 업로드로 복구(선택)**")
    step2_txt = st.file_uploader("2차시 백업 TXT 업로드", type=["txt"], key="step3_step2_txt_upload")
with colB:
    st.markdown("**CSV 업로드(그래프/계산용)**")
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
        st.success("TXT에서 2차시 정보를(부분적으로) 불러왔습니다. (py_model 등은 세션 저장값을 우선 사용)")
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

# ============================================================
# 1) 데이터 열 자동 설정 + 시간축(t) 자동 변환
# ============================================================
st.subheader("1) 데이터 열 자동 설정")

cols = list(df.columns)
if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다. CSV를 다시 확인하세요.")
    st.stop()

x_prev, y_prev = get_xy()
x_col = (step2.get("x_col") or step1.get("x_col") or (x_prev if x_prev in cols else "")).strip()
y_col = (step2.get("y_col") or step1.get("y_col") or (y_prev if y_prev in cols else "")).strip()

if x_col not in cols:
    x_col = cols[0]
if y_col not in cols:
    y_col = cols[1] if len(cols) > 1 else cols[0]
if y_col == x_col:
    y_col = cols[1] if len(cols) > 1 and cols[1] != x_col else cols[0]

st.caption(f"X축(시간): **{x_col}**  |  Y축(수치): **{y_col}** (2차시 선택값 기반 자동 설정)")

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
    st.warning(f"유효 데이터 점이 {len(xv)}개입니다. (최소 {MIN_VALID_POINTS}개 권장)")
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

# ============================================================
# 2) 모델식 확인 + 적분 구간 선택
# ============================================================
st.subheader("2) 모델식 확인 & 적분 구간 선택")

py_model_default = (step2.get("py_model") or "").strip()
py_model = st.text_input(
    "모델식 f(t) (numpy 사용, 변수는 t)",
    value=py_model_default,
    placeholder="예: 22 - 0.017*t + 6*np.cos(2*np.pi*t/12) + 4*np.sin(2*np.pi*t/12)",
)

n = len(t_all)
default_i0 = int(step3_prev.get("i0", 0) or 0)
default_i1 = int(step3_prev.get("i1", n - 1) or (n - 1))
default_i0 = max(0, min(n - 2, default_i0))
default_i1 = max(default_i0 + 1, min(n - 1, default_i1))

i0, i1 = st.slider(
    "적분 구간(인덱스)",
    min_value=0,
    max_value=n - 1,
    value=(default_i0, default_i1),
    step=1,
)

t = t_all[i0 : i1 + 1]
st.divider()

# ============================================================
# 3) f(t)로 보는 정적분 근사(직사각형 합 / 사다리꼴) + 오차
# ============================================================
st.subheader("3) 데이터 기반 수치적분 vs 모델 정적분 비교")

# --- (1) 데이터 기반 수치적분 값 ---
# 직사각형: 좌측 리만합(데이터 y_i를 구간 [t_i,t_{i+1}] 높이로)
A_rect = _data_rect_left(y, t)

# 사다리꼴: 데이터 점을 선분으로 연결한 사다리꼴 합
A_trap = _data_trap(y, t)

# --- (2) 모델 정적분 값 ---
a, b = float(t[0]), float(t[-1])

I_model = None
I_err_msg = ""
if py_model.strip():
    try:
        I_model = _sympy_definite_integral(py_model, a, b)
    except Exception as e:
        I_err_msg = str(e)

# 모델 정적분이 안 되면(예: sympy 미해결) 수치적분 fallback(선택)
if I_model is None:
    st.warning("sympy 정적분 계산에 실패했습니다. 모델 정적분 값은 수치적으로 근사합니다.")
    if I_err_msg:
        st.caption(f"sympy 오류: {I_err_msg}")
    tt_ref = np.linspace(a, b, 20001, dtype=float)
    ff_ref = _eval_model_expr(py_model, tt_ref)
    I_model = _trapz(ff_ref, tt_ref)  # 매우 촘촘한 사다리꼴(기준값)

# --- (3) 값 표시(윗줄 3칸) ---
c1, c2, c3 = st.columns(3)
c1.metric("직사각형 값(데이터)", f"{A_rect:,.6g}")
c2.metric("사다리꼴 값(데이터)", f"{A_trap:,.6g}")
c3.metric("정적분 값(모델)", f"{I_model:,.6g}")

# --- (4) 오차 표시(아랫줄 3칸: 마지막은 빈칸) ---
e_rect = abs(A_rect - I_model)
e_trap = abs(A_trap - I_model)

d1, d2, d3 = st.columns(3)
d1.metric("직사각형 오차 |A-I|", f"{e_rect:,.6g}")
d2.metric("사다리꼴 오차 |A-I|", f"{e_trap:,.6g}")
d3.metric("", "")  # 열 맞추기용 빈칸

# --- (5) 그래프: 데이터 점 + 모델 곡선 + 데이터 도형 ---
tt = np.linspace(a, b, 600, dtype=float)
ff = _eval_model_expr(py_model, tt)

if PLOTLY_AVAILABLE:
    fig = go.Figure()

    # 모델 곡선
    fig.add_trace(go.Scatter(x=tt, y=ff, mode="lines", name="모델 f(t)"))

    # 데이터 점(선택 구간)
    fig.add_trace(go.Scatter(x=t, y=y, mode="markers+lines", name="데이터(구간)"))

    # 도형 표시 방식 선택(직사각형 or 사다리꼴)
    vis_mode = st.radio("도형 표시", ["직사각형(좌측)", "사다리꼴"], horizontal=True)

    if vis_mode == "직사각형(좌측)":
        for i in range(len(t) - 1):
            x0 = float(t[i]); x1 = float(t[i + 1])
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
            x0 = float(t[i]); x1 = float(t[i + 1])
            yL = float(y[i]); yR = float(y[i + 1])
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
        xaxis_title="t (모델 시간축)",
        yaxis_title="y",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Plotly가 없어 도형 시각화는 생략됩니다. (값/오차는 위에서 확인 가능)")

st.divider()

# ============================================================
# 4) 종합 결론(장점/한계/개선 제안)
# ============================================================
st.subheader("4) 종합 결론: 이 모델의 장점과 한계")

st.info(
    "아래 내용을 모두 포함해 서술하세요.\n"
    "• n이 커질수록 오차(|S-I|)가 어떻게 변하는가?\n"
    "• 좌/중/우/사다리꼴 중 어떤 방법이 더 빠르게 정확해졌는가?\n"
    "• 이 모델의 장점 1가지, 한계 1가지(근거 포함)\n"
    "• 개선 제안 1가지(변수/모델/구간/방법 등)\n"
)

conclusion = st.text_area(
    "종합 서술(필수)",
    value=step3_prev.get("conclusion", ""),
    height=220,
)
note = st.text_area(
    "추가 메모(선택)",
    value=step3_prev.get("note", ""),
    height=100,
)

st.divider()

# ============================================================
# 5) 저장 및 백업
# ============================================================
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
    # 핵심 결과(학습목표에 맞춤)
    "I_ref": float(I_ref),
    "n_div": int(n_div),
    "S_left": float(S_left),
    "S_mid": float(S_mid),
    "S_right": float(S_right),
    "S_trap": float(S_trap),
    "err_left": float(eL),
    "err_mid": float(eM),
    "err_right": float(eR),
    "err_trap": float(eT),
    # 구글시트 기존 컬럼 호환(유지): A_model=I_ref, relative_error=중점 상대오차(대표값)
    "A_data": "",  # Step3 핵심에서 제외(공란 저장)
    "A_model": float(I_ref),
    "relative_error": float(rM),
    "conclusion": conclusion.strip(),
    "note": note.strip(),
}

col1, col2, col3 = st.columns([1, 1, 1.2])
save_clicked = col1.button("💾 저장(구글시트)", use_container_width=True)
download_clicked = col2.button("⬇️ TXT 백업 만들기", use_container_width=True)
go_next = col3.button("➡️ 종료/제출", use_container_width=True)

backup_bytes = build_step3_backup(payload)
st.download_button(
    label="📄 (다운로드) 3차시 백업 TXT",
    data=backup_bytes,
    file_name=f"미적분_수행평가_3차시_{student_id}.txt",
    mime="text/plain; charset=utf-8",
)

def _validate_step3() -> bool:
    if not payload["conclusion"]:
        st.warning("종합 서술을 입력하세요.")
        return False
    if not payload["py_model"]:
        st.warning("모델식(py_model)을 확인하세요.")
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
                A_data=payload["A_data"],
                A_model=payload["A_model"],
                relative_error=payload["relative_error"],
                py_model=payload["py_model"],
                conclusion=payload["conclusion"],
                note=payload["note"],
            )
            st.success("✅ 구글 시트에 성공적으로 저장되었습니다.")
        except Exception as e:
            st.error(f"⚠️ 구글 시트 저장 오류: {e}")
            st.stop()

    if go_next:
        st.success("제출/종료 처리되었습니다. (필요 시 다음 페이지 이동 로직을 연결하세요.)")

with st.expander("계산 세부값(검토용)", expanded=False):
    st.write(
        {
            "x_col": x_col,
            "y_col": y_col,
            "x_type": x_type,
            "n_valid": int(len(t_all)),
            "range": (int(i0), int(i1)),
            "n_div": int(n_div),
            "I_ref": float(I_ref),
            "S_left": float(S_left),
            "S_mid": float(S_mid),
            "S_right": float(S_right),
            "S_trap": float(S_trap),
            "err_abs": {"left": float(eL), "mid": float(eM), "right": float(eR), "trap": float(eT)},
            "err_rel": {"left": float(rL), "mid": float(rM), "right": float(rR), "trap": float(rT)},
        }
    )
