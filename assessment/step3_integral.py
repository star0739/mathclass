# assessment/step3_integral.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 3차시: 적분(누적) 관점에서 모델 평가 + 장점/한계 정리
#
# UX 목표(1/2차시와 유사):
# 0) 2차시 기록 불러오기(백업 TXT 업로드) + CSV 업로드(그래프/적분 계산용)
# 1) X/Y 선택 및 시간축 해석 방식 선택
# 2) 누적량(수치적분) vs 모델 정적분 비교
# 3) 누적 그래프 비교
# 4) 종합 결론(장점/한계/개선) 작성
# 5) 저장 및 백업(구글시트 + TXT) + (선택) 다음 페이지 이동
# ------------------------------------------------------------

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

from assessment.common import (
    init_assessment_session,
    require_student_id,
    set_df,
    get_df,
    get_df_preview,
    set_xy,
    get_xy,
    get_step1_summary,
)

from assessment.google_sheets import append_step3_row


# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 5  # 적분 비교는 구간이 있으니 MVP는 낮게


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
# 년/월/년월 파서 (Step2와 동일 계열)
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
#  - Step2의 build_step2_backup 포맷을 대략적으로 읽어 필요한 값만 추출
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
    # 데이터 정보
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

    # LaTeX/py 식은 섹션 기반 추출이 포맷 변화에 취약하니,
    # MVP에서는 "키워드 라인"을 직접 찾지 않고, Step2 앱 저장을 우선 사용.
    # (필요하면 Step2 백업 포맷을 key:value로 통일하는 리팩토링에서 개선)
    out["py_model"] = ""  # 백업에서 안정적으로 뽑기 어려움 → Step2 세션 값 우선
    return out


# -----------------------------
# 수치적분(사다리꼴) + 누적 사다리꼴
# -----------------------------
def _trapz(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.trapz(y, t))


def _cumtrapz(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = np.zeros_like(y, dtype=float)
    for k in range(1, len(y)):
        dt = t[k] - t[k - 1]
        A[k] = A[k - 1] + 0.5 * (y[k] + y[k - 1]) * dt
    return A


# -----------------------------
# 모델 평가: Step2 py_model(표현식) eval
# -----------------------------
def _eval_model_expr(expr: str, t: np.ndarray) -> np.ndarray:
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("py_model이 비어 있습니다.")

    # 구글시트에서 '='로 시작하면 수식으로 오해할 수 있으니 Step2에서 텍스트로 저장했을 수 있음
    # 화면에서는 그대로 오지만, 혹시 '=...'이면 앞 '=' 제거는 하지 않고 오류로 처리(학생 수정 유도)
    if expr.startswith("="):
        raise ValueError("py_model이 '='로 시작합니다. 수식이 아니라 '표현식'만 입력하세요.")

    # 최소한의 위험 토큰 차단(MVP)
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

    # 스칼라면 브로드캐스트
    if y_hat.shape == ():
        y_hat = np.full_like(t, float(y_hat), dtype=float)

    if len(y_hat) != len(t):
        raise ValueError("모델 결과 길이가 t와 일치하지 않습니다.")
    return y_hat


# -----------------------------
# Step3 백업 생성/파서 (Step2 UX와 유사)
# -----------------------------
def build_step3_backup(payload: dict) -> bytes:
    lines = []
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
    lines.append(payload.get("py_model","").strip() or "(미입력)")
    lines.append("")

    lines.append("[적분 결과]")
    lines.append(f"- 데이터 누적량(근사): {payload.get('A_data','')}")
    lines.append(f"- 모델 누적량(근사): {payload.get('A_model','')}")
    lines.append(f"- 상대오차: {payload.get('relative_error','')}")
    lines.append("")

    lines.append("[종합 결론(학생 작성)]")
    lines.append(payload.get("conclusion","").strip() or "(미입력)")
    lines.append("")
    lines.append("[추가 메모]")
    lines.append(payload.get("note","").strip() or "(없음)")
    lines.append("")
    lines.append("※ 이 파일은 학생 개인 백업용입니다. 필요 시 다시 앱에 업로드하여 복구할 수 있습니다.")
    return "\n".join(lines).encode("utf-8-sig")


def parse_step3_backup_txt(text: str) -> dict:
    out = {}
    lines = [ln.strip() for ln in (text or "").splitlines()]

    def find_value(prefix: str) -> str:
        for ln in lines:
            if ln.startswith(prefix):
                return ln.replace(prefix, "", 1).strip()
        return ""

    out["student_id"] = find_value("학번:")
    out["data_source"] = find_value("- 데이터 출처:")
    out["x_col"] = ""
    out["y_col"] = ""
    for ln in lines:
        if ln.startswith("- X축:"):
            m = re.search(r"- X축:\s*(.*?)\s*\|\s*Y축:\s*(.*)$", ln)
            if m:
                out["x_col"] = m.group(1).strip()
                out["y_col"] = m.group(2).strip()
    out["valid_n"] = find_value("- 유효 데이터 점:")
    # 구간
    rng = find_value("- 적분 구간 인덱스:")
    m = re.search(r"(\d+)\s*~\s*(\d+)", rng)
    if m:
        out["i0"] = m.group(1)
        out["i1"] = m.group(2)

    # 결론 섹션
    try:
        i = lines.index("[종합 결론(학생 작성)]")
        j = lines.index("[추가 메모]")
        out["conclusion"] = "\n".join(lines[i + 1 : j]).strip()
    except ValueError:
        out["conclusion"] = ""

    out["note"] = ""  # 메모는 MVP에서 생략(필요하면 섹션 파싱 추가)
    return out


# ============================================================
# UI 시작
# ============================================================
init_assessment_session()
student_id = require_student_id("학번을 입력하세요.")

st.title("(3차시) 적분(누적) 관점에서 모델 평가")
st.caption("데이터 누적량과 모델 정적분을 비교하고, 최종적으로 모델의 장점과 한계를 정리합니다.")
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
    st.markdown("**CSV 업로드(그래프/적분 계산용)**")
    csv_file = st.file_uploader("CSV 업로드", type=["csv"], key="step3_csv_upload")

if step2_txt is not None:
    try:
        raw = step2_txt.getvalue().decode("utf-8", errors="replace")
        parsed2 = parse_step2_backup_txt(raw)

        # step2_state 보강(가능한 범위만)
        # (py_model 등은 백업 포맷에서 안정적으로 못 뽑으므로 기존 step2 세션 값을 유지)
        step2 = {
            **step2,
            "student_id": parsed2.get("student_id") or step2.get("student_id") or student_id,
            "data_source": parsed2.get("data_source") or step2.get("data_source") or step1.get("data_source", ""),
            "x_col": parsed2.get("x_col") or step2.get("x_col") or step1.get("x_col", ""),
            "y_col": parsed2.get("y_col") or step2.get("y_col") or step1.get("y_col", ""),
            "valid_n": parsed2.get("valid_n") or step2.get("valid_n") or step1.get("valid_n", ""),
        }
        _set_step2_state(step2)
        st.success("TXT에서 2차시 정보를(부분적으로) 불러왔습니다. (수식 py_model 등은 세션 저장값을 우선 사용)")
    except Exception as e:
        st.error("2차시 TXT를 읽는 중 오류가 발생했습니다.")
        st.exception(e)

# CSV 업로드 시 DF 저장
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
    st.info("CSV를 업로드하면 다음 단계(적분 비교)로 진행할 수 있습니다.")
    st.stop()

with st.expander("참고: 데이터 미리보기", expanded=False):
    st.dataframe(get_df_preview(df), use_container_width=True)

st.divider()

# ============================================================
# 1) X/Y 선택(통일 규칙) + X축 해석 방식
# ============================================================
st.subheader("1) X/Y 선택")

cols = list(df.columns)
if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다. CSV를 다시 확인하세요.")
    st.stop()

# ✅ 통일 규칙: Step2 저장값 → Step1 summary → get_xy()
x_prev, y_prev = get_xy()
x_init = step2.get("x_col") or step1.get("x_col") or (x_prev if x_prev in cols else cols[0])
y_init = step2.get("y_col") or step1.get("y_col") or (y_prev if y_prev in cols else (cols[1] if len(cols) > 1 else cols[0]))

if x_init not in cols:
    x_init = cols[0]
if y_init not in cols:
    y_init = cols[1] if len(cols) > 1 else cols[0]
if y_init == x_init:
    y_init = cols[1] if len(cols) > 1 and cols[1] != x_init else cols[0]

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    x_col = st.selectbox("X축 선택", cols, index=cols.index(x_init), key="step3_x_col")
with col_sel2:
    y_col = st.selectbox("Y축 선택", cols, index=cols.index(y_init), key="step3_y_col")

set_xy(x_col, y_col)

x_mode = st.radio("X축 해석 방식", ["자동(권장)", "날짜(년월)", "숫자"], horizontal=True, key="step3_x_mode")

# 데이터 숫자화
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

if len(xv) < MIN_VALID_POINTS:
    st.warning(f"유효 데이터 점이 {len(xv)}개입니다. (최소 {MIN_VALID_POINTS}개 권장)")
    if len(xv) < 2:
        st.stop()

# 정렬
if len(xv) >= 2:
    order = np.argsort(xv.values) if x_type == "datetime" else np.argsort(xv.to_numpy())
    xv = xv.iloc[order]
    yv = yv.iloc[order]

# t 수치화(적분/모델 계산용)
if x_type == "datetime":
    base = xv.iloc[0]
    # 월 인덱스
    t_all = ((xv.dt.year - base.year) * 12 + (xv.dt.month - base.month)).to_numpy(dtype=float)

    # 연도만 들어온 듯하면 경고(YYYY만 많은 경우)
    raw = df.loc[valid, x_col].astype(str).str.strip().iloc[order]
    if (raw.str.fullmatch(r"\d{4}")).mean() >= 0.8:
        st.warning("시간 데이터가 '연도(YYYY)' 중심으로 보입니다. 월 단위(1월 가정)로 변환되어 해석이 거칠 수 있습니다.")
else:
    t_all = xv.to_numpy(dtype=float)

y_all = yv.to_numpy(dtype=float)

st.metric("유효 데이터 점(숫자쌍) 개수", int(len(t_all)))

st.divider()

# ============================================================
# 2) 모델식(py_model) 확인 + 적분 구간 선택
# ============================================================
st.subheader("2) 모델식 확인 & 적분 구간 선택")

py_model_default = (step2.get("py_model") or "").strip()
py_model = st.text_input(
    "모델식 f(t) (numpy 사용, 변수는 t)",
    value=py_model_default,
    placeholder="예: 22 - 0.017*t + 6*np.cos(2*np.pi*t/12) + 4*np.sin(2*np.pi*t/12)",
)

n = len(t_all)
# Step3 백업 복구(선택)
restored3 = {}
st.markdown("**(선택) 3차시 백업 TXT로 복구**")
step3_txt = st.file_uploader("3차시 백업 TXT 업로드", type=["txt"], key="step3_txt_upload")
if step3_txt is not None:
    try:
        raw3 = step3_txt.getvalue().decode("utf-8", errors="replace")
        restored3 = parse_step3_backup_txt(raw3)
        st.success("3차시 백업에서 일부 값을 불러왔습니다.")
    except Exception as e:
        st.error("3차시 TXT를 읽는 중 오류가 발생했습니다.")
        st.exception(e)

def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default

default_i0 = _safe_int(restored3.get("i0", step3_prev.get("i0", 0)), 0)
default_i1 = _safe_int(restored3.get("i1", step3_prev.get("i1", n - 1)), n - 1)
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
y = y_all[i0 : i1 + 1]

# x축 표시용(가능하면 datetime을 유지)
x_display = xv.iloc[i0 : i1 + 1]  # datetime 또는 numeric 시리즈

st.divider()

# ============================================================
# 3) 누적량(정적분) 비교
# ============================================================
st.subheader("3) 누적량(정적분) 비교")

A_data = _trapz(y, t)

A_model = None
y_hat = None
model_err_msg = ""

if py_model.strip():
    try:
        y_hat_all = _eval_model_expr(py_model, t_all)
        y_hat = y_hat_all[i0 : i1 + 1]
        A_model = _trapz(y_hat, t)
    except Exception as e:
        model_err_msg = str(e)

c1, c2, c3 = st.columns(3)
c1.metric("데이터 누적량  ∫y dt(근사)", f"{A_data:,.6g}")

if A_model is None:
    c2.metric("모델 누적량  ∫f dt(근사)", "—")
    c3.metric("상대오차", "—")
    if model_err_msg:
        st.warning(f"모델 적분을 계산하지 못했습니다: {model_err_msg}")
else:
    c2.metric("모델 누적량  ∫f dt(근사)", f"{A_model:,.6g}")
    rel = abs(A_data - A_model) / (abs(A_data) + 1e-12)
    c3.metric("상대오차", f"{rel:.3%}")

st.divider()

# ============================================================
# 4) 누적 그래프(누적 적분 곡선) 비교
# ============================================================
st.subheader("4) 누적 그래프 비교")

cum_data = _cumtrapz(y, t)
cum_model = None if y_hat is None else _cumtrapz(y_hat, t)

if PLOTLY_AVAILABLE:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_display, y=cum_data, mode="lines", name="누적(데이터)"))
    if cum_model is not None:
        fig.add_trace(go.Scatter(x=x_display, y=cum_model, mode="lines", name="누적(모델)"))
    fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title=str(x_col),
        yaxis_title="누적량",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(cum_data)), cum_data, label="누적(데이터)")
    if cum_model is not None:
        ax.plot(np.arange(len(cum_model)), cum_model, label="누적(모델)")
    ax.set_xlabel("index")
    ax.set_ylabel("누적량")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

st.divider()

# ============================================================
# 5) 종합 결론(장점/한계/개선 제안)
# ============================================================
st.subheader("5) 종합 결론: 이 모델의 장점과 한계")

st.info(
    "아래 내용을 모두 포함해 서술하세요.\n"
    "• 누적 관점에서 데이터와 모델이 얼마나 일치하는가(근거: 누적량/누적 그래프)\n"
    "• 장점 1가지(근거 포함)\n"
    "• 한계 1가지(근거 포함)\n"
    "• 개선 제안 1가지(변수/모델/구간/방법 등)\n"
)

conclusion_default = (
    restored3.get("conclusion")
    or step3_prev.get("conclusion", "")
)

conclusion = st.text_area(
    "종합 서술(필수)",
    value=conclusion_default,
    height=220,
)

note = st.text_area(
    "추가 메모(선택)",
    value=step3_prev.get("note", ""),
    height=100,
)

st.divider()

# ============================================================
# 6) 저장 및 백업 (Step1/2와 유사)
# ============================================================
st.subheader("6) 저장 및 백업")

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
    "A_data": float(A_data),
    "A_model": "" if A_model is None else float(A_model),
    "relative_error": "" if A_model is None else float(abs(A_data - A_model) / (abs(A_data) + 1e-12)),
    "py_model": py_model.strip(),
    "conclusion": conclusion.strip(),
    "note": note.strip(),
}

col1, col2, col3 = st.columns([1, 1, 1.2])
save_clicked = col1.button("💾 저장(구글시트)", use_container_width=True)
download_clicked = col2.button("⬇️ TXT 백업 만들기", use_container_width=True)
go_next = col3.button("➡️ 종료/제출", use_container_width=True)

# 다운로드 버튼은 항상 렌더링(최신 payload 반영)
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
    return True

if save_clicked or download_clicked or go_next:
    if not _validate_step3():
        st.stop()

    # (1) 세션 저장: 다운로드 클릭 시에도 실행(2차시 UX와 동일)
    _set_step3_state({**payload, "saved_at": pd.Timestamp.now().isoformat()})

    if download_clicked:
        st.success("✅ 백업 데이터가 준비되었습니다. 위 '다운로드' 버튼을 눌러주세요.")

    # (2) 구글 시트 저장: 저장 버튼이나 종료/제출 버튼 클릭 시 실행
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

    # (3) 종료/제출 처리(페이지 이동이 필요하면 switch_page로 변경)
    if go_next:
        st.success("제출/종료 처리되었습니다. (필요 시 다음 페이지로 이동 로직을 연결하세요.)")

# 검토용
with st.expander("계산 세부값(검토용)", expanded=False):
    st.write(
        {
            "x_col": x_col,
            "y_col": y_col,
            "x_type": x_type,
            "n_valid": int(len(t_all)),
            "range": (int(i0), int(i1)),
            "A_data": float(A_data),
            "A_model": None if A_model is None else float(A_model),
            "py_model_preview": (py_model[:120] + ("..." if len(py_model) > 120 else "")),
        }
    )
