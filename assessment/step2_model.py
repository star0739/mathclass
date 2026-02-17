# assessment/step2_model.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 2차시: AI 모델식 도출 + 미분 기반 검증(비판적 검토)
#
# 핵심:
# - 1차시 기록이 날아가도 복구 가능: (1) 1차시 TXT 업로드 (2) CSV 업로드
# - AI가 제안한 모델식을 "LaTeX($$...$$)" 형태로 받도록 안내 + 입력
# - 데이터 기반 근사 변화율(Δy/Δt)과 비교하여 학생이 비판적으로 검토
# - 저장 시: Google Sheet(미적분_수행평가_2차시) + TXT 백업 다운로드
# ------------------------------------------------------------

import re
import streamlit as st
import pandas as pd
import numpy as np

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

from assessment.google_sheets import append_step2_row

# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 30


# -----------------------------
# 세션용 step2 저장(간단)
# -----------------------------
def _get_step2_state() -> dict:
    return st.session_state.get("assessment_step2", {})


def _set_step2_state(d: dict) -> None:
    st.session_state["assessment_step2"] = d


# -----------------------------
# CSV 로더 (1차시와 동일하게 관대)
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
    return dt


# -----------------------------
# 1차시 TXT 백업 파서(최소)
#  - step1 txt의 섹션 제목을 이용해 값 추출
# -----------------------------
def parse_step1_backup_txt(text: str) -> dict:
    # 매우 관대한 파서: 키워드 라인을 찾아 값 추출
    out = {}
    lines = [ln.strip() for ln in text.splitlines()]

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
            # "- X축: A  |  Y축: B"
            m = re.search(r"- X축:\s*(.*?)\s*\|\s*Y축:\s*(.*)$", ln)
            if m:
                out["x_col"] = m.group(1).strip()
                out["y_col"] = m.group(2).strip()
    out["x_mode"] = find_value("- X축 해석 방식:")
    out["valid_n"] = find_value("- 유효 데이터 점 개수:")
    out["features"] = ""

    # [그래프 관찰 특징] 섹션 추출
    try:
        i = lines.index("[그래프 관찰 특징]")
        j = lines.index("[모델링 가설]")
        out["features"] = "\n".join(lines[i + 1 : j]).strip()
    except ValueError:
        pass

    out["model_primary"] = find_value("- 주된 모델:")
    # 주된 모델 근거 섹션
    try:
        i = lines.index("- 주된 모델 근거:")
        # 다음 섹션까지
        j = lines.index("[추가 메모]")
        out["model_primary_reason"] = "\n".join(lines[i + 1 : j]).strip()
    except ValueError:
        out["model_primary_reason"] = ""

    return out


# -----------------------------
# LaTeX 블록 추출/미리보기용
# -----------------------------
LATEX_BLOCK = re.compile(r"\${1,2}(.+?)\${1,2}", re.DOTALL)

def extract_latex_blocks(s: str) -> list[str]:
    if not s:
        return []
    return [m.group(1).strip() for m in LATEX_BLOCK.finditer(s)]


# -----------------------------
# 데이터 기반 근사 도함수(차분/gradient)
# -----------------------------
def compute_derivatives(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # t가 등간격이 아닐 수도 있어 gradient에 t를 넣어 안정화
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return dy, d2y


# -----------------------------
# TXT 백업 생성(2차시)
# -----------------------------
def build_step2_backup(payload: dict) -> bytes:
    lines = []
    lines.append("공공데이터 분석 수행 (2차시) 백업")
    lines.append("=" * 40)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번/식별코드: {payload.get('student_id','')}")
    lines.append("")

    lines.append("[가설 재평가]")
    lines.append(f"- 1차시 가설 모델: {payload.get('model_hypothesis_step1','')}")
    lines.append(f"- 가설 판단: {payload.get('hypothesis_decision','')}")
    if payload.get("hypothesis_decision") == "가설 수정":
        lines.append(f"- 수정한 가설 모델: {payload.get('revised_model','')}")
    lines.append("")

    lines.append("[데이터 정보]")
    lines.append(f"- 데이터 출처: {payload.get('data_source','')}")
    lines.append(f"- X축: {payload.get('x_col','')} | Y축: {payload.get('y_col','')}")
    lines.append(f"- 유효 데이터 점: {payload.get('valid_n','')}")
    lines.append("")

    lines.append("[AI 프롬프트]")
    lines.append(payload.get("ai_prompt","").strip() or "(미입력)")
    lines.append("")

    lines.append("[AI 모델식/미분식(LaTeX)]")
    lines.append(payload.get("ai_model_latex","").strip() or "(미입력)")
    lines.append(payload.get("ai_derivative_latex","").strip() or "")
    lines.append(payload.get("ai_second_derivative_latex","").strip() or "")
    lines.append("")

    lines.append("[미분 관점의 모델 해석(학생 작성)]")
    lines.append(payload.get("student_analysis","").strip() or "(미입력)")
    lines.append("")

    lines.append("[추가 메모]")
    lines.append(payload.get("note","").strip() or "(없음)")
    lines.append("")
    lines.append("※ 수식은 $$...$$ 형태의 LaTeX로 유지하는 것이 안전합니다.")

    return "\n".join(lines).encode("utf-8-sig")

# ============================================================
# UI 시작
# ============================================================
init_assessment_session()
student_id = require_student_id("학번을 입력하세요.")

st.title("(2차시) AI 모델식 도출 & 미분 기반 검증")
st.caption("AI가 제안한 모델식을 입력하고, 데이터 변화율(근사 도함수)과 비교해 비판적으로 검토합니다.")
st.divider()

# ============================================================
# 0) 1차시 기록 불러오기/복구
# ============================================================
st.subheader("0) 1차시 기록 불러오기")

step1 = get_step1_summary() or {}
step2_prev = _get_step2_state()

colA, colB = st.columns([1.2, 1])
with colA:
    st.markdown("**(권장) 1차시 TXT 업로드로 복구**")
    txt_file = st.file_uploader("1차시 백업 TXT 업로드", type=["txt"], key="step2_txt_upload")

with colB:
    st.markdown("**(선택) CSV 다시 업로드(그래프/도함수 계산용)**")
    csv_file = st.file_uploader("CSV 업로드", type=["csv"], key="step2_csv_upload")

if txt_file is not None:
    try:
        raw = txt_file.getvalue().decode("utf-8", errors="replace")
        parsed = parse_step1_backup_txt(raw)
        # step1 dict 보강
        step1.update({
            "student_id": parsed.get("student_id") or step1.get("student_id") or student_id,
            "data_source": parsed.get("data_source") or step1.get("data_source",""),
            "x_col": parsed.get("x_col") or step1.get("x_col",""),
            "y_col": parsed.get("y_col") or step1.get("y_col",""),
            "x_mode": parsed.get("x_mode") or step1.get("x_mode",""),
            "valid_n": parsed.get("valid_n") or step1.get("valid_n",""),
            "features": parsed.get("features") or step1.get("features",""),
            "model_primary": parsed.get("model_primary") or step1.get("model_primary",""),
            "model_primary_reason": parsed.get("model_primary_reason") or step1.get("model_primary_reason",""),
        })
        st.success("TXT에서 1차시 정보를 불러왔습니다.")
    except Exception as e:
        st.error("TXT를 읽는 중 오류가 발생했습니다.")
        st.exception(e)

# CSV 업로드 시 DF 저장
if csv_file is not None:
    try:
        df = read_csv_kosis(csv_file)
        set_df(df)
        st.success(f"CSV 업로드 완료 ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
    except Exception as e:
        st.error("CSV를 읽지 못했습니다.")
        st.exception(e)

# df = get_df()
# if df is not None:
#    st.markdown("#### 참고: 현재 데이터 미리보기")
#    st.dataframe(get_df_preview(df), use_container_width=True)

st.divider()

# ============================================================
# 1) AI 프롬프트 자동 생성
# ============================================================
st.subheader("1) AI로 모델식(y=f(t)) 제안 받기")

st.info(
    "1차시에서 세운 가설 모델과 그 근거를 바탕으로 AI에게 모델식을 제안받습니다.\n\n"
    "⚠ 반드시 **파이썬 계산용 식**도 함께 출력하도록 요청하세요."
)

# 1차시 정보 자동 불러오기 (common.py 연동)
model_hypothesis = step1.get("model_primary", "")
model_reason = step1.get("model_primary_reason", "")

st.markdown("### 🔹 1차시 가설 확인")
st.write(f"**가설 모델:** {model_hypothesis or '(기록 없음)'}")
st.write(f"**가설 근거:** {model_reason or '(기록 없음)'}")

additional_context = st.text_area("추가 설명(선택)", height=80)

if st.button("📌 프롬프트 자동 생성", use_container_width=True):
    # AI에게 LaTeX와 Python 식을 모두 요구하는 템플릿
    generated_prompt = build_unified_prompt(model_hypothesis, model_reason, additional_context)
    st.session_state["step2_ai_prompt"] = generated_prompt

ai_prompt = st.text_area("AI에 입력할 프롬프트", value=st.session_state.get("step2_ai_prompt", ""), height=200)

st.divider()

# ============================================================
# 2) AI 출력 결과 입력
# ============================================================
st.subheader("2) AI 출력 식 입력")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**LaTeX 수식 (보고서용)**")
    ai_model_latex = st.text_area("AI 모델식 f(t) (LaTeX)", value=step2_prev.get("ai_model_latex", ""), height=100, placeholder="$$ y = ... $$")
    ai_derivative_latex = st.text_area("AI 도함수 f'(t) (LaTeX)", value=step2_prev.get("ai_derivative_latex", ""), height=100)
    ai_second_derivative_latex = st.text_area("AI 이계도함수 f''(t) (LaTeX)", value=step2_prev.get("ai_second_derivative_latex", ""), height=100)

with col2:
    st.markdown("**파이썬 수식 (그래프 시뮬레이션용)**")
    py_model = st.text_input("모델식 f(t) 식('f=' 이후 식 붙여넣기)", value=step2_prev.get("py_model", ""), placeholder="3.2 * np.exp(0.04 * t)")
    py_d1 = st.text_input("도함수 f'(t) 식('d1=' 이후 식 붙여넣기)", value=step2_prev.get("py_d1", ""), placeholder="0.128 * np.exp(0.04 * t)")
    py_d2 = st.text_input("이계도함수 f''(t) 식('d2=' 이후 식 붙여넣기)", value=step2_prev.get("py_d2", ""), placeholder="0.00512 * np.exp(0.04 * t)")

st.subheader("가설 재평가")
hypothesis_decision = st.radio("가설 판단", ["가설 유지", "가설 수정"], horizontal=True, key="hypothesis_decision")

revised_model = ""
if hypothesis_decision == "가설 수정":
    revised_model = st.text_input("수정한 모델 유형", placeholder="예: 다항함수", key="revised_model")
    st.warning("모델을 수정했다면 위 항목 2)의 수식들을 수정된 모델 기준으로 다시 입력하세요.")

st.divider()

# ============================================================
# 3) 데이터 및 AI 모델 그래프 확인
# ============================================================
st.subheader("3) 데이터 기반 변화율 및 AI 모델 비교")

df = get_df()
if df is None:
    st.info("CSV를 업로드하면 그래프를 확인할 수 있습니다.")
else:
    # --- [데이터 전처리 로직 시작] ---
    cols = list(df.columns)
    x_prev, y_prev = get_xy()
    
    # 세션 또는 1차시 기록에서 초기값 설정
    x_init = step1.get("x_col") if step1.get("x_col") in cols else (x_prev if x_prev in cols else cols[0])
    y_init = step1.get("y_col") if step1.get("y_col") in cols else (y_prev if y_prev in cols else (cols[1] if len(cols) > 1 else cols[0]))

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        x_col = st.selectbox("X축 선택", cols, index=cols.index(x_init), key="step2_x_col")
    with col_sel2:
        y_col = st.selectbox("Y축 선택", cols, index=cols.index(y_init), key="step2_y_col")
    
    set_xy(x_col, y_col)

    x_mode = st.radio("X축 해석 방식", ["자동(권장)", "날짜(년월)", "숫자"], horizontal=True, key="step2_x_mode")

    # 데이터 필터링 및 변수(xv, yv) 정의
    y = pd.to_numeric(df[y_col], errors="coerce")
    if x_mode == "숫자":
        x = pd.to_numeric(df[x_col], errors="coerce")
        x_type = "numeric"
    else:
        x_dt = parse_year_month(df[x_col])
        if x_mode == "자동(권장)" and x_dt.notna().mean() < 0.6:
            x = pd.to_numeric(df[x_col], errors="coerce")
            x_type = "numeric"
        else:
            x = x_dt
            x_type = "datetime"

    valid = x.notna() & y.notna()
    xv = x[valid]
    yv = y[valid]
    # --- [데이터 전처리 로직 끝] ---

    if len(xv) < 30:
        st.warning(f"유효 데이터 점이 {len(xv)}개입니다. 변화율 계산을 위해 최소 30점 이상이 필요합니다.")
    else:
        # 데이터 정렬 및 t 수치화
        order = np.argsort(xv.values) if x_type == "datetime" else np.argsort(xv.to_numpy())
        xv = xv.iloc[order]
        yv = yv.iloc[order]
        
        if x_type == "datetime":
            base = xv.iloc[0]
            t = ((xv.dt.year - base.year) * 12 + (xv.dt.month - base.month)).to_numpy(dtype=float)
        else:
            t = xv.to_numpy(dtype=float)

        y_arr = yv.to_numpy(dtype=float)
        dy, d2y = compute_derivatives(t, y_arr)
        st.session_state["step2_valid_n"] = int(len(t))

        # --- eval()을 이용한 AI 수식 계산 ---
        eval_env = {"np": np, "t": t, "exp": np.exp, "sin": np.sin, "cos": np.cos, "log": np.log}
        ai_y, ai_dy, ai_d2y = None, None, None
        
        # UI에서 입력받은 py_model 등 변수 가져오기
        try:
            if py_model: ai_y = eval(py_model, eval_env)
            if py_d1: ai_dy = eval(py_d1, eval_env)
            if py_d2: ai_d2y = eval(py_d2, eval_env)
        except Exception as e:
            st.error(f"수식 계산 오류: {e}")

        # --- 그래프 출력 (Plotly) ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=xv, y=y_arr, mode="markers", name="실제 데이터", marker=dict(color='gray', opacity=0.5)))
        if ai_y is not None:
            fig1.add_trace(go.Scatter(x=xv, y=ai_y, mode="lines", name="AI 모델식", line=dict(color='red', width=2)))
        fig1.update_layout(height=320, title="원데이터 vs AI 모델 비교", margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=xv, y=dy, mode="markers", name="데이터 변화율", marker=dict(color='gray', opacity=0.5)))
        if ai_dy is not None:
            fig2.add_trace(go.Scatter(x=xv, y=ai_dy, mode="lines", name="AI 도함수", line=dict(color='blue', width=2)))
        fig2.update_layout(height=320, title="변화율 비교 분석", margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=xv, y=d2y, mode="markers", name="데이터 이계변화율", marker=dict(color='gray', opacity=0.5)))
        if ai_d2y is not None:
            fig3.add_trace(go.Scatter(x=xv, y=ai_d2y, mode="lines", name="AI 이계도함수", line=dict(color='green', width=2)))
        fig3.update_layout(height=320, title="곡률(오목·볼록) 비교 분석", margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig3, use_container_width=True)
        
st.divider()

# ============================================================
# 4) 학생 검증/비판(핵심 제출물)
# ============================================================
st.subheader("4) 미분 관점의 모델 해석")

st.info(
    "🔹 변화율 비교\n\n"
    "데이터의 변화율($\\Delta y/\\Delta t$) 그래프에서 특징 두 가지를 찾고, "
    "AI가 제시한 도함수 $f'(t)$가 이를 얼마나 잘 설명하는지 서술하시오.\n\n"
    "🔹 곡선의 모양 분석\n\n"
    "데이터의 이계변화율($\\Delta^2 y/\\Delta t^2$) 그래프에 나타난 오목·볼록 상태를 "
    "AI의 이계도함수 $f''(t)$와 비교하여 분석하시오.\n\n"
    "🔹 모델의 한계\n\n"
    "실제 데이터와 모델 식의 차이가 큰 구간을 한 곳 제시하고, "
    "모델링 과정에서 누락되었을 가능성이 있는 변수나 환경적 요인을 추론하여 서술해 봅시다."
)


student_critical_review = st.text_area(
    "분석 내용(필수)",
    value=step2_prev.get("student_critical_review", ""),
    height=220,
    placeholder=(
        "수식은 반드시 LaTeX 형식($$ ... $$)으로 입력하세요."
    ),
)

note = st.text_area("추가 메모(선택)", value=step2_prev.get("note", ""), height=100)

st.divider()

# ============================================================
# 5) 저장(구글시트) + TXT 백업 다운로드
# ============================================================
st.subheader("5) 저장 및 백업")

# step1에서 가져올 수 있는 기본 정보
data_source = (step1.get("data_source") or "").strip()
model_hypothesis_step1 = (step1.get("model_primary") or "").strip()

# X/Y 컬럼(있다면)
x_col_now = st.session_state.get("step2_x_col", step1.get("x_col",""))
y_col_now = st.session_state.get("step2_y_col", step1.get("y_col",""))

valid_n_now = st.session_state.get("step2_valid_n")

revised_model_safe = revised_model.strip() if hypothesis_decision == "가설 수정" else ""

payload = {
    "student_id": student_id,
    "data_source": data_source,
    "x_col": x_col_now,
    "y_col": y_col_now,
    "valid_n": valid_n_now,
    "model_hypothesis_step1": model_hypothesis_step1,
    "hypothesis_decision": hypothesis_decision,
    "revised_model": revised_model_safe,
    "ai_prompt": ai_prompt,
    "ai_model_latex": ai_model_latex,
    "ai_derivative_latex": ai_derivative_latex,
    "ai_second_derivative_latex": ai_second_derivative_latex,
    "student_analysis": student_critical_review,  # UI 변수명 그대로 쓰되, 키는 analysis로
    "note": note,
}

backup_bytes = build_step2_backup(payload)
st.download_button(
    label="📄 (다운로드) 2차시 백업 TXT",
    data=backup_bytes,
    file_name=f"미적분_수행평가_2차시_{student_id}.txt",
    mime="text/plain; charset=utf-8",
)

colS, colN = st.columns([1, 1])
save_clicked = colS.button("💾 저장(구글시트)", use_container_width=True)
go_next = colN.button("➡️ 3차시로 이동(추후)", use_container_width=True)


def _validate_step2() -> bool:
    # --- 가설 수정 검증 ---
    if hypothesis_decision == "가설 수정" and not revised_model_safe:
        st.warning("가설을 수정했다면, 수정한 모델 유형을 입력하세요.")
        return False

    # --- AI 입력 검증 ---
    if not ai_prompt.strip():
        st.warning("AI 프롬프트(원문)를 입력하세요.")
        return False

    if not ai_model_latex.strip():
        st.warning("AI 모델식(LaTeX)을 입력하세요.")
        return False

    if not student_critical_review.strip():
        st.warning("분석 내용을 입력하세요.")
        return False

    return True


if save_clicked or go_next:
    if not _validate_step2():
        st.stop()

    # 세션 저장(새로고침 대비용)
    _set_step2_state(payload)

    # 구글 시트 저장
    try:
        append_step2_row(
            student_id=payload["student_id"],
            data_source=payload["data_source"],
            x_col=payload["x_col"],
            y_col=payload["y_col"],
            valid_n=payload["valid_n"],
            model_hypothesis_step1=payload["model_hypothesis_step1"],
            hypothesis_decision=payload["hypothesis_decision"],
            revised_model=payload["revised_model"],
            ai_prompt=payload["ai_prompt"],
            ai_model_latex=payload["ai_model_latex"],
            ai_derivative_latex=payload["ai_derivative_latex"],
            ai_second_derivative_latex=payload["ai_second_derivative_latex"],
            student_analysis=payload["student_analysis"],
            note=payload["note"],
        )
        st.success("✅ 저장 완료! (Google Sheet에 기록되었습니다)")

    except Exception as e:
        st.error("⚠️ Google Sheet 저장 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

    if go_next:
        st.info("3차시는 아직 페이지를 만들기 전이라 이동은 나중에 연결하면 됩니다.")
        # st.switch_page("assessment/step3_integral.py")
