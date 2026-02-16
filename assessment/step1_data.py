# assessment/step1_data.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 1차시: 데이터 탐색
#
# 요구 반영:
# - CSV(인코딩 UTF-8) 필수
# - 유효 데이터 점(숫자 쌍) 최소 30개 이상
# - "좋은 데이터" 자동 점검 + 경고(유효점수, 선형 R², 곡률 지표)
# - Step 구조(1~5) 명시
# - KOSIS 링크 버튼
# - 업로드 전 형식 규칙 강조
# - 추세 중심 해석 유도 문구 강화
# - plotly 없을 때 matplotlib로 폴백(의존성 안정)
#
# NOTE:
# - 멀티페이지 구조이므로 st.set_page_config() 호출하지 않음
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

from assessment.common import (
    init_assessment_session,
    require_student_id,
    set_df,
    get_df,
    get_df_preview,
    numeric_columns,
    to_numeric_series,
    set_xy,
    get_xy,
    set_step1_summary,
    get_step1_summary,
)

# -----------------------------
# 설정(운영 기준)
# -----------------------------
MIN_VALID_POINTS = 30              # 유효 데이터 점 최소 기준
LINEAR_R2_HIGH = 0.985             # 선형에 과도하게 잘 맞으면(거의 직선) 경고
CURVATURE_LOW = 0.05               # 곡률 지표가 너무 낮으면(거의 직선/변화 미약) 경고(상대적)

# -----------------------------
# 세션 초기화 + 학생 식별
# -----------------------------
init_assessment_session()
student_id = require_student_id("1차시를 시작하기 전에 학번/식별 코드를 입력하세요.")

st.title("Step1) 🔎 데이터 탐색")
st.caption("CSV(UTF-8) 데이터를 업로드하고, (X, Y) 그래프의 추세를 관찰하여 특징과 분석 질문을 작성합니다.")
st.divider()


# ============================================================
# Step1) 공공데이터 선택하기(링크)
# ============================================================
st.subheader("Step1) 🔎 공공데이터 선택하기")

st.link_button(
    "📊 여기를 클릭하여 국가통계포털 KOSIS에서 데이터 다운로드",
    "https://kosis.kr",
)

st.markdown(
    """
- **연도(또는 시간)에 따른 변화 추이**를 분석할 수 있는 데이터를 선택하세요.
- 데이터는 **반드시 숫자 데이터**(예: 인구 수, 참여율, 농도, 비율, 금액 등)여야 합니다.
- 너무 짧은 데이터는(점이 적은 데이터) 비선형 모델 비교가 어렵습니다.
"""
)

# ============================================================
# Step2) 업로드 전 규칙 안내(강조)
# ============================================================
st.subheader("Step2) 🛠️ 데이터 전처리(업로드 전 확인)")

with st.expander("✅ 업로드 파일 규칙(필수): CSV / UTF-8 / 2열 / 숫자", expanded=True):
    st.markdown(
        f"""
**반드시 아래 조건을 만족해야 합니다.**

- 파일 형식: **CSV만 허용**
- 인코딩: **UTF-8로 다운로드**
- 데이터 구성: **2개의 열(컬럼)** 로 정리  
  - 1열: X축(예: 연도/시간/기간)
  - 2열: Y축(예: 측정값/비율/수치)
- 모든 값은 **숫자 데이터**여야 합니다.
- 첫 번째 행(1행)은 **열 이름(헤더)** 이어야 합니다.
- 불필요한 행/열(주석, 합계, 공백 행 등)은 **삭제** 후 업로드하세요.

**권장 조건**
- 유효 데이터 점(숫자 쌍) **최소 {MIN_VALID_POINTS}개 이상**
"""
    )

st.divider()


# ============================================================
# Step3) 데이터 업로드 (CSV만)
# ============================================================
st.subheader("Step3) 📁 데이터 업로드 (CSV / UTF-8 필수)")

uploaded = st.file_uploader(
    "CSV 파일 업로드 (인코딩: UTF-8)",
    type=["csv"],  # ✅ CSV만 허용
)

def read_csv_utf8_only(file) -> pd.DataFrame:
    """
    UTF-8 계열로만 읽도록 강제(실패 시 안내).
    - utf-8-sig: BOM 포함 가능성 대응
    - utf-8: 일반 UTF-8
    """
    try:
        return pd.read_csv(file, encoding="utf-8-sig")
    except Exception:
        file.seek(0)
        return pd.read_csv(file, encoding="utf-8")

if uploaded is not None:
    try:
        df = read_csv_utf8_only(uploaded)

        # 세션 저장(메모리 안전 검증은 common.py에서 수행)
        meta = {
            "uploaded_filename": uploaded.name,
            "uploaded_at": pd.Timestamp.now().isoformat(),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "encoding_note": "utf-8 / utf-8-sig",
        }
        set_df(df, meta=meta)
        st.success(f"업로드 완료: {uploaded.name}  ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
    except UnicodeDecodeError:
        st.error("CSV 인코딩 오류입니다. **UTF-8로 다시 다운로드**한 뒤 업로드하세요.")
        st.stop()
    except Exception as e:
        st.error("파일을 읽는 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

df = get_df()
if df is None:
    st.info("CSV 파일을 업로드하면 Step4(시각화)와 Step5(해석 작성)로 진행할 수 있습니다.")
    st.stop()

# ============================================================
# 참고: 업로드 데이터 확인
# ============================================================
st.markdown("#### 참고: 업로드한 데이터 확인하기")
c1, c2 = st.columns([3, 2])

with c1:
    st.dataframe(get_df_preview(df), use_container_width=True)

with c2:
    st.write("**요약 정보**")
    st.write(f"- 행 수: **{df.shape[0]:,}**")
    st.write(f"- 열 수: **{df.shape[1]:,}**")
    st.caption("※ Step4에서 X/Y 선택 시 숫자형(변환 가능) 열이 우선 추천됩니다.")

st.divider()


# ============================================================
# Step4) 데이터 시각화 (X/Y 선택 + 그래프)
# ============================================================
st.subheader("Step4) 📈 데이터 시각화")

all_cols = list(df.columns)
if len(all_cols) < 2:
    st.warning("열이 2개 이상 있어야 X/Y를 선택할 수 있습니다.")
    st.stop()

num_cols = numeric_columns(df)
x_prev, y_prev = get_xy()

x_candidates = num_cols if len(num_cols) >= 1 else all_cols
y_candidates = num_cols if len(num_cols) >= 1 else all_cols

x_default = x_prev if x_prev in x_candidates else x_candidates[0]
y_default = y_prev if y_prev in y_candidates else None
if y_default == x_default:
    y_default = None

left, right = st.columns([2, 3])

with left:
    st.markdown("**X/Y 축 선택**")

    x_col = st.selectbox(
        "📊 X축 데이터(연도/시간 등)",
        options=x_candidates,
        index=x_candidates.index(x_default),
        key="step1_x_col",
    )

    y_options = [c for c in y_candidates if c != x_col] or y_candidates
    y_col = st.selectbox(
        "📊 Y축 데이터(수치/비율 등)",
        options=y_options,
        index=(y_options.index(y_default) if (y_default in y_options) else 0),
        key="step1_y_col",
    )

    st.markdown("**(선택) X축 단위/해석**")
    x_unit = st.text_input("X축 단위(예: 년, 월, 일 등)", key="step1_x_unit", placeholder="예: 년")
    x_note = st.text_input("X축 해석 메모(예: 2010~2024)", key="step1_x_note", placeholder="예: 2010~2024")

    set_xy(x_col, y_col)

# --- 숫자 변환 및 유효값 정리 ---
x = to_numeric_series(df, x_col)
y = to_numeric_series(df, y_col)
valid = x.notna() & y.notna()
xv = x[valid].to_numpy()
yv = y[valid].to_numpy()

if len(xv) >= 2:
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]

with right:
    if len(xv) < 2:
        st.warning("유효한 숫자 데이터가 부족하여 그래프를 그릴 수 없습니다. (X/Y 열 값 확인)")
    else:
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines+markers", name="Data"))
            fig.update_layout(
                height=520,
                margin=dict(l=40, r=20, t=30, b=40),
                xaxis_title=f"{x_col}" + (f" ({x_unit})" if x_unit else ""),
                yaxis_title=f"{y_col}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            ax.plot(xv, yv, marker="o")
            ax.set_xlabel(f"{x_col}" + (f" ({x_unit})" if x_unit else ""))
            ax.set_ylabel(f"{y_col}")
            st.pyplot(fig, use_container_width=True)

st.divider()


# ============================================================
# A. 좋은 데이터 자동 점검 + 경고(요청 반영)
# ============================================================
st.subheader("A. ✅ 데이터 품질 자동 점검")

def linear_r2(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """
    1차 회귀 y = ax + b 의 R^2 계산 (간단/가벼운 구현)
    """
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    if len(x_arr) < 2:
        return float("nan")

    a, b = np.polyfit(x_arr, y_arr, deg=1)
    y_hat = a * x_arr + b

    ss_res = np.sum((y_arr - y_hat) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)

def curvature_score(y_arr: np.ndarray) -> float:
    """
    간단 곡률 지표: 2차 차분의 평균 절댓값을 y 범위로 정규화.
    - 값이 0에 가까우면 거의 직선/변화 미약 가능성
    - 값이 클수록 오목/볼록(가속/감속)이 존재할 가능성
    """
    y_arr = np.asarray(y_arr, dtype=float)
    if len(y_arr) < 3:
        return float("nan")
    d2 = np.diff(y_arr, n=2)  # Δ²y
    denom = np.ptp(y_arr)  # max-min
    if denom == 0:
        return 0.0
    return float(np.mean(np.abs(d2)) / denom)

valid_n = int(len(xv))
r2 = linear_r2(xv, yv) if valid_n >= 2 else float("nan")
curv = curvature_score(yv) if valid_n >= 3 else float("nan")

cA, cB, cC = st.columns(3)
with cA:
    st.metric("유효 데이터 점(숫자 쌍) N", f"{valid_n}")
with cB:
    st.metric("선형 적합 R²(참고)", "-" if np.isnan(r2) else f"{r2:.4f}")
with cC:
    st.metric("곡률 지표(참고)", "-" if np.isnan(curv) else f"{curv:.4f}")

# 경고/가이드
quality_ok = True

if valid_n < MIN_VALID_POINTS:
    quality_ok = False
    st.error(
        f"유효 데이터 점이 **{MIN_VALID_POINTS}개 미만**입니다. "
        "데이터가 너무 짧으면 지수/로그/삼각 등 비선형 모델 비교가 어렵습니다. "
        "더 긴 기간/더 많은 관측값이 있는 데이터를 선택하세요."
    )

if not np.isnan(r2) and r2 >= LINEAR_R2_HIGH:
    st.warning(
        f"현재 데이터는 선형 모델에 매우 잘 맞습니다(R²≈{r2:.3f}). "
        "이 경우 비선형 모델이 큰 의미가 없을 수 있습니다. "
        "포화/가속/주기성이 보이는 다른 데이터를 선택하거나, 2차시에서 모델 비교 근거를 더 명확히 제시하세요."
    )

if not np.isnan(curv) and curv < CURVATURE_LOW:
    st.warning(
        f"곡률 지표가 낮습니다({curv:.3f}). "
        "그래프가 거의 직선이거나 변화 속도 변화가 약할 수 있습니다. "
        "비선형 해석을 원하면 기간을 늘리거나 다른 지표를 고려하세요."
    )

st.caption("※ 자동 점검은 ‘참고용’입니다. 최종 모델 선택은 2차시에서 근거와 함께 결정합니다.")

st.divider()


# ============================================================
# Step5) 그래프 해석 작성(필수) - 추세 중심 유도
# ============================================================
st.subheader("Step5) 💬 그래프 해석 작성(필수)")

st.info(
    "🔎 **그래프의 추세(증가/감소/증가속도 변화/포화/주기성)** 를 중심으로 관찰 내용을 작성하세요. "
    "2차시에서 함수 모델로 더 정밀하게 해석합니다."
)

prev = get_step1_summary()

data_source = st.text_input(
    "데이터 출처(링크/기관명 등) (필수)",
    value=prev.get("data_source", ""),
    placeholder="예: KOSIS / 공공데이터포털 / URL 등",
    key="step1_data_source",
)

feature1 = st.text_area(
    "추세 기반 특징 1 (필수)",
    value=prev.get("feature1", ""),
    height=90,
    placeholder="예: 시간이 지날수록 y가 증가(감소)한다. 특정 구간에서 변화가 급격해진다.",
    key="step1_feature1",
)

feature2 = st.text_area(
    "추세 기반 특징 2 (필수)",
    value=prev.get("feature2", ""),
    height=90,
    placeholder="예: 증가 속도가 커진다(오목 위). 증가 속도가 줄어든다(포화/로그). 주기성이 보인다(삼각).",
    key="step1_feature2",
)

feature3 = st.text_area(
    "추가 특징 (선택)",
    value=prev.get("feature3", ""),
    height=80,
    placeholder="추가로 관찰한 특징이 있으면 작성",
    key="step1_feature3",
)

question = st.text_area(
    "분석 질문(문장) (필수)",
    value=prev.get("question", ""),
    height=90,
    placeholder="예: 이 추세는 선형/지수/로그 중 무엇에 가까운가? 변화율은 시간이 지날수록 어떻게 달라지는가?",
    key="step1_question",
)

col_a, col_b, col_c = st.columns([2, 2, 3])
with col_a:
    save_clicked = st.button("💾 1차시 내용 저장", use_container_width=True)
with col_b:
    go_next = st.button("➡️ 2차시로 이동", use_container_width=True)
with col_c:
    st.caption("※ ‘유효 데이터 점 30개 이상’ 조건을 만족해야 2차시로 이동할 수 있습니다.")

def _validate_step1_inputs() -> bool:
    if not str(data_source).strip():
        st.warning("데이터 출처(링크/기관명)를 입력하세요.")
        return False
    if not str(feature1).strip():
        st.warning("특징 1을 입력하세요.")
        return False
    if not str(feature2).strip():
        st.warning("특징 2를 입력하세요.")
        return False
    if not str(question).strip():
        st.warning("분석 질문(문장)을 입력하세요.")
        return False
    return True

if save_clicked or go_next:
    if not _validate_step1_inputs():
        st.stop()

    # 저장 payload
    payload = {
        "data_source": str(data_source).strip(),
        "x_col": st.session_state.get("step1_x_col"),
        "y_col": st.session_state.get("step1_y_col"),
        "x_unit": str(st.session_state.get("step1_x_unit", "")).strip(),
        "x_note": str(st.session_state.get("step1_x_note", "")).strip(),
        "feature1": str(feature1).strip(),
        "feature2": str(feature2).strip(),
        "feature3": str(feature3).strip(),
        "question": str(question).strip(),
        "saved_at": pd.Timestamp.now().isoformat(),
        "quality_check": {
            "valid_n": valid_n,
            "linear_r2": None if np.isnan(r2) else float(r2),
            "curvature": None if np.isnan(curv) else float(curv),
            "min_valid_required": MIN_VALID_POINTS,
        },
    }
    set_step1_summary(payload)
    st.success("1차시 내용이 저장되었습니다.")

    if go_next:
        if not quality_ok:
            st.error(
                f"2차시로 이동할 수 없습니다. "
                f"유효 데이터 점이 **{MIN_VALID_POINTS}개 이상**이어야 합니다. "
                "데이터를 다시 선택/다운로드하여 업로드하세요."
            )
            st.stop()
        st.switch_page("assessment/step2_model.py")

st.divider()
st.markdown(
    """
### 다음 단계(2차시) 예고
- 선택한 데이터의 추세를 설명할 **함수 모델(지수/로그/삼각/선형/합성 템플릿)**을 정하고,
- 도함수/이계도함수로 변화(증가·감소, 오목·볼록)를 해석합니다.
"""
)
