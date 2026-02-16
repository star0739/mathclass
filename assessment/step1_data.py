# assessment/step1_data.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 1차시: 데이터 탐색
# (안정성 최우선 버전)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import csv
from assessment.google_sheets import append_step1_row

# 그래프 라이브러리 (plotly 있으면 사용, 없으면 matplotlib)
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
    set_step1_summary,
    get_step1_summary,
)

# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 30   # 최소 데이터 개수

# -----------------------------
# CSV 로더 (KOSIS 대응, 최대한 관대)
# -----------------------------
def read_csv_kosis(file) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding=enc,
                sep=None,            # 구분자 자동
                engine="python",
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise ValueError("CSV를 읽을 수 없습니다. (구분자/형식 문제 가능)")

# -----------------------------
# 년·월 파서
# -----------------------------
def parse_year_month(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.+$", "", regex=True)
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)

    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m")

    mask = dt.isna()
    if mask.any():
        digits = s[mask].str.replace(r"\D", "", regex=True)
        m6 = digits.str.len() == 6
        dt.loc[mask[m6].index] = pd.to_datetime(
            digits[m6], format="%Y%m", errors="coerce"
        )

    return dt

# -----------------------------
# 세션 초기화
# -----------------------------
init_assessment_session()
student_id = require_student_id("학번 또는 식별 코드를 입력하세요.")

st.title("Step1) 데이터 탐색")
st.caption("CSV 데이터를 업로드하고 그래프의 추세를 관찰합니다.")
st.divider()

# ============================================================
# Step1. 데이터 선택
# ============================================================
st.subheader("Step1) 공공데이터 선택")
st.link_button("📊 KOSIS 바로가기", "https://kosis.kr")

st.markdown(
    """
- **CSV 파일**만 업로드하세요  
- 인코딩은 **UTF-8 권장** (앱이 자동 처리 시도)
- **데이터 점 30개 이상** 권장
- X축이 `2015.01` 같은 **년·월 형식이어도 괜찮습니다**
"""
)

# ============================================================
# Step2. 데이터 업로드
# ============================================================
st.subheader("Step2) CSV 업로드")

uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded:
    try:
        df = read_csv_kosis(uploaded)
        set_df(df)
        st.success(f"업로드 성공 ({df.shape[0]}행 × {df.shape[1]}열)")
    except Exception as e:
        st.error("CSV 파일을 읽지 못했습니다.")
        st.exception(e)
        st.stop()

df = get_df()
if df is None:
    st.stop()

st.dataframe(get_df_preview(df), use_container_width=True)
st.divider()

# ============================================================
# Step3. X / Y 선택
# ============================================================
st.subheader("Step3) X / Y 선택")

cols = list(df.columns)
x_prev, y_prev = get_xy()

x_col = st.selectbox("X축 (시간/연도/년월)", cols, index=cols.index(x_prev) if x_prev in cols else 0)
y_col = st.selectbox("Y축 (수치)", cols, index=cols.index(y_prev) if y_prev in cols else 1)

set_xy(x_col, y_col)

x_mode = st.radio(
    "X축 해석 방식",
    ["자동(권장)", "날짜(년월)", "숫자"],
    horizontal=True,
)

# Y는 항상 숫자
y = pd.to_numeric(df[y_col], errors="coerce")

# X 처리
if x_mode == "숫자":
    x = pd.to_numeric(df[x_col], errors="coerce")
else:
    x_dt = parse_year_month(df[x_col])
    if x_mode == "자동(권장)" and x_dt.notna().mean() < 0.6:
        x = pd.to_numeric(df[x_col], errors="coerce")
    else:
        x = x_dt

valid = x.notna() & y.notna()
xv = x[valid]
yv = y[valid]

# 정렬
if np.issubdtype(xv.dtype, np.datetime64):
    order = np.argsort(xv.values)
else:
    order = np.argsort(xv.to_numpy())

xv = xv.iloc[order]
yv = yv.iloc[order]

# ============================================================
# Step4. 그래프
# ============================================================
st.subheader("Step4) 데이터 시각화")

if len(xv) < 2:
    st.warning("그래프를 그릴 수 있는 데이터가 부족합니다.")
else:
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines+markers"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(xv, yv, marker="o")
        st.pyplot(fig)

st.divider()

# ============================================================
# A. 데이터 품질 점검
# ============================================================
st.subheader("A. 데이터 품질 점검")

valid_n = len(xv)
st.metric("유효 데이터 점 개수", valid_n)

quality_ok = valid_n >= MIN_VALID_POINTS

if not quality_ok:
    st.error(f"데이터가 {MIN_VALID_POINTS}개 미만입니다. 더 긴 데이터를 사용하세요.")

# ============================================================
# Step5. 해석 작성
# ============================================================
st.subheader("Step5) 그래프 해석")

prev = get_step1_summary()

data_source = st.text_input(
    "데이터 출처(링크/기관명 등) (필수)",
    value=prev.get("data_source", ""),
    placeholder="예: KOSIS / 공공데이터포털 / URL 등",
    key="step1_data_source",
)

feature1 = st.text_area(
    "추세 기반 특징 (필수)",
    value=prev.get("feature1", ""),
    height=90,
    placeholder="예: 시간이 지날수록 y가 증가(감소)한다. 특정 구간에서 변화가 급격해진다. 주기성이 나타난다.",
    key="step1_feature1",
)

question = st.text_area(
    "분석 질문(문장) (필수)",
    value=prev.get("question", ""),
    height=90,
    placeholder="예: 이 추세는 선형/지수/로그 중 무엇에 가까운가? 변화율은 시간이 지날수록 어떻게 달라지는가?",
    key="step1_question",
)

save = st.button("💾 저장")
next_step = st.button("➡️ 2차시로 이동")

if save or next_step:
    if not all([data_source.strip(), feature1.strip(), question.strip()]):
        st.warning("모든 항목을 입력하세요.")
        st.stop()

    # 세션 저장(다음 차시용)
    set_step1_summary(
        {
            "data_source": data_source,
            "feature1": feature1,
            "question": question,
            "valid_n": valid_n,
        }
    )

    # 🔥 Google Sheet에 한 줄 추가
    try:
        append_step1_row(
            student_id=student_id,
            data_source=data_source,
            feature1=feature1,
            question=question,
            valid_n=valid_n,
        )
        st.success("✅ 저장 완료! (Google Sheet에 기록되었습니다)")
    except Exception as e:
        st.error("⚠️ Google Sheet 저장 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

    if next_step:
        if valid_n < MIN_VALID_POINTS:
            st.error("데이터 개수 조건을 만족해야 2차시로 이동할 수 있습니다.")
            st.stop()
        st.switch_page("assessment/step2_model.py")

