# assessment/step1_data.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 1차시: 데이터 탐색
#
# 반영 사항(요청 1~4)
# 1) Step 구조를 명시적으로 드러내기 (Step1~Step5 헤더/구성)
# 2) 외부 데이터 출처 링크 버튼(KOSIS) 제공
# 3) 데이터 형식 규칙을 업로드 이전에 강하게 안내 (CSV, 2열, 숫자, 헤더 등)
# 4) '추세(Trend)' 중심 해석을 유도하는 문구/입력 안내 강화
#
# NOTE
# - 멀티페이지 구조이므로 이 페이지에서는 st.set_page_config()를 호출하지 않습니다.
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from assessment.common import (
    init_assessment_session,
    require_student_id,
    read_uploaded_file,
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
# 세션 초기화 + 학생 식별
# -----------------------------
init_assessment_session()
student_id = require_student_id("1차시를 시작하기 전에 학번/식별 코드를 입력하세요.")

st.title("Step1) 🔎 데이터 탐색")
st.caption("공공데이터를 업로드하고, (X, Y) 그래프의 추세를 관찰하여 특징과 분석 질문을 작성합니다.")
st.divider()

# ============================================================
# Step1) 데이터 탐색: 공공데이터 받기(링크)
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
- (권장) **10분 안에** 데이터 다운로드를 완료하세요.
"""
)

# ============================================================
# Step2) 데이터 전처리: 업로드 전 규칙 안내(강조)
# ============================================================
st.subheader("Step2) 🛠️ 데이터 전처리(업로드 전 확인)")

with st.expander("✅ 업로드 파일 형식 규칙(필수)", expanded=True):
    st.markdown(
        """
**아래 조건을 만족하지 않으면 다음 단계(시각화/해석) 진행이 어렵습니다.**

- 파일 형식: **CSV 권장** (Excel도 가능하지만 최종은 CSV 권장)
- 데이터 구성: **2개의 열(컬럼)** 로 정리  
  - 1열: X축(예: 연도/시간/기간)
  - 2열: Y축(예: 측정값/비율/수치)
- 모든 값은 **숫자 데이터**여야 합니다.
- 첫 번째 행(1행)은 **열 이름(헤더)** 이어야 합니다.
- 불필요한 행/열(주석, 합계, 공백 행 등)은 **삭제** 후 업로드하세요.

예시)
| 연도 | 인터넷 이용률 |
|---:|---:|
| 2019 | 91.8 |
| 2020 | 91.9 |
"""
    )

# ============================================================
# Step3) 데이터 업로드
# ============================================================
st.subheader("Step3) 📁 데이터 업로드")

uploaded = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    try:
        df = read_uploaded_file(uploaded)
        meta = {
            "uploaded_filename": uploaded.name,
            "uploaded_at": pd.Timestamp.now().isoformat(),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
        set_df(df, meta=meta)
        st.success(f"업로드 완료: {uploaded.name}  ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
    except Exception as e:
        st.error("파일을 읽는 중 오류가 발생했습니다.")
        st.exception(e)

df = get_df()
if df is None:
    st.info("파일을 업로드하면 Step4(시각화)와 Step5(해석 작성)로 진행할 수 있습니다.")
    st.stop()

# ============================================================
# 업로드 데이터 확인(참고)
# ============================================================
st.markdown("#### 참고: 업로드한 데이터 확인하기")
c1, c2 = st.columns([3, 2])

with c1:
    st.dataframe(get_df_preview(df), use_container_width=True)

with c2:
    st.write("**요약 정보**")
    st.write(f"- 행 수: **{df.shape[0]:,}**")
    st.write(f"- 열 수: **{df.shape[1]:,}**")
    st.caption("※ 아래 Step4에서 X/Y를 선택할 때 숫자형(변환 가능) 열이 우선 추천됩니다.")

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

# 후보 목록: 가능하면 숫자형 후보 우선
x_candidates = num_cols if len(num_cols) >= 1 else all_cols
y_candidates = num_cols if len(num_cols) >= 1 else all_cols

# 기본값 설정
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

    # X축 해석 보조
    st.markdown("**(선택) X축 단위/해석**")
    x_unit = st.text_input("X축 단위(예: 년, 월, 일, 초 등)", key="step1_x_unit", placeholder="예: 년")
    x_note = st.text_input("X축 해석 메모(예: 2010~2024)", key="step1_x_note", placeholder="예: 2010~2024")

    set_xy(x_col, y_col)

with right:
    x = to_numeric_series(df, x_col)
    y = to_numeric_series(df, y_col)
    valid = x.notna() & y.notna()

    xv = x[valid].to_numpy()
    yv = y[valid].to_numpy()

    if len(xv) < 2:
        st.warning("유효한 숫자 데이터가 부족하여 그래프를 그릴 수 없습니다. (X/Y 열 값 확인)")
    else:
        order = xv.argsort()
        xv = xv[order]
        yv = yv[order]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines+markers", name="Data"))
        fig.update_layout(
            height=520,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title=f"{x_col}" + (f" ({x_unit})" if x_unit else ""),
            yaxis_title=f"{y_col}",
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Step5) 데이터 분석(1차시 범위: 관찰 + 질문 작성)
# ============================================================
st.subheader("Step5) 💬 그래프 해석 작성(필수)")

st.info("🔎 **그래프의 추세(증가/감소/증가속도 변화/주기성 등)** 를 중심으로 관찰 내용을 작성하세요. "
        "2차시에서 함수 모델로 더 정밀하게 해석합니다.")

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
    placeholder="예: 시간이 지날수록 y가 증가한다(감소한다). 특정 구간에서 급격한 변화가 있다.",
    key="step1_feature1",
)

feature2 = st.text_area(
    "추세 기반 특징 2 (필수)",
    value=prev.get("feature2", ""),
    height=90,
    placeholder="예: 증가 속도가 점점 커진다(오목 위처럼 보임). 증가 속도가 줄어든다(포화에 가까움). 주기성이 보인다.",
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
    placeholder="예: 이 데이터의 추세는 선형/지수/로그 중 무엇에 가까운가? 변화율은 시간이 지날수록 어떻게 달라지는가?",
    key="step1_question",
)

col_a, col_b, col_c = st.columns([2, 2, 3])

with col_a:
    save_clicked = st.button("💾 1차시 내용 저장", use_container_width=True)

with col_b:
    go_next = st.button("➡️ 2차시로 이동", use_container_width=True)

with col_c:
    st.caption("※ 저장 후 2차시로 이동하는 것을 권장합니다. (세션에 저장됨)")

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
    if _validate_step1_inputs():
        x_col = st.session_state.get("step1_x_col")
        y_col = st.session_state.get("step1_y_col")

        payload = {
            "data_source": str(data_source).strip(),
            "x_col": x_col,
            "y_col": y_col,
            "x_unit": str(st.session_state.get("step1_x_unit", "")).strip(),
            "x_note": str(st.session_state.get("step1_x_note", "")).strip(),
            "feature1": str(feature1).strip(),
            "feature2": str(feature2).strip(),
            "feature3": str(feature3).strip(),
            "question": str(question).strip(),
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        set_step1_summary(payload)
        st.success("1차시 내용이 저장되었습니다.")

        if go_next:
            st.switch_page("assessment/step2_model.py")
    else:
        st.stop()

st.divider()

st.markdown(
    """
### 다음 단계(2차시) 예고
- 선택한 데이터의 추세를 설명할 **함수 모델(지수/로그/삼각/선형/합성 템플릿)**을 정하고,
- 도함수/이계도함수로 변화(증가·감소, 오목·볼록)를 해석합니다.
"""
)

