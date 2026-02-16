# assessment/step1_data.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 1차시: 데이터 탐색 & 모델링 가설
#
# 목표(1차시):
# 1) 공공데이터(KOSIS 등)에서 CSV를 내려받아 업로드
# 2) X/Y를 선택해 그래프를 시각화(년월 포함 가능)
# 3) 그래프 특징을 관찰하고, "함수 모델링 가설"을 세운다
#    - 주된 모델 1개 + 근거
#    - 대안 모델 1개 + 덜 적절한 근거
# 4) 저장 버튼을 누르면 Google Sheet(미적분_수행평가_1차시)에 1행 추가 저장
#
# 설계:
# - 인코딩 검사는 하지 않음(권장만 안내). KOSIS CSV에 관대하게 대응.
# - plotly가 없으면 matplotlib로 폴백.
# - 유효 데이터 점(숫자 쌍) 30개 이상 권장(미만이면 2차시 이동 제한).
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

# plotly 있으면 사용, 없으면 matplotlib 폴백
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

from assessment.google_sheets import append_step1_row


# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 30


# -----------------------------
# CSV 로더 (KOSIS 대응: 관대하게)
# -----------------------------
def read_csv_kosis(file) -> pd.DataFrame:
    """
    KOSIS/공공데이터 CSV에서 자주 생기는 문제(구분자/인코딩/깨진 행)를 최대한 흡수.
    - sep 자동 감지(sep=None + engine=python)
    - on_bad_lines='skip'로 깨진 행은 스킵
    - 인코딩은 utf-8 계열/국문 계열을 순서대로 시도
    """
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
            # 최소 2열 이상이어야 X/Y 선택이 의미 있음
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
            continue

    raise last_err if last_err else ValueError("CSV를 읽을 수 없습니다.")


# -----------------------------
# 년월 파서 (2015.01 / 2015-01 / 201501 등)
# -----------------------------
def parse_year_month(s: pd.Series) -> pd.Series:
    """
    X축이 '년월' 문자열일 때 datetime으로 변환.
    지원 예:
    - 2015.01 / 2015.01. / 2015-01 / 2015/01 / 201501
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.+$", "", regex=True)  # 끝 점 제거: 2015.01. -> 2015.01
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)

    # 1차 시도: YYYY-MM
    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m")

    # 2차 시도: YYYYMM (숫자 6자리)
    mask = dt.isna()
    if mask.any():
        digits = s[mask].str.replace(r"\D", "", regex=True)
        m6 = digits.str.fullmatch(r"\d{6}")
        if m6.any():
            dt2 = pd.to_datetime(digits[m6], errors="coerce", format="%Y%m")
            dt.loc[digits[m6].index] = dt2

    return dt


# -----------------------------
# 세션 초기화 + 학생 식별
# -----------------------------
init_assessment_session()
student_id = require_student_id("학번 또는 식별 코드를 입력하세요.")


# -----------------------------
# UI 시작
# -----------------------------
st.title("공공데이터 분석 수행 (1차시) — 데이터 탐색 & 모델링 가설")
st.caption("데이터를 업로드하고 그래프를 관찰한 뒤, 어떤 함수 모델이 적절할지 ‘가설’을 세웁니다.")
st.divider()

# ============================================================
# Step1) 공공데이터 선택
# ============================================================
st.subheader("Step1) 🔎 공공데이터 선택")
st.link_button("📊 국가통계포털 KOSIS에서 데이터 다운로드", "https://kosis.kr")

st.markdown(
    """
- **연도/월 등 시간에 따른 변화**를 분석할 수 있는 데이터를 선택하세요.
- 데이터는 **숫자 데이터**여야 합니다. (예: 인구 수, 비율, 농도, 금액 등)
- 다운로드 파일은 **CSV 권장(UTF-8 권장)**  
  *(단, 앱은 자동으로 읽기를 시도합니다.)*
- 너무 짧은 데이터는 비선형 모델 비교가 어렵습니다. **유효 데이터 점 30개 이상 권장**
"""
)

# ============================================================
# Step2) 업로드 전 전처리 규칙
# ============================================================
st.subheader("Step2) 🛠️ 업로드 전 전처리(권장)")
with st.expander("파일 규칙(권장) — 꼭 확인하세요", expanded=True):
    st.markdown(
        """
- 파일 형식: **CSV**
- 첫 행: **열 이름(헤더)**
- **불필요한 행/열(주석, 합계, 공백 행 등)** 삭제
- X축, Y축으로 사용할 **2개의 열**이 포함되어 있어야 함
- X축이 `2015.01`처럼 **년월**인 경우 그대로 두어도 됩니다.
"""
    )

st.divider()

# ============================================================
# Step3) 데이터 업로드
# ============================================================
st.subheader("Step3) 📁 데이터 업로드")
uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded is not None:
    try:
        df = read_csv_kosis(uploaded)
        set_df(df)
        st.success(f"업로드 완료: {uploaded.name}  ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
    except Exception as e:
        st.error("CSV 파일을 읽지 못했습니다. (구분자/형식 문제일 수 있습니다)")
        st.exception(e)
        st.stop()

df = get_df()
if df is None:
    st.info("CSV 파일을 업로드하면 다음 단계로 진행할 수 있습니다.")
    st.stop()

# 업로드 데이터 미리보기
st.markdown("#### 참고: 업로드한 데이터 확인")
st.dataframe(get_df_preview(df), use_container_width=True)

# ============================================================
# Step4) 데이터 시각화
# ============================================================
st.divider()
st.subheader("Step4) 📈 데이터 시각화 (X/Y 선택)")

cols = list(df.columns)
if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다. CSV를 다시 확인하세요.")
    st.stop()

x_prev, y_prev = get_xy()

x_col = st.selectbox(
    "X축(시간/연도/년월)",
    cols,
    index=cols.index(x_prev) if x_prev in cols else 0,
)

# y는 x와 다른 열을 기본 선택
y_default_idx = 1 if len(cols) > 1 else 0
if y_prev in cols and y_prev != x_col:
    y_default_idx = cols.index(y_prev)
elif y_default_idx < len(cols) and cols[y_default_idx] == x_col:
    y_default_idx = 0

y_col = st.selectbox(
    "Y축(수치 데이터)",
    cols,
    index=y_default_idx,
)

set_xy(x_col, y_col)

x_mode = st.radio(
    "X축 해석 방식",
    ["자동(권장)", "날짜(년월)", "숫자"],
    horizontal=True,
    help="‘자동(권장)’은 년월로 인식되면 날짜로, 아니면 숫자로 처리합니다.",
)

# 숫자 변환 (Y는 숫자 필수)
y = pd.to_numeric(df[y_col], errors="coerce")

# X 처리
if x_mode == "숫자":
    x = pd.to_numeric(df[x_col], errors="coerce")
    x_type = "numeric"
else:
    x_dt = parse_year_month(df[x_col])
    if x_mode == "자동(권장)":
        # 파싱 성공률이 낮으면 숫자로 fallback
        if x_dt.notna().mean() < 0.6:
            x = pd.to_numeric(df[x_col], errors="coerce")
            x_type = "numeric"
        else:
            x = x_dt
            x_type = "datetime"
    else:
        x = x_dt
        x_type = "datetime"

valid = x.notna() & y.notna()
xv = x[valid]
yv = y[valid]

# 정렬
if len(xv) >= 2:
    if x_type == "datetime":
        order = np.argsort(xv.values)
    else:
        order = np.argsort(xv.to_numpy())
    xv = xv.iloc[order]
    yv = yv.iloc[order]

# 그래프 출력
if len(xv) < 2:
    st.warning("유효한 숫자 데이터가 부족하여 그래프를 그릴 수 없습니다. (X/Y 열 값을 확인하세요)")
else:
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines+markers", name="Data"))
        fig.update_layout(
            height=520,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title=str(x_col),
            yaxis_title=str(y_col),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(xv, yv, marker="o")
        ax.set_xlabel(str(x_col))
        ax.set_ylabel(str(y_col))
        st.pyplot(fig, use_container_width=True)

# ============================================================
# A) 데이터 품질 점검(간단)
# ============================================================
st.divider()
st.subheader("A) ✅ 데이터 품질 점검(간단)")

valid_n = int(len(xv))
st.metric("유효 데이터 점(숫자 쌍) 개수", valid_n)

quality_ok = valid_n >= MIN_VALID_POINTS
if not quality_ok:
    st.error(
        f"유효 데이터 점이 {MIN_VALID_POINTS}개 미만입니다. "
        "데이터가 너무 짧으면 비선형 모델 비교가 제한적일 수 있습니다."
    )
st.caption("※ 2차시로 이동하려면 유효 데이터 점 30개 이상을 권장합니다.")

# ============================================================
# Step5) 그래프 해석 & 모델링 가설
# ============================================================
st.divider()
st.subheader("Step5) 🧠 그래프 특징 관찰 & 함수 모델링 가설(핵심)")

st.info(
    "1차시의 목표는 ‘정답’을 내는 것이 아니라, "
    "그래프에서 보이는 특징을 근거로 **어떤 함수 모델이 적절할지 가설을 세우는 것**입니다."
)

prev = get_step1_summary()

data_source = st.text_input(
    "데이터 출처(필수) — 예: KOSIS, 공공데이터포털, URL 등",
    value=str(prev.get("data_source", "")),
)

feature1 = st.text_area(
    "그래프에서 관찰한 특징 1 (필수)",
    value=str(prev.get("feature1", "")),
    height=90,
    placeholder="예: 시간이 지날수록 증가한다 / 감소한다 / 특정 구간에서 증가 속도가 빨라진다 / 주기성이 보인다 등",
)

feature2 = st.text_area(
    "그래프에서 관찰한 특징 2 (필수)",
    value=str(prev.get("feature2", "")),
    height=90,
    placeholder="예: 증가 속도가 줄어든다(포화) / 오목·볼록이 바뀌는 지점이 있다(변곡) / 12개월 주기 패턴 등",
)

model_primary = st.selectbox(
    "가설 모델(주된 모델) 선택 (필수)",
    ["선형(직선)", "지수함수", "로그함수", "삼각함수(주기)", "합성함수(조합)", "기타(직접 입력)"],
    index=0,
)

model_primary_custom = ""
if model_primary == "기타(직접 입력)":
    model_primary_custom = st.text_input(
        "주된 모델 이름/형식(직접 입력)",
        value=str(prev.get("model_primary_custom", "")),
        placeholder="예: 2차함수 / 포화형(로지스틱) / y = a + b*log(t) 등",
    )

model_primary_reason = st.text_area(
    "주된 모델이 적절하다고 생각한 근거 (필수)",
    value=str(prev.get("model_primary_reason", "")),
    height=110,
    placeholder="예: 12개월마다 반복되는 패턴이 있어 삼각함수가 적절하다. 증가 속도가 감소하므로 로그가 더 적절할 수 있다 등",
)

model_alt = st.selectbox(
    "대안 모델(다른 후보 1개) 선택 (필수)",
    ["선형(직선)", "지수함수", "로그함수", "삼각함수(주기)", "합성함수(조합)", "기타(직접 입력)"],
    index=1 if model_primary != "선형(직선)" else 2,
)

model_alt_custom = ""
if model_alt == "기타(직접 입력)":
    model_alt_custom = st.text_input(
        "대안 모델 이름/형식(직접 입력)",
        value=str(prev.get("model_alt_custom", "")),
        placeholder="예: 2차함수 / 포화형(로지스틱) / y = a + b*log(t) 등",
    )

model_alt_reason = st.text_area(
    "대안 모델이 덜 적절하다고 생각한 근거 (필수)",
    value=str(prev.get("model_alt_reason", "")),
    height=110,
    placeholder="예: 직선 모델은 주기성을 설명하지 못한다. 지수 모델은 후반부 완만해지는 추세와 맞지 않는다 등",
)

note = st.text_area(
    "추가 메모(선택)",
    value=str(prev.get("note", "")),
    height=80,
    placeholder="예: 데이터가 특정 구간에서 급변하는 이유(정책/외부 요인)를 추가로 조사해볼 수 있음",
)

# 저장/이동 버튼
col1, col2 = st.columns(2)
with col1:
    save_clicked = st.button("💾 저장(구글시트 기록)", use_container_width=True)
with col2:
    go_next = st.button("➡️ 2차시로 이동", use_container_width=True)


def _validate_inputs() -> bool:
    if not data_source.strip():
        st.warning("데이터 출처를 입력하세요.")
        return False
    if not feature1.strip():
        st.warning("특징 1을 입력하세요.")
        return False
    if not feature2.strip():
        st.warning("특징 2를 입력하세요.")
        return False
    if not model_primary_reason.strip():
        st.warning("주된 모델 근거를 입력하세요.")
        return False
    if not model_alt_reason.strip():
        st.warning("대안 모델 근거를 입력하세요.")
        return False
    return True


def _final_model_label(choice: str, custom: str) -> str:
    if choice == "기타(직접 입력)":
        return custom.strip() if custom.strip() else "기타(미입력)"
    return choice


if save_clicked or go_next:
    if not _validate_inputs():
        st.stop()

    mp = _final_model_label(model_primary, model_primary_custom)
    ma = _final_model_label(model_alt, model_alt_custom)

    # 세션 저장(다음 차시에서도 참조 가능)
    payload = {
        "student_id": student_id,
        "data_source": data_source.strip(),
        "x_col": x_col,
        "y_col": y_col,
        "x_mode": x_mode,
        "valid_n": valid_n,
        "feature1": feature1.strip(),
        "feature2": feature2.strip(),
        "model_primary": mp,
        "model_primary_reason": model_primary_reason.strip(),
        "model_alt": ma,
        "model_alt_reason": model_alt_reason.strip(),
        "note": note.strip(),
        "saved_at": pd.Timestamp.now().isoformat(),
    }
    set_step1_summary(payload)

    # Google Sheet에 append_row 저장
    try:
        append_step1_row(
            student_id=payload["student_id"],
            data_source=payload["data_source"],
            x_col=payload["x_col"],
            y_col=payload["y_col"],
            x_mode=payload["x_mode"],
            valid_n=payload["valid_n"],
            feature1=payload["feature1"],
            feature2=payload["feature2"],
            model_primary=payload["model_primary"],
            model_primary_reason=payload["model_primary_reason"],
            model_alt=payload["model_alt"],
            model_alt_reason=payload["model_alt_reason"],
            note=payload["note"],
        )
        st.success("✅ 저장 완료! (Google Sheet에 기록되었습니다)")
    except Exception as e:
        st.error("⚠️ Google Sheet 저장 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

    if go_next:
        if not quality_ok:
            st.error(f"유효 데이터 점이 {MIN_VALID_POINTS}개 이상이어야 2차시로 이동할 수 있습니다.")
            st.stop()
        st.switch_page("assessment/step2_model.py")
