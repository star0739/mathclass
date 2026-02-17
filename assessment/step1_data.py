# assessment/step1_data.py
# ------------------------------------------------------------
# 공공데이터 분석 수행 - 1차시: 데이터 탐색 & 모델링 가설 (단순화 + TXT 백업)
# ------------------------------------------------------------

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
    set_step1_summary,
    get_step1_summary,
)

from assessment.google_sheets import append_step1_row

MIN_VALID_POINTS = 30


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


def build_backup_text(payload: dict) -> str:
    # TXT 백업 내용(학생이 스스로 복구할 수 있게 핵심만)
    lines = []
    lines.append("공공데이터 분석 수행 (1차시) 백업")
    lines.append("=" * 40)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")
    lines.append("[데이터 정보]")
    lines.append(f"- 데이터 출처: {payload.get('data_source','')}")
    lines.append(f"- X축: {payload.get('x_col','')}  |  Y축: {payload.get('y_col','')}")
    lines.append(f"- X축 해석 방식: {payload.get('x_mode','')}")
    lines.append(f"- 유효 데이터 점 개수: {payload.get('valid_n','')}")
    lines.append("")
    lines.append("[그래프 관찰 특징]")
    lines.append(payload.get("features","").strip() or "(미입력)")
    lines.append("")
    lines.append("[모델링 가설]")
    lines.append(f"- 주된 모델: {payload.get('model_primary','')}")
    lines.append("- 주된 모델 근거:")
    lines.append(payload.get("model_primary_reason","").strip() or "(미입력)")
    lines.append("")
    lines.append("※ 이 파일은 학생 개인 백업용입니다. 필요 시 다시 앱에 입력할 때 참고하세요.")
    return "\n".join(lines)


# -----------------------------
# 세션 초기화
# -----------------------------
init_assessment_session()
student_id = require_student_id("학번을 입력하세요.")

st.title("(1차시) 데이터 탐색 & 모델링 가설")
st.caption("그래프를 보고 특징을 정리한 뒤, 어떤 함수 모델이 적절할지 가설을 세웁니다.")
st.divider()

# Step1
st.subheader("1) 공공데이터 선택")
st.link_button("📊 KOSIS에서 데이터 다운로드", "https://kosis.kr")
st.markdown(
    """
- 공공데이터포털(data.go.kr), 서울 열린데이터 광장(data.seoul.go.kr) 등 다른 사이트도 가능
- **연도/월 등 시간에 따른 변화**를 분석할 수 있는 데이터를 선택하세요.
- 데이터는 **숫자 데이터**여야 합니다. (예: 인구 수, 비율, 농도, 금액 등)
- 다운로드 파일은 **CSV(UTF-8 권장)**  
- 너무 짧은 데이터는 비선형 모델 비교가 어렵습니다. **유효 데이터 점 30개 이상 권장**"""
)

# Step2
st.subheader("2) 업로드 전 전처리")
with st.expander("파일 규칙(권장)", expanded=True):
    st.markdown(
        """
- 파일 형식: **CSV(UTF-8 권장)**
- 첫 행: **열 이름(헤더)**
- **불필요한 행/열(주석, 합계, 공백 행 등)** 삭제
- X축, Y축으로 사용할 **2개의 열**이 포함되어 있어야 함
- X축이 `2015.01`처럼 **년월**인 경우 그대로 두어도 됩니다.
"""
    )

st.divider()

# Step3
st.subheader("3) CSV 업로드")
uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded is not None:
    try:
        df = read_csv_kosis(uploaded)
        set_df(df)
        st.success(f"업로드 완료: {uploaded.name}  ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
    except Exception as e:
        st.error("CSV 파일을 읽지 못했습니다. (구분자/형식 문제 가능)")
        st.exception(e)
        st.stop()

df = get_df()
if df is None:
    st.info("CSV를 업로드하면 다음 단계로 진행할 수 있습니다.")
    st.stop()

st.markdown("#### 참고: 데이터 미리보기")
st.dataframe(get_df_preview(df), use_container_width=True)


# Step4
st.divider()
st.subheader("4) 시각화 (X/Y 선택)")

cols = list(df.columns)
if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다. CSV를 다시 확인하세요.")
    st.stop()

x_prev, y_prev = get_xy()

x_col = st.selectbox("X축(시간/연도/년월)", cols, index=cols.index(x_prev) if x_prev in cols else 0)

# y 기본 선택
y_default_idx = 1 if len(cols) > 1 else 0
if y_prev in cols and y_prev != x_col:
    y_default_idx = cols.index(y_prev)
elif cols[y_default_idx] == x_col:
    y_default_idx = 0

y_col = st.selectbox("Y축(수치)", cols, index=y_default_idx)

set_xy(x_col, y_col)

x_mode = st.radio(
    "X축 해석 방식",
    ["자동(권장)", "날짜(년월)", "숫자"],
    horizontal=True,
)

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

if len(xv) >= 2:
    order = np.argsort(xv.values) if x_type == "datetime" else np.argsort(xv.to_numpy())
    xv = xv.iloc[order]
    yv = yv.iloc[order]

if len(xv) < 2:
    st.warning("유효한 데이터가 부족하여 그래프를 그릴 수 없습니다. (X/Y 열 값 확인)")
else:
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="lines+markers", name="Data"))
        fig.update_layout(height=520, margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(xv, yv, marker="o")
        ax.set_xlabel(str(x_col))
        ax.set_ylabel(str(y_col))
        st.pyplot(fig, use_container_width=True)

# 품질 점검
st.divider()
st.subheader("✅ 데이터 개수 점검")
valid_n = int(len(xv))
st.metric("유효 데이터 점(숫자 쌍) 개수", valid_n)
quality_ok = valid_n >= MIN_VALID_POINTS
if not quality_ok:
    st.error(f"유효 데이터 점이 {MIN_VALID_POINTS}개 미만입니다. (2차시 이동 제한)")
st.caption("※ 2차시 이동은 유효 데이터 점 30개 이상일 때만 허용합니다.")

# Step5
st.divider()
st.subheader("5) 그래프 특징 & 함수 모델링 가설")

prev = get_step1_summary()

data_source = st.text_input("데이터 출처(필수)", value=str(prev.get("data_source", "")))

features = st.text_area(
    "그래프에서 관찰한 특징(필수) — 한 칸에 자유롭게 작성",
    value=str(prev.get("features", "")),
    height=140,
    placeholder="예: 전체적으로 증가하지만 후반부에 증가 속도가 줄어든다(포화). 12개월 주기 패턴이 반복된다 등",
)

model_primary = st.selectbox(
    "가설 모델(주된 모델) 선택 (필수)",
    ["선형(직선)", "다항함수(직선 외)", "지수함수", "로그함수", "삼각함수", "기타(직접 입력)"],
    index=0,
)

model_primary_custom = ""
if model_primary == "기타(직접 입력)":
    model_primary_custom = st.text_input(
        "주된 모델 이름/형식(직접 입력)",
        value=str(prev.get("model_primary_custom", "")),
        placeholder="예: 포화형(로지스틱) / y = a + b*log(t) 등",
    )

model_primary_reason = st.text_area(
    "주된 모델이 적절하다고 생각한 근거(필수)",
    value=str(prev.get("model_primary_reason", "")),
    height=140,
    placeholder="예: 12개월마다 반복되는 패턴이 있어 삼각함수가 적절. 선형은 주기성을 설명 못함 등",
)


col1, col2, col3 = st.columns([1, 1, 1.2])
save_clicked = col1.button("💾 저장(구글시트)", use_container_width=True)
download_clicked = col2.button("⬇️ TXT 백업 만들기", use_container_width=True)
go_next = col3.button("➡️ 2차시로 이동", use_container_width=True)


def _final_model(choice: str, custom: str) -> str:
    if choice == "기타(직접 입력)":
        return custom.strip() if custom.strip() else "기타(미입력)"
    return choice


def _validate() -> bool:
    if not data_source.strip():
        st.warning("데이터 출처를 입력하세요.")
        return False
    if not features.strip():
        st.warning("관찰한 특징을 입력하세요.")
        return False
    if not model_primary_reason.strip():
        st.warning("주된 모델 근거를 입력하세요.")
        return False
    return True


# 저장 payload 구성(다운로드/시트 저장 공통)
payload = {
    "student_id": student_id,
    "data_source": data_source.strip(),
    "x_col": x_col,
    "y_col": y_col,
    "x_mode": x_mode,
    "valid_n": valid_n,
    "features": features.strip(),
    "model_primary": _final_model(model_primary, model_primary_custom),
    "model_primary_reason": model_primary_reason.strip(),
}

# TXT 백업 다운로드 버튼(즉시 다운로드 버튼을 표시하기 위해 항상 렌더)
backup_text = build_backup_text(payload)
backup_bytes = backup_text.encode("utf-8-sig")  # ✅ 한글 안전(윈도우 메모장 호환 ↑)
st.download_button(
    label="📄 (다운로드) 1차시 백업 TXT",
    data=backup_bytes,
    file_name=f"미적분_수행평가_1차시_{student_id}.txt",
    mime="text/plain; charset=utf-8",
)

if save_clicked or go_next:
    if not _validate():
        st.stop()

    # 세션 저장(다음 차시 연동)
    set_step1_summary({**payload, "saved_at": pd.Timestamp.now().isoformat()})

    # Google Sheet 저장
    try:
        append_step1_row(
            student_id=payload["student_id"],
            data_source=payload["data_source"],
            x_col=payload["x_col"],
            y_col=payload["y_col"],
            x_mode=payload["x_mode"],
            valid_n=payload["valid_n"],
            features=payload["features"],
            model_primary=payload["model_primary"],
            model_primary_reason=payload["model_primary_reason"],
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
