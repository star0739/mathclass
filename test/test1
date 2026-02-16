import streamlit as st
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="Test1 | Google Sheets 연동 테스트",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 Test1 : Google Sheets 연동 테스트")
st.caption("제출 버튼을 누르면 구글 시트에 한 줄이 추가됩니다.")

st.divider()

# -----------------------------
# Google Sheets 연결
# -----------------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

try:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,
    )
    gc = gspread.authorize(creds)

    # ⚠️ 여기 시트 이름을 실제 사용하는 시트 이름으로 맞추세요
    sheet = gc.open("미적분_수행평가_제출").sheet1

except Exception as e:
    st.error("❌ Google Sheets에 연결할 수 없습니다.")
    st.exception(e)
    st.stop()

st.success("✅ Google Sheets 연결 완료")

st.divider()

# -----------------------------
# 입력 폼
# -----------------------------
with st.form("test_submit_form"):
    student_id = st.text_input(
        "학번 또는 식별 코드",
        placeholder="예: 30215",
    )

    test_message = st.text_area(
        "테스트 메시지",
        placeholder="예: Test1 페이지에서 정상 제출 확인",
        height=120,
    )

    submitted = st.form_submit_button("제출하기")

# -----------------------------
# 제출 처리
# -----------------------------
if submitted:
    if not student_id.strip():
        st.warning("⚠️ 학번(식별 코드)을 입력하세요.")
        st.stop()

    try:
        sheet.append_row(
            [
                datetime.now().isoformat(),
                student_id,
                "TEST_PAGE",
                test_message,
            ],
            value_input_option="USER_ENTERED",
        )

        st.success("🎉 제출 완료! 구글 시트에 정상적으로 저장되었습니다.")

    except Exception as e:
        st.error("❌ 제출 중 오류가 발생했습니다.")
        st.exception(e)

st.divider()

# -----------------------------
# 안내 문구
# -----------------------------
st.markdown(
    """
### 📌 확인 방법
- 교사용 Google Sheets에서 **새 행이 추가되는지**
- 여러 명이 동시에 제출해도 **누락 없이 쌓이는지**
- 새로고침/뒤로가기 후에도 **이미 제출한 기록은 유지되는지**

이 3가지만 확인되면  
👉 **실제 수행평가 페이지에서도 같은 방식으로 데이터 보존 가능**합니다.
"""
)
