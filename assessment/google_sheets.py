import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
from datetime import datetime

# -----------------------------
# Google Sheets 연결
# -----------------------------
def get_worksheet(sheet_name: str):
    """
    Streamlit secrets에 저장된 서비스 계정 정보로
    Google Sheet worksheet 객체 반환
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )

    client = gspread.authorize(creds)
    sheet = client.open(sheet_name)
    return sheet.sheet1  # 첫 번째 시트 사용


# -----------------------------
# Step1 결과 저장
# -----------------------------
def append_step1_row(
    student_id: str,
    data_source: str,
    feature1: str,
    feature2: str,
    question: str,
    valid_n: int,
):
    ws = get_worksheet("미적분_수행평가_1차시")

    ws.append_row(
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            student_id,
            data_source,
            feature1,
            feature2,
            question,
            valid_n,
        ],
        value_input_option="USER_ENTERED",
    )
