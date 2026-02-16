# assessment/google_sheets.py
# ------------------------------------------------------------
# Google Sheets 저장 유틸 (1차시: 모델링 가설 중심)
#
# 목표:
# - Google Sheet: "미적분_수행평가_1차시" 에 학생 기록을 append_row로 누적 저장
# - (Step1) 데이터 출처 + 그래프 특징 + 모델링 가설(모델 선택/이유/대안 모델)
#
# 준비 사항(교사용 체크):
# 1) Streamlit Secrets에 st.secrets["gcp_service_account"] 가 존재해야 함
# 2) 서비스계정 이메일을 Google Sheet에 "편집자"로 공유해야 함
# 3) 시트 첫 행(헤더) 권장 예:
#    timestamp | student_id | data_source | x_col | y_col | x_mode | valid_n
#    feature1 | feature2 | model_primary | model_primary_reason | model_alt | model_alt_reason | note
# ------------------------------------------------------------

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# 내부 설정
# -----------------------------
SHEET_NAME_STEP1 = "미적분_수행평가_1차시"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# -----------------------------
# Google Sheets 연결
# -----------------------------
def _get_gspread_client() -> gspread.Client:
    """
    Streamlit secrets에 들어있는 서비스 계정 JSON으로 gspread client 생성
    """
    if "gcp_service_account" not in st.secrets:
        raise KeyError(
            "st.secrets['gcp_service_account'] 가 없습니다. "
            "Streamlit Cloud > Settings > Secrets에 서비스계정 JSON을 gcp_service_account로 넣어주세요."
        )

    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def get_worksheet(sheet_name: str = SHEET_NAME_STEP1, worksheet_index: int = 0):
    """
    Google Sheet 이름으로 열고, 첫 번째 worksheet(기본)를 반환
    """
    client = _get_gspread_client()
    sh = client.open(sheet_name)
    return sh.get_worksheet(worksheet_index)


# -----------------------------
# (선택) 헤더 자동 생성/검증
# -----------------------------
DEFAULT_STEP1_HEADER: List[str] = [
    "timestamp",
    "student_id",
    "data_source",
    "x_col",
    "y_col",
    "x_mode",
    "valid_n",
    "feature1",
    "feature2",
    "model_primary",
    "model_primary_reason",
    "model_alt",
    "model_alt_reason",
    "note",
]


def ensure_step1_header(ws) -> None:
    """
    시트가 비어있거나 헤더가 없을 때 기본 헤더를 1행에 넣는다.
    - 이미 헤더가 있으면 아무것도 하지 않음
    """
    values = ws.get_all_values()
    if not values:
        ws.append_row(DEFAULT_STEP1_HEADER, value_input_option="USER_ENTERED")
        return

    first_row = values[0]
    if len(first_row) == 0 or all((c.strip() == "" for c in first_row)):
        ws.update("A1", [DEFAULT_STEP1_HEADER])
        return


# -----------------------------
# Step1 저장(모델링 가설 중심)
# -----------------------------
def append_step1_row(
    *,
    student_id: str,
    data_source: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    x_mode: Optional[str] = None,
    valid_n: Optional[int] = None,
    feature1: str = "",
    feature2: str = "",
    model_primary: str = "",
    model_primary_reason: str = "",
    model_alt: str = "",
    model_alt_reason: str = "",
    note: str = "",
    sheet_name: str = SHEET_NAME_STEP1,
) -> None:
    """
    Step1 결과를 Google Sheet에 한 줄 추가.

    필수:
    - student_id
    - data_source

    권장:
    - feature1, feature2
    - model_primary + reason
    - model_alt + reason
    """

    if not str(student_id).strip():
        raise ValueError("student_id는 비어 있을 수 없습니다.")
    if not str(data_source).strip():
        raise ValueError("data_source는 비어 있을 수 없습니다.")

    ws = get_worksheet(sheet_name=sheet_name, worksheet_index=0)

    # 헤더가 없으면 만들어줌(원치 않으면 이 줄을 주석처리 가능)
    ensure_step1_header(ws)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(student_id).strip(),
        str(data_source).strip(),
        "" if x_col is None else str(x_col),
        "" if y_col is None else str(y_col),
        "" if x_mode is None else str(x_mode),
        "" if valid_n is None else int(valid_n),
        str(feature1).strip(),
        str(feature2).strip(),
        str(model_primary).strip(),
        str(model_primary_reason).strip(),
        str(model_alt).strip(),
        str(model_alt_reason).strip(),
        str(note).strip(),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")


# -----------------------------
# (선택) Step1 payload 저장(딕셔너리 기반)
# -----------------------------
def append_step1_payload(
    *,
    payload: Dict[str, Any],
    sheet_name: str = SHEET_NAME_STEP1,
) -> None:
    """
    step1_data.py에서 payload(dict)를 만들어두는 방식이면,
    그 payload를 그대로 받아 append_step1_row로 연결해주는 편의 함수.
    """
    append_step1_row(
        student_id=payload.get("student_id", ""),
        data_source=payload.get("data_source", ""),
        x_col=payload.get("x_col"),
        y_col=payload.get("y_col"),
        x_mode=payload.get("x_mode"),
        valid_n=payload.get("valid_n"),
        feature1=payload.get("feature1", ""),
        feature2=payload.get("feature2", ""),
        model_primary=payload.get("model_primary", ""),
        model_primary_reason=payload.get("model_primary_reason", ""),
        model_alt=payload.get("model_alt", ""),
        model_alt_reason=payload.get("model_alt_reason", ""),
        note=payload.get("note", ""),
        sheet_name=sheet_name,
    )
