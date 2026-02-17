# assessment/google_sheets.py
# ------------------------------------------------------------
# Google Sheets 저장 유틸 (1차시: 모델링 가설 중심 / 단순화 버전)
#
# 저장 Sheet: "미적분_수행평가_1차시"
# 저장 항목:
# - timestamp, student_id
# - data_source
# - x_col, y_col, x_mode, valid_n
# - features(관찰 특징 통합)
# - model_primary(주된 모델), model_primary_reason(근거)
# - note(선택)
#
# (교사용) 시트 1행(헤더) 권장:
# timestamp | student_id | data_source | x_col | y_col | x_mode | valid_n | features | model_primary | model_primary_reason | note
# ------------------------------------------------------------

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

SHEET_NAME_STEP1 = "미적분_수행평가_1차시"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _get_gspread_client() -> gspread.Client:
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
    client = _get_gspread_client()
    sh = client.open(sheet_name)
    return sh.get_worksheet(worksheet_index)


DEFAULT_STEP1_HEADER: List[str] = [
    "timestamp",
    "student_id",
    "data_source",
    "x_col",
    "y_col",
    "x_mode",
    "valid_n",
    "features",
    "model_primary",
    "model_primary_reason",
    "note",
]


def ensure_step1_header(ws) -> None:
    values = ws.get_all_values()
    if not values:
        ws.append_row(DEFAULT_STEP1_HEADER, value_input_option="USER_ENTERED")
        return

    first_row = values[0]
    if len(first_row) == 0 or all((c.strip() == "" for c in first_row)):
        ws.update("A1", [DEFAULT_STEP1_HEADER])
        return


def append_step1_row(
    *,
    student_id: str,
    data_source: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    x_mode: Optional[str] = None,
    valid_n: Optional[int] = None,
    features: str = "",
    model_primary: str = "",
    model_primary_reason: str = "",
    note: str = "",
    sheet_name: str = SHEET_NAME_STEP1,
) -> None:
    if not str(student_id).strip():
        raise ValueError("student_id는 비어 있을 수 없습니다.")
    if not str(data_source).strip():
        raise ValueError("data_source는 비어 있을 수 없습니다.")

    ws = get_worksheet(sheet_name=sheet_name, worksheet_index=0)
    ensure_step1_header(ws)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(student_id).strip(),
        str(data_source).strip(),
        "" if x_col is None else str(x_col),
        "" if y_col is None else str(y_col),
        "" if x_mode is None else str(x_mode),
        "" if valid_n is None else int(valid_n),
        str(features).strip(),
        str(model_primary).strip(),
        str(model_primary_reason).strip(),
        str(note).strip(),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")


# --- (추가) 2차시 저장용 ---
SHEET_NAME_STEP2 = "미적분_수행평가_2차시"

DEFAULT_STEP2_HEADER = [
    "timestamp",
    "student_id",
    "data_source",
    "x_col",
    "y_col",
    "valid_n",
    "model_hypothesis_step1",
    "hypothesis_decision",
    "revised_model",
    "ai_model_latex",
    "ai_derivative_latex",
    "ai_second_derivative_latex",
    "py_model",
    "py_d1",
    "py_d2",
    "student_analysis",
    "note",
]


def ensure_step2_header(ws) -> None:
    values = ws.get_all_values()
    if not values:
        ws.append_row(DEFAULT_STEP2_HEADER, value_input_option="USER_ENTERED")
        return
    first_row = values[0]
    if len(first_row) == 0 or all((c.strip() == "" for c in first_row)):
        ws.update("A1", [DEFAULT_STEP2_HEADER])
        return


def append_step2_row(
    *,
    student_id: str,
    data_source: str = "",
    x_col: str = "",
    y_col: str = "",
    valid_n: int | None = None,
    model_hypothesis_step1: str = "",
    hypothesis_decision: str = "",
    revised_model: str = "",
    ai_model_latex: str = "",
    ai_derivative_latex: str = "",
    ai_second_derivative_latex: str = "",
    py_model: str = "",
    py_d1: str = "",
    py_d2: str = "",
    student_analysis: str = "",
    note: str = "",
    sheet_name: str = SHEET_NAME_STEP2,
) -> None:

    if not str(student_id).strip():
        raise ValueError("student_id는 비어 있을 수 없습니다.")

    ws = get_worksheet(sheet_name=sheet_name, worksheet_index=0)
    ensure_step2_header(ws)

    # '='로 시작하면 구글시트가 수식으로 오해할 수 있어 텍스트로 고정
    def _as_text(v: str) -> str:
        v = (v or "").strip()
        if v.startswith("="):
            return "'" + v
        return v

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(student_id).strip(),
        _as_text(data_source),
        _as_text(x_col),
        _as_text(y_col),
        "" if valid_n is None else int(valid_n),
        _as_text(model_hypothesis_step1),
        _as_text(hypothesis_decision),
        _as_text(revised_model),
        _as_text(ai_model_latex),
        _as_text(ai_derivative_latex),
        _as_text(ai_second_derivative_latex),
        _as_text(py_model),
        _as_text(py_d1),
        _as_text(py_d2),
        _as_text(student_analysis),
        _as_text(note),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")
