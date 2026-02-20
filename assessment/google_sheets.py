
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
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")

# --- (추가) 3차시 저장용 ---
SHEET_NAME_STEP3 = "미적분_수행평가_3차시"

DEFAULT_STEP3_HEADER: List[str] = [
    "timestamp",
    "student_id",
    "data_source",
    "x_col",
    "y_col",
    "valid_n",
    "i0",
    "i1",
    "py_model",
    "A_rect",
    "A_trap",
    "I_model",
    "err_rect",
    "err_trap",
    "rel_trap",
    "student_critical_review2",
]


def ensure_step3_header(ws) -> None:
    values = ws.get_all_values()
    if not values:
        ws.append_row(DEFAULT_STEP3_HEADER, value_input_option="USER_ENTERED")
        return

    first_row = values[0]
    if len(first_row) == 0 or all((c.strip() == "" for c in first_row)):
        ws.update("A1", [DEFAULT_STEP3_HEADER])
        return


def append_step3_row(
    *,
    student_id: str,
    data_source: str = "",
    x_col: str = "",
    y_col: str = "",
    valid_n: int | None = None,
    i0: int | None = None,
    i1: int | None = None,
    py_model: str = "",
    A_rect: float | None = None,
    A_trap: float | None = None,
    I_model: float | None = None,
    err_rect: float | None = None,
    err_trap: float | None = None,
    rel_trap: float | None = None,
    student_critical_review2: str = "",
    sheet_name: str = SHEET_NAME_STEP3,
) -> None:
    """
    3차시 전용 저장:
    - 데이터 기반 수치적분(직사각형/사다리꼴)과 모델 정적분 비교 결과를 저장
    """
    if not str(student_id).strip():
        raise ValueError("student_id는 비어 있을 수 없습니다.")

    ws = get_worksheet(sheet_name=sheet_name, worksheet_index=0)
    ensure_step3_header(ws)

    def _as_text(v: str) -> str:
        v = (v or "").strip()
        # 구글시트 수식 오해 방지
        if v.startswith("="):
            return "'" + v
        return v

    def _as_num(v) -> str | float | int:
        if v is None or v == "":
            return ""
        # gspread USER_ENTERED로 넘겨도 숫자형으로 들어가게 float로 정리
        return float(v)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(student_id).strip(),
        _as_text(data_source),
        _as_text(x_col),
        _as_text(y_col),
        "" if valid_n is None else int(valid_n),
        "" if i0 is None else int(i0),
        "" if i1 is None else int(i1),
        _as_text(py_model),
        _as_num(A_rect),
        _as_num(A_trap),
        _as_num(I_model),
        _as_num(err_rect),
        _as_num(err_trap),
        _as_num(rel_trap),
        _as_text(student_critical_review2),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")


# ------------------------------------------------------------
# 인공지능수학 수행평가 저장용 (추가)
# ------------------------------------------------------------

# --- AI 1차시 저장용 ---
SHEET_NAME_AI_STEP1 = "인공지능수학_수행평가_1차시"

DEFAULT_AI_STEP1_HEADER: List[str] = [
    "timestamp",
    "student_id",
    "alpha",
    "beta",
    "a0",
    "b0",
    "obs_shape",
    "obs_sensitivity",
    "obs_zigzag",
]


def ensure_ai_step1_header(ws) -> None:
    values = ws.get_all_values()
    if not values:
        ws.append_row(DEFAULT_AI_STEP1_HEADER, value_input_option="USER_ENTERED")
        return

    first_row = values[0]
    if len(first_row) == 0 or all((c.strip() == "" for c in first_row)):
        ws.update("A1", [DEFAULT_AI_STEP1_HEADER])
        return


def append_ai_step1_row(
    *,
    student_id: str,
    alpha: float,
    beta: float,
    a0: float,
    b0: float,
    obs_shape: str = "",
    obs_sensitivity: str = "",
    obs_zigzag: str = "",
    sheet_name: str = SHEET_NAME_AI_STEP1,
) -> None:
    if not str(student_id).strip():
        raise ValueError("student_id는 비어 있을 수 없습니다.")

    ws = get_worksheet(sheet_name=sheet_name, worksheet_index=0)
    ensure_ai_step1_header(ws)

    # '='로 시작하면 구글시트가 수식으로 오해할 수 있어 텍스트로 고정
    def _as_text(v: str) -> str:
        v = (v or "").strip()
        if v.startswith("="):
            return "'" + v
        return v

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(student_id).strip(),
        float(alpha),
        float(beta),
        float(a0),
        float(b0),
        _as_text(obs_shape),
        _as_text(obs_sensitivity),
        _as_text(obs_zigzag),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")


# --- AI 2차시 저장용 ---
SHEET_NAME_AI_STEP2 = "인공지능수학_수행평가_2차시"

DEFAULT_AI_STEP2_HEADER = [
    "timestamp",
    "student_id",
    "alpha",
    "beta",
    "start_a",
    "start_b",
    "step_size",
    "dE_da",          # ✅ 추가
    "dE_db",          # ✅ 추가
    "direction_desc",
    "direction_reason",
    "result_reflection",
    "final_a",
    "final_b",
    "steps_used",
    "final_E",
]


def ensure_ai_step2_header(ws) -> None:
    values = ws.get_all_values()
    if not values:
        ws.append_row(DEFAULT_AI_STEP2_HEADER, value_input_option="USER_ENTERED")
        return

    first_row = values[0]
    if len(first_row) == 0 or all((c.strip() == "" for c in first_row)):
        ws.update("A1", [DEFAULT_AI_STEP2_HEADER])
        return


def append_ai_step2_row(
    *,
    student_id: str,
    alpha: float,
    beta: float,
    start_a: float,
    start_b: float,
    step_size: float,
    dE_da: str = "",      # ✅ 추가
    dE_db: str = "",      # ✅ 추가
    direction_desc: str = "",
    direction_reason: str = "",
    result_reflection: str = "",
    final_a: float | None = None,
    final_b: float | None = None,
    steps_used: int | None = None,
    final_E: float | None = None,
    sheet_name: str = SHEET_NAME_AI_STEP2,
) -> None:
    if not str(student_id).strip():
        raise ValueError("student_id는 비어 있을 수 없습니다.")

    ws = get_worksheet(sheet_name=sheet_name, worksheet_index=0)
    ensure_ai_step2_header(ws)

    def _as_text(v: str) -> str:
        v = (v or "").strip()
        if v.startswith("="):
            return "'" + v
        return v

    def _as_num(v):
        if v is None or v == "":
            return ""
        return float(v)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(student_id).strip(),
        float(alpha),
        float(beta),
        float(start_a),
        float(start_b),
        float(step_size),
        _as_text(dE_da),
        _as_text(dE_db),
        _as_text(direction_desc),
        _as_text(direction_reason),
        _as_text(result_reflection),
        _as_num(final_a),
        _as_num(final_b),
        "" if steps_used is None else int(steps_used),
        _as_num(final_E),
    ]

    ws.append_row(row, value_input_option="USER_ENTERED")

