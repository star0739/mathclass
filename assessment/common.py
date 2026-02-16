# assessment/common.py
# ------------------------------------------------------------
# 공공데이터 분석 수행(수행평가) 공통 유틸 모듈
#
# 설계 목적
# - 1~3차시 페이지(step1/2/3)에서 공통으로 쓰는 세션 키/검증/로딩/저장 유틸을 한 곳에 모음
# - 대용량 객체를 과도하게 session_state에 넣지 않도록 가드(메모리 안전)
#
# NOTE(추후 제거/정리)
# - 수행평가 기간 종료 시 assessment 폴더를 통째로 제거하면 됨
# - 단, 다른 기능에서 import 하고 있다면 그 부분도 함께 삭제
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st


# ============================
# 세션 키(통일 관리)
# ============================
class SKeys:
    # 학생 식별
    STUDENT_ID = "assess_student_id"
    # 업로드 데이터
    DF = "assess_df"
    DF_META = "assess_df_meta"  # 파일명/업로드시간/행열수 등

    # 1차시(데이터 탐색)
    STEP1_SUMMARY = "assess_step1_summary"  # 그래프 특징/분석 질문/출처 등

    # 공통 축 선택(모델링/그래프)
    X_COL = "assess_x_col"
    Y_COL = "assess_y_col"

    # 2차시(모델링)
    MODEL_TYPE = "assess_model_type"       # "선형", "지수", "로그", "삼각", "합성(템플릿)" 등
    MODEL_EXPR = "assess_model_expr"       # 학생이 확정한 함수식(문자열)
    MODEL_PARAMS = "assess_model_params"   # 추정 계수(딕트/리스트) - 필요 시
    STEP2_INTERP = "assess_step2_interp"   # 도함수/이계도함수 해석 서술

    # 3차시(누적량)
    ACC_METHOD = "assess_acc_method"       # "정적분", "급수(합)" 등
    ACC_RANGE = "assess_acc_range"         # (start, end)
    ACC_VALUE = "assess_acc_value"         # 계산 결과
    STEP3_EVAL = "assess_step3_eval"       # 모델 평가/결론 서술

    # 제출/저장 상태
    LAST_SAVE_TS = "assess_last_save_ts"
    LAST_SAVE_STATUS = "assess_last_save_status"  # "OK"/"ERR"/None
    LAST_SAVE_MSG = "assess_last_save_msg"


# ============================
# 기본 설정(메모리 안전)
# ============================
@dataclass(frozen=True)
class Limits:
    # 과도한 데이터 업로드 방지(메모리/성능)
    max_rows: int = 200_000
    max_cols: int = 200
    # 미리보기 행 수
    preview_rows: int = 50


DEFAULT_LIMITS = Limits()


# ============================
# 세션 초기화/조회 유틸
# ============================
def init_assessment_session() -> None:
    """
    수행평가 관련 session_state 키들을 기본값으로 초기화(없을 때만).
    각 step 페이지 상단에서 호출해 두면 키 누락으로 인한 예외를 줄일 수 있음.
    """
    defaults = {
        SKeys.STUDENT_ID: "",
        SKeys.DF: None,
        SKeys.DF_META: None,
        SKeys.STEP1_SUMMARY: {},
        SKeys.X_COL: None,
        SKeys.Y_COL: None,
        SKeys.MODEL_TYPE: None,
        SKeys.MODEL_EXPR: "",
        SKeys.MODEL_PARAMS: None,
        SKeys.STEP2_INTERP: {},
        SKeys.ACC_METHOD: None,
        SKeys.ACC_RANGE: None,
        SKeys.ACC_VALUE: None,
        SKeys.STEP3_EVAL: {},
        SKeys.LAST_SAVE_TS: None,
        SKeys.LAST_SAVE_STATUS: None,
        SKeys.LAST_SAVE_MSG: None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_student_id(student_id: str) -> None:
    init_assessment_session()
    st.session_state[SKeys.STUDENT_ID] = (student_id or "").strip()


def get_student_id() -> str:
    init_assessment_session()
    return (st.session_state.get(SKeys.STUDENT_ID) or "").strip()


def require_student_id(message: str = "학번/식별 코드를 먼저 입력하세요.") -> str:
    """
    학생 식별이 없으면 입력을 유도하고, 이후 단계 진행을 막는다.
    step1~3 공통 상단에서 호출 권장.
    """
    init_assessment_session()
    sid = get_student_id()
    if sid:
        return sid

    st.warning(message)
    with st.form("assess_student_id_form"):
        sid_in = st.text_input("학번/식별 코드", placeholder="예: 30215")
        ok = st.form_submit_button("저장")
    if ok:
        set_student_id(sid_in)
        st.rerun()

    st.stop()


# ============================
# 데이터 로딩/검증
# ============================
def _validate_df(df: pd.DataFrame, limits: Limits = DEFAULT_LIMITS) -> Tuple[bool, str]:
    if df is None:
        return False, "데이터가 비어 있습니다."
    if df.shape[0] == 0 or df.shape[1] == 0:
        return False, "행/열이 비어 있는 데이터입니다."
    if df.shape[0] > limits.max_rows:
        return False, f"행 수가 너무 많습니다({df.shape[0]:,}행). 최대 {limits.max_rows:,}행까지 허용."
    if df.shape[1] > limits.max_cols:
        return False, f"열 수가 너무 많습니다({df.shape[1]:,}열). 최대 {limits.max_cols:,}열까지 허용."
    return True, "OK"


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    업로드 파일(csv/xlsx)을 DataFrame으로 읽는다.
    step1에서만 사용하는 것을 권장.
    """
    name = (uploaded_file.name or "").lower()
    if name.endswith(".csv"):
        # 인코딩 문제를 줄이기 위해 utf-8-sig 우선, 실패 시 기본
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8-sig")
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 Excel을 업로드하세요.")


def set_df(df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None, limits: Limits = DEFAULT_LIMITS) -> None:
    """
    DataFrame을 세션에 저장. (메모리 안전 검증 포함)
    """
    init_assessment_session()
    ok, msg = _validate_df(df, limits=limits)
    if not ok:
        raise ValueError(msg)

    st.session_state[SKeys.DF] = df
    st.session_state[SKeys.DF_META] = meta or {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }


def get_df() -> Optional[pd.DataFrame]:
    init_assessment_session()
    df = st.session_state.get(SKeys.DF)
    return df if isinstance(df, pd.DataFrame) else None


def require_df(
    step1_path: str = "assessment/step1_data.py",
    message: str = "업로드된 데이터가 없습니다. 1차시(데이터 탐색)에서 먼저 데이터를 업로드하세요.",
) -> pd.DataFrame:
    """
    df가 없으면 안내 후 step1로 이동. (페이지 분리 구조에서 안정적)
    """
    df = get_df()
    if df is not None:
        return df

    st.warning(message)
    if st.button("1차시로 이동", use_container_width=True, key="go_step1"):
        st.switch_page(step1_path)
    st.stop()


def get_df_preview(df: pd.DataFrame, limits: Limits = DEFAULT_LIMITS) -> pd.DataFrame:
    """
    화면 미리보기용(상위 N행). 대용량에서도 가볍게.
    """
    n = min(limits.preview_rows, len(df))
    return df.head(n).copy()


def numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    숫자형으로 변환 가능한 컬럼 목록(후보).
    실제로는 데이터에 따라 coerce 후 NaN 비율이 높을 수 있어,
    step 페이지에서 선택 후 유효값 개수를 추가 검증하는 것을 권장.
    """
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 2:
            cols.append(c)
    return cols


def to_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


# ============================
# 상태 저장(간단)
# ============================
def set_xy(x_col: Optional[str], y_col: Optional[str]) -> None:
    init_assessment_session()
    st.session_state[SKeys.X_COL] = x_col
    st.session_state[SKeys.Y_COL] = y_col


def get_xy() -> Tuple[Optional[str], Optional[str]]:
    init_assessment_session()
    return st.session_state.get(SKeys.X_COL), st.session_state.get(SKeys.Y_COL)


def set_step1_summary(summary: Dict[str, Any]) -> None:
    init_assessment_session()
    st.session_state[SKeys.STEP1_SUMMARY] = summary or {}


def get_step1_summary() -> Dict[str, Any]:
    init_assessment_session()
    return st.session_state.get(SKeys.STEP1_SUMMARY) or {}


def set_step2_interp(payload: Dict[str, Any]) -> None:
    init_assessment_session()
    st.session_state[SKeys.STEP2_INTERP] = payload or {}


def get_step2_interp() -> Dict[str, Any]:
    init_assessment_session()
    return st.session_state.get(SKeys.STEP2_INTERP) or {}


def set_step3_eval(payload: Dict[str, Any]) -> None:
    init_assessment_session()
    st.session_state[SKeys.STEP3_EVAL] = payload or {}


def get_step3_eval() -> Dict[str, Any]:
    init_assessment_session()
    return st.session_state.get(SKeys.STEP3_EVAL) or {}


# ============================
# 제출/저장 상태 표시용
# ============================
def set_save_status(ok: bool, msg: str = "") -> None:
    init_assessment_session()
    st.session_state[SKeys.LAST_SAVE_TS] = datetime.now().isoformat(timespec="seconds")
    st.session_state[SKeys.LAST_SAVE_STATUS] = "OK" if ok else "ERR"
    st.session_state[SKeys.LAST_SAVE_MSG] = msg


def render_save_status() -> None:
    """
    상단에 '마지막 저장 상태'를 작게 보여주고 싶을 때 사용.
    """
    init_assessment_session()
    ts = st.session_state.get(SKeys.LAST_SAVE_TS)
    status = st.session_state.get(SKeys.LAST_SAVE_STATUS)
    msg = st.session_state.get(SKeys.LAST_SAVE_MSG)

    if not ts:
        st.info("저장 상태: 아직 제출/저장 기록이 없습니다.")
        return

    if status == "OK":
        st.success(f"저장 상태: OK ({ts}) {msg}".strip())
    else:
        st.error(f"저장 상태: 오류 ({ts}) {msg}".strip())


# ============================
# (추후) PDF 생성 준비용 데이터 패키징
# ============================
def build_report_payload() -> Dict[str, Any]:
    """
    step1~3에서 작성/계산된 내용을 PDF 생성용으로 하나의 dict로 모아준다.
    step3에서 PDF 다운로드 버튼을 만들 때 이 payload를 사용하면 됨.
    """
    init_assessment_session()
    df_meta = st.session_state.get(SKeys.DF_META) or {}
    x_col, y_col = get_xy()

    payload = {
        "student_id": get_student_id(),
        "df_meta": df_meta,
        "x_col": x_col,
        "y_col": y_col,
        "step1": get_step1_summary(),
        "model": {
            "type": st.session_state.get(SKeys.MODEL_TYPE),
            "expr": st.session_state.get(SKeys.MODEL_EXPR),
            "params": st.session_state.get(SKeys.MODEL_PARAMS),
        },
        "step2": get_step2_interp(),
        "accumulation": {
            "method": st.session_state.get(SKeys.ACC_METHOD),
            "range": st.session_state.get(SKeys.ACC_RANGE),
            "value": st.session_state.get(SKeys.ACC_VALUE),
        },
        "step3": get_step3_eval(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return payload


def reset_assessment(keep_student_id: bool = True) -> None:
    """
    전체 수행평가 세션 상태 초기화(데이터/입력 내용 포함).
    - keep_student_id=True면 학번은 남기고 나머지 초기화.
    """
    init_assessment_session()
    sid = get_student_id() if keep_student_id else ""
    for k in list(st.session_state.keys()):
        if k.startswith("assess_"):
            del st.session_state[k]
    init_assessment_session()
    if keep_student_id:
        st.session_state[SKeys.STUDENT_ID] = sid
