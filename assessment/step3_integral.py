# step3_integral.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

from assessment.common import (
    init_assessment_session,
    require_student_id,
    get_df,
    get_xy,
)

# ✅ Step2와 동일한 방식으로 Step3도 구글시트 저장을 붙일 거면,
# assessment/google_sheets.py에 append_step3_row를 만들어서 아래 import를 활성화하세요.
# from assessment.google_sheets import append_step3_row


# -----------------------------
# 운영 기준 (MVP)
# -----------------------------
MIN_VALID_POINTS = 5


# -----------------------------
# Step2/Step3 세션 저장
# -----------------------------
def _get_step2_state() -> dict:
    return st.session_state.get("assessment_step2", {})


def _get_step3_state() -> dict:
    return st.session_state.get("assessment_step3", {})


def _set_step3_state(d: dict) -> None:
    st.session_state["assessment_step3"] = d


# -----------------------------
# 백업 TXT (MVP: key: value 라인)
# -----------------------------
def build_step3_backup(payload: dict) -> bytes:
    # 줄바꿈 포함 필드(conclusion 등)는 안전하게 \n 치환
    def _ser(v):
        if v is None:
            return ""
        if isinstance(v, float):
            # 너무 긴 과학표기 방지
            return f"{v:.12g}"
        s = str(v)
        return s.replace("\n", "\\n")

    lines = [f"{k}: {_ser(v)}" for k, v in payload.items()]
    return ("\n".join(lines)).encode("utf-8")


def parse_step3_backup(text: str) -> dict:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().replace("\\n", "\n")
    return out


# -----------------------------
# 시간축: 년/월/년월 혼재 -> '시작 시점으로부터 지난 개월 수'
# -----------------------------
def make_month_index(t_series: pd.Series) -> np.ndarray:
    """
    t_series가 다음 중 어떤 형식이든 월 단위 인덱스(개월 수)로 변환한다.
    - datetime / 'YYYY-MM' / 'YYYY/MM'
    - 정수/문자 'YYYYMM' (예: 202401)
    - 'YYYY' (연도만) -> 1월로 간주(해석 주의)
    실패 시: 순번(0..n-1) fallback
    """
    s = t_series.astype(str).str.strip()
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # 1) YYYYMM 우선
    yyyymm = s.str.replace(r"[^\d]", "", regex=True)
    mask_yyyymm = yyyymm.str.fullmatch(r"\d{6}")
    if mask_yyyymm.any():
        y = yyyymm[mask_yyyymm].str.slice(0, 4).astype(int)
        m = yyyymm[mask_yyyymm].str.slice(4, 6).astype(int)
        dt.loc[mask_yyyymm] = pd.to_datetime(
            dict(year=y, month=m, day=1), errors="coerce"
        )

    # 2) 나머지 to_datetime
    remain = dt.isna()
    if remain.any():
        dt.loc[remain] = pd.to_datetime(s[remain], errors="coerce")

    # 3) 연도만(YYYY) -> 1월로 처리
    remain = dt.isna()
    if remain.any():
        mask_year = s[remain].str.fullmatch(r"\d{4}")
        if mask_year.any():
            y = s[remain][mask_year].astype(int)
            idx = s[remain].index[mask_year]
            dt.loc[idx] = pd.to_datetime(dict(year=y, month=1, day=1), errors="coerce")

    # 4) 실패 fallback
    if dt.isna().any():
        return np.arange(len(t_series), dtype=float)

    p = dt.dt.to_period("M")
    p0 = p.iloc[0]
    months = (p - p0).astype(int).to_numpy(dtype=float)
    return months


# -----------------------------
# 수치적분(사다리꼴) + 누적 사다리꼴
# -----------------------------
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    A = np.zeros_like(y, dtype=float)
    for k in range(1, len(y)):
        dx = x[k] - x[k - 1]
        A[k] = A[k - 1] + 0.5 * (y[k] + y[k - 1]) * dx
    return A


# -----------------------------
# Step2 py_model(표현식 문자열) -> f(t)
# 예: "22 - 0.017*t + 6*np.cos(2*np.pi*t/12) + ..."
# -----------------------------
def _compile_model_expr(py_model: str):
    expr = (py_model or "").strip()
    if not expr:
        return None, "py_model이 비어 있습니다."

    # MVP 수준의 위험 토큰 차단(완전한 샌드박스는 아님)
    blocked = [
        "__", "import", "open(", "exec(", "eval(",
        "os.", "sys.", "subprocess", "pickle", "globals", "locals",
    ]
    if any(tok in expr for tok in blocked):
        return None, "허용되지 않는 토큰이 포함되어 모델식을 사용할 수 없습니다."

    safe_globals = {
        "__builtins__": {},
        "np": np,
        "numpy": np,
        "math": __import__("math"),
    }
    safe_locals = {}

    code = "def f(t):\n    return " + expr.replace("\n", " ")
    try:
        exec(code, safe_globals, safe_locals)
        f = safe_locals.get("f")
        if not callable(f):
            return None, "f(t) 생성에 실패했습니다."
        return f, "표현식 py_model을 f(t)=...로 변환해 사용합니다."
    except Exception as e:
        return None, f"모델식 컴파일 실패: {e}"


# -----------------------------
# 입력 검증
# -----------------------------
def _validate_step3(conclusion: str) -> bool:
    if not (conclusion or "").strip():
        st.warning("종합 결론(장점/한계/개선 제안)을 입력하세요.")
        return False
    return True


# -----------------------------
# 메인
# -----------------------------
def run():
    st.title("Step 3. 적분(누적) 관점에서 모델의 장점과 한계 정리 (MVP)")

    init_assessment_session()
    student_id = require_student_id()

    df = get_df()
    xy = get_xy()
    step2 = _get_step2_state()

    if df is None or xy is None:
        st.error("Step1/Step2 정보가 없습니다. 먼저 Step1~Step2를 완료해주세요.")
        st.stop()

    t_col, y_col = xy["t"], xy["y"]
    if t_col not in df.columns or y_col not in df.columns:
        st.error("선택된 t/y 컬럼을 데이터프레임에서 찾을 수 없습니다.")
        st.stop()

    # -----------------------------
    # 0) Step3 백업 복구(선택)
    # -----------------------------
    st.subheader("0) Step3 백업 복구(선택)")
    restored: dict[str, str] = {}
    up = st.file_uploader("Step3 백업 TXT가 있으면 업로드하세요.", type=["txt"])
    if up is not None:
        txt = up.read().decode("utf-8", errors="replace")
        restored = parse_step3_backup(txt)
        st.success("Step3 백업을 읽었습니다. 아래 입력값에 반영됩니다.")

    # -----------------------------
    # 데이터 준비
    # -----------------------------
    d = df[[t_col, y_col]].copy().dropna()
    if len(d) < MIN_VALID_POINTS:
        st.error("유효 데이터가 너무 적습니다.")
        st.stop()

    # ✅ 년/월/년월 혼재 -> 개월 인덱스
    x_all = make_month_index(d[t_col])

    # y 숫자화
    y_all = pd.to_numeric(d[y_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y_all).any():
        mask = ~np.isnan(y_all)
        d = d.loc[mask].copy()
        x_all = x_all[mask]
        y_all = y_all[mask]

    if len(d) < MIN_VALID_POINTS:
        st.error("유효 데이터가 너무 적습니다(숫자 변환 후).")
        st.stop()

    # 연도만(YYYY)으로 처리된 가능성 안내(간단 경고)
    # (정확한 감지는 어렵지만, 원자료가 4자리 숫자만 많으면 경고)
    s_raw = d[t_col].astype(str).str.strip()
    ratio_year_only = (s_raw.str.fullmatch(r"\d{4}")).mean()
    if ratio_year_only >= 0.8:
        st.warning("시간 데이터가 '연도(YYYY)' 중심으로 보입니다. 월 단위(1월 가정)로 변환되어 해석이 거칠 수 있습니다.")

    # -----------------------------
    # 1) 분석 구간 선택
    # -----------------------------
    st.subheader("1) 분석 구간 선택")
    n = len(d)

    def _safe_int(v, default):
        try:
            return int(v)
        except Exception:
            return default

    default_i0 = _safe_int(restored.get("i0", 0), 0)
    default_i1 = _safe_int(restored.get("i1", n - 1), n - 1)
    default_i0 = max(0, min(n - 2, default_i0))
    default_i1 = max(default_i0 + 1, min(n - 1, default_i1))

    i0, i1 = st.slider(
        "적분 구간(인덱스)",
        min_value=0,
        max_value=n - 1,
        value=(default_i0, default_i1),
        step=1,
    )

    x = x_all[i0 : i1 + 1]
    y = y_all[i0 : i1 + 1]

    # -----------------------------
    # 모델 함수 준비 (Step2 py_model: 표현식)
    # -----------------------------
    st.subheader("2) 누적량 비교 (데이터 수치적분 vs 모델 정적분)")
    py_model = (step2.get("py_model") or "").strip()
    f_func, model_msg = _compile_model_expr(py_model)
    st.caption(f"모델 로딩: {model_msg}")

    # 누적량 계산
    A_data = _trapz(y, x)

    A_model = None
    f_vals = None
    if callable(f_func):
        try:
            f_vals_all = np.asarray(f_func(x_all), dtype=float)
            if len(f_vals_all) == len(x_all):
                f_vals = f_vals_all[i0 : i1 + 1]
                A_model = _trapz(f_vals, x)
            else:
                st.warning("모델값 길이가 데이터와 일치하지 않아 모델 적분 비교를 생략합니다.")
        except Exception as e:
            st.warning(f"모델값 계산 실패로 모델 적분 비교를 생략합니다: {e}")

    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 누적량  ∫y dt(근사)", f"{A_data:,.6g}")
    if A_model is None:
        c2.metric("모델 누적량  ∫f dt(근사)", "—")
        c3.metric("상대오차", "—")
    else:
        c2.metric("모델 누적량  ∫f dt(근사)", f"{A_model:,.6g}")
        rel = abs(A_data - A_model) / (abs(A_data) + 1e-12)
        c3.metric("상대오차", f"{rel:.3%}")

    # -----------------------------
    # 3) 누적 그래프
    # -----------------------------
    st.subheader("3) 누적 그래프 (누적 적분 곡선)")
    cum_data = _cumtrapz(y, x)
    cum_model = None if f_vals is None else _cumtrapz(f_vals, x)

    # x축 라벨 안내(개월 인덱스)
    st.caption("x축 t는 '시작 시점으로부터 지난 개월 수'로 변환해 사용합니다. (예: 0, 1, 2, ... 또는 누락 시 0,1,3,...)")

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=cum_data, mode="lines", name="누적(데이터)"))
        if cum_model is not None:
            fig.add_trace(go.Scatter(x=x, y=cum_model, mode="lines", name="누적(모델)"))
        fig.update_layout(
            height=420,
            xaxis_title="t (개월 인덱스)",
            yaxis_title="누적량",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(x, cum_data, label="누적(데이터)")
        if cum_model is not None:
            plt.plot(x, cum_model, label="누적(모델)")
        plt.xlabel("t (개월 인덱스)")
        plt.ylabel("누적량")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    # -----------------------------
    # 4) 종합 결론(장점/한계/개선)
    # -----------------------------
    st.subheader("4) 종합 결론: 이 모델의 장점과 한계")
    st.info(
        "아래 내용을 모두 포함해 서술하세요.\n"
        "• 누적 관점에서 데이터와 모델이 얼마나 일치하는가(근거: 누적량/누적 그래프)\n"
        "• 장점 1가지(근거 포함)\n"
        "• 한계 1가지(근거 포함)\n"
        "• 개선 제안 1가지(변수/모델/구간/방법 등)\n"
    )

    conclusion_default = restored.get("conclusion", _get_step3_state().get("conclusion", ""))
    conclusion = st.text_area("서술 입력", value=conclusion_default, height=220)

    note_default = restored.get("note", _get_step3_state().get("note", ""))
    note = st.text_input("메모(선택)", value=note_default)

    # -----------------------------
    # 5) 저장 및 백업 (Step2 패턴)
    # -----------------------------
    st.subheader("5) 저장 및 백업")

    payload = {
        "student_id": student_id,
        "data_source": (step2.get("data_source") or "").strip(),
        "x_col": step2.get("x_col", t_col),
        "y_col": step2.get("y_col", y_col),
        "valid_n": step2.get("valid_n", ""),
        "i0": int(i0),
        "i1": int(i1),
        "A_data": float(A_data),
        "A_model": "" if A_model is None else float(A_model),
        "relative_error": "" if A_model is None else float(abs(A_data - A_model) / (abs(A_data) + 1e-12)),
        "py_model": py_model,
        "conclusion": conclusion.strip(),
        "note": note.strip(),
    }

    col1, col2, col3 = st.columns([1, 1, 1.2])
    save_clicked = col1.button("💾 저장(구글시트)", use_container_width=True)
    download_clicked = col2.button("⬇️ TXT 백업 만들기", use_container_width=True)
    go_next = col3.button("🏁 제출/종료", use_container_width=True)

    # 다운로드 버튼은 항상 노출(최신 payload 반영)
    backup_bytes = build_step3_backup(payload)
    st.download_button(
        label="📄 (다운로드) 3차시 백업 TXT",
        data=backup_bytes,
        file_name=f"미적분_수행평가_3차시_{student_id}.txt",
        mime="text/plain; charset=utf-8",
    )

    if save_clicked or download_clicked or go_next:
        if not _validate_step3(conclusion):
            st.stop()

        _set_step3_state({**payload, "saved_at": pd.Timestamp.now().isoformat()})

        if download_clicked:
            st.success("✅ 백업 데이터가 준비되었습니다. 위 '다운로드' 버튼을 눌러주세요.")

        if save_clicked or go_next:
            try:
                # ✅ 구글시트 저장을 붙이려면 assessment/google_sheets.py에 append_step3_row를 구현해서 호출하세요.
                # append_step3_row(
                #     student_id=payload["student_id"],
                #     data_source=payload["data_source"],
                #     x_col=payload["x_col"],
                #     y_col=payload["y_col"],
                #     valid_n=payload["valid_n"],
                #     i0=payload["i0"],
                #     i1=payload["i1"],
                #     A_data=payload["A_data"],
                #     A_model=payload["A_model"],
                #     relative_error=payload["relative_error"],
                #     py_model=payload["py_model"],
                #     conclusion=payload["conclusion"],
                #     note=payload["note"],
                # )
                st.success("✅ (임시) 구글시트 저장 위치입니다. append_step3_row 연결 후 활성화하세요.")
            except Exception as e:
                st.error(f"⚠️ 구글 시트 저장 오류: {e}")
                st.stop()

    # -----------------------------
    # 검토용
    # -----------------------------
    with st.expander("계산 세부값(검토용)", expanded=False):
        st.write(
            {
                "t_col": t_col,
                "y_col": y_col,
                "n_valid": len(d),
                "range": (int(i0), int(i1)),
                "A_data": A_data,
                "A_model": A_model,
                "py_model": py_model[:120] + ("..." if len(py_model) > 120 else ""),
            }
        )


if __name__ == "__main__":
    run()

