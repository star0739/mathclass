# assessment/ai_step2_path.py
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import re


PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

from assessment.common import (
    init_assessment_session,
    require_student_id,
    set_save_status,
    render_save_status,
)

# ✅ NEW: 선택형 손실함수
from assessment.ai_loss import (
    make_loss_spec,
    E as E_loss,
    grad as grad_loss,
    latex_E,
)

TITLE = "2차시: 경로(손실을 줄이는 방향) 탐색"

# (1차시와 동일한 범위 설정)
A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0

GRID_N = 121  # 고정 해상도(학생 선택 X) — 안정성 우선
LEARNING_RATE = 0.18  # ✅ 기존 STEP_SIZE를 교과서 용어로 통일(학습률)
MAX_PATH_POINTS = 250  # 렌더/메모리 안전 상한

PRESET_STARTS = [
    (2.2, 2.2),
    (-2.2, 2.0),
    (2.5, -1.8),
    (-2.4, -2.1),
]

_STATE_KEY = "ai_step2_path_state"
_BACKUP_STATE_KEY = "ai_step2_backup_payload"


# -----------------------------
# 1차시 선택 손실함수 로더(호환 포함)
# -----------------------------
def _load_loss_spec_from_step1() -> tuple[object, str]:
    """
    returns:
      (loss_spec: LossSpec, display_latex: str)
    호환:
      - 신형: st.session_state["ai_step1_structure"]["loss_spec"] = {type, params, ...}
      - 구형: st.session_state["ai_step1_structure"] has alpha/beta -> quad(alpha=alpha, b^2 coefficient fixed in ai_loss)
    """
    step1 = st.session_state.get("ai_step1_structure", {}) or {}
    loss_info = step1.get("loss_spec", None)

    # 신형 구조
    if isinstance(loss_info, dict) and loss_info.get("type"):
        loss_type = str(loss_info.get("type"))
        params = loss_info.get("params", {}) or {}
        spec = make_loss_spec(loss_type, params)
        return spec, latex_E(spec)

    # 구형 구조(예: alpha/beta)
    # -> quad(alpha=alpha)로 매핑 (ai_loss의 quad는 b^2 계수 1로 설계)
    alpha = step1.get("alpha", None)
    if alpha is not None:
        spec = make_loss_spec("quad", {"alpha": float(alpha)})
        return spec, latex_E(spec)

    # 마지막 fallback: ai_loss_spec(1차시에서 별도로 저장했을 수도 있음)
    raw = st.session_state.get("ai_loss_spec", None)
    if isinstance(raw, dict) and raw.get("type"):
        spec = make_loss_spec(str(raw["type"]), raw.get("params", {}) or {})
        return spec, latex_E(spec)

    return None, None

def parse_step1_backup_txt(uploaded_file):
    content = uploaded_file.read().decode("utf-8-sig")

    match_type = re.search(r"loss_type:\s*(\w+)", content)
    if not match_type:
        return None, "loss_type을 찾을 수 없습니다."

    loss_type = match_type.group(1).strip()

    match_params = re.search(r"params:\s*(\{.*?\})", content)
    if not match_params:
        return None, "params 정보를 찾을 수 없습니다."

    try:
        params = eval(match_params.group(1))
    except Exception:
        return None, "params 해석 실패"

    return {"type": loss_type, "params": params}, None

def _get_state() -> dict:
    return st.session_state.get(_STATE_KEY, {})


def _set_state(d: dict) -> None:
    st.session_state[_STATE_KEY] = d


def _init_state(student_id: str, start_a: float, start_b: float, start_e: float) -> dict:
    s = _get_state()
    if not isinstance(s, dict) or s.get("student_id") != student_id:
        s = {
            "student_id": student_id,
            "theta_deg": 225.0,
            "start_a": float(start_a),
            "start_b": float(start_b),
            "path": [(float(start_a), float(start_b), float(start_e))],
            "last_delta": None,
        }
        _set_state(s)
    return s


def _clip(a: float, b: float) -> tuple[float, float]:
    return float(np.clip(a, A_MIN, A_MAX)), float(np.clip(b, B_MIN, B_MAX))


def _unit_from_angle_deg(theta_deg: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return math.cos(t), math.sin(t)


def recommended_direction(a: float, b: float, loss_spec) -> tuple[float, float]:
    """
    현재 점에서 손실을 줄이는(가장 빨리 줄이는) 방향(정규화)을 계산.
    - (∂E/∂a, ∂E/∂b)의 반대 방향을 사용.
    """
    da, db = grad_loss(a, b, loss_spec)
    vx, vy = -float(da), -float(db)
    norm = math.hypot(vx, vy)
    if norm < 1e-12:
        return 0.0, 0.0
    return vx / norm, vy / norm


def _append_point(path: list[tuple[float, float, float]], a: float, b: float, loss_spec) -> list[tuple[float, float, float]]:
    a, b = _clip(a, b)
    e = float(E_loss(np.array(a), np.array(b), loss_spec))
    new_path = path + [(a, b, e)]
    if len(new_path) > MAX_PATH_POINTS:
        new_path = new_path[-MAX_PATH_POINTS:]
    return new_path


@st.cache_data(show_spinner=False)
def build_grid(a_min: float, a_max: float, b_min: float, b_max: float, n: int, loss_type: str, params_items: tuple):
    """
    cache key에 loss_type/params가 반영되도록 params_items(정렬된 튜플)로 받음
    """
    params = dict(params_items)
    spec = make_loss_spec(loss_type, params)

    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    Z = E_loss(A, B, spec)
    return A, B, Z


def coord_axis_path(a0: float, b0: float, steps: int, learning_rate: float, loss_spec) -> list[tuple[float, float]]:
    """
    1차시처럼 'a만, b만 번갈아' 움직이는 경로(점선 표시용)
    """
    a, b = float(a0), float(b0)
    pts = [(a, b)]
    for k in range(steps):
        da, db = grad_loss(a, b, loss_spec)
        if k % 2 == 0:
            a = a - learning_rate * float(da)
        else:
            b = b - learning_rate * float(db)
        a, b = _clip(a, b)
        pts.append((a, b))
    return pts


def build_backup_text(payload: dict) -> str:
    lines: list[str] = []
    lines.append("인공지능수학 수행평가 (2차시) 백업")
    lines.append("=" * 46)
    lines.append(f"저장시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")

    lines.append("[함수 설정]")
    lines.append(f"- loss_type: {payload.get('loss_type','')}")
    lines.append(f"- params: {payload.get('loss_params',{})}")
    lines.append(f"- {payload.get('loss_latex','')}")
    lines.append("")

    lines.append("[이동 설정]")
    lines.append(f"- 시작점: ({payload.get('start_a','')}, {payload.get('start_b','')})")
    lines.append(f"- 학습률(learning rate): {payload.get('learning_rate','')}")
    lines.append(f"- 사용 step 수: {payload.get('steps_used','')}")
    lines.append(f"- 최종점: ({payload.get('final_a','')}, {payload.get('final_b','')})")
    lines.append(f"- 최종 손실: {payload.get('final_E','')}")
    lines.append("")

    lines.append("[학생 입력(서술)]")
    lines.append("1) 편미분 값 계산:")
    lines.append((payload.get("partials_input", "") or "").strip())
    lines.append("")
    lines.append("2) 방향 성분 판단 + 선택한 이동 방향:")
    lines.append((payload.get("direction_desc", "") or "").strip())
    lines.append("")
    lines.append("3) 1 step 이동 결과 해석:")
    lines.append((payload.get("result_reflection", "") or "").strip())
    lines.append("")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title=TITLE, layout="wide")

    init_assessment_session()
    student_id = require_student_id()

    # --------------------------------------------
    # 0) 1차시 손실함수 불러오기 (세션 우선 → TXT 업로드 대안)
    # --------------------------------------------
    loss_spec = None
    loss_latex = None

    # (1) 세션에서 먼저 시도
    try:
        loss_spec, loss_latex = _load_loss_spec_from_step1()
    except Exception:
        loss_spec = None
        loss_latex = None

    # (2) 세션에 없으면: 1차시 백업 TXT 업로드로 복원
    if loss_spec is None:
        st.subheader("① 1차시 백업 파일 업로드")

        uploaded_file = st.file_uploader(
            "1차시 백업 TXT 파일을 업로드하세요",
            type=["txt"],
        )

        if uploaded_file is None:
            st.info("1차시를 완료한 같은 세션이 아니면, 1차시 백업 TXT 업로드가 필요합니다.")
            st.stop()

        parsed, error_msg = parse_step1_backup_txt(uploaded_file)
        if error_msg:
            st.error(error_msg)
            st.stop()

        loss_spec = make_loss_spec(parsed["type"], parsed["params"])
        loss_latex = latex_E(loss_spec)

        st.success("1차시 손실함수 복원 완료")
        st.latex(loss_latex)

    # (선택) 현재 선택된 함수 표시(세션으로 불러온 경우에도 보이게)
    if loss_latex:
        st.caption("현재 적용된 손실함수")
        st.latex(loss_latex)

    st.title(TITLE)


    # 초기 시작점: 프리셋 1
    a_init, b_init = _clip(PRESET_STARTS[0][0], PRESET_STARTS[0][1])
    e_init = float(E_loss(np.array(a_init), np.array(b_init), loss_spec))

    s = _init_state(student_id, a_init, b_init, e_init)

    st.markdown(
        r"""
이번 시간은 **등고선(2D)** 위에서, 시작점에서 **손실을 줄이는 방향**을 직접 선택하고 1 step 이동을 반복하며 경로를 관찰합니다.
"""
    )

    left, right = st.columns([1, 2], gap="large")

    # -------------------------
    # 좌측: ① 시작점 / ② 이동
    # -------------------------
    with left:
        st.subheader("① 시작점 설정")

        preset_labels = [f"프리셋 {i+1}: ({a:g}, {b:g})" for i, (a, b) in enumerate(PRESET_STARTS)]
        preset_idx = st.selectbox(
            "시작점 선택",
            options=list(range(len(PRESET_STARTS))),
            format_func=lambda i: preset_labels[i],
            key="ai_step2_preset_idx",
        )

        c1, c2 = st.columns(2, gap="small")
        with c1:
            apply_preset = st.button("적용", use_container_width=True)
        with c2:
            reset_path = st.button("경로 초기화", use_container_width=True)

        if apply_preset:
            a0, b0 = PRESET_STARTS[int(preset_idx)]
            a0, b0 = _clip(a0, b0)
            s["start_a"], s["start_b"] = a0, b0
            s["path"] = [(a0, b0, float(E_loss(np.array(a0), np.array(b0), loss_spec)))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        if reset_path:
            a0, b0 = float(s.get("start_a", PRESET_STARTS[0][0])), float(s.get("start_b", PRESET_STARTS[0][1]))
            a0, b0 = _clip(a0, b0)
            s["path"] = [(a0, b0, float(E_loss(np.array(a0), np.array(b0), loss_spec)))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        st.divider()
        st.subheader("② 방향 선택 & 1 step 이동")

        theta = st.slider(
            "내가 고른 방향(각도, 도)",
            min_value=0.0,
            max_value=360.0,
            value=float(s.get("theta_deg", 225.0)),
            step=1.0,
        )
        s["theta_deg"] = float(theta)
        _set_state(s)

        path = s.get("path", [])
        cur_a, cur_b, cur_e = path[-1]
        st.metric("현재 손실", f"{cur_e:.6f}")

        # ✅ 교과서 용어로 표시(학습률)
        st.markdown(f"- 학습률(learning rate): **{LEARNING_RATE:g}** (고정)")
        st.caption("※ 학습률은 한 번 이동할 때 기울기 방향으로 얼마나 움직일지 결정합니다.")

        # 추천 방향 벡터
        reco_vx, reco_vy = recommended_direction(cur_a, cur_b, loss_spec)

        # 학생이 선택한 방향
        ux, uy = _unit_from_angle_deg(theta)

        move = st.button("➡️ 1 step 이동", type="primary", use_container_width=True)

        if move:
            # 1 step 이동: 학생이 고른 방향으로 이동
            na = cur_a + LEARNING_RATE * ux
            nb = cur_b + LEARNING_RATE * uy
            new_path = _append_point(path, na, nb, loss_spec)
            s["path"] = new_path
            s["last_delta"] = (float(new_path[-1][0] - cur_a), float(new_path[-1][1] - cur_b))
            _set_state(s)
            st.rerun()

    # -------------------------
    # 우측: 시각화(등고선 + 경로 + 방향 화살표)
    # -------------------------
    with right:
        st.subheader("시각화(등고선 + 경로)")

        # cache key 안정화를 위해 params를 정렬 튜플로
        params_items = tuple(sorted(dict(loss_spec.params).items()))
        A, B, Z = build_grid(A_MIN, A_MAX, B_MIN, B_MAX, GRID_N, loss_spec.type, params_items)

        path = s.get("path", [])
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        cur_a, cur_b, cur_e = path[-1]
        ux, uy = _unit_from_angle_deg(float(s.get("theta_deg", 225.0)))
        reco_vx, reco_vy = recommended_direction(cur_a, cur_b, loss_spec)

        # 1차시식 축-번갈아 경로(점선)
        axis_path = coord_axis_path(cur_a, cur_b, steps=8, learning_rate=LEARNING_RATE, loss_spec=loss_spec)

        arrow_len = 0.6

        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(
                go.Contour(
                    x=np.linspace(A_MIN, A_MAX, GRID_N),
                    y=np.linspace(B_MIN, B_MAX, GRID_N),
                    z=Z,
                    contours=dict(showlabels=False),
                    line=dict(width=1),
                    name="등고선",
                )
            )

            # 경로
            if len(xs) >= 2:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="이동 경로"))
            else:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="현재"))

            # 축-번갈아 경로(점선)
            if axis_path is not None and len(axis_path) >= 2:
                ax_x = [p[0] for p in axis_path]
                ax_y = [p[1] for p in axis_path]
                fig.add_trace(go.Scatter(x=ax_x, y=ax_y, mode="lines", line=dict(dash="dot"), name="축만 번갈아(참고)"))

            # 화살표(학생 선택)
            fig.add_trace(
                go.Scatter(
                    x=[cur_a, cur_a + arrow_len * ux],
                    y=[cur_b, cur_b + arrow_len * uy],
                    mode="lines",
                    name="내가 고른 방향",
                )
            )
            # 화살표(추천)
            fig.add_trace(
                go.Scatter(
                    x=[cur_a, cur_a + arrow_len * reco_vx],
                    y=[cur_b, cur_b + arrow_len * reco_vy],
                    mode="lines",
                    name="추천 방향",
                )
            )

            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="a",
                yaxis_title="b",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots()
            cs = ax.contour(A, B, Z, levels=18)
            ax.clabel(cs, inline=True, fontsize=8)

            if len(xs) >= 2:
                ax.plot(xs, ys, marker="o")
            else:
                ax.scatter(xs, ys, s=60)

            if axis_path is not None and len(axis_path) >= 2:
                ax_x = [p[0] for p in axis_path]
                ax_y = [p[1] for p in axis_path]
                ax.plot(ax_x, ax_y, linestyle=":", linewidth=2)

            ax.scatter([cur_a], [cur_b], s=70)
            ax.text(cur_a, cur_b, "현재", fontsize=10)

            ax.arrow(cur_a, cur_b, arrow_len * ux, arrow_len * uy, head_width=0.08, length_includes_head=True)
            ax.arrow(cur_a, cur_b, arrow_len * reco_vx, arrow_len * reco_vy, head_width=0.08, length_includes_head=True)

            ax.set_xlabel("a")
            ax.set_ylabel("b")
            ax.set_title("Contour + Path")
            st.pyplot(fig, clear_figure=True)

    # -------------------------
    # 하단(전체 폭): ③ 서술 + 백업 + 저장/상태
    # -------------------------
    st.divider()
    st.subheader("③ 관찰 기록 서술")

    # ✅ 고정식 제거 -> 선택된 함수 표시
    st.markdown(
        rf"""
1) 선택한 손실함수 $E(a,b)$에 대해 시작점 $(a,b)$에서의 $\dfrac{{\partial E}}{{\partial a}}$, $\dfrac{{\partial E}}{{\partial b}}$를 구하시오.  

현재 선택된 함수:
$$
{loss_latex}
$$
"""
    )

    colp1, colp2 = st.columns(2, gap="large")
    with colp1:
        st.markdown(r"$$\frac{\partial E}{\partial a} = $$")
        dE_da = st.text_input("편미분 식에 시작점 a좌표 값 대입", key="ai_step2_dE_da", label_visibility="collapsed")
    with colp2:
        st.markdown(r"$$\frac{\partial E}{\partial b} = $$")
        dE_db = st.text_input("편미분 식에 시작점 b좌표 값 대입", key="ai_step2_dE_db", label_visibility="collapsed")

    direction_desc = st.text_area(
        "2) 위에서 구한 두 값의 부호를 관찰하고, 손실을 줄이기 위해 각 변수를 어떤 방향(증가/감소)으로 변화시켜야 하는지 서술하시오.",
        height=120,
        placeholder="예: ∂E/∂a의 부호가 +이면 a를 감소시키면 E가 줄어든다. ∂E/∂b의 부호가 -이면 b를 증가시키면 E가 줄어든다. ...",
        key="ai_step2_direction_desc",
    )

    reflection = st.text_area(
        "3) 내가 선택한 방향으로 1 step씩 이동한 결과(경로)를 해석하시오.",
        height=120,
        placeholder="예: 처음에는 손실이 빠르게 감소했지만, 이후에는 감소 폭이 줄었다. 추천 방향과 비교했을 때... 등",
        key="ai_step2_reflection",
    )

    # ---- 유효성 검사 ----
    def _validate_inputs() -> tuple[bool, str]:
        if not (dE_da or "").strip():
            return False, "1) ∂E/∂a 입력이 비어 있습니다."
        if not (dE_db or "").strip():
            return False, "1) ∂E/∂b 입력이 비어 있습니다."
        if not (direction_desc or "").strip():
            return False, "2) 방향 성분 판단 서술이 비어 있습니다."
        if not (reflection or "").strip():
            return False, "3) 이동 결과 해석이 비어 있습니다."
        return True, "OK"

    # 현재 경로 결과 요약
    path = s.get("path", [])
    start_a = float(s.get("start_a", path[0][0]))
    start_b = float(s.get("start_b", path[0][1]))
    final_a, final_b, final_e = path[-1]
    steps_used = max(0, len(path) - 1)

    payload = {
        "student_id": student_id,
        "loss_type": loss_spec.type,
        "loss_params": dict(loss_spec.params),
        "loss_latex": loss_latex,
        "learning_rate": float(LEARNING_RATE),
        "start_a": float(start_a),
        "start_b": float(start_b),
        "final_a": float(final_a),
        "final_b": float(final_b),
        "final_E": float(final_e),
        "steps_used": int(steps_used),
        "partials_input": f"∂E/∂a: {dE_da} / ∂E/∂b: {dE_db}",
        "direction_desc": direction_desc,
        "result_reflection": reflection,
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }

    backup_text = build_backup_text(payload)

    # 다운로드 버튼은 항상 표시(학생 UX 안정)
    st.download_button(
        label="📄 (다운로드) 2차시 백업 TXT",
        data=backup_text.encode("utf-8-sig"),
        file_name=f"인공지능_수행평가_2차시_{student_id}.txt",
        mime="text/plain; charset=utf-8",
        use_container_width=True,
    )

    # 버튼 3개(기존 UX 유지)
    cA, cB, cC = st.columns([1, 1, 1], gap="small")
    with cA:
        backup_make_clicked = st.button("⬇️ TXT 백업 만들기", use_container_width=True)
    with cB:
        save_clicked = st.button("✅ 제출/저장", use_container_width=True)
    with cC:
        go_next = st.button("➡️ 최종 보고서로 이동", use_container_width=True)

    # ---- 공통 검증(세 버튼 모두) ----
    if save_clicked or backup_make_clicked or go_next:
        ok, msg = _validate_inputs()
        if not ok:
            st.error(msg)
            st.stop()

    # ---- 백업 만들기 버튼: 세션에 payload 저장(선택: 보고서 자동채움/복구용) ----
    if backup_make_clicked:
        st.session_state[_BACKUP_STATE_KEY] = payload
        st.success("백업 내용을 준비했습니다. 위의 다운로드 버튼을 눌러 저장하세요.")

    # ---- 저장 버튼: 구글시트 기록 ----
    if save_clicked:
        try:
            # late import: 페이지 로딩 안정
            from assessment.google_sheets import append_ai_step2_row

            # ✅ 신형 컬럼이 있으면 신형으로
            try:
                append_ai_step2_row(
                    student_id=student_id,
                    loss_type=loss_spec.type,
                    loss_params=str(dict(loss_spec.params)),
                    start_a=float(start_a),
                    start_b=float(start_b),
                    learning_rate=float(LEARNING_RATE),
                    dE_da=str(dE_da).strip(),
                    dE_db=str(dE_db).strip(),
                    direction_desc=str(direction_desc).strip(),
                    result_reflection=str(reflection).strip(),
                    final_a=float(final_a),
                    final_b=float(final_b),
                    steps_used=int(steps_used),
                    final_E=float(final_e),
                )
            except TypeError:
                # ✅ 구형 시트(alpha/beta)만 받는 경우: quad(alpha=...)일 때만 의미 있게 저장
                alpha_fallback = float(loss_spec.params.get("alpha", 10.0)) if loss_spec.type == "quad" else 10.0
                append_ai_step2_row(
                    student_id=student_id,
                    alpha=float(alpha_fallback),
                    beta=float(1.0),
                    start_a=float(start_a),
                    start_b=float(start_b),
                    step_size=float(LEARNING_RATE),
                    dE_da=str(dE_da).strip(),
                    dE_db=str(dE_db).strip(),
                    direction_desc=str(direction_desc).strip(),
                    result_reflection=str(reflection).strip(),
                    final_a=float(final_a),
                    final_b=float(final_b),
                    steps_used=int(steps_used),
                    final_E=float(final_e),
                )

            set_save_status(True, "구글시트 저장 완료")
        except Exception as e:
            set_save_status(False, f"구글시트 저장 실패: {e}")
            st.stop()

        # 저장 상태가 바로 보이게
        st.rerun()

    # ---- 최종보고서 이동 ----
    if go_next:
        # (선택) 보고서 페이지에서 자동 채움에 활용 가능
        st.session_state[_BACKUP_STATE_KEY] = payload
        st.switch_page("assessment/ai_final_report.py")

    render_save_status()


if __name__ == "__main__":
    main()
