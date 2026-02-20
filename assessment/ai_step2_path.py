# assessment/ai_step2_path.py
from __future__ import annotations

import math
from datetime import datetime

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
    set_save_status,
    render_save_status,
)

TITLE = "2차시: 경로(손실을 줄이는 방향) 탐구"

# (1차시와 동일한 함수/범위 설정)
ALPHA = 10.0
BETA = 1.0
A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0

GRID_N = 121  # 고정 해상도(학생 선택 X) — 안정성 우선
STEP_SIZE = 0.18  # 1 step 이동 거리(고정)
MAX_PATH_POINTS = 250  # 렌더/메모리 안전 상한

PRESET_STARTS = [
    (2.2, 2.2),
    (-2.2, 2.0),
    (2.5, -1.8),
    (-2.4, -2.1),
]

_STATE_KEY = "ai_step2_path_state"
_BACKUP_STATE_KEY = "ai_step2_backup_payload"


def E(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ALPHA * (a**2) + BETA * (b**2)


def _partials(a: float, b: float) -> tuple[float, float]:
    # 편미분(용어는 활동에서 필요한 만큼만 노출)
    return 2.0 * ALPHA * a, 2.0 * BETA * b


@st.cache_data(show_spinner=False)
def build_grid(a_min: float, a_max: float, b_min: float, b_max: float, n: int):
    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    Z = E(A, B)
    return A, B, Z


def _clip(a: float, b: float) -> tuple[float, float]:
    return float(np.clip(a, A_MIN, A_MAX)), float(np.clip(b, B_MIN, B_MAX))


def _unit_from_angle_deg(theta_deg: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return math.cos(t), math.sin(t)


def recommended_direction(a: float, b: float) -> tuple[float, float]:
    """
    현재 점에서 손실을 줄이는(가장 빨리 줄이는) 방향(정규화)을 계산.
    - (∂E/∂a, ∂E/∂b)의 반대 방향을 사용.
    """
    da, db = _partials(a, b)
    vx, vy = -da, -db
    norm = math.hypot(vx, vy)
    if norm < 1e-12:
        return 0.0, 0.0
    return vx / norm, vy / norm


def coord_axis_path(a0: float, b0: float, steps: int, step_size: float) -> list[tuple[float, float, float]]:
    """
    비교용(1차시 방식): 좌표축 방향 이동(번갈아) 경로
    - k 짝수: a만 이동
    - k 홀수: b만 이동
    """
    a, b = _clip(a0, b0)
    pts: list[tuple[float, float, float]] = [(a, b, float(E(np.array(a), np.array(b))))]

    for k in range(steps):
        da, db = _partials(a, b)
        if k % 2 == 0:
            a = a - step_size * da
        else:
            b = b - step_size * db
        a, b = _clip(a, b)
        pts.append((a, b, float(E(np.array(a), np.array(b)))))

    return pts


def _get_state() -> dict:
    return st.session_state.get(_STATE_KEY, {})


def _set_state(d: dict) -> None:
    st.session_state[_STATE_KEY] = d


def _init_state(student_id: str) -> dict:
    s = _get_state()
    if isinstance(s, dict) and s.get("student_id") == student_id and "path" in s:
        return s

    # 1차시에서 시작점 저장된 경우 그걸 우선 사용
    step1 = st.session_state.get("ai_step1_structure", {})
    if isinstance(step1, dict) and step1.get("student_id") == student_id:
        a0 = float(step1.get("start_point", {}).get("a", PRESET_STARTS[0][0]))
        b0 = float(step1.get("start_point", {}).get("b", PRESET_STARTS[0][1]))
    else:
        a0, b0 = PRESET_STARTS[0]

    a0, b0 = _clip(a0, b0)
    e0 = float(E(np.array(a0), np.array(b0)))

    s = {
        "student_id": student_id,
        "start_a": a0,
        "start_b": b0,
        "theta_deg": 225.0,
        "last_delta": None,
        "path": [(a0, b0, e0)],
    }
    _set_state(s)
    return s


def _append_point(s: dict, a: float, b: float) -> None:
    a, b = _clip(a, b)
    e = float(E(np.array(a), np.array(b)))
    path = list(s.get("path", []))
    path.append((a, b, e))
    if len(path) > MAX_PATH_POINTS:
        path = path[-MAX_PATH_POINTS:]
    s["path"] = path


def build_backup_text(payload: dict) -> str:
    """
    payload 기대 키:
    - student_id
    - start_a, start_b
    - step_size
    - theta_deg
    - path_final_a, path_final_b, path_final_e, steps_used
    - dE_da, dE_db
    - direction_desc
    - result_reflection
    """
    lines: list[str] = []
    lines.append("인공지능수학 수행평가 (2차시) 백업")
    lines.append("=" * 46)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")
    lines.append("[함수/조건]")
    lines.append(f"- E(a,b) = {ALPHA:g} a^2 + {BETA:g} b^2")
    lines.append(f"- 관찰 범위: a∈[{A_MIN:g},{A_MAX:g}], b∈[{B_MIN:g},{B_MAX:g}]")
    lines.append(f"- step_size = {payload.get('step_size', STEP_SIZE)}")
    lines.append("")
    lines.append("[시작점/결과]")
    lines.append(f"- 시작점: ({payload.get('start_a', '')}, {payload.get('start_b', '')})")
    lines.append(f"- 최종점: ({payload.get('final_a', '')}, {payload.get('final_b', '')})")
    lines.append(f"- 사용 step 수: {payload.get('steps_used', '')}")
    lines.append(f"- 최종 손실 E: {payload.get('final_E', '')}")
    lines.append("")
    lines.append("[학생 입력(서술)]")
    lines.append("1) 편미분 계산:")
    lines.append(f"∂E/∂a = {payload.get('dE_da','')}".strip())
    lines.append(f"∂E/∂b = {payload.get('dE_db','')}".strip())
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

    st.title(TITLE)

    s = _init_state(student_id)

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
            s["path"] = [(a0, b0, float(E(np.array(a0), np.array(b0))))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        if reset_path:
            a0, b0 = float(s.get("start_a", PRESET_STARTS[0][0])), float(s.get("start_b", PRESET_STARTS[0][1]))
            a0, b0 = _clip(a0, b0)
            s["path"] = [(a0, b0, float(E(np.array(a0), np.array(b0))))]
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
        st.metric("현재 위치", f"({cur_a:.3f}, {cur_b:.3f})")
        st.metric("현재 손실 E", f"{cur_e:.6f}")

        b1, b2 = st.columns(2, gap="small")
        with b1:
            step_move = st.button("▶ 내가 고른 방향으로 1 step", type="primary", use_container_width=True)
        with b2:
            step_reco = st.button("★ 추천 방향으로 1 step", use_container_width=True)

        if step_move or step_reco:
            if step_reco:
                ux, uy = recommended_direction(cur_a, cur_b)
            else:
                ux, uy = _unit_from_angle_deg(theta)

            na = cur_a + STEP_SIZE * ux
            nb = cur_b + STEP_SIZE * uy

            prev_e = float(cur_e)
            _append_point(s, na, nb)
            new_e = float(s["path"][-1][2])
            s["last_delta"] = float(new_e - prev_e)
            _set_state(s)
            st.rerun()

        if s.get("last_delta") is not None:
            dE = float(s["last_delta"])
            if dE < 0:
                st.success(f"손실이 감소했습니다.  ΔE = {dE:.6f}")
            elif dE > 0:
                st.warning(f"손실이 증가했습니다.  ΔE = +{dE:.6f}")
            else:
                st.info("손실 변화가 거의 없습니다. (ΔE ≈ 0)")

    # -------------------------
    # 우측: 시각화
    # -------------------------
    with right:
        st.subheader("등고선 위 경로 관찰(핵심)")

        A, B, Z = build_grid(A_MIN, A_MAX, B_MIN, B_MAX, GRID_N)

        path = s.get("path", [])
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        cur_a, cur_b, _ = path[-1]
        reco_vx, reco_vy = recommended_direction(cur_a, cur_b)
        ux, uy = _unit_from_angle_deg(float(s.get("theta_deg", 0.0)))
        arrow_len = 0.55

        show_axis_compare = st.checkbox("좌표축 방향 이동(지그재그) 경로도 함께 보기", value=False, key="ai_step2_show_axis_compare")
        axis_path = None
        if show_axis_compare:
            steps_for_compare = max(0, len(path) - 1)
            axis_path = coord_axis_path(float(s.get("start_a", cur_a)), float(s.get("start_b", cur_b)), steps_for_compare, STEP_SIZE)

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

            # 내 경로
            if len(xs) >= 2:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", marker=dict(size=6), name="내 경로"))
            else:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+text",
                        text=["시작"],
                        textposition="top center",
                        marker=dict(size=10),
                        name="시작점",
                    )
                )

            # 비교 경로
            if axis_path is not None and len(axis_path) >= 2:
                ax_x = [p[0] for p in axis_path]
                ax_y = [p[1] for p in axis_path]
                fig.add_trace(go.Scatter(x=ax_x, y=ax_y, mode="lines", line=dict(dash="dot", width=2), name="좌표축 이동(비교)"))

            # 현재점
            fig.add_trace(
                go.Scatter(
                    x=[cur_a],
                    y=[cur_b],
                    mode="markers+text",
                    text=["현재"],
                    textposition="top center",
                    marker=dict(size=10),
                    name="현재",
                )
            )

            # 내 방향(화살표 대신 선)
            fig.add_trace(go.Scatter(x=[cur_a, cur_a + arrow_len * ux], y=[cur_b, cur_b + arrow_len * uy], mode="lines", name="내 방향"))
            # 추천 방향
            fig.add_trace(
                go.Scatter(x=[cur_a, cur_a + arrow_len * reco_vx], y=[cur_b, cur_b + arrow_len * reco_vy], mode="lines", name="추천 방향")
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

    st.markdown(
        r"""
1) 손실함수 $E(a,b)=10 a^2+ b^2$에 대해 시작점 $(a,b)$에서의 $\dfrac{\partial E}{\partial a}$, $\dfrac{\partial E}{\partial b}$를 구하시오.  
"""
    )

    colp1, colp2 = st.columns(2, gap="large")
    with colp1:
        st.markdown(r"$$\frac{\partial E}{\partial a} = $$")
        dE_da = st.text_input("편미분 식에 시작점 a좌표 값 대입", key="ai_step2_dE_da", label_visibility="collapsed")
    with colp2:
        st.markdown(r"$$\frac{\partial E}{\partial b} = $$")
        dE_db = st.text_input("편미분 식에 시작점 a좌표 값 대입", key="ai_step2_dE_db", label_visibility="collapsed")

    direction_desc = st.text_area(
        "2) 위에서 구한 두 값의 부호를 관찰하고, 손실을 줄이기 위해 각 변수를 어떤 방향(증가/감소)으로 변화시켜야 하는지 서술하시오.",
        height=100,
        placeholder="예: 각 값의 부호를 확인하여 a와 b의 값을 키울지 줄일지 결정하고, 그에 따라 내가 선택한 이동 방향을 서술",
        key="ai_step2_direction_desc",
    )

    reflection = st.text_area(
        "3) 1 step 이동 결과 손실값은 어떻게 변하였는가? 기울기의 부호를 이용한 나의 판단이 결과와 일치하였는지 그 이유를 설명하시오.",
        height=120,
        placeholder="예: 이동 후 손실의 변화와 그 원인을 자신의 판단과 연결하여 서술",
        key="ai_step2_reflection",
    )

    st.divider()

    def _validate_inputs() -> tuple[bool, str]:
        if not str(dE_da).strip():
            return False, "1) ∂E/∂a 값을 입력하세요."
        if not str(dE_db).strip():
            return False, "1) ∂E/∂b 값을 입력하세요."
        if not str(direction_desc).strip():
            return False, "2) 방향 성분/이동 방향 서술을 입력하세요."
        if not str(reflection).strip():
            return False, "3) 결과 해석을 입력하세요."
        return True, "OK"

    # (저장/백업 패턴) 백업 만들기 → 다운로드
    colA, colB = st.columns([1, 1], gap="large")
    with colA:
        backup_make_clicked = st.button("⬇️ TXT 백업 만들기", use_container_width=True)
        # 다운로드 데이터는 "확정된 payload" 우선
        saved_payload = st.session_state.get(_BACKUP_STATE_KEY) or None
        payload_for_download = saved_payload if isinstance(saved_payload, dict) and saved_payload.get("student_id") == student_id else None

        # 현재 상태 기반 payload
        path = s.get("path", [])
        start_a = float(s.get("start_a", path[0][0] if path else 0.0))
        start_b = float(s.get("start_b", path[0][1] if path else 0.0))
        final_a, final_b, final_e = path[-1] if path else (start_a, start_b, float(E(np.array(start_a), np.array(start_b))))
        steps_used = max(0, len(path) - 1)

        if payload_for_download is None:
            payload_for_download = {
                "student_id": student_id,
                "start_a": f"{start_a:.4f}",
                "start_b": f"{start_b:.4f}",
                "step_size": STEP_SIZE,
                "final_a": f"{float(final_a):.4f}",
                "final_b": f"{float(final_b):.4f}",
                "final_E": f"{float(final_e):.6f}",
                "steps_used": int(steps_used),
                "dE_da": str(dE_da).strip(),
                "dE_db": str(dE_db).strip(),
                "direction_desc": str(direction_desc).strip(),
                "result_reflection": str(reflection).strip(),
            }

        backup_text = build_backup_text(payload_for_download)

        st.download_button(
            label="📄 (다운로드) 2차시 백업 TXT",
            data=backup_text.encode("utf-8-sig"),
            file_name=f"인공지능_수행평가_2차시_{student_id}.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
        )

    with colB:
        save_clicked = st.button("✅ 제출/저장", use_container_width=True)

    # ✅ 저장 상태 알림: 버튼 바로 아래
    render_save_status()

    if backup_make_clicked:
        ok, msg = _validate_inputs()
        if not ok:
            st.error(msg)
            st.stop()

        path = s.get("path", [])
        start_a = float(s.get("start_a", path[0][0] if path else 0.0))
        start_b = float(s.get("start_b", path[0][1] if path else 0.0))
        final_a, final_b, final_e = path[-1] if path else (start_a, start_b, float(E(np.array(start_a), np.array(start_b))))
        steps_used = max(0, len(path) - 1)

        st.session_state[_BACKUP_STATE_KEY] = {
            "student_id": student_id,
            "start_a": f"{start_a:.4f}",
            "start_b": f"{start_b:.4f}",
            "step_size": STEP_SIZE,
            "final_a": f"{float(final_a):.4f}",
            "final_b": f"{float(final_b):.4f}",
            "final_E": f"{float(final_e):.6f}",
            "steps_used": int(steps_used),
            "dE_da": str(dE_da).strip(),
            "dE_db": str(dE_db).strip(),
            "direction_desc": str(direction_desc).strip(),
            "result_reflection": str(reflection).strip(),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.rerun()

    if save_clicked:
        ok, msg = _validate_inputs()
        if not ok:
            st.error(msg)
            st.stop()

        path = s.get("path", [])
        start_a = float(s.get("start_a", path[0][0] if path else 0.0))
        start_b = float(s.get("start_b", path[0][1] if path else 0.0))
        final_a, final_b, final_e = path[-1] if path else (start_a, start_b, float(E(np.array(start_a), np.array(start_b))))
        steps_used = max(0, len(path) - 1)

        # 세션 저장(보고서/복구용)
        st.session_state[_STATE_KEY] = s

        # 구글시트 저장(인공지능수학 전용)
        try:
            from assessment.google_sheets import append_ai_step2_row  # late import

            # direction_reason은 "편미분 계산식"을 함께 남기기 위해 사용(컬럼 의미는 내부 운영 기준)
            direction_reason = f"∂E/∂a = {str(dE_da).strip()}\n∂E/∂b = {str(dE_db).strip()}"

            append_ai_step2_row(
                student_id=student_id,
                alpha=ALPHA,
                beta=BETA,
                start_a=float(start_a),
                start_b=float(start_b),
                step_size=float(STEP_SIZE),
                dE_da=str(dE_da).strip(),
                dE_db=str(dE_db).strip(),
                direction_desc=str(direction_desc).strip(),
                direction_reason=direction_reason.strip(),
                result_reflection=str(reflection).strip(),
                final_a=float(final_a),
                final_b=float(final_b),
                steps_used=int(steps_used),
                final_E=float(final_e),
            )
            set_save_status(True, "구글시트 저장 완료")
        except Exception as e:
            set_save_status(False, f"구글시트 저장 실패: {e}")

        st.rerun()


if __name__ == "__main__":
    main()
