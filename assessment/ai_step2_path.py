# assessment/ai_step2_path.py
# ------------------------------------------------------------
# 인공지능수학 수행평가 - 2차시: 경로(path) 탐구
# 목표:
# - 임의 시작점에서 "손실을 줄이는 방향"을 스스로 추론
# - 2D 등고선 위에서 1-step 이동을 반복하며 경로를 관찰
# - (용어 언급 없이) 추천 방향과 비교해 자신의 추론을 점검(선택)
# - 최소 서술(2~3개) + 구글시트 저장 + TXT 백업
# ------------------------------------------------------------

from __future__ import annotations

import math
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

# (추가한 함수)
# from assessment.google_sheets import append_ai_step2_row


# -----------------------------
# 기본 설정(고정)
# -----------------------------
TITLE = "2차시: 경로(Path) 탐색"

DEFAULT_ALPHA = 10.0
DEFAULT_BETA = 1.0

A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0

GRID_N = 121  # 고정 해상도(학생 선택 X)

# 이동 길이(학습률 역할) — 수행평가 조건 통제를 위해 고정 권장
STEP_SIZE = 0.18

# 경로 저장 상한(메모리/렌더 안전)
MAX_PATH_POINTS = 250

# 시작점 후보(교실 운영 안정)
PRESET_STARTS = [
    (2.2, 2.2),
    (-2.2, 2.0),
    (2.5, -1.8),
    (-2.4, -2.1),
]

_BACKUP_STATE_KEY = "ai_step2_backup_payload"


# -----------------------------
# 계산 유틸
# -----------------------------
def E(alpha: float, beta: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return alpha * (a ** 2) + beta * (b ** 2)


def _partials(alpha: float, beta: float, a: float, b: float) -> tuple[float, float]:
    # 용어는 쓰지 않되, 현재 위치에서의 a방향/ b방향 변화율
    return 2.0 * alpha * a, 2.0 * beta * b


@st.cache_data(show_spinner=False)
def build_grid(alpha: float, beta: float, a_min: float, a_max: float, b_min: float, b_max: float, n: int):
    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    Z = E(alpha, beta, A, B)
    return A, B, Z


def _unit_from_angle_deg(theta_deg: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return math.cos(t), math.sin(t)


def _clip(a: float, b: float) -> tuple[float, float]:
    return float(np.clip(a, A_MIN, A_MAX)), float(np.clip(b, B_MIN, B_MAX))


def recommended_direction(alpha: float, beta: float, a: float, b: float) -> tuple[float, float]:
    """
    (표현상) '현재 위치에서 손실을 가장 빨리 줄이는 방향'을 계산해 제공(선택 힌트).
    실제론 -[a방향 변화율, b방향 변화율]의 방향.
    """
    da, db = _partials(alpha, beta, a, b)
    vx, vy = -da, -db
    norm = math.hypot(vx, vy)
    if norm < 1e-12:
        return 0.0, 0.0
    return vx / norm, vy / norm


def coord_axis_path(alpha: float, beta: float, a0: float, b0: float, steps: int, step_size: float) -> list[tuple[float, float, float]]:
    """
    1차시의 '좌표축 방향 이동(지그재그)'과 같은 비교 경로
    - k 짝수: a만 이동
    - k 홀수: b만 이동
    """
    a, b = float(a0), float(b0)
    a, b = _clip(a, b)
    e0 = float(E(alpha, beta, np.array(a), np.array(b)))
    pts: list[tuple[float, float, float]] = [(a, b, e0)]

    for k in range(steps):
        da, db = _partials(alpha, beta, a, b)
        if k % 2 == 0:
            a = a - step_size * da
        else:
            b = b - step_size * db
        a, b = _clip(a, b)
        e = float(E(alpha, beta, np.array(a), np.array(b)))
        pts.append((a, b, e))

    return pts


# -----------------------------
# 상태 관리
# -----------------------------
def _get_state() -> dict:
    return st.session_state.get("ai_step2_path", {})


def _set_state(d: dict) -> None:
    st.session_state["ai_step2_path"] = d


def _init_state_if_needed(student_id: str) -> dict:
    s = _get_state()
    if s:
        return s

    # 1차시에서 저장된 시작점이 있으면 그걸 기본으로 사용(있어도 강제는 아님)
    step1 = st.session_state.get("ai_step1_structure", {})
    if isinstance(step1, dict) and step1.get("student_id") == student_id:
        alpha = float(step1.get("alpha", DEFAULT_ALPHA))
        beta = float(step1.get("beta", DEFAULT_BETA))
        a0 = float(step1.get("start_point", {}).get("a", PRESET_STARTS[0][0]))
        b0 = float(step1.get("start_point", {}).get("b", PRESET_STARTS[0][1]))
    else:
        alpha = DEFAULT_ALPHA
        beta = DEFAULT_BETA
        a0, b0 = PRESET_STARTS[0]

    a0, b0 = _clip(a0, b0)
    e0 = float(E(alpha, beta, np.array(a0), np.array(b0)))

    s = {
        "student_id": student_id,
        "alpha": alpha,
        "beta": beta,
        "start_a": a0,
        "start_b": b0,
        "step_size": STEP_SIZE,
        "theta_deg": 225.0,  # 기본(대각선 아래쪽)
        "path": [(a0, b0, e0)],
        "last_delta": None,  # (전 step) 손실 변화량
        "hint_on": False,
        "saved_at": None,
    }
    _set_state(s)
    return s


def _append_point(s: dict, a: float, b: float) -> None:
    alpha = float(s["alpha"])
    beta = float(s["beta"])
    e = float(E(alpha, beta, np.array(a), np.array(b)))
    path = list(s.get("path", []))
    path.append((float(a), float(b), float(e)))
    if len(path) > MAX_PATH_POINTS:
        path = path[-MAX_PATH_POINTS:]
    s["path"] = path


# -----------------------------
# TXT 백업
# -----------------------------
def build_backup_text(s: dict, direction_desc: str, direction_reason: str, reflection: str) -> str:
    alpha = float(s.get("alpha", DEFAULT_ALPHA))
    beta = float(s.get("beta", DEFAULT_BETA))
    start_a = float(s.get("start_a", 0.0))
    start_b = float(s.get("start_b", 0.0))
    step_size = float(s.get("step_size", STEP_SIZE))
    path = s.get("path", [])
    steps_used = max(0, len(path) - 1)
    final_a, final_b, final_e = path[-1] if path else (start_a, start_b, float(E(alpha, beta, np.array(start_a), np.array(start_b))))

    lines: list[str] = []
    lines.append("인공지능수학 수행평가 (2차시) 백업")
    lines.append("=" * 46)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {s.get('student_id','')}")
    lines.append("")
    lines.append("[함수/조건]")
    lines.append(f"- E(a,b) = {alpha:g} a^2 + {beta:g} b^2")
    lines.append(f"- 관찰 범위: a∈[{A_MIN:g},{A_MAX:g}], b∈[{B_MIN:g},{B_MAX:g}]")
    lines.append(f"- step_size = {step_size:g}")
    lines.append("")
    lines.append("[시작점/결과]")
    lines.append(f"- 시작점: ({start_a:.4f}, {start_b:.4f})")
    lines.append(f"- 최종점: ({final_a:.4f}, {final_b:.4f})")
    lines.append(f"- 사용 step 수: {steps_used}")
    lines.append(f"- 최종 손실 E: {final_e:.6f}")
    lines.append("")
    lines.append("[학생 입력(서술)]")
    lines.append("1) 내가 선택한 방향(설명):")
    lines.append((direction_desc or "").strip())
    lines.append("")
    lines.append("2) 그 방향을 선택한 기준(내 규칙/판단):")
    lines.append((direction_reason or "").strip())
    lines.append("")
    lines.append("3) 실행 결과에 대한 해석(일치/불일치 + 이유):")
    lines.append((reflection or "").strip())
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# 메인
# -----------------------------
def main():
    st.set_page_config(page_title=TITLE, layout="wide")

    init_assessment_session()
    student_id = require_student_id()

    st.title(TITLE)

    s = _init_state_if_needed(student_id)

    alpha = float(s["alpha"])
    beta = float(s["beta"])

    st.markdown(
        """
이번 시간은 **등고선(2D)** 을 중심으로, 시작점에서 **손실을 줄이는 방향**을 스스로 추론해 봅니다.

- 먼저 **내가 생각한 방향**으로 한 번 이동해 보고,
- 필요하면 **추천 방향(힌트)** 과 비교해 보세요.
"""
    )

    # -------------------------
    # 상단: 좌(①②) / 우(시각화)
    # -------------------------
    left, right = st.columns([1, 2], gap="large")

    # ------------------ 좌측: ① 시작점 + ② 이동 ------------------
    with left:
        st.subheader("① 시작점 설정")

        preset_labels = [f"프리셋 {i+1}: ({a:g}, {b:g})" for i, (a, b) in enumerate(PRESET_STARTS)]
        preset_idx = st.selectbox(
            "시작점 선택",
            options=list(range(len(PRESET_STARTS))),
            format_func=lambda i: preset_labels[i],
            key="ai_step2_preset_idx",
        )

        c1, c2 = st.columns(2)
        with c1:
            apply_preset = st.button("적용", use_container_width=True)
        with c2:
            reset_path = st.button("경로 초기화", use_container_width=True)

        if apply_preset:
            a0, b0 = PRESET_STARTS[int(preset_idx)]
            a0, b0 = _clip(a0, b0)
            s["start_a"], s["start_b"] = a0, b0
            s["path"] = [(a0, b0, float(E(alpha, beta, np.array(a0), np.array(b0))))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        if reset_path:
            a0, b0 = float(s.get("start_a", PRESET_STARTS[0][0])), float(s.get("start_b", PRESET_STARTS[0][1]))
            a0, b0 = _clip(a0, b0)
            s["path"] = [(a0, b0, float(E(alpha, beta, np.array(a0), np.array(b0))))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        st.divider()
        st.subheader("② 방향 선택 & 1 step 이동")

        theta = st.slider("내가 고른 방향(각도, 도)", min_value=0.0, max_value=360.0, value=float(s.get("theta_deg", 225.0)), step=1.0)
        s["theta_deg"] = float(theta)
        _set_state(s)

        path = s.get("path", [])
        cur_a, cur_b, cur_e = path[-1]
        st.metric("현재 위치", f"({cur_a:.3f}, {cur_b:.3f})")
        st.metric("현재 손실 E", f"{cur_e:.6f}")

        c3, c4 = st.columns(2)
        with c3:
            step_move = st.button("▶ 내가 고른 방향으로 1 step", type="primary", use_container_width=True)
        with c4:
            step_reco = st.button("★ 추천 방향으로 1 step", use_container_width=True)

        if step_move or step_reco:
            if step_reco:
                ux, uy = recommended_direction(alpha, beta, cur_a, cur_b)
            else:
                ux, uy = _unit_from_angle_deg(theta)

            na = cur_a + STEP_SIZE * ux
            nb = cur_b + STEP_SIZE * uy
            na, nb = _clip(na, nb)

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

    # ------------------ 우측: 시각화 ------------------
    with right:
        st.subheader("등고선 위 경로 관찰(핵심)")

        A, B, Z = build_grid(alpha, beta, A_MIN, A_MAX, B_MIN, B_MAX, GRID_N)

        path = s.get("path", [])
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        cur_a, cur_b, cur_e = path[-1]
        reco_vx, reco_vy = recommended_direction(alpha, beta, cur_a, cur_b)
        ux, uy = _unit_from_angle_deg(float(s.get("theta_deg", 0.0)))

        arrow_len = 0.55

        show_axis_compare = st.checkbox("좌표축 방향 이동(지그재그) 경로도 함께 보기", value=False, key="ai_step2_show_axis_compare")

        axis_path = None
        if show_axis_compare:
            steps_for_compare = max(0, len(path) - 1)
            axis_path = coord_axis_path(alpha, beta, float(s.get("start_a", cur_a)), float(s.get("start_b", cur_b)), steps_for_compare, STEP_SIZE)

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

            if len(xs) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        marker=dict(size=6),
                        name="내 경로",
                    )
                )
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

            if axis_path is not None and len(axis_path) >= 2:
                ax_x = [p[0] for p in axis_path]
                ax_y = [p[1] for p in axis_path]
                fig.add_trace(
                    go.Scatter(
                        x=ax_x,
                        y=ax_y,
                        mode="lines",
                        line=dict(dash="dot", width=2),
                        name="좌표축 이동(비교)",
                    )
                )

            # 현재점 표시
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

            # 내가 고른 방향 화살표
            fig.add_trace(
                go.Scatter(
                    x=[cur_a, cur_a + arrow_len * ux],
                    y=[cur_b, cur_b + arrow_len * uy],
                    mode="lines",
                    name="내 방향",
                )
            )

            # 추천 방향 화살표
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
    # 하단(전체 폭): ③ 서술 + 백업 + 저장 + 저장상태
    # -------------------------
    st.divider()
    st.subheader("③ 서술(최소)")

    direction_desc = st.text_area(
        "1) 내가 선택한 방향(설명)",
        height=70,
        placeholder="예: b도 줄이되, a를 더 빨리 줄이는 성분이 큰 방향으로 이동하려고 대략 ↙ 방향을 선택했다.",
        key="ai_step2_direction_desc",
    )

    direction_reason = st.text_area(
        "2) 그 방향을 선택한 기준(내 규칙/판단)",
        height=100,
        placeholder="예: 현재 위치에서 a의 영향이 더 크다고 보고, a가 감소하는 성분이 큰 방향을 우선했다. (필요하면 등고선 근거도 함께)",
        key="ai_step2_direction_reason",
    )

    reflection = st.text_area(
        "3) 실행 결과 해석(일치/불일치 + 이유)",
        height=110,
        placeholder="예: 실제로 ΔE가 줄었다/늘었다. 내 판단과 결과가 일치/불일치한 이유는 …",
        key="ai_step2_reflection",
    )

    st.divider()

    # 버튼 레이아웃(1차시와 동일한 감각)
    col1, col2, col3 = st.columns([1, 1, 1.2], gap="small")
    with col1:
        save_clicked = st.button("✅ 제출/저장", use_container_width=True)
    with col2:
        backup_make_clicked = st.button("⬇️ TXT 백업 만들기", use_container_width=True)
    with col3:
        pass  # (2차시는 다음 차시 이동 버튼을 강제하지 않음)

    # 검증(기존 수준 유지)
    def _validate_step2() -> bool:
        if not direction_desc.strip():
            st.error("서술 1) 방향(설명)을 입력하세요.")
            return False
        if not direction_reason.strip():
            st.error("서술 2) 기준(내 규칙/판단)을 입력하세요.")
            return False
        if not reflection.strip():
            st.error("서술 3) 결과 해석을 입력하세요.")
            return False
        return True

    # 다운로드 버튼은 항상 렌더링(단, '백업 만들기'로 확정된 payload가 있으면 그걸 사용)
    saved_payload = st.session_state.get(_BACKUP_STATE_KEY) or None
    payload_for_download = saved_payload if isinstance(saved_payload, dict) and saved_payload.get("student_id") == student_id else None

    if payload_for_download is None:
        payload_for_download = {
            "s": dict(s),
            "direction_desc": direction_desc,
            "direction_reason": direction_reason,
            "reflection": reflection,
        }

    backup_text = build_backup_text(
        payload_for_download["s"],
        payload_for_download.get("direction_desc", ""),
        payload_for_download.get("direction_reason", ""),
        payload_for_download.get("reflection", ""),
    )

    st.download_button(
        label="📄 (다운로드) 2차시 백업 TXT",
        data=backup_text.encode("utf-8-sig"),
        file_name=f"인공지능_수행평가_2차시_{student_id}.txt",
        mime="text/plain; charset=utf-8",
        use_container_width=True,
    )

    if backup_make_clicked:
        if not _validate_step2():
            st.stop()
        st.session_state[_BACKUP_STATE_KEY] = {
            "student_id": student_id,
            "s": dict(s),
            "direction_desc": direction_desc.strip(),
            "direction_reason": direction_reason.strip(),
            "reflection": reflection.strip(),
            "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        st.rerun()

    if save_clicked:
        if not _validate_step2():
            st.stop()

        # 최종 상태 값
        path = s.get("path", [])
        start_a = float(s.get("start_a", path[0][0] if path else 0.0))
        start_b = float(s.get("start_b", path[0][1] if path else 0.0))
        final_a, final_b, final_e = path[-1] if path else (start_a, start_b, float(E(alpha, beta, np.array(start_a), np.array(start_b))))
        steps_used = max(0, len(path) - 1)

        # 세션 저장(추후 리포트/복구용)
        s["saved_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        _set_state(s)

        # 구글시트 저장(인공지능수학 전용)
        try:
            from assessment.google_sheets import append_ai_step2_row  # late import

            append_ai_step2_row(
                student_id=student_id,
                alpha=alpha,
                beta=beta,
                start_a=start_a,
                start_b=start_b,
                step_size=float(s.get("step_size", STEP_SIZE)),
                direction_desc=direction_desc.strip(),
                direction_reason=direction_reason.strip(),
                result_reflection=reflection.strip(),
                final_a=float(final_a),
                final_b=float(final_b),
                steps_used=int(steps_used),
                final_E=float(final_e),
            )
            set_save_status(True, "구글시트 저장 완료")
        except Exception as e:
            set_save_status(False, f"구글시트 저장 실패: {e}")

        st.rerun()

    # ✅ 저장 상태 알림: 버튼 아래(1차시와 같은 흐름)
    render_save_status()


if __name__ == "__main__":
    main()
