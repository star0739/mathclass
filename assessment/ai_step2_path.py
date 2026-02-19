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
TITLE = "인공지능수학 수행평가 (2차시) — 경로(Path) 탐구"

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
    lines.append("2) 그 방향을 선택한 근거(등고선 등):")
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
    st.title(TITLE)

    init_assessment_session()
    student_id = require_student_id()
    render_save_status()

    s = _init_state_if_needed(student_id)

    alpha = float(s["alpha"])
    beta = float(s["beta"])

    st.markdown(
        """
이번 시간은 **등고선(2D)** 을 중심으로, 시작점에서 **손실을 줄이는 방향**을 스스로 추론해 봅니다.

- 먼저 **내가 생각한 방향**으로 한 번 이동해 보고,
- 필요하면 **추천 방향(힌트)** 과 비교해 보세요.

> 오늘도 용어(그래디언트)는 쓰지 않습니다. 대신 **등고선 간격/모양**을 근거로 설명합니다.
"""
    )

    left, right = st.columns([1, 2], gap="large")

    # ------------------ 좌측: 조작/서술/저장 ------------------
    with left:
        st.subheader("① 시작점 설정")

        # 시작점 선택(프리셋/랜덤)
        preset_labels = [f"프리셋 {i+1}: ({a:g}, {b:g})" for i, (a, b) in enumerate(PRESET_STARTS)]
        preset_idx = st.selectbox("시작점 선택", options=list(range(len(PRESET_STARTS))), format_func=lambda i: preset_labels[i])

        c1, c2 = st.columns(2)
        with c1:
            apply_preset = st.button("적용", use_container_width=True)
        with c2:
            reset_path = st.button("경로 초기화", use_container_width=True)

        if apply_preset:
            a0, b0 = PRESET_STARTS[int(preset_idx)]
            a0, b0 = _clip(a0, b0)
            s["start_a"], s["start_b"] = a0, b0
            s["theta_deg"] = float(s.get("theta_deg", 225.0))
            s["path"] = [(a0, b0, float(E(alpha, beta, np.array(a0), np.array(b0))))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        if reset_path:
            a0, b0 = float(s["start_a"]), float(s["start_b"])
            s["path"] = [(a0, b0, float(E(alpha, beta, np.array(a0), np.array(b0))))]
            s["last_delta"] = None
            _set_state(s)
            st.rerun()

        st.divider()
        st.subheader("② 방향 선택 → 1 step 이동")

        theta = st.slider("방향(각도, 도)", min_value=0.0, max_value=360.0, value=float(s.get("theta_deg", 225.0)), step=1.0)
        s["theta_deg"] = float(theta)

        # 힌트(추천 방향) 토글
        hint_on = st.checkbox("힌트 보기(추천 방향 표시)", value=bool(s.get("hint_on", False)))
        s["hint_on"] = bool(hint_on)

        # 현재 위치/손실
        path = s.get("path", [])
        cur_a, cur_b, cur_e = path[-1]
        st.metric("현재 위치 (a,b)", f"({cur_a:.3f}, {cur_b:.3f})")
        st.metric("현재 손실 E", f"{cur_e:.6f}")

        # 이동 버튼
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

            # 1 step 이동
            na = cur_a + STEP_SIZE * ux
            nb = cur_b + STEP_SIZE * uy
            na, nb = _clip(na, nb)

            prev_e = float(cur_e)
            _append_point(s, na, nb)
            new_e = float(s["path"][-1][2])
            s["last_delta"] = float(new_e - prev_e)

            _set_state(s)
            st.rerun()

        # 전 step 피드백
        if s.get("last_delta") is not None:
            dE = float(s["last_delta"])
            if dE < 0:
                st.success(f"손실이 감소했습니다.  ΔE = {dE:.6f}")
            elif dE > 0:
                st.warning(f"손실이 증가했습니다.  ΔE = +{dE:.6f}")
            else:
                st.info("손실 변화가 거의 없습니다. (ΔE ≈ 0)")

        st.divider()
        st.subheader("③ 서술(최소)")

        direction_desc = st.text_area(
            "1) 내가 선택한 방향(설명)",
            height=70,
            placeholder="예: 등고선이 가장 촘촘한 쪽으로 향하도록 대략 남서쪽(↙) 방향을 선택했다.",
            key="ai_step2_direction_desc",
        )

        direction_reason = st.text_area(
            "2) 근거(등고선 모양/간격을 근거로)",
            height=100,
            placeholder="예: 현재 위치에서 등고선이 a방향으로 더 촘촘하므로, a를 빠르게 줄이는 성분이 큰 방향이 유리하다고 판단했다.",
            key="ai_step2_direction_reason",
        )

        reflection = st.text_area(
            "3) 실행 결과 해석(일치/불일치 + 이유)",
            height=110,
            placeholder="예: 예상대로 손실이 줄었지만, 경로가 직선이 되지 않고 조금씩 꺾인다. 이유는 …",
            key="ai_step2_reflection",
        )

        st.divider()

        # 백업 TXT
        backup_text = build_backup_text(s, direction_desc, direction_reason, reflection)
        st.download_button(
            label="📄 (다운로드) 2차시 백업 TXT",
            data=backup_text.encode("utf-8-sig"),
            file_name=f"인공지능_수행평가_2차시_{student_id}.txt",
            mime="text/plain; charset=utf-8",
        )

        st.divider()

        # 저장/제출
        save_clicked = st.button("✅ 제출/저장", use_container_width=True)

        if save_clicked:
            # 최소 검증(부담 최소화: 핵심 2개는 필수, 3개째도 필수로 두되 길이 제한은 안 둠)
            if not direction_desc.strip():
                st.error("서술 1) 방향(설명)을 입력하세요.")
                st.stop()
            if not direction_reason.strip():
                st.error("서술 2) 근거를 입력하세요.")
                st.stop()
            if not reflection.strip():
                st.error("서술 3) 결과 해석을 입력하세요.")
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

    # ------------------ 우측: 시각화 ------------------
    with right:
        st.subheader("등고선 위 경로 관찰(핵심)")

        A, B, Z = build_grid(alpha, beta, A_MIN, A_MAX, B_MIN, B_MAX, GRID_N)

        path = s.get("path", [])
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        # 현재점/추천 방향(힌트) 벡터
        cur_a, cur_b, cur_e = path[-1]
        reco_vx, reco_vy = recommended_direction(alpha, beta, cur_a, cur_b)

        # 내가 고른 방향 벡터
        ux, uy = _unit_from_angle_deg(float(s.get("theta_deg", 0.0)))

        # 화살표 길이(시각용)
        arrow_len = 0.55

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
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        marker=dict(size=5),
                        name="이동 경로",
                    )
                )

            # 현재점
            fig.add_trace(
                go.Scatter(
                    x=[cur_a],
                    y=[cur_b],
                    mode="markers+text",
                    text=["현재"],
                    textposition="top center",
                    marker=dict(size=10),
                    name="현재 위치",
                )
            )

            # 내가 고른 방향(항상 표시)
            fig.add_annotation(
                x=cur_a + arrow_len * ux,
                y=cur_b + arrow_len * uy,
                ax=cur_a,
                ay=cur_b,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
                text="내 방향",
            )

            # 추천 방향(힌트)
            if bool(s.get("hint_on", False)) and (abs(reco_vx) + abs(reco_vy) > 0):
                fig.add_annotation(
                    x=cur_a + arrow_len * reco_vx,
                    y=cur_b + arrow_len * reco_vy,
                    ax=cur_a,
                    ay=cur_b,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=2,
                    text="추천",
                )

            fig.update_layout(
                height=560,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="a",
                yaxis_title="b",
                xaxis=dict(range=[A_MIN, A_MAX]),
                yaxis=dict(range=[B_MIN, B_MAX]),
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # matplotlib fallback
            fig, ax = plt.subplots()
            cs = ax.contour(A, B, Z, levels=18)
            ax.clabel(cs, inline=True, fontsize=8)

            if len(xs) >= 2:
                ax.plot(xs, ys, marker="o")

            ax.scatter([cur_a], [cur_b], s=60)
            ax.text(cur_a, cur_b, "현재")

            # 내 방향 화살표
            ax.annotate(
                "내 방향",
                xy=(cur_a + arrow_len * ux, cur_b + arrow_len * uy),
                xytext=(cur_a, cur_b),
                arrowprops=dict(arrowstyle="->", lw=2),
            )

            # 추천 방향 화살표
            if bool(s.get("hint_on", False)) and (abs(reco_vx) + abs(reco_vy) > 0):
                ax.annotate(
                    "추천",
                    xy=(cur_a + arrow_len * reco_vx, cur_b + arrow_len * reco_vy),
                    xytext=(cur_a, cur_b),
                    arrowprops=dict(arrowstyle="->", lw=2),
                )

            ax.set_xlim(A_MIN, A_MAX)
            ax.set_ylim(B_MIN, B_MAX)
            ax.set_xlabel("a")
            ax.set_ylabel("b")
            ax.set_title("Contour + Path")
            st.pyplot(fig, clear_figure=True)

        st.caption("팁: 등고선이 촘촘한 쪽으로 향하는 방향일수록, 손실이 더 빠르게 줄어드는 경향이 있습니다.")


if __name__ == "__main__":
    main()
