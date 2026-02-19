# assessment/ai_step1_structure.py
# ------------------------------------------------------------
# 인공지능수학 수행평가 - 1차시: 비등방 이차함수의 구조(지형) 관찰
# 목표:
# - E(a,b)=αa^2+βb^2 (α≠β) 손실곡면과 등고선을 연결해 해석
# - 방향에 따른 민감도(가파름/완만함)를 관찰하고 근거를 서술
# - "한 변수만" 줄이는 이동이 왜 비효율(지그재그)인지 관찰
# ------------------------------------------------------------

from __future__ import annotations

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

# ✅ (추후) assessment/google_sheets.py에 추가 예정
# from assessment.google_sheets import append_ai_step1_row


# -----------------------------
# 활동 설정(고정)
# -----------------------------
TITLE = "인공지능수학 수행평가 (1차시) — 구조(손실 지형) 관찰"
ALPHA = 10.0
BETA = 1.0

A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0

GRID_N = 121  # 고정 해상도(학생 선택 X) — 메모리/렌더 안전
DEFAULT_START_A = 2.2
DEFAULT_START_B = 2.2

# 좌표축 방향 이동(“한 변수만”) 실험 파라미터
COORD_STEPS = 18
STEP_SIZE = 0.15  # 너무 크면 튐, 너무 작으면 변화가 안 보임(고정)


# -----------------------------
# 계산 유틸
# -----------------------------
def E(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ALPHA * (a ** 2) + BETA * (b ** 2)


def _partials(a: float, b: float) -> tuple[float, float]:
    """
    (용어는 피하고) 현재 위치에서 a만 변할 때, b만 변할 때의 변화율(기울기)을 계산.
    E(a,b)=αa^2+βb^2 이므로:
    a방향 변화율: 2αa
    b방향 변화율: 2βb
    """
    da = 2.0 * ALPHA * a
    db = 2.0 * BETA * b
    return da, db


@st.cache_data(show_spinner=False)
def build_grid(a_min: float, a_max: float, b_min: float, b_max: float, n: int):
    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    Z = E(A, B)
    return A, B, Z


def coord_descent_path(a0: float, b0: float, steps: int, step_size: float) -> np.ndarray:
    """
    '한 번에 한 변수만' 줄이는 이동을 번갈아 수행(지그재그 유도).
    - 홀수 스텝: a만 이동
    - 짝수 스텝: b만 이동
    """
    a, b = float(a0), float(b0)
    pts = [(a, b, float(E(np.array(a), np.array(b))))]

    for k in range(steps):
        da, db = _partials(a, b)
        if k % 2 == 0:
            # a만 이동
            a = a - step_size * da
        else:
            # b만 이동
            b = b - step_size * db

        # 범위를 너무 벗어나면 잘라서 시각화 안정
        a = float(np.clip(a, A_MIN, A_MAX))
        b = float(np.clip(b, B_MIN, B_MAX))
        pts.append((a, b, float(E(np.array(a), np.array(b)))))

    return np.array(pts, dtype=float)


# -----------------------------
# TXT 백업
# -----------------------------
def build_backup_text(payload: dict) -> str:
    lines: list[str] = []
    lines.append("인공지능수학 수행평가 (1차시) 백업")
    lines.append("=" * 46)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")
    lines.append("[함수 설정]")
    lines.append(f"- E(a,b) = {ALPHA:g} a^2 + {BETA:g} b^2")
    lines.append(f"- 관찰 범위: a∈[{A_MIN:g},{A_MAX:g}], b∈[{B_MIN:g},{B_MAX:g}]")
    lines.append("")
    lines.append("[학생 입력(서술)]")
    lines.append("1) 전체 형태/최소점 관찰:")
    lines.append(payload.get("obs_shape", "").strip())
    lines.append("")
    lines.append("2) 민감도 큰 방향 + 근거(등고선/단면 등):")
    lines.append(payload.get("obs_sensitivity", "").strip())
    lines.append("")
    lines.append("3) 한 변수만 줄이는 이동(지그재그) 관찰 + 이유:")
    lines.append(payload.get("obs_zigzag", "").strip())
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

    # 상단 저장 상태(공통)
    render_save_status()

    st.markdown(
        """
이번 시간은 **손실함수 \(E(a,b)\)** 를 하나의 **지형(landscape)** 으로 보고,
- **최소점**
- **대칭성**
- **방향에 따른 가파름(민감도)**
을 관찰·기록합니다.

> 오늘은 용어(그래디언트)는 쓰지 않습니다. 대신 **등고선 간격**과 **한 변수만 움직였을 때 변화량**으로 근거를 제시합니다.
"""
    )

    # ---------
    # 좌측: 조작 / 우측: 시각화
    # ---------
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("① 현재 위치 선택")

        a0 = st.slider("a 값", min_value=A_MIN, max_value=A_MAX, value=DEFAULT_START_A, step=0.05)
        b0 = st.slider("b 값", min_value=B_MIN, max_value=B_MAX, value=DEFAULT_START_B, step=0.05)

        e0 = float(E(np.array(a0), np.array(b0)))
        st.metric("현재 손실 E(a,b)", f"{e0:.4f}")

        st.divider()
        st.subheader("② '한 변수만' 줄이는 이동 실험")

        st.caption("버튼을 누르면 **a만, b만 번갈아** 이동한 경로가 등고선에 표시됩니다(지그재그 관찰).")

        run_coord = st.button("▶ 좌표축만 번갈아 이동(지그재그 관찰)", type="primary")

        st.divider()
        st.subheader("③ 관찰 기록(서술 3개)")

        obs_shape = st.text_area(
            "1) 손실곡면의 전체 형태/대칭성/최소점 위치를 한 문장으로 설명",
            height=90,
            placeholder="예: (0,0) 부근이 가장 낮고, a방향으로 더 가파르게 솟아오른다 …",
            key="ai_step1_obs_shape",
        )

        obs_sensitivity = st.text_area(
            "2) 더 가파른(민감도 큰) 방향은 어느 쪽인가? 근거(등고선 간격 등) 포함",
            height=110,
            placeholder="예: 등고선이 a방향으로 더 촘촘하므로 a가 변할 때 손실 변화가 더 크다 …",
            key="ai_step1_obs_sensitivity",
        )

        obs_zigzag = st.text_area(
            "3) '한 변수만' 줄이는 경로의 특징과, 그렇게 되는 이유",
            height=120,
            placeholder="예: a만 줄이다가 b만 줄이면 방향이 번갈아 꺾이며 지그재그가 나타난다 …",
            key="ai_step1_obs_zigzag",
        )

        st.divider()

        # 백업 TXT(항상 제공)
        payload_for_backup = {
            "student_id": student_id,
            "obs_shape": obs_shape,
            "obs_sensitivity": obs_sensitivity,
            "obs_zigzag": obs_zigzag,
        }
        backup_text = build_backup_text(payload_for_backup)
        st.download_button(
            label="📄 (다운로드) 1차시 백업 TXT",
            data=backup_text.encode("utf-8-sig"),
            file_name=f"인공지능_수행평가_1차시_{student_id}.txt",
            mime="text/plain; charset=utf-8",
        )

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            save_clicked = st.button("✅ 제출/저장", use_container_width=True)
        with c2:
            go_next = st.button("➡️ 2차시로 이동", use_container_width=True)

    # ---------
    # 시각화 패널
    # ---------
    with right:
        st.subheader("손실 지형 시각화")

        A, B, Z = build_grid(A_MIN, A_MAX, B_MIN, B_MAX, GRID_N)

        # 경로 계산(버튼 트리거 시)
        path = None
        if run_coord:
            path = coord_descent_path(a0, b0, steps=COORD_STEPS, step_size=STEP_SIZE)

        tab1, tab2 = st.tabs(["2D 등고선(핵심)", "3D 손실곡면(형태 보기)"])

        with tab1:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(
                    go.Contour(
                        x=np.linspace(A_MIN, A_MAX, GRID_N),
                        y=np.linspace(B_MIN, B_MAX, GRID_N),
                        z=Z,
                        contours=dict(showlabels=False),
                        line=dict(width=1),
                    )
                )
                # 현재 위치
                fig.add_trace(
                    go.Scatter(
                        x=[a0],
                        y=[b0],
                        mode="markers+text",
                        text=["현재"],
                        textposition="top center",
                        marker=dict(size=10),
                        name="현재 위치",
                    )
                )
                # 경로(있으면)
                if path is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=path[:, 0],
                            y=path[:, 1],
                            mode="lines+markers",
                            marker=dict(size=5),
                            name="축만 번갈아 이동 경로",
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
                # matplotlib fallback
                fig, ax = plt.subplots()
                cs = ax.contour(A, B, Z, levels=18)
                ax.clabel(cs, inline=True, fontsize=8)
                ax.scatter([a0], [b0], s=60)
                ax.text(a0, b0, "현재", fontsize=10)
                if path is not None:
                    ax.plot(path[:, 0], path[:, 1], marker="o")
                ax.set_xlabel("a")
                ax.set_ylabel("b")
                ax.set_title("Contour (E(a,b))")
                st.pyplot(fig, clear_figure=True)

            st.caption("등고선이 **더 촘촘한 방향**일수록, 같은 거리 이동에서 손실 변화가 더 큽니다.")

        with tab2:
            if PLOTLY_AVAILABLE:
                # 3D는 형태 파악용(과도한 상호작용/재계산 방지 위해 그리드 고정)
                surf = go.Surface(x=A, y=B, z=Z, showscale=False, opacity=0.95)
                fig3d = go.Figure(data=[surf])
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[a0],
                        y=[b0],
                        z=[e0],
                        mode="markers+text",
                        text=["현재"],
                        textposition="top center",
                        marker=dict(size=5),
                        name="현재 위치",
                    )
                )
                fig3d.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=10, b=10),
                    scene=dict(
                        xaxis_title="a",
                        yaxis_title="b",
                        zaxis_title="E(a,b)",
                    ),
                )
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("3D 표면은 Plotly가 필요합니다. (현재 환경에서는 2D 등고선으로 충분합니다.)")

            st.caption("3D는 ‘전체 형태’를 보는 용도입니다. 실제 방향 추론은 2D 등고선이 핵심입니다.")

    # ---------
    # 저장/이동 처리
    # ---------
    def _validate_inputs() -> tuple[bool, str]:
        if not obs_shape.strip():
            return False, "서술 1) 전체 형태/최소점 관찰을 입력하세요."
        if not obs_sensitivity.strip():
            return False, "서술 2) 민감도 방향과 근거를 입력하세요."
        if not obs_zigzag.strip():
            return False, "서술 3) 지그재그 관찰과 이유를 입력하세요."
        return True, "OK"

    if save_clicked or go_next:
        ok, msg = _validate_inputs()
        if not ok:
            st.error(msg)
            st.stop()

        # 세션 저장(2차시에 필요하면 참조 가능)
        st.session_state["ai_step1_structure"] = {
            "student_id": student_id,
            "alpha": ALPHA,
            "beta": BETA,
            "range": {"a": [A_MIN, A_MAX], "b": [B_MIN, B_MAX]},
            "start_point": {"a": float(a0), "b": float(b0)},
            "obs_shape": obs_shape.strip(),
            "obs_sensitivity": obs_sensitivity.strip(),
            "obs_zigzag": obs_zigzag.strip(),
            "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }

        # Google Sheet 저장(인공지능수학 전용)
        try:
            from assessment.google_sheets import append_ai_step1_row  # late import

            append_ai_step1_row(
                student_id=student_id,
                alpha=ALPHA,
                beta=BETA,
                a0=float(a0),
                b0=float(b0),
                obs_shape=obs_shape.strip(),
                obs_sensitivity=obs_sensitivity.strip(),
                obs_zigzag=obs_zigzag.strip(),
            )
            set_save_status(True, "구글시트 저장 완료")
        except Exception as e:
            # 시트 함수/권한 문제 등으로 실패해도 페이지는 동작하게
            set_save_status(False, f"구글시트 저장 실패: {e}")

        if go_next:
            st.switch_page("assessment/ai_step2_path.py")
        else:
            st.rerun()


if __name__ == "__main__":
    main()
