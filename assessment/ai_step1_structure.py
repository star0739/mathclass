# assessment/ai_step1_structure.py
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

TITLE = r"1차시: 구조(손실 지형) 관찰"

ALPHA = 10.0
BETA = 1.0

A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0

GRID_N = 121  # 고정 해상도(학생 선택 X) — 안정성 우선
DEFAULT_START_A = 2.2
DEFAULT_START_B = 2.2

COORD_STEPS = 18
STEP_SIZE = 0.15


def E(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ALPHA * (a**2) + BETA * (b**2)


def _partials(a: float, b: float) -> tuple[float, float]:
    # a방향 변화율, b방향 변화율 (용어는 페이지에서 언급하지 않음)
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
    한 번에 한 변수만 줄이는 이동(번갈아): 지그재그 관찰용
    - k 짝수: a만 이동
    - k 홀수: b만 이동
    """
    a, b = float(a0), float(b0)
    pts = [(a, b, float(E(np.array(a), np.array(b))))]

    for k in range(steps):
        da, db = _partials(a, b)
        if k % 2 == 0:
            a = a - step_size * da
        else:
            b = b - step_size * db

        a = float(np.clip(a, A_MIN, A_MAX))
        b = float(np.clip(b, B_MIN, B_MAX))
        pts.append((a, b, float(E(np.array(a), np.array(b)))))

    return np.array(pts, dtype=float)


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


def main():
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)

    init_assessment_session()
    student_id = require_student_id()

    st.markdown(
        rf"""
손실함수의 그래프 $z=E(a,b)$는 하나의 곡면이며,

이를 직관적으로 **손실 지형(loss landscape)**이라고 부릅니다.

다음 손실 지형을 관찰하며 손실함수 값이 최소가 되는 지점과 그 방향적 특징을 분석해 봅시다.
$$
E(a,b) = \alpha a^2 + \beta b^2
$$

$$
\alpha = {ALPHA:g}, \quad \beta = {BETA:g}
$$

관찰 포인트:
- 전역 최소점(global minimum)의 위치
- 좌표축에 대한 대칭성
- 방향에 따른 기울기 크기(가파름)
- 한 변수만 줄이는 이동(좌표축 방향 이동)에서 나타나는 경로의 특징
"""
    )

    # -------------------------
    # 상단: 좌(①②) / 우(시각화)
    # -------------------------
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("① 현재 위치 선택")

        a0 = st.slider("a 값", min_value=A_MIN, max_value=A_MAX, value=DEFAULT_START_A, step=0.05)
        b0 = st.slider("b 값", min_value=B_MIN, max_value=B_MAX, value=DEFAULT_START_B, step=0.05)

        e0 = float(E(np.array(a0), np.array(b0)))
        st.metric("현재 손실", f"{e0:.6f}")

        st.markdown(
            r"""
참고(해석의 기준):
- 등고선 간격이 **더 촘촘한 방향**일수록, 같은 거리 이동에서 손실 변화가 더 큽니다.
"""
        )

        st.divider()
        st.subheader("② 한 변수만 줄이는 이동 실험")

        st.markdown(
            r"""
아래 버튼은 **a만, b만 번갈아** 이동하는 경로를 그립니다.  
이 경로의 특징을 ③에서 서술하세요.
"""
        )

        run_coord = st.button("▶ 좌표축만 번갈아 이동(지그재그 관찰)", type="primary")

    with right:
        st.subheader("손실 지형 시각화")

        A, B, Z = build_grid(A_MIN, A_MAX, B_MIN, B_MAX, GRID_N)

        path = coord_descent_path(a0, b0, steps=COORD_STEPS, step_size=STEP_SIZE) if run_coord else None

        tab1, tab2 = st.tabs(["2D 등고선", "3D 손실곡면"])

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
                fig, ax = plt.subplots()
                cs = ax.contour(A, B, Z, levels=18)
                ax.clabel(cs, inline=True, fontsize=8)
                ax.scatter([a0], [b0], s=60)
                ax.text(a0, b0, "현재", fontsize=10)
                if path is not None:
                    ax.plot(path[:, 0], path[:, 1], marker="o")
                ax.set_xlabel("a")
                ax.set_ylabel("b")
                ax.set_title("Contour of E(a,b)")
                st.pyplot(fig, clear_figure=True)

            st.markdown(
                r"""
$$
\text{등고선 간격이 촘촘한 방향 } \Rightarrow \text{ 더 가파른 방향}
$$
"""
            )

        with tab2:
            if PLOTLY_AVAILABLE:
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
                    scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="E(a,b)"),
                )
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("3D 표면은 Plotly가 필요합니다. (현재 환경에서는 2D 등고선으로 충분합니다.)")

            st.markdown(
                r"""
$$
E(a,b) \text{ 는 그릇(bowl) 모양의 손실 지형으로 해석할 수 있습니다.}
$$
"""
            )

    # -------------------------
    # 하단(전체 폭): ③ 서술 + 백업 + 저장/이동 + 저장상태
    # -------------------------
    st.divider()
    st.subheader("③ 관찰 기록 서술")

    obs_shape = st.text_area(
        "1) 전역 최소점의 위치와 손실 지형의 전체적인 형태를 함께 설명하시오.",
        height=90,
        placeholder="예: 전역 최소점의 좌표, 그 주변에서 함숫값이 어떻게 변하는지, 손실 지형의 전체적인 형태 서",
        key="ai_step1_obs_shape",
    )

    obs_sensitivity = st.text_area(
        "2) 같은 거리만큼 이동했을 때 손실이 더 크게 변하는 방향은 어느 쪽인가? 등고선의 모양 또는 간격을 근거로 설명하시오.",
        height=110,
        placeholder="예: 어느 방향이 더 가파른지, 등고선 간격이나 모양이 어떤지 서술",
        key="ai_step1_obs_sensitivity",
    )

    obs_zigzag = st.text_area(
        "3) a와 b를 한 번에 하나씩만 줄이는 방식으로 이동했을 때 경로는 어떤 특징을 보이는가? 그 이유를 설명하시오.",
        height=120,
        placeholder="예: 경로의 모양 설명, 그렇게 되는 수학적 이유 서",
        key="ai_step1_obs_zigzag",
    )

    st.caption("※ 구체적인 좌표, 방향, 등고선 근거를 포함하여 작성하세요.")
    
    payload_for_backup = {
        "student_id": student_id,
        "obs_shape": obs_shape,
        "obs_sensitivity": obs_sensitivity,
        "obs_zigzag": obs_zigzag,
    }
    backup_text = build_backup_text(payload_for_backup)

    cA, cB = st.columns([1, 1], gap="large")
    with cA:
        st.download_button(
            label="📄 (다운로드) 1차시 백업 TXT",
            data=backup_text.encode("utf-8-sig"),
            file_name=f"인공지능_수행평가_1차시_{student_id}.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
        )

    with cB:
        # 저장 / 이동 버튼을 같은 줄에 (미적분 수행평가 UI와 유사한 느낌)
        btn1, btn2 = st.columns(2, gap="small")
        with btn1:
            save_clicked = st.button("✅ 제출/저장", use_container_width=True)
        with btn2:
            go_next = st.button("➡️ 2차시로 이동", use_container_width=True)

    # ✅ 저장 상태 알림: 버튼 바로 아래로 이동
    render_save_status()

    # -------------------------
    # 저장/이동 처리
    # -------------------------
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

        # 세션 저장(2차시에 필요하면 참조)
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
            set_save_status(False, f"구글시트 저장 실패: {e}")

        if go_next:
            st.switch_page("assessment/ai_step2_path.py")
        else:
            st.rerun()


if __name__ == "__main__":
    main()
