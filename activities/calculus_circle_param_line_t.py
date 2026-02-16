from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = r"단위원 매개화"


def _xy_from_t(t: float) -> tuple[float, float]:
    denom = 1.0 + t * t
    x = (1.0 - t * t) / denom
    y = (2.0 * t) / denom
    return float(x), float(y)


@st.cache_data
def _unit_circle(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0, 2 * np.pi, n)
    return np.cos(th), np.sin(th)


@st.cache_data
def _line_points(t: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.3, 1.3, n)
    ys = t * (xs + 1.0)
    return xs, ys


def render(show_title: bool = True, key_prefix: str = "circle_param_t") -> None:
    if show_title:
        st.title(TITLE)

    st.markdown(
        r"""
원 $x^2+y^2=1$ 위의 점을, 점 $(-1,0)$을 지나는 직선 $y=t(x+1)$과 원의 **다른 교점**으로 잡습니다.
$$
x=\frac{1-t^2}{1+t^2},\qquad y=\frac{2t}{1+t^2}\quad (t\in\mathbb{R})
$$
"""
    )

    # ----------------------------
    # 상태(누적 점, 마지막 t) 초기화
    # ----------------------------
    pts_key = f"{key_prefix}_pts"
    last_t_key = f"{key_prefix}_last_t"

    if pts_key not in st.session_state:
        st.session_state[pts_key] = np.empty((0, 2), dtype=float)  # (N,2)
    if last_t_key not in st.session_state:
        st.session_state[last_t_key] = None

    # ----------------------------
    # 입력 UI
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2, col3 = st.columns([1.4, 1.0, 1.0])

        with col1:
            t = st.slider(
                "기울기 t",
                min_value=-10.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key=f"{key_prefix}_t",
            )

        with col2:
            show_line = st.checkbox(
                "직선 $y=t(x+1)$ 표시",
                value=True,
                key=f"{key_prefix}_showline",
            )
            accumulate = st.checkbox(
                "점 누적 표시",
                value=True,
                key=f"{key_prefix}_accum",
            )

        with col3:
            if st.button("누적 점 초기화", key=f"{key_prefix}_clear"):
                st.session_state[pts_key] = np.empty((0, 2), dtype=float)
                st.session_state[last_t_key] = None
                st.rerun()


    # ----------------------------
    # 현재 점 계산 + 누적(가벼운 샘플링)
    # ----------------------------
    x, y = _xy_from_t(float(t))

    if accumulate:
        last_t = st.session_state[last_t_key]

        # ✅ t가 일정 이상 변할 때만 누적(과밀 방지)
        ADD_T_STEP = 0.05

        should_add = (last_t is None) or (abs(float(t) - float(last_t)) >= ADD_T_STEP)

        if should_add:
            pts = st.session_state[pts_key]
            new_pt = np.array([[x, y]], dtype=float)
            st.session_state[pts_key] = np.vstack([pts, new_pt])
            st.session_state[last_t_key] = float(t)

            # ✅ 누적 상한을 낮춰 브라우저/메모리 부담 줄이기
            MAX_POINTS = 1200
            if st.session_state[pts_key].shape[0] > MAX_POINTS:
                st.session_state[pts_key] = st.session_state[pts_key][-MAX_POINTS:, :]

    # ----------------------------
    # 수치 표시
    # ----------------------------
    c1, c2 = st.columns(2)
    c1.metric("x(t)", f"{x:.6f}")
    c2.metric("y(t)", f"{y:.6f}")

    # ----------------------------
    # 시각화
    # ----------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # 원(희미하게) - 캐시 사용
    cx, cy = _unit_circle(600)
    ax.plot(cx, cy, linewidth=1, alpha=0.25)

    # 축
    ax.axhline(0, linewidth=1, alpha=0.4)
    ax.axvline(0, linewidth=1, alpha=0.4)

    # (-1,0) 기준점
    ax.plot([-1], [0], marker="o", markersize=6)
    ax.text(-1, 0, "  (-1,0)", va="center")

    # 직선 표시(캐시 사용)
    if show_line:
        xs, ys = _line_points(float(t), 200)
        ax.plot(xs, ys, linewidth=1, alpha=0.6)

    # 누적 점(벡터화)
    pts = st.session_state[pts_key]
    if accumulate and pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.6)

    # 현재 점(강조)
    ax.scatter([x], [y], s=80)
    ax.text(x, y, "  P(t)", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    st.pyplot(fig)
    plt.close(fig)
