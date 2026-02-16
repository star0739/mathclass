
from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = r"직선의 기울기 $t$로 원을 매개화하기"


def _xy_from_t(t: float) -> tuple[float, float]:
    """
    점 (-1,0)을 지나고 기울기가 t인 직선 y = t(x+1) 과 원 x^2+y^2=1의
    다른 교점은
      x = (1 - t^2)/(1 + t^2),  y = (2t)/(1 + t^2)
    """
    denom = 1.0 + t * t
    x = (1.0 - t * t) / denom
    y = (2.0 * t) / denom
    return float(x), float(y)


def render(show_title: bool = True, key_prefix: str = "circle_param_t") -> None:
    if show_title:
        st.title(TITLE)

    # ----------------------------
    # 개념 안내
    # ----------------------------
    st.markdown(
        r"""
원 $x^2+y^2=1$ 위의 점을, 점 $(-1,0)$을 지나는 직선
$$
y=t(x+1)
$$
과 원의 **다른 교점**으로 잡습니다.

연립하면(교과서 결과)
$$
x=\frac{1-t^2}{1+t^2},\qquad y=\frac{2t}{1+t^2}\quad (t\in\mathbb{R})
$$
가 되어, $t$를 바꾸면 원 위의 점을 계속 만들어낼 수 있습니다.
"""
    )

    # ----------------------------
    # 상태(누적 점) 초기화
    # ----------------------------
    pts_key = f"{key_prefix}_pts"
    if pts_key not in st.session_state:
        st.session_state[pts_key] = []  # list[tuple[float,float]]

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
                step=0.01,
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
                st.session_state[pts_key] = []
                st.rerun()

        st.markdown(
            f"""
현재 선택:
$$
t = {t:.2f}
$$
"""
        )

    # ----------------------------
    # 현재 점 계산 + 누적
    # ----------------------------
    x, y = _xy_from_t(float(t))

    if accumulate:
        # 중복이 너무 많이 쌓이는 것 방지(동일/근접 점은 생략)
        pts = st.session_state[pts_key]
        if not pts:
            pts.append((x, y))
        else:
            px, py = pts[-1]
            if (x - px) ** 2 + (y - py) ** 2 > 1e-6:
                pts.append((x, y))
        # 너무 길어지면 메모리 보호(상한)
        if len(pts) > 3000:
            st.session_state[pts_key] = pts[-3000:]

    # ----------------------------
    # 수치 표시
    # ----------------------------
    c1, c2 = st.columns(2)
    c1.metric("x(t)", f"{x:.6f}")
    c2.metric("y(t)", f"{y:.6f}")

    st.markdown(
        f"""
검산:
$$
x^2+y^2 \approx {(x*x + y*y):.6f}
$$
(부동소수점 오차로 1에 매우 가깝게 나옵니다.)
"""
    )

    # ----------------------------
    # 시각화
    # ----------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # 원(희미하게)
    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.cos(th), np.sin(th), linewidth=1, alpha=0.25)

    # 축
    ax.axhline(0, linewidth=1, alpha=0.4)
    ax.axvline(0, linewidth=1, alpha=0.4)

    # (-1,0) 기준점
    ax.plot([-1], [0], marker="o", markersize=6)
    ax.text(-1, 0, "  (-1,0)", va="center")

    # 직선 y = t(x+1) (표시 옵션)
    if show_line:
        # 보기 좋게 x 범위를 약간 넓힘
        xs = np.linspace(-1.3, 1.3, 200)
        ys = t * (xs + 1.0)
        ax.plot(xs, ys, linewidth=1, alpha=0.6)

    # 누적 점
    pts = st.session_state[pts_key]
    if accumulate and pts:
        pxs = [p[0] for p in pts]
        pys = [p[1] for p in pts]
        ax.scatter(pxs, pys, s=12, alpha=0.6)

    # 현재 점(강조)
    ax.scatter([x], [y], s=80)
    ax.text(x, y, "  P(t)", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Tip: t를 크게 키우면 P(t)가 (−1,0) 근처로, t=0이면 (1,0)으로 이동합니다. t를 빠르게 바꿔 점들이 원 전체로 퍼지는지 관찰해보세요.")
