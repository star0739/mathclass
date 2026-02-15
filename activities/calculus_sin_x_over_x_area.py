
from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = r"삼각함수의 극한(넓이 비교) : $\dfrac{\sin x}{x}$"


def _safe_sin_over_x(x: float) -> float:
    if abs(x) < 1e-12:
        return 1.0  # 연속연장 값
    return float(np.sin(x) / x)


def render(show_title: bool = True, key_prefix: str = "sinx_over_x_area") -> None:
    if show_title:
        st.title(TITLE)

    # ----------------------------
    # 개념 설명(수식은 $ / $$ 만)
    # ----------------------------
    st.markdown(
        r"""
단위원(반지름 1)에서 $0<x<\dfrac{\pi}{2}$일 때, 다음 세 넓이를 비교합니다.

- 내접삼각형 넓이: $$\dfrac12\sin x$$
- 부채꼴 넓이(라디안): $$\dfrac12 x$$
- 외접삼각형 넓이: $$\dfrac12\tan x$$

따라서
$$
\dfrac12\sin x \;<\; \dfrac12 x \;<\; \dfrac12\tan x
$$
즉,
$$
\sin x \;<\; x \;<\; \tan x
$$

양변을 $\sin x$로 나누면
$$
1 \;<\; \dfrac{x}{\sin x} \;<\; \dfrac{1}{\cos x}
$$

역수를 취하면(모두 양수이므로 부등호 방향 유지)
$$
\cos x \;<\; \dfrac{\sin x}{x} \;<\; 1
$$

그리고 $x\to 0$일 때 $\cos x\to 1$ 이므로
$$
\lim_{x\to 0}\dfrac{\sin x}{x}=1
$$
"""
    )

    st.markdown("### 관찰 포인트")
    st.markdown(
        r"""
- $x$를 $0$에 가깝게 할수록 $\dfrac{\sin x}{x}$ 값이 $1$에 가까워지는지 확인해보세요.
- $x$를 작게 할수록 $\cos x$와 $\dfrac{\sin x}{x}$가 모두 $1$에 가까워지는지 확인해보세요.
"""
    )

    # ----------------------------
    # 입력 UI (부등식 옵션 삭제)
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        x = st.slider(
            "각 x (라디안, 0 < x < π/2)",
            min_value=0.001,
            max_value=float(np.pi / 2 - 0.01),
            value=0.5,
            step=0.001,
            key=f"{key_prefix}_x",
        )

        st.caption("단위: 라디안(rad)")

        # 현재 선택값 (가장 안정적인 $$ 블록)
        st.markdown(
            f"""
현재 선택 값:

$$
x = {x:.3f}
$$
"""
        )

    # ----------------------------
    # 계산
    # ----------------------------
    sinx = float(np.sin(x))
    cosx = float(np.cos(x))
    tanx = float(np.tan(x))
    val = _safe_sin_over_x(x)

    # 단위원에서의 넓이
    area_triangle_inner = 0.5 * sinx
    area_sector = 0.5 * x
    area_triangle_outer = 0.5 * tanx

    # ----------------------------
    # 수치 요약
    # ----------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("cos x", f"{cosx:.10f}")
    c2.metric("sin x / x", f"{val:.10f}")
    c3.metric("1", "1.0000000000")

    # 오차도 같이 보여주면 관찰이 쉬움
    st.markdown(
        f"""
$$
|1-\cos x| \approx {abs(1 - cosx):.3e}, \qquad
\\left|1-\\dfrac{{\sin x}}{{x}}\\right| \approx {abs(1 - val):.3e}
$$
"""
    )

    # ----------------------------
    # 넓이 비교(수치)
    # ----------------------------
    st.markdown("### 넓이 비교(단위원)")

    st.markdown(
        f"""
$$
\\dfrac12\\sin x \\approx {area_triangle_inner:.6f}, \qquad
\\dfrac12 x \\approx {area_sector:.6f}, \qquad
\\dfrac12\\tan x \\approx {area_triangle_outer:.6f}
$$
"""
    )

    # ----------------------------
    # 도식(단위원)
    # ----------------------------
    st.markdown("### 단위원 도식(개념 확인용)")

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    # 단위원
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1)

    # 기준축
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    # 점 P=(cos x, sin x)
    px, py = cosx, sinx
    ax.plot([0, px], [0, py], linewidth=2)  # 반지름 OP
    ax.plot([px], [py], marker="o")         # 점 P
    ax.text(px, py, "  P", va="center")

    # A=(1,0)
    ax.plot([1], [0], marker="o")
    ax.text(1, 0, "  A", va="center")

    # 호(0..x) 강조
    t2 = np.linspace(0, x, 200)
    ax.plot(np.cos(t2), np.sin(t2), linewidth=3)

    # 내접삼각형 O-A-P
    ax.plot([0, 1, px, 0], [0, 0, py, 0], linewidth=1)

    # 외접삼각형: 접선 x=1과 OP 연장선의 교점 T=(1, tan x)
    tx, ty = 1.0, tanx
    ax.plot([1, 1], [0, ty], linewidth=1)   # 접선 일부
    ax.plot([0, 1], [0, ty], linewidth=1)   # O-T

    # 보기 설정
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.2, max(1.1, ty + 0.2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Tip: x를 0.5 → 0.2 → 0.1 → 0.05처럼 줄이며 sin x / x 값이 1로 가까워지는지 확인해보세요.")
