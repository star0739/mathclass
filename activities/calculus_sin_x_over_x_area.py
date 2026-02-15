
from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "삼각함수의 극한  :  $\\dfrac{\\sin x}{x}$"


def _safe_sin_over_x(x: float) -> float:
    if abs(x) < 1e-12:
        return 1.0  # 정의상 연속연장 값
    return float(np.sin(x) / x)


def render(show_title: bool = True, key_prefix: str = "sinx_over_x_area") -> None:
    if show_title:
        st.title(TITLE)

    st.markdown(
        r"""
단위원(반지름 1)에서 \(0<x<\frac{\pi}{2}\)일 때,

- 내접삼각형 넓이: \(\displaystyle \frac{1}{2}\sin x\)
- 부채꼴 넓이: \(\displaystyle \frac{1}{2}x\)   (라디안!)
- 외접삼각형 넓이: \(\displaystyle \frac{1}{2}\tan x\)

이므로
\[
\frac{1}{2}\sin x \;<\; \frac{1}{2}x \;<\; \frac{1}{2}\tan x
\]
즉,
\[
\sin x \;<\; x \;<\; \tan x
\]
양변을 \(\sin x\)로 나누면
\[
1 \;<\; \frac{x}{\sin x} \;<\; \frac{1}{\cos x}
\]
따라서 (역수를 취하면)
\[
\cos x \;<\; \frac{\sin x}{x} \;<\; 1
\]
이제 \(x\to 0\)일 때 \(\cos x\to 1\) 이므로
\[
\lim_{x\to 0}\frac{\sin x}{x}=1
\]
"""
    )

    # ----------------------------
    # 입력 UI
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2 = st.columns([1.2, 1.0])

        with col1:
            # 0<x<pi/2 범위에서 관찰(넓이 비교 조건)
            x = st.slider(
                "각 x (라디안, 0 < x < π/2)",
                min_value=0.001,
                max_value=float(np.pi / 2 - 0.01),
                value=0.5,
                step=0.001,
                key=f"{key_prefix}_x",
            )

        with col2:
            show_inequality = st.checkbox(
                "부등식(상·하한) 함께 표시",
                value=True,
                key=f"{key_prefix}_ineq",
            )

        # 학생이 보는 값 명시(LaTeX 블록이 가장 안정적)
        st.markdown(
            rf"""
현재 선택:
\[
x = {x:.3f}\ \text{{(rad)}}
\]
"""
        )

    # ----------------------------
    # 계산
    # ----------------------------
    sinx = float(np.sin(x))
    cosx = float(np.cos(x))
    tanx = float(np.tan(x))

    val = _safe_sin_over_x(x)

    # 면적(단위원, r=1)
    area_triangle_inner = 0.5 * sinx     # 내접삼각형
    area_sector = 0.5 * x               # 부채꼴(라디안)
    area_triangle_outer = 0.5 * tanx    # 외접삼각형

    # ----------------------------
    # 수치 요약
    # ----------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric(r"$\cos x$", f"{cosx:.10f}")
    c2.metric(r"$\dfrac{\sin x}{x}$", f"{val:.10f}")
    c3.metric("상한 1", "1.0000000000")

    # 면적 비교를 표처럼 보여주기(학생 직관용)
    st.markdown("### 넓이 비교(단위원)")
    st.markdown(
        rf"""
- 내접삼각형 넓이: \(\frac12\sin x \approx {area_triangle_inner:.6f}\)
- 부채꼴 넓이: \(\frac12 x \approx {area_sector:.6f}\)
- 외접삼각형 넓이: \(\frac12\tan x \approx {area_triangle_outer:.6f}\)
"""
    )

    if show_inequality:
        st.markdown(
            rf"""
### 부등식 관찰
\[
\sin x < x < \tan x
\]
\[
\cos x < \frac{{\sin x}}{{x}} < 1
\]
현재 값으로 확인하면
\[
{cosx:.6f}\;<\;{val:.6f}\;<\;1
\]
"""
        )

    # ----------------------------
    # 도식(단위원 + 3개 영역 느낌)
    # - 복잡한 채색 대신, 핵심 선/점만 표시(수업용 안정)
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
    ax.plot([0, px], [0, py], linewidth=2)          # 반지름 OP
    ax.plot([px], [py], marker="o")                  # 점 P
    ax.text(px, py, "  P", va="center")

    # x축 위의 점 A=(1,0), O=(0,0)
    ax.plot([1], [0], marker="o")
    ax.text(1, 0, "  A", va="center")

    # 부채꼴 경계(호) 일부 강조: 0..x
    t2 = np.linspace(0, x, 200)
    ax.plot(np.cos(t2), np.sin(t2), linewidth=3)

    # 내접삼각형: O-A-P (A=(1,0))
    ax.plot([0, 1, px, 0], [0, 0, py, 0], linewidth=1)

    # 외접삼각형: O-A-T, T는 x=1에서 접선과 OP 연장선 만나는 점
    # 단위원에서 x=1의 접선은 수직선 x=1, OP의 기울기 = tan x -> y = (tan x) * 1
    tx, ty = 1.0, tanx
    ax.plot([1, 1], [0, ty], linewidth=1)           # 접선의 일부(세로)
    ax.plot([0, 1], [0, ty], linewidth=1)           # O-T
    ax.plot([1, tx], [0, ty], linewidth=1)          # A-T(세로와 동일)

    # 보기 설정
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.2, max(1.1, ty + 0.2))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Tip: x를 0.5 → 0.2 → 0.1 → 0.05처럼 줄이며  cos x < (sin x)/x < 1  이 1로 모이는지 확인해보세요.")
