# activities/calculus_series.py
# 정적분과 급수(리만합) 관계 탐구활동
# - 분할 개수 n: 최대 80
# - 오른쪽/왼쪽 끝점 리만합 선택 가능 (동시 표시 불가)
# - LaTeX 중심 표현
# - 수열 수렴 그래프 없음

from __future__ import annotations

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "정적분과 급수(리만합)"


def _cases():
    return {
        "x^2": {
            "a": 0.0,
            "b": 1.0,
            "f": lambda x: x**2,
            "integral_exact": 1.0 / 3.0,
            "problem_tex": r"f(x)=x^2,\;\; \text{적분구간 }[0,1]",
            "integral_tex": r"\int_{0}^{1}x^2\,dx=\frac{1}{3}",
            "sigma_right": r"S_n=\sum_{k=1}^{n}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
            "sigma_left": r"S_n=\sum_{k=1}^{n}\left(\frac{k-1}{n}\right)^2\cdot\frac{1}{n}",
        },
        "1/x": {
            "a": 1.0,
            "b": math.e,
            "f": lambda x: 1.0 / x,
            "integral_exact": 1.0,
            "problem_tex": r"f(x)=\frac{1}{x},\;\; \text{적분구간 }[1,e]",
            "integral_tex": r"\int_{1}^{e}\frac{1}{x}\,dx=1",
            "sigma_right": r"S_n=\sum_{k=1}^{n}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
            "sigma_left": r"S_n=\sum_{k=1}^{n}\frac{1}{1+\frac{(e-1)(k-1)}{n}}\cdot\frac{e-1}{n}",
        },
        "sin x": {
            "a": 0.0,
            "b": math.pi,
            "f": lambda x: math.sin(x),
            "integral_exact": 2.0,
            "problem_tex": r"f(x)=\sin x,\;\; \text{적분구간 }[0,\pi]",
            "integral_tex": r"\int_{0}^{\pi}\sin x\,dx=2",
            "sigma_right": r"S_n=\sum_{k=1}^{n}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",
            "sigma_left": r"S_n=\sum_{k=1}^{n}\sin\!\left(\frac{(k-1)\pi}{n}\right)\cdot\frac{\pi}{n}",
        },
    }


def _riemann_sum(f, a: float, b: float, n: int, mode: str) -> float:
    dx = (b - a) / n
    if mode == "right":
        k = np.arange(1, n + 1, dtype=float)
        xk = a + k * dx
    else:  # left
        k = np.arange(0, n, dtype=float)
        xk = a + k * dx

    yk = np.array([f(float(x)) for x in xk], dtype=float)
    return float(np.sum(yk) * dx)


def render(show_title: bool = True, key_prefix: str = "cal_series") -> None:
    if show_title:
        st.title(TITLE)

    cases = _cases()

    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2 = st.columns([2, 1])

        with col1:
            case_key = st.radio(
                "예제 선택",
                options=list(cases.keys()),
                format_func=lambda k: {"x^2": r"$x^2$", "1/x": r"$\frac{1}{x}$", "sin x": r"$\sin x$"}[k],
                key=f"{key_prefix}_case",
            )

        with col2:
            n = st.slider(
                "분할 개수 n",
                min_value=5,
                max_value=80,
                value=40,
                step=1,
                key=f"{key_prefix}_n",
            )

        mode = st.radio(
            "리만합 방식 선택",
            options=["right", "left"],
            format_func=lambda m: "오른쪽 끝점" if m == "right" else "왼쪽 끝점",
            key=f"{key_prefix}_mode",
        )

    cfg = cases[case_key]
    a, b, f = float(cfg["a"]), float(cfg["b"]), cfg["f"]

    st.markdown("### 선택한 예제")
    st.latex(cfg["problem_tex"])

    st.markdown("### 정적분 값")
    st.latex(cfg["integral_tex"])

    st.markdown("### 구간에 따른 급수 표현")

    if mode == "right":
        st.latex(cfg["sigma_right"])
    else:
        st.latex(cfg["sigma_left"])

    Sn = _riemann_sum(f, a, b, int(n), mode)
    st.caption(f"현재 선택한 n에서의 리만합 값:  S_{n} = {Sn:.8f}")

    # ----------------------------
    # 그래프
    # ----------------------------
    st.markdown("### 그래프 확인")

    fig = plt.figure(figsize=(6.2, 4.0))
    ax = fig.add_subplot(111)

    xs = np.linspace(a, b, 500)
    ys = np.array([f(float(x)) for x in xs], dtype=float)

    ax.plot(xs, ys, linewidth=1.5)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    dx = (b - a) / n

    if mode == "right":
        for k in range(1, n + 1):
            x_left = a + (k - 1) * dx
            x_right = a + k * dx
            height = f(x_right)

            ax.plot([x_left, x_right], [height, height], linewidth=0.8)
            ax.plot([x_left, x_left], [0, height], linewidth=0.6)
            ax.plot([x_right, x_right], [0, height], linewidth=0.6)
    else:
        for k in range(0, n):
            x_left = a + k * dx
            x_right = a + (k + 1) * dx
            height = f(x_left)

            ax.plot([x_left, x_right], [height, height], linewidth=0.8)
            ax.plot([x_left, x_left], [0, height], linewidth=0.6)
            ax.plot([x_right, x_right], [0, height], linewidth=0.6)

    st.pyplot(fig)

    st.markdown("### 정리")
    st.latex(r"S_n=\sum_{k=1}^{n} f\!\big(a+k\Delta x\big)\,\Delta x \quad \text{또는} \quad f\!\big(a+(k-1)\Delta x\big)\,\Delta x")
    st.caption("끝점 선택에 따라 시그마 식이 어떻게 달라지는지 비교해보세요.")
