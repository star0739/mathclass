# activities/calculus_series.py
# 정적분과 구분구적법 탐구활동
# - Δx, x_k 한 줄 제시
# - S_n은 등호로 연결하여 전개
# - 왼쪽/오른쪽 끝점 선택
# - n ≤ 80
# - 교과서식 lim 표현 포함

from __future__ import annotations

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "정적분과 구분구적법"


def _cases():
    return {
        "x^2": {
            "a": 0.0,
            "b": 1.0,
            "f": lambda x: x**2,
            "problem_tex": r"f(x)=x^2,\;\; \text{적분구간 }[0,1]",
            "integral_tex": r"\int_{0}^{1}x^2\,dx=\frac{1}{3}",
            "dx_right_tex":
                r"\Delta x=\frac{b-a}{n}=\frac{1-0}{n},\quad "
                r"x_k=a+k\Delta x=0+k\frac{1-0}{n}=\frac{k}{n}"
                r"\;(k=1,2,\dots,n)",
            
            "dx_left_tex":
                r"\Delta x=\frac{b-a}{n}=\frac{1-0}{n},\quad "
                r"x_k=a+k\Delta x=0+k\frac{1-0}{n}=\frac{k}{n}"
                r"\;(k=0,1,\dots,n-1)",
            
            "sigma_right_tex":
                r"S_n=\sum_{k=1}^{n}f(x_k)\Delta x"
                r"=\sum_{k=1}^{n}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
            
            "sigma_left_tex":
                r"S_n=\sum_{k=0}^{n-1}f(x_k)\Delta x"
                r"=\sum_{k=0}^{n-1}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
        },
        "1/x": {
            "a": 1.0,
            "b": math.e,
            "f": lambda x: 1.0 / x,
            "problem_tex": r"f(x)=\frac{1}{x},\;\; \text{적분구간 }[1,e]",
            "integral_tex": r"\int_{1}^{e}\frac{1}{x}\,dx=1",

            "dx_right_tex":
                r"\Delta x=\frac{b-a}{n}=\frac{e-1}{n},\quad "
                r"x_k=a+k\Delta x=1+k\frac{e-1}{n}"
                r"\;(k=1,2,\dots,n)",

            "dx_left_tex":
                r"\Delta x=\frac{b-a}{n}=\frac{e-1}{n},\quad "
                r"x_k=a+k\Delta x=1+k\frac{e-1}{n}"
                r"\;(k=0,1,\dots,n-1)",

            "sigma_right_tex":
                r"S_n=\sum_{k=1}^{n}f(x_k)\Delta x"
                r"=\sum_{k=1}^{n}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
            
            "sigma_left_tex":
                r"S_n=\sum_{k=0}^{n-1}f(x_k)\Delta x"
                r"=\sum_{k=0}^{n-1}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
        },
        "sin x": {
            "a": 0.0,
            "b": math.pi,
            "f": lambda x: math.sin(x),
            "problem_tex": r"f(x)=\sin x,\;\; \text{적분구간 }[0,\pi]",
            "integral_tex": r"\int_{0}^{\pi}\sin x\,dx=2",
        
            "dx_right_tex":
                r"\Delta x=\frac{b-a}{n}=\frac{\pi-0}{n},\quad "
                r"x_k=a+k\Delta x=0+k\frac{\pi-0}{n}=\frac{k\pi}{n}"
                r"\;(k=1,2,\dots,n)",

            "dx_left_tex":
                r"\Delta x=\frac{b-a}{n}=\frac{\pi-0}{n},\quad "
                r"x_k=a+k\Delta x=0+k\frac{\pi-0}{n}=\frac{k\pi}{n}"
                r"\;(k=0,1,\dots,n-1)",

            "sigma_right_tex":
                r"S_n=\sum_{k=1}^{n}f(x_k)\Delta x"
                r"=\sum_{k=1}^{n}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",

            "sigma_left_tex":
                r"S_n=\sum_{k=0}^{n-1}f(x_k)\Delta x"
                r"=\sum_{k=0}^{n-1}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",
        },
    }


def _sum_value(f, a: float, b: float, n: int, mode: str) -> float:
    dx = (b - a) / n
    if mode == "right":
        k = np.arange(1, n + 1, dtype=float)
    else:
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
                "구간을 나누는 개수 n",
                min_value=5,
                max_value=80,
                value=40,
                step=1,
                key=f"{key_prefix}_n",
            )

        mode = st.radio(
            "대표값 선택",
            options=["right", "left"],
            format_func=lambda m: "오른쪽 끝점" if m == "right" else "왼쪽 끝점",
            key=f"{key_prefix}_mode",
        )

    cfg = cases[case_key]
    a, b, f = float(cfg["a"]), float(cfg["b"]), cfg["f"]

    st.markdown("### 선택한 함수와 적분구간")
    st.latex(cfg["problem_tex"])

    st.markdown("### 구간을 n등분하여 구한 합")

    if mode == "right":
        st.latex(cfg["dx_right_tex"])
        st.latex(cfg["sigma_right_tex"])
    else:
        st.latex(cfg["dx_left_tex"])
        st.latex(cfg["sigma_left_tex"])

    st.latex(r"\lim_{n\to\infty} S_n=\int_a^b f(x)\,dx")

    Sn = _sum_value(f, a, b, int(n), mode)
    st.caption(f"현재 선택한 n에서의 합:  S_{n} = {Sn:.8f}")

    st.markdown("### 정적분 값")
    st.latex(cfg["integral_tex"])

    # 그래프
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
