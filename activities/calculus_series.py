# activities/calculus_series.py
# 정적분과 구분구적법(급수→정적분 변형) 탐구활동
# - 교과서식 전개: Δx, x_k 정의 → Σ f(x_k)Δx → lim → ∫
# - 왼쪽/오른쪽 끝점(대표값) 중 하나만 선택
# - n ≤ 80 (성능 안정)
# - "리만합" 용어 사용하지 않음

from __future__ import annotations

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "구분구적법"


def _cases():
    return {
        "x^2": {
            "a": 0.0,
            "b": 1.0,
            "f": lambda x: x**2,
            "problem_tex": r"f(x)=x^2,\;\; \text{적분구간 }[0,1]",
            "integral_tex": r"\int_{0}^{1}x^2\,dx=\frac{1}{3}",
            # 오른쪽/왼쪽 끝점에서의 x_k (표현용)
            "dx_tex": r"\Delta x=\frac{1}{n}",
            "xk_right_tex": r"x_k=\frac{k}{n}\quad (k=1,2,\dots,n)",
            "xk_left_tex": r"x_k=\frac{k}{n}\quad (k=0,1,\dots,n-1)",
            # 교과서 연습용으로 “대입한 시그마 형태”도 함께 제시
            "sigma_right_tex": r"S_n=\sum_{k=1}^{n}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
            "sigma_left_tex": r"S_n=\sum_{k=0}^{n-1}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
        },
        "1/x": {
            "a": 1.0,
            "b": math.e,
            "f": lambda x: 1.0 / x,
            "problem_tex": r"f(x)=\frac{1}{x},\;\; \text{적분구간 }[1,e]",
            "integral_tex": r"\int_{1}^{e}\frac{1}{x}\,dx=1",
            "dx_tex": r"\Delta x=\frac{e-1}{n}",
            "xk_right_tex": r"x_k=1+k\frac{e-1}{n}\quad (k=1,2,\dots,n)",
            "xk_left_tex": r"x_k=1+k\frac{e-1}{n}\quad (k=0,1,\dots,n-1)",
            "sigma_right_tex": r"S_n=\sum_{k=1}^{n}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
            "sigma_left_tex": r"S_n=\sum_{k=0}^{n-1}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
        },
        "sin x": {
            "a": 0.0,
            "b": math.pi,
            "f": lambda x: math.sin(x),
            "problem_tex": r"f(x)=\sin x,\;\; \text{적분구간 }[0,\pi]",
            "integral_tex": r"\int_{0}^{\pi}\sin x\,dx=2",
            "dx_tex": r"\Delta x=\frac{\pi}{n}",
            "xk_right_tex": r"x_k=\frac{k\pi}{n}\quad (k=1,2,\dots,n)",
            "xk_left_tex": r"x_k=\frac{k\pi}{n}\quad (k=0,1,\dots,n-1)",
            "sigma_right_tex": r"S_n=\sum_{k=1}^{n}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",
            "sigma_left_tex": r"S_n=\sum_{k=0}^{n-1}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",
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
            "소구간 기준점 설정",
            options=["right", "left"],
            format_func=lambda m: "오른쪽 끝점" if m == "right" else "왼쪽 끝점",
            key=f"{key_prefix}_mode",
        )

    cfg = cases[case_key]
    a, b, f = float(cfg["a"]), float(cfg["b"]), cfg["f"]

    # ----------------------------
    # 1) 선택한 함수/구간
    # ----------------------------
    st.markdown("### 선택한 함수와 적분구간")
    st.latex(cfg["problem_tex"])

    # ----------------------------
    # 2) 구간을 n등분하여 구한 합 (교과서식 전개)
    # ----------------------------
    st.markdown("### 구간을 n등분하여 구한 합")

    # (1) Δx
    st.latex(cfg["dx_tex"])

    # (2) x_k (대표값 정의)
    if mode == "right":
        st.latex(cfg["xk_right_tex"])
        sum_template = r"S_n=\sum_{k=1}^{n} f(x_k)\,\Delta x"
        st.latex(sum_template)
        st.latex(cfg["sigma_right_tex"])
    else:
        st.latex(cfg["xk_left_tex"])
        sum_template = r"S_n=\sum_{k=0}^{n-1} f(x_k)\,\Delta x"
        st.latex(sum_template)
        st.latex(cfg["sigma_left_tex"])

    # (3) lim → 정적분 (교과서식)
    st.latex(r"\lim_{n\to\infty} S_n=\int_{a}^{b} f(x)\,dx")

    # 참고값(현재 n에서 계산된 합) — 관찰을 강요하지 않는 정도로만
    Sn = _sum_value(f, a, b, int(n), mode)
    st.caption(f"현재 선택한 n에서의 합:  S_{n} = {Sn:.8f}")

    # ----------------------------
    # 3) 정적분 값 (교과서 답)
    # ----------------------------
    st.markdown("### 정적분 값")
    st.latex(cfg["integral_tex"])

    # ----------------------------
    # 4) 그래프 (함수 + 직사각형)
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

    # ----------------------------
    # 5) 정리 (교과서식 문장 최소)
    # ----------------------------
    st.markdown("### 정리")
    st.latex(r"\Delta x=\frac{b-a}{n}")
    if mode == "right":
        st.latex(r"x_k=a+k\Delta x\quad (k=1,2,\dots,n)")
        st.latex(r"S_n=\sum_{k=1}^{n} f(x_k)\,\Delta x")
    else:
        st.latex(r"x_k=a+k\Delta x\quad (k=0,1,\dots,n-1)")
        st.latex(r"S_n=\sum_{k=0}^{n-1} f(x_k)\,\Delta x")
    st.latex(r"\lim_{n\to\infty} S_n=\int_a^b f(x)\,dx")
