# 정적분과 급수(리만합) 관계 탐구활동 (교과서 예제 3종 고정)
# - 분할 수 n이 커질수록 리만합 S_n이 정적분값 I에 가까워짐을 시각화
# - 구간/함수에 따라 "구체적인 급수(시그마) 표현"을 익히도록 식을 명시
# - 성능 안정: 직사각형은 n이 너무 크면 자동으로 생략

from __future__ import annotations

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TITLE = "정적분과 급수"


# -----------------------------
# 예제 정의 (교과서 고정 3종)
# -----------------------------
def _cases():
    return {
        "x^2  on  [0,1]": {
            "a": 0.0,
            "b": 1.0,
            "f": lambda x: x**2,
            "integral_exact": 1.0 / 3.0,
            "sigma_tex": r"S_n=\sum_{k=1}^{n}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
            "integral_tex": r"\int_{0}^{1}x^2\,dx=\frac{1}{3}",
            "func_tex": r"f(x)=x^2,\quad [0,1]",
        },
        "1/x  on  [1,e]": {
            "a": 1.0,
            "b": math.e,
            "f": lambda x: 1.0 / x,
            "integral_exact": 1.0,  # ln(e)-ln(1)=1
            "sigma_tex": r"S_n=\sum_{k=1}^{n}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
            "integral_tex": r"\int_{1}^{e}\frac{1}{x}\,dx=\ln e-\ln 1=1",
            "func_tex": r"f(x)=\frac{1}{x},\quad [1,e]",
        },
        "sin x  on  [0,π]": {
            "a": 0.0,
            "b": math.pi,
            "f": lambda x: math.sin(x),
            "integral_exact": 2.0,
            "sigma_tex": r"S_n=\sum_{k=1}^{n}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",
            "integral_tex": r"\int_{0}^{\pi}\sin x\,dx=2",
            "func_tex": r"f(x)=\sin x,\quad [0,\pi]",
        },
    }


# -----------------------------
# 리만합(오른쪽 끝점) 계산
# -----------------------------
def _riemann_right_sum(f, a: float, b: float, n: int) -> float:
    n = int(n)
    dx = (b - a) / n
    # 오른쪽 끝점: x_k = a + k*dx (k=1..n)
    k = np.arange(1, n + 1, dtype=float)
    xk = a + k * dx
    yk = np.array([f(float(x)) for x in xk], dtype=float)
    return float(np.sum(yk) * dx)


def _series_values(f, a: float, b: float, n_max: int) -> np.ndarray:
    # S_1, S_2, ..., S_nmax
    vals = np.empty(int(n_max), dtype=float)
    for n in range(1, int(n_max) + 1):
        vals[n - 1] = _riemann_right_sum(f, a, b, n)
    return vals


def render(show_title: bool = True, key_prefix: str = "cal_series") -> None:
    if show_title:
        st.title(TITLE)

    cases = _cases()

    with st.container(border=True):
        st.subheader("입력값 설정")

        c1, c2 = st.columns([2, 1])

        with c1:
            case_name = st.radio(
                "예제 선택",
                options=list(cases.keys()),
                index=0,
                key=f"{key_prefix}_case",
            )

        with c2:
            n = st.slider(
                "분할 개수 n",
                min_value=5,
                max_value=300,
                value=60,
                step=1,
                key=f"{key_prefix}_n",
            )

        # 직사각형은 너무 많으면 화면/성능이 부담 → 자동 제한
        show_rect = st.checkbox(
            "리만합 직사각형(오른쪽 끝점) 표시",
            value=True,
            key=f"{key_prefix}_rect",
        )

    cfg = cases[case_name]
    a, b, f = float(cfg["a"]), float(cfg["b"]), cfg["f"]
    I = float(cfg["integral_exact"])

    Sn = _riemann_right_sum(f, a, b, int(n))
    err = abs(Sn - I)

    st.markdown(f"**선택된 예제:** {cfg['func_tex']}")
    st.latex(cfg["integral_tex"])
    st.latex(cfg["sigma_tex"])

    st.success(f"현재 리만합  S_{n} = {Sn:.8f}   |   정적분 I = {I:.8f}   |   오차 |S_n - I| = {err:.8f}")

    # ----------------------------
    # Plot A: f(x)와 (선택) 직사각형
    # ----------------------------
    fig1 = plt.figure(figsize=(6.2, 4.0))
    ax1 = fig1.add_subplot(111)

    xs = np.linspace(a, b, 500)
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    ax1.plot(xs, ys, linewidth=1.5)
    ax1.axhline(0, linewidth=1)

    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.set_title("함수 그래프와 리만합(오른쪽 끝점)")

    if show_rect:
        if n <= 120:
            dx = (b - a) / n
            # 각 구간 [a+(k-1)dx, a+k dx]에서 높이는 f(a+k dx)
            for k in range(1, n + 1):
                x_left = a + (k - 1) * dx
                x_right = a + k * dx
                height = f(x_right)
                # 직사각형(테두리만 느낌) : 상단선 + 양쪽선
                ax1.plot([x_left, x_right], [height, height], linewidth=0.8)
                ax1.plot([x_left, x_left], [0, height], linewidth=0.6)
                ax1.plot([x_right, x_right], [0, height], linewidth=0.6)
        else:
            st.info("직사각형 표시: n이 커지면 너무 촘촘해져 자동으로 생략합니다. (n ≤ 120에서 표시)")

    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    st.pyplot(fig1)

    # ----------------------------
    # Plot B: S_n 수열이 I로 수렴
    # ----------------------------
    n_max_plot = 200  # 과도 확장 방지(그래도 충분히 관찰됨)
    seq = _series_values(f, a, b, n_max_plot)
    ns = np.arange(1, n_max_plot + 1)

    fig2 = plt.figure(figsize=(6.2, 3.8))
    ax2 = fig2.add_subplot(111)

    ax2.plot(ns, seq, marker="o", linestyle="None", markersize=3.5)
    ax2.axhline(I, linewidth=1.2)  # 정적분값 기준선

    # 현재 선택한 n 위치 표시(범위 내일 때)
    if 1 <= n <= n_max_plot:
        ax2.plot([n], [seq[n - 1]], marker="o", markersize=7, linestyle="None")

    ax2.set_xlabel("n")
    ax2.set_ylabel(r"$S_n$")
    ax2.set_title(r"리만합 $S_n$은 $n$이 커질수록 정적분값 $I$에 가까워진다")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig2)

    st.markdown("### 정리")
    st.markdown(
        r"""
- 이 활동에서는 **오른쪽 끝점 리만합**으로
  $$S_n=\sum_{k=1}^{n} f\!\big(a+k\Delta x\big)\,\Delta x,\quad \Delta x=\frac{b-a}{n}$$
  를 계산했습니다.
- **n이 커질수록** 직사각형의 폭이 줄어들어, 근사값 \(S_n\)이 정적분값 \(I\)에 가까워집니다.
"""
    )
    st.caption("Tip: n을 10 → 30 → 100으로 바꾸며, Plot B에서 점들이 기준선(I) 쪽으로 모이는지 확인해보세요.")
