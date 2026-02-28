# activities/calculus_series.py
# 정적분과 급수(리만합) 관계 탐구활동 (교과서 예제 3종 고정)
# 요청 반영:
# 1) 예제 표기는 LaTeX 중심
# 2) "on [0,1]" 대신 "적분구간 [0,1]"로 표시
# 3) 분할 개수 n: 최대 120 (기본 60) + 과부하 방지(직사각형은 더 보수적으로)
# 4) plot title 제거, 대신 markdown 안내 문구로 구분
# 5) S_n이 정적분값에 가까워지는 "관찰용 수열 그래프(Plot B)" 제거

from __future__ import annotations

import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "정적분과 급수(리만합)"


# -----------------------------
# 예제 정의 (교과서 고정 3종)
# - 표기용 LaTeX: 문제/구간/급수표현/정적분값
# -----------------------------
def _cases():
    return {
        "x^2": {
            "a": 0.0,
            "b": 1.0,
            "f": lambda x: x**2,
            "integral_exact": 1.0 / 3.0,
            "problem_tex": r"f(x)=x^2,\;\; \text{적분구간 }[0,1]",
            "integral_tex": r"\int_{0}^{1}x^2\,dx=\frac{1}{3}",
            # 오른쪽 끝점 리만합(교과서/활동 공통)
            "sigma_tex": r"S_n=\sum_{k=1}^{n}\left(\frac{k}{n}\right)^2\cdot\frac{1}{n}",
        },
        "1/x": {
            "a": 1.0,
            "b": math.e,
            "f": lambda x: 1.0 / x,
            "integral_exact": 1.0,  # ln(e)-ln(1)=1
            "problem_tex": r"f(x)=\frac{1}{x},\;\; \text{적분구간 }[1,e]",
            "integral_tex": r"\int_{1}^{e}\frac{1}{x}\,dx=\ln e-\ln 1=1",
            # x_k = 1 + k*(e-1)/n,  Δx=(e-1)/n
            "sigma_tex": r"S_n=\sum_{k=1}^{n}\frac{1}{1+\frac{(e-1)k}{n}}\cdot\frac{e-1}{n}",
        },
        "sin x": {
            "a": 0.0,
            "b": math.pi,
            "f": lambda x: math.sin(x),
            "integral_exact": 2.0,
            "problem_tex": r"f(x)=\sin x,\;\; \text{적분구간 }[0,\pi]",
            "integral_tex": r"\int_{0}^{\pi}\sin x\,dx=2",
            "sigma_tex": r"S_n=\sum_{k=1}^{n}\sin\!\left(\frac{k\pi}{n}\right)\cdot\frac{\pi}{n}",
        },
    }


# -----------------------------
# 오른쪽 끝점 리만합
# -----------------------------
def _riemann_right_sum(f, a: float, b: float, n: int) -> float:
    n = int(n)
    dx = (b - a) / n
    k = np.arange(1, n + 1, dtype=float)
    xk = a + k * dx
    yk = np.array([f(float(x)) for x in xk], dtype=float)
    return float(np.sum(yk) * dx)


def render(show_title: bool = True, key_prefix: str = "cal_series") -> None:
    if show_title:
        st.title(TITLE)

    cases = _cases()

    with st.container(border=True):
        st.subheader("입력값 설정")

        c1, c2 = st.columns([2, 1])

        with c1:
            case_key = st.radio(
                "예제 선택",
                options=list(cases.keys()),
                index=0,
                format_func=lambda k: {"x^2": r"$x^2$", "1/x": r"$\frac{1}{x}$", "sin x": r"$\sin x$"}[k],
                key=f"{key_prefix}_case",
            )

        with c2:
            # ✅ 최대 120 (요청 반영)
            n = st.slider(
                "분할 개수 n",
                min_value=5,
                max_value=120,
                value=60,
                step=1,
                key=f"{key_prefix}_n",
            )

        # 직사각형은 n이 커지면 선이 너무 많아져 부담 → 더 보수적으로 제한
        show_rect = st.checkbox(
            "리만합 직사각형(오른쪽 끝점) 표시",
            value=True,
            key=f"{key_prefix}_rect",
        )

    cfg = cases[case_key]
    a, b, f = float(cfg["a"]), float(cfg["b"]), cfg["f"]
    I = float(cfg["integral_exact"])

    # ----------------------------
    # 1) 수식 안내 (LaTeX 중심)
    # ----------------------------
    st.markdown("### 선택한 예제")
    st.latex(cfg["problem_tex"])

    st.markdown("### 정적분 값")
    st.latex(cfg["integral_tex"])

    st.markdown("### 구간에 따른 급수(리만합) 표현")
    st.latex(cfg["sigma_tex"])

    # 현재 n에서의 근사값은 "참고" 정도로만 (관찰 유도는 하지 않음)
    Sn = _riemann_right_sum(f, a, b, int(n))
    st.caption(f"현재 선택한 n에서의 리만합 값:  S_{n} = {Sn:.8f}")

    # ----------------------------
    # 2) 그래프: 함수 + (선택) 직사각형
    # ----------------------------
    st.markdown("### 그래프 확인")
    st.markdown("- 곡선은 $f(x)$ 그래프입니다.\n- 체크를 켜면 오른쪽 끝점 리만합 직사각형을 함께 표시합니다.")

    fig = plt.figure(figsize=(6.2, 4.0))
    ax = fig.add_subplot(111)

    xs = np.linspace(a, b, 500)
    ys = np.array([f(float(x)) for x in xs], dtype=float)

    ax.plot(xs, ys, linewidth=1.5)
    ax.axhline(0, linewidth=1)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if show_rect:
        # ✅ 직사각형은 n이 80 이하일 때만 (더 안정적으로)
        if n <= 80:
            dx = (b - a) / n
            for k in range(1, n + 1):
                x_left = a + (k - 1) * dx
                x_right = a + k * dx
                height = f(x_right)

                # 선으로만 표현(가볍게)
                ax.plot([x_left, x_right], [height, height], linewidth=0.8)
                ax.plot([x_left, x_left], [0, height], linewidth=0.6)
                ax.plot([x_right, x_right], [0, height], linewidth=0.6)
        else:
            st.info("직사각형 표시는 선이 너무 많아질 수 있어 n ≤ 80에서만 표시합니다.")

    st.pyplot(fig)

    # ----------------------------
    # 3) 정리(간단)
    # ----------------------------
    st.markdown("### 정리")
    st.latex(r"S_n=\sum_{k=1}^{n} f\!\big(a+k\Delta x\big)\,\Delta x,\quad \Delta x=\frac{b-a}{n}")
    st.caption("Tip: 예제별로 Δx와 x_k가 어떻게 정해지는지, 위의 시그마 식을 보고 직접 확인해보세요.")
