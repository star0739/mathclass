# activities/calculus_volume.py
# 입체도형의 부피(교과서 표기 S(x), V_n 사용)
# - 단면 넓이를 S(x)로 두고, V_n = Σ S(x_k) Δx 를 시각화(직사각형)
# - 교과서식 기호/전개:
#   a=x_0,...,x_n=b,  Δx=(b-a)/n,  x_k=a+kΔx,
#   V_n=Σ_{k=1}^n S(x_k)Δx,  V=lim V_n=∫_a^b S(x)dx
# - 기능: 함수 선택 + 단면 모양(원/정사각형) 선택 + n(≤60)
# - 표시용(a_tex, b_tex)과 계산용(a, b) 분리

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "입체도형의 부피"


@dataclass(frozen=True)
class Case:
    key: str
    f: Callable[[float], float]
    f_tex: str
    a: float
    b: float
    a_tex: str
    b_tex: str


def _cases() -> Dict[str, Case]:
    return {
        "x": Case(
            key="x",
            f=lambda x: x,
            f_tex=r"x",
            a=0.0,
            b=1.0,
            a_tex="0",
            b_tex="1",
        ),
        "x^2": Case(
            key="x^2",
            f=lambda x: x**2,
            f_tex=r"x^2",
            a=0.0,
            b=1.0,
            a_tex="0",
            b_tex="1",
        ),
        "sqrt(x)": Case(
            key="sqrt(x)",
            f=lambda x: math.sqrt(x),
            f_tex=r"\sqrt{x}",
            a=0.0,
            b=1.0,
            a_tex="0",
            b_tex="1",
        ),
        "sin x": Case(
            key="sin x",
            f=lambda x: math.sin(x),
            f_tex=r"\sin x",
            a=0.0,
            b=math.pi,
            a_tex="0",
            b_tex=r"\pi",
        ),
    }


def _S_builder(shape: str, f: Callable[[float], float]) -> Tuple[Callable[[float], float], str]:
    """
    shape:
      - "circle": 단면이 원(반지름 = f(x))
      - "square": 단면이 정사각형(한 변 = f(x))
    """
    if shape == "circle":
        return (lambda x: math.pi * (f(x) ** 2)), r"S(x)=\pi\{f(x)\}^2"
    return (lambda x: (f(x) ** 2)), r"S(x)=\{f(x)\}^2"


def _Vn_right_sum(S: Callable[[float], float], a: float, b: float, n: int) -> float:
    dx = (b - a) / n
    k = np.arange(1, n + 1, dtype=float)
    xk = a + k * dx
    Sk = np.array([S(float(x)) for x in xk], dtype=float)
    return float(np.sum(Sk) * dx)


def _exact_volume(case_key: str, shape: str) -> Optional[str]:
    """
    예제별로 간단히 정리되는 경우만 '정적분 값(텍스트)' 제공.
    (원 단면이면 π가 곱해지는 형태)
    """
    if case_key == "x":
        return r"\frac{1}{3}" if shape == "square" else r"\frac{\pi}{3}"
    if case_key == "x^2":
        return r"\frac{1}{5}" if shape == "square" else r"\frac{\pi}{5}"
    if case_key == "sqrt(x)":
        return r"\frac{1}{2}" if shape == "square" else r"\frac{\pi}{2}"
    if case_key == "sin x":
        return r"\frac{\pi}{2}" if shape == "square" else r"\frac{\pi^{2}}{2}"
    return None


def render(show_title: bool = True, key_prefix: str = "cal_volume") -> None:
    if show_title:
        st.title(TITLE)

    cases = _cases()

    with st.container(border=True):
        st.subheader("입력값 설정")

        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            case_key = st.radio(
                "함수 선택",
                options=list(cases.keys()),
                index=0,
                format_func=lambda k: {
                    "x": r"$f(x)=x$",
                    "x^2": r"$f(x)=x^2$",
                    "sqrt(x)": r"$f(x)=\sqrt{x}$",
                    "sin x": r"$f(x)=\sin x$",
                }[k],
                key=f"{key_prefix}_case",
            )

        with c2:
            shape = st.radio(
                "단면 모양",
                options=["circle", "square"],
                format_func=lambda v: "원" if v == "circle" else "정사각형",
                key=f"{key_prefix}_shape",
            )

        with c3:
            n = st.slider(
                "구간을 나누는 개수 n",
                min_value=5,
                max_value=60,
                value=30,
                step=1,
                key=f"{key_prefix}_n",
            )

    case = cases[case_key]
    a, b = float(case.a), float(case.b)
    a_tex, b_tex = case.a_tex, case.b_tex

    # 단면 넓이 S(x)
    S, S_tex = _S_builder("circle" if shape == "circle" else "square", case.f)

    # ----------------------------
    # 1) 교과서식 기호/정의
    # ----------------------------
    st.markdown("### 설정")
    st.latex(rf"f(x)={case.f_tex},\qquad [a,b]=[{a_tex},{b_tex}]")
    st.latex(S_tex)

    st.markdown("### 구간을 n등분")
    st.latex(rf"a=x_0,\;x_1,\;\dots,\;x_n=b")
    st.latex(
        rf"""
\Delta x=\frac{{b-a}}{{n}}=\frac{{{b_tex}-{a_tex}}}{{n}},\qquad
x_k=a+k\Delta x={a_tex}+k\frac{{{b_tex}-{a_tex}}}{{n}}\;\;(k=0,1,2,\dots,n)
"""
    )

    st.markdown("### 부피의 근삿값")
    st.latex(r"V_n=\sum_{k=1}^{n} S(x_k)\Delta x")

    Vn = _Vn_right_sum(S, a, b, int(n))
    st.caption(f"현재 선택한 n에서의 부피 근삿값:  V_{n} = {Vn:.8f}")

    st.markdown("### 정적분으로 나타내면")
    rhs = _exact_volume(case_key, "circle" if shape == "circle" else "square")
    if rhs is None:
        st.latex(r"V=\lim_{n\to\infty}V_n=\lim_{n\to\infty}\sum_{k=1}^{n}S(x_k)\Delta x=\int_a^b S(x)\,dx")
    else:
        st.latex(
            r"V=\lim_{n\to\infty}V_n=\lim_{n\to\infty}\sum_{k=1}^{n}S(x_k)\Delta x=\int_a^b S(x)\,dx="
            + rhs
        )

    # ----------------------------
    # 2) 시각화
    # ----------------------------
    st.markdown("### 그래프 확인")
    st.markdown("- 위: $f(x)$ 그래프\n- 아래: 단면 넓이 $S(x)$와 $V_n=\\sum S(x_k)\\Delta x$를 나타내는 직사각형")

    # Plot 1: f(x)
    fig1 = plt.figure(figsize=(6.2, 3.6))
    ax1 = fig1.add_subplot(111)
    xs = np.linspace(a, b, 600)
    ys = np.array([case.f(float(x)) for x in xs], dtype=float)
    ax1.plot(xs, ys, linewidth=1.5)
    ax1.axhline(0, linewidth=1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    st.pyplot(fig1)

    # Plot 2: S(x) + rectangles (right endpoints, k=1..n)
    fig2 = plt.figure(figsize=(6.2, 3.9))
    ax2 = fig2.add_subplot(111)
    Ss = np.array([S(float(x)) for x in xs], dtype=float)
    ax2.plot(xs, Ss, linewidth=1.5)

    dx = (b - a) / n
    # 직사각형 (폭 Δx, 높이 S(x_k), 오른쪽 끝점)
    for k in range(1, n + 1):
        x_left = a + (k - 1) * dx
        x_right = a + k * dx
        h = S(x_right)
        ax2.plot([x_left, x_right], [h, h], linewidth=0.8)
        ax2.plot([x_left, x_left], [0, h], linewidth=0.6)
        ax2.plot([x_right, x_right], [0, h], linewidth=0.6)

    ax2.axhline(0, linewidth=1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("S(x)")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    st.pyplot(fig2)
