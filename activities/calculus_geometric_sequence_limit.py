# activities/calculus_geometric_sequence_limit.py
# 등비수열 r^n (초항 1 고정) 수렴/발산 시뮬레이션
# - 입력은 공비 r, 항의 개수 n만
# - 그래프는 "점만" 표시
# - 입력은 본문 상단 박스(container border)로 표시
# - 교과서 조건( r>1, r=1, -1<r<1, r<=-1 )에 맞춘 판정/정리 포함

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "등비수열의 수렴과 발산 (rⁿ, 초항 1)"


def _classify_textbook(r: float) -> tuple[str, str]:
    """
    교과서 조건:
    1) r > 1    : lim r^n = ∞ (발산)
    2) r = 1    : lim r^n = 1 (수렴)
    3) -1 < r < 1 : lim r^n = 0 (수렴)
    4) r <= -1  : 진동한다 (발산)
    """
    eps = 1e-12

    if r > 1 + eps:
        return "발산", r"$r>1$ 이면 $r^n$은 계속 커져 $\lim_{n\to\infty} r^n = \infty$ (발산)."
    if abs(r - 1) <= eps:
        return "수렴", r"$r=1$ 이면 모든 항이 $1$이므로 $\lim_{n\to\infty} r^n = 1$ (수렴)."
    if (-1 + eps) < r < (1 - eps):
        return "수렴", r"$-1<r<1$ 이면 $r^n\to 0$ 이므로 $\lim_{n\to\infty} r^n = 0$ (수렴)."
    # r <= -1 (또는 수치오차로 -1 근처 포함)
    return "발산", r"$r\le -1$ 이면 부호가 번갈아 바뀌거나 크기가 커지며 진동하므로 극한이 없어 발산."


def _safe_sequence_r_pow_n(r: float, n_max: int) -> np.ndarray:
    """
    a_n = r^n (n=1..n_max) 계산.
    overflow 방지: 너무 큰 값/inf는 NaN 처리.
    """
    n = np.arange(1, n_max + 1, dtype=float)

    # r=0이면 0,0,0,... (n>=1)
    if r == 0:
        return np.zeros_like(n)

    with np.errstate(over="ignore", invalid="ignore"):
        arr = r ** n

    bad = np.isinf(arr) | (np.abs(arr) > 1e308)
    arr[bad] = np.nan
    return arr


def render():
    st.title(TITLE)

    st.markdown(
        r"""
학생들이 배우는 등비수열을 **초항 1로 고정**하여  
\[
a_n = r^n \quad (n=1,2,3,\dots)
\]
의 수렴/발산을 공비 \(r\) 값에 따라 관찰합니다.
"""
    )

    # ----------------------------
    # 입력 UI (본문 상단 + 박스)
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2 = st.columns([2, 1])

        with col1:
            r = st.slider(
                "공비 r",
                min_value=-2.0,
                max_value=2.0,
                value=0.7,
                step=0.01,
            )

        with col2:
            n_max = st.slider(
                "표시할 항의 개수 n",
                min_value=5,
                max_value=200,
                value=60,
                step=1,
            )

        show_abs = st.checkbox("|rⁿ|도 함께 보기(보조 그래프)", value=False)

    # --- 판정 표시 (교과서 기준) ---
    verdict, desc = _classify_textbook(float(r))
    if verdict == "수렴":
        st.success(f"판정: {verdict}")
    else:
        st.warning(f"판정: {verdict}")
    st.markdown(desc)

    # --- 수열 계산 ---
    n = np.arange(1, int(n_max) + 1)
    a_n = _safe_sequence_r_pow_n(float(r), int(n_max))

    # ----------------------------
    # Plot 1: r^n (점만 표시)
    # ----------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.plot(n, a_n, marker="o", linestyle="None")  # ✅ 점만 표시
    ax.axhline(0, linewidth=1)

    ax.set_xlabel("n")
    ax.set_ylabel(r"$r^n$")
    ax.set_title(r"$r^n$의 변화 (점 그래프)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)

    # ----------------------------
    # Plot 2: |r^n| optional
    # ----------------------------
    if show_abs:
        fig2 = plt.figure(figsize=(6, 3.5))
        ax2 = fig2.add_subplot(111)

        abs_vals = np.abs(a_n)
        ax2.plot(n, abs_vals, marker="o", linestyle="None")
        ax2.set_xlabel("n")
        ax2.set_ylabel(r"$|r^n|$")
        ax2.set_title(r"$|r^n|$의 변화 (점 그래프)")
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        st.pyplot(fig2)

    # --- 핵심 정리: 교과서 문장/조건 반영 ---
    st.markdown("### 교과서 핵심 정리(조건별)")
    st.markdown(r"""
- $$r>1 \quad \Rightarrow \quad \lim_{n\to\infty} r^n = \infty \;(\text{발산})$$
- $$r=1 \quad \Rightarrow \quad \lim_{n\to\infty} r^n = 1 \;(\text{수렴})$$
- $$-1<r<1 \quad \Rightarrow \quad \lim_{n\to\infty} r^n = 0 \;(\text{수렴})$$
- $$r\le -1 \quad \Rightarrow \quad \text{진동한다} \;(\text{발산})$$
""")

    st.caption("Tip: r을 0.99→1.01, -0.99→-1.01로 바꿔 경계에서 변화가 어떻게 달라지는지 관찰해보세요.")
