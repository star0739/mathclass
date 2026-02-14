# activities/calculus_geometric_sequence_limit.py
# 등비수열 r^n (초항 1 고정) 수렴/발산 시뮬레이션
# - n 범위 축소, r 범위 축소
# - 가이드선: x축은 정수(1 간격) 고정, y축은 "적당한 정수 간격" 자동
# - 그래프는 점만 표시
# - 입력은 본문 상단 박스(container border)로 표시
# - 교과서 조건( r>1, r=1, -1<r<1, r<=-1 ) 반영

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TITLE = "등비수열 {r^n}의 수렴과 발산"


def _classify_textbook(r: float) -> tuple[str, str]:
    """
    교과서 조건:
    1) r > 1      : lim r^n = ∞ (발산)
    2) r = 1      : lim r^n = 1 (수렴)
    3) -1 < r < 1 : lim r^n = 0 (수렴)
    4) r <= -1    : 진동한다 (발산)
    """
    eps = 1e-12

    if r > 1 + eps:
        return "발산", r"$r>1$ 이면 $r^n$은 계속 커져 $\lim_{n\to\infty} r^n = \infty$ (발산)."
    if abs(r - 1) <= eps:
        return "수렴", r"$r=1$ 이면 모든 항이 $1$이므로 $\lim_{n\to\infty} r^n = 1$ (수렴)."
    if (-1 + eps) < r < (1 - eps):
        return "수렴", r"$-1<r<1$ 이면 $r^n\to 0$ 이므로 $\lim_{n\to\infty} r^n = 0$ (수렴)."
    return "발산", r"$r\le -1$ 이면 진동하므로 극한이 없어 발산."


def _safe_sequence_r_pow_n(r: float, n_max: int) -> np.ndarray:
    """a_n = r^n (n=1..n_max). overflow/inf는 NaN 처리."""
    n = np.arange(1, n_max + 1, dtype=float)

    if r == 0:
        return np.zeros_like(n)

    with np.errstate(over="ignore", invalid="ignore"):
        arr = r ** n

    bad = np.isinf(arr) | (np.abs(arr) > 1e308)
    arr[bad] = np.nan
    return arr


def _nice_int_step(y_min: float, y_max: float, target_ticks: int = 7) -> int:
    """
    y 범위에 맞춰 '정수 간격'을 자동 결정.
    1, 2, 5, 10 * 10^k 형태의 간격 중에서 target_ticks에 가깝게 선택.
    """
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return 1

    span = abs(y_max - y_min)
    if span <= 0:
        return 1

    raw = span / max(target_ticks, 1)
    if raw <= 1:
        return 1

    k = 10 ** int(np.floor(np.log10(raw)))
    candidates = np.array([1, 2, 5, 10]) * k
    step = float(candidates[np.argmin(np.abs(candidates - raw))])
    return max(1, int(step))


def render():
    st.title(TITLE)

    # ----------------------------
    # 입력 UI (본문 상단 + 박스)
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2 = st.columns([2, 1])

        with col1:
            # ✅ r 범위 축소 (발산/수렴 경계 관찰에 충분)
            r = st.slider(
                "공비 r",
                min_value=-1.5,
                max_value=1.5,
                value=0.7,
                step=0.01,
            )

        with col2:
            # ✅ n 범위 축소 (수치 폭주 및 축 눈금 폭발 예방)
            n_max = st.slider(
                "표시할 항의 개수 n",
                min_value=5,
                max_value=80,
                value=40,
                step=1,
            )

        show_abs = st.checkbox("|rⁿ|", value=False)

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

    # ✅ x축은 정수 1 간격
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # ✅ y축은 "적당한 정수 간격" 자동
    finite_y = a_n[np.isfinite(a_n)]
    if finite_y.size > 0:
        y_min, y_max = float(np.min(finite_y)), float(np.max(finite_y))
        y_step = _nice_int_step(y_min, y_max, target_ticks=7)
    else:
        y_step = 1
    ax.yaxis.set_major_locator(MultipleLocator(y_step))

    # ✅ major tick(정수 눈금) 기준 grid
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

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

        ax2.xaxis.set_major_locator(MultipleLocator(1))

        finite_y2 = abs_vals[np.isfinite(abs_vals)]
        if finite_y2.size > 0:
            y_min2, y_max2 = float(np.min(finite_y2)), float(np.max(finite_y2))
            y_step2 = _nice_int_step(y_min2, y_max2, target_ticks=7)
        else:
            y_step2 = 1
        ax2.yaxis.set_major_locator(MultipleLocator(y_step2))

        ax2.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

        st.pyplot(fig2)

    # --- 핵심 정리: 교과서 문장/조건 반영 ---
    st.markdown("### 등비수열의 수렴과 발산")
    st.markdown(r"""
- $$r>1 \Rightarrow \lim_{n\to\infty} r^n = \infty \;(\text{발산})$$
- $$r=1 \Rightarrow \lim_{n\to\infty} r^n = 1 \;(\text{수렴})$$
- $$-1<r<1 \Rightarrow \lim_{n\to\infty} r^n = 0 \;(\text{수렴})$$
- $$r\le -1 \Rightarrow \text{진동한다} \;(\text{발산})$$
""")

    st.caption("Tip: r을 0.99→1.01, -0.99→-1.01로 바꿔 경계에서 변화를 관찰해보세요.")
