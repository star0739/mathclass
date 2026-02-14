# activities/calculus_geometric_sequence_limit.py
# 등비수열 r^n (초항 1 고정) 수렴/발산 탐구활동
# - 탭 환경에서 위젯 충돌 방지: key_prefix 사용
# - 라우터에서 탭이 제목 역할을 하므로 show_title 옵션 제공
# - 메모리 안정형: tick/grid 폭발 방지
# - 정수 기준 가이드선: x축(정수) + y축(정수 간격)
# - 그래프는 점만 표시
# - 입력은 본문 상단 박스(container border)

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TITLE = "등비수열의 수렴과 발산"


def _classify_textbook(r: float) -> tuple[str, str]:
    eps = 1e-12
    if r > 1 + eps:
        return "발산", r"$r>1$ 이면 $r^n$은 계속 커져 $\lim_{n\to\infty} r^n = \infty$ (발산)."
    if abs(r - 1) <= eps:
        return "수렴", r"$r=1$ 이면 모든 항이 $1$이므로 $\lim_{n\to\infty} r^n = 1$ (수렴)."
    if (-1 + eps) < r < (1 - eps):
        return "수렴", r"$-1<r<1$ 이면 $r^n\to 0$ 이므로 $\lim_{n\to\infty} r^n = 0$ (수렴)."
    return "발산", r"$r\le -1$ 이면 진동하므로 극한이 없어 발산."


def _safe_sequence_r_pow_n(r: float, n_max: int) -> np.ndarray:
    n = np.arange(1, n_max + 1, dtype=float)
    if r == 0:
        return np.zeros_like(n)

    with np.errstate(over="ignore", invalid="ignore"):
        arr = r ** n

    bad = np.isinf(arr) | (np.abs(arr) > 1e308)
    arr[bad] = np.nan
    return arr


def _nice_int_step(y_min: float, y_max: float, target_ticks: int = 7, min_step: int = 1) -> int:
    """y 범위에 맞춰 '정수 간격' 자동 결정(틱 폭발 방지)."""
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return min_step

    span = abs(y_max - y_min)
    if span <= 0:
        return min_step

    raw = span / max(target_ticks, 1)
    if raw <= 1:
        return max(min_step, 1)

    k = 10 ** int(np.floor(np.log10(raw)))
    candidates = np.array([1, 2, 5, 10]) * k
    step = float(candidates[np.argmin(np.abs(candidates - raw))])
    return max(min_step, int(step))


def _x_step_for_integers(n_max: int, max_ticks: int = 15) -> int:
    """x축 정수 눈금 간격을 자동으로 키워 tick 개수 제한."""
    if n_max <= max_ticks:
        return 1
    return int(np.ceil(n_max / max_ticks))


def _set_reasonable_ylim(ax, y: np.ndarray, pad_ratio: float = 0.08) -> None:
    """finite 값 기준으로 y축 범위를 고정(오토스케일 폭주 방지)."""
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        ax.set_ylim(-1, 1)
        return

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))

    if y_min == y_max:
        pad = 1.0 if y_min == 0 else abs(y_min) * 0.2
        ax.set_ylim(y_min - pad, y_max + pad)
        return

    span = y_max - y_min
    pad = span * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)


def render(show_title: bool = True, key_prefix: str = "geom_seq") -> None:
    if show_title:
        st.title(TITLE)

    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2 = st.columns([2, 1])

        with col1:
            # 범위 축소: 값 폭주 완화
            r = st.slider(
                "공비 r",
                min_value=-1.3,
                max_value=1.3,
                value=0.7,
                step=0.01,
                key=f"{key_prefix}_r",
            )

        with col2:
            # n 범위 축소
            n_max = st.slider(
                "표시할 항의 개수 n",
                min_value=5,
                max_value=50,
                value=35,
                step=1,
                key=f"{key_prefix}_n",
            )

        show_abs = st.checkbox("|rⁿ|", value=False, key=f"{key_prefix}_abs")

    verdict, desc = _classify_textbook(float(r))
    if verdict == "수렴":
        st.success(f"판정: {verdict}")
    else:
        st.warning(f"판정: {verdict}")
    st.markdown(desc)

    n = np.arange(1, int(n_max) + 1)
    a_n = _safe_sequence_r_pow_n(float(r), int(n_max))

    # ----------------------------
    # Plot 1: r^n (점만 표시)
    # ----------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.plot(n, a_n, marker="o", linestyle="None")
    ax.axhline(0, linewidth=1)

    ax.set_xlabel("n")
    ax.set_ylabel(r"$r^n$")

    # x축 정수 grid 유지하되 tick 개수 제한
    x_step = _x_step_for_integers(int(n_max), max_ticks=15)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))

    # y축 정수 간격 자동
    finite_y = a_n[np.isfinite(a_n)]
    if finite_y.size > 0:
        y_min, y_max = float(np.min(finite_y)), float(np.max(finite_y))
        y_step = _nice_int_step(y_min, y_max, target_ticks=7, min_step=1)
    else:
        y_step = 1
    ax.yaxis.set_major_locator(MultipleLocator(y_step))

    _set_reasonable_ylim(ax, a_n)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)

    # ----------------------------
    # Plot 2: |r^n|
    # ----------------------------
    if show_abs:
        fig2 = plt.figure(figsize=(6, 3.5))
        ax2 = fig2.add_subplot(111)

        abs_vals = np.abs(a_n)
        ax2.plot(n, abs_vals, marker="o", linestyle="None")

        ax2.set_xlabel("n")
        ax2.set_ylabel(r"$|r^n|$")

        x_step2 = _x_step_for_integers(int(n_max), max_ticks=15)
        ax2.xaxis.set_major_locator(MultipleLocator(x_step2))

        finite_y2 = abs_vals[np.isfinite(abs_vals)]
        if finite_y2.size > 0:
            y_min2, y_max2 = float(np.min(finite_y2)), float(np.max(finite_y2))
            y_step2 = _nice_int_step(y_min2, y_max2, target_ticks=7, min_step=1)
        else:
            y_step2 = 1
        ax2.yaxis.set_major_locator(MultipleLocator(y_step2))

        _set_reasonable_ylim(ax2, abs_vals)
        ax2.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

        st.pyplot(fig2)

    st.markdown(r"### 등비수열 $r^n$의 수렴과 발산")
    st.markdown(r"""
- $$r>1 \Rightarrow \lim_{n\to\infty} r^n = \infty \;(\text{발산})$$
- $$r=1 \Rightarrow \lim_{n\to\infty} r^n = 1 \;(\text{수렴})$$
- $$-1<r<1 \Rightarrow \lim_{n\to\infty} r^n = 0 \;(\text{수렴})$$
- $$r\le -1 \Rightarrow \text{진동한다} \;(\text{발산})$$
""")

    st.caption("Tip: r을 0.99→1.01, -0.99→-1.01로 바꿔 경계에서 변화를 관찰해보세요.")
