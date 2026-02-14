# activities/calculus_geometric_series_sum.py
# 등비급수의 수렴과 발산 시뮬레이션
# - 교과서: |r|<1 수렴, 합 a/(1-r); |r|>=1 발산
# - 시각화: 부분합 S_n = a + ar + ... + ar^(n-1)
# - 입력: 본문 상단 박스(container border)
# - 점 그래프(선 없음), 정수 기준 grid(안정형)

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TITLE = "등비급수의 수렴과 발산"


def _classify_series(a: float, r: float) -> tuple[str, str]:
    """
    교과서 정리:
    1) |r| < 1 이면 수렴, 합 = a/(1-r) (a≠0)
    2) |r| >= 1 이면 발산
    """
    eps = 1e-12

    if abs(a) < eps:
        # a=0이면 모든 항이 0 -> 모든 부분합도 0, 수렴(합 0)
        return "수렴", r"$a=0$ 이면 모든 항이 0이므로 부분합도 0, 합은 0입니다."

    if abs(r) < 1 - 1e-12:
        return "수렴", r"$|r|<1$ 이므로 수렴하며, 무한등비급수의 합은 $\frac{a}{1-r}$ 입니다."
    return "발산", r"$|r|\ge 1$ 이면 등비급수는 발산합니다."


def _partial_sums(a: float, r: float, n_max: int) -> np.ndarray:
    """
    S_n = sum_{k=0}^{n-1} a r^k 계산 (n=1..n_max)
    overflow/inf는 NaN 처리
    """
    n = np.arange(1, n_max + 1, dtype=int)

    # 직접 누적합(수치적으로 안정적, 단 n이 크면 커짐)
    terms = np.empty(n_max, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        terms[:] = a * (r ** np.arange(0, n_max, dtype=float))

    terms[np.isinf(terms) | (np.abs(terms) > 1e308)] = np.nan

    # nan이 섞이면 cumsum이 nan으로 퍼질 수 있으니 구간 누적
    S = np.empty(n_max, dtype=float)
    running = 0.0
    for i in range(n_max):
        t = terms[i]
        if np.isnan(t):
            S[i] = np.nan
        else:
            running += t
            if np.isinf(running) or abs(running) > 1e308:
                S[i] = np.nan
            else:
                S[i] = running
    return S


def _nice_int_step(y_min: float, y_max: float, target_ticks: int = 7, min_step: int = 1) -> int:
    """y 범위에 맞춘 정수 tick 간격(폭발 방지용)"""
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
    """x축 정수 tick 간격 자동(최대 tick 수 제한)"""
    if n_max <= max_ticks:
        return 1
    return int(np.ceil(n_max / max_ticks))


def _set_reasonable_ylim(ax, y: np.ndarray, pad_ratio: float = 0.08):
    """finite 값 기준으로 y축 범위 고정(오토스케일 폭주 방지)"""
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


def render():
    st.title(TITLE)

    # ----------------------------
    # 입력 UI (본문 상단 + 박스)
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            a = st.number_input("초항 a", value=2.0, step=0.5, format="%.6f")

        with col2:
            # 발산/수렴 경계 관찰을 위해 -1.3~1.3 정도면 충분
            r = st.slider("공비 r", min_value=-1.3, max_value=1.3, value=0.7, step=0.01)

        with col3:
            n_max = st.slider("부분합 항의 개수 n", min_value=5, max_value=50, value=40, step=1)

    verdict, desc = _classify_series(float(a), float(r))
    if verdict == "수렴":
        st.success(f"판정: {verdict}")
    else:
        st.warning(f"판정: {verdict}")
    st.markdown(desc)

    # ----------------------------
    # 계산: 부분합
    # ----------------------------
    n = np.arange(1, int(n_max) + 1)
    S = _partial_sums(float(a), float(r), int(n_max))

    # 수렴일 때 극한값
    limit_value = None
    if abs(float(a)) > 1e-12 and abs(float(r)) < 1 - 1e-12:
        limit_value = float(a) / (1.0 - float(r))

    # ----------------------------
    # Plot: S_n 점 그래프
    # ----------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.plot(n, S, marker="o", linestyle="None")
    ax.axhline(0, linewidth=1)

    if limit_value is not None and np.isfinite(limit_value):
        ax.axhline(limit_value, linewidth=1)
        ax.text(
            0.99,
            0.97,
            r"$\frac{a}{1-r}$" + f" = {limit_value:.4g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    ax.set_xlabel("n")
    ax.set_ylabel(r"$S_n$")

    # x축 정수 tick (제한)
    x_step = _x_step_for_integers(int(n_max), max_ticks=15)
    ax.xaxis.set_major_locator(MultipleLocator(x_step))

    # y축 정수 tick 간격 자동
    finite_y = S[np.isfinite(S)]
    if finite_y.size > 0:
        y_min, y_max = float(np.min(finite_y)), float(np.max(finite_y))
        y_step = _nice_int_step(y_min, y_max, target_ticks=7, min_step=1)
    else:
        y_step = 1
    ax.yaxis.set_major_locator(MultipleLocator(y_step))

    _set_reasonable_ylim(ax, S)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)

    # ----------------------------
    # 교과서 핵심 정리
    # ----------------------------
    st.markdown("### 등비급수의 수렴과 발산")
    st.markdown(r"""
- $$|r|<1 \Rightarrow \sum_{n=1}^{\infty} ar^{n-1} \text{ 는 수렴하고, 합은 } \frac{a}{1-r} \text{ 이다.}$$
- $$|r|\ge 1 \Rightarrow \sum_{n=1}^{\infty} ar^{n-1} \text{ 는 발산한다.}$$
""")

    st.caption("Tip: r을 0.99→1.01, -0.99→-1.01로 바꿔 부분합이 어떻게 변하는지 비교해보세요.")
