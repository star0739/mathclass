# activities/calculus_geometric_series_sum.py
# 등비급수의 수렴과 발산 탐구활동
# - 교과서 정의: sum_{n=1}^∞ a_n = lim_{n→∞} S_n
# - r=1 / r≠1 부분합 공식 분기
# - |r|<1일 때만 극한값 수평선 표시
# - 점 그래프, 정수 기준 grid (안정형)
# - 탭 환경 위젯 충돌 방지: key_prefix 사용
# - 라우터에서 탭이 제목 역할 → show_title 옵션 제공

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TITLE = "등비급수의 수렴과 발산"


# --------------------------------------------------
# 부분합 계산
# --------------------------------------------------
def _partial_sums(a: float, r: float, n_max: int) -> np.ndarray:
    n = np.arange(1, n_max + 1)

    if abs(r - 1) < 1e-12:
        return a * n  # S_n = na

    with np.errstate(over="ignore", invalid="ignore"):
        S = a * (1 - r**n) / (1 - r)

    S[np.isinf(S) | (np.abs(S) > 1e308)] = np.nan
    return S


# --------------------------------------------------
# 판정
# --------------------------------------------------
def _classify_series(a: float, r: float) -> tuple[str, str]:
    eps = 1e-12

    if abs(a) < eps:
        return "수렴", r"$a=0$ 이므로 모든 부분합이 0이고, 급수는 0으로 수렴합니다."

    if abs(r) < 1 - eps:
        return "수렴", r"$|r|<1$이면  $\lim_{n\to\infty} S_n$이 수렴하므로 급수는 수렴"
    return "발산", r"$|r|\ge 1$이면  $\lim_{n\to\infty} S_n$이 발산하므로 급수는 발산"


# --------------------------------------------------
# 정수 tick 간격 자동
# --------------------------------------------------
def _nice_int_step(y_min: float, y_max: float, target_ticks: int = 7) -> int:
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


def _x_step(n_max: int, max_ticks: int = 15) -> int:
    if n_max <= max_ticks:
        return 1
    return int(np.ceil(n_max / max_ticks))


def _set_ylim(ax, y: np.ndarray) -> None:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        ax.set_ylim(-1, 1)
        return

    y_min, y_max = float(np.min(finite)), float(np.max(finite))

    if y_min == y_max:
        pad = 1 if y_min == 0 else abs(y_min) * 0.2
        ax.set_ylim(y_min - pad, y_max + pad)
        return

    span = y_max - y_min
    pad = span * 0.08
    ax.set_ylim(y_min - pad, y_max + pad)


# --------------------------------------------------
# 메인
# --------------------------------------------------
def render(show_title: bool = True, key_prefix: str = "geom_series") -> None:
    if show_title:
        st.title(TITLE)

    # ----------------------------
    # 부분합과 급수의 관계
    # ----------------------------
    st.markdown(r""" ### 부분합과 급수의 관계
    $$ S_n = \sum_{k=1}^{n} a_k $$ 
    
    $$ \lim_{n \to \infty} S_n = S \;\Longleftrightarrow\; \sum_{k=1}^{\infty} a_k = S$$
    """)
    st.markdown("", unsafe_allow_html=True)

    # ----------------------------
    # 등비급수의 제 n항까지의 부분합
    # ----------------------------
    st.markdown(r""" ### 등비급수의 제 $n$항까지의 부분합
    $$ S_n = a + ar + ar^2 + \cdots + ar^{n-1} = \sum_{k=1}^{n} a r^{k-1}$$이다. """)
    st.markdown("", unsafe_allow_html=True)

    # ----------------------------
    # 입력 UI
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            a = st.number_input(
                "초항 a",
                min_value=-5.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key=f"{key_prefix}_a",
            )

        with col2:
            r = st.slider(
                "공비 r",
                -1.3,
                1.3,
                0.7,
                0.01,
                key=f"{key_prefix}_r",
            )

        with col3:
            n_max = st.slider(
                "부분합 항의 개수 n",
                5,
                50,
                35,
                1,
                key=f"{key_prefix}_n",
            )

    verdict, desc = _classify_series(float(a), float(r))
    if verdict == "수렴":
        st.success(f"판정: {verdict}")
    else:
        st.warning(f"판정: {verdict}")
    st.markdown(desc)

    # ----------------------------
    # 현재 S_n 표시 (r=1 / r≠1)
    # ----------------------------
    if abs(float(r) - 1.0) < 1e-12:
        
        current = float(a) * int(n_max)
        st.markdown(rf"$S_{{{int(n_max)}}}={int(n_max)}\times {float(a):.4g}={current:.6g}$")
    else:
        current = float(a) * (1 - float(r) ** int(n_max)) / (1 - float(r))
        st.markdown(
            rf"$S_{{{int(n_max)}}}=\frac{{{float(a):.4g}\left(1-({float(r):.4g})^{{{int(n_max)}}}\right)}}{{1-{float(r):.4g}}}={current:.6g}$"
        )

    # ----------------------------
    # 부분합 계산
    # ----------------------------
    n = np.arange(1, int(n_max) + 1)
    S = _partial_sums(float(a), float(r), int(n_max))

    # 극한값(|r|<1일 때)
    limit_val = None
    if abs(float(r)) < 1 - 1e-12:
        limit_val = float(a) / (1 - float(r))

    # ----------------------------
    # 그래프
    # ----------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    ax.plot(n, S, marker="o", linestyle="None")
    ax.set_xlabel("$n$")
    ax.set_ylabel("$S_n$")
    ax.axhline(0, linewidth=1)

    if limit_val is not None and np.isfinite(limit_val):
        ax.axhline(limit_val, linewidth=1)

    ax.xaxis.set_major_locator(MultipleLocator(_x_step(int(n_max))))
    finite_y = S[np.isfinite(S)]
    if finite_y.size > 0:
        y_step = _nice_int_step(float(np.min(finite_y)), float(np.max(finite_y)))
    else:
        y_step = 1
    ax.yaxis.set_major_locator(MultipleLocator(y_step))

    _set_ylim(ax, S)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)

    # ----------------------------
    # 교과서 정리
    # ----------------------------
    st.markdown(r"### 등비급수 $\sum_{n=1}^{\infty} ar^{n-1}$ ($a \neq 0$)의 수렴과 발산")
    st.markdown(r"""
- $$|r|<1 \Rightarrow  \text{ 는 수렴하고 그 합은 } \frac{a}{1-r}$$
- $$|r|\ge1 \Rightarrow  \text{  발산한다.}$$
""")
