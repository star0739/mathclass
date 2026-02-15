# activities/calculus_e_definition_limit.py
# Ⅱ. 미분법 - 자연상수 e (정의/극한) 탐구활동
# - 탭 환경 위젯 충돌 방지: key_prefix 사용
# - show_title 옵션 제공(라우터 탭이 제목 역할)
# - 메모리 안정: 샘플 수/틱 간격 제한, figure close

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TITLE = "자연상수 $e$"


def _safe_f_of_x(x: np.ndarray) -> np.ndarray:
    r"""
    \( f(x) = (1+x)^{1/x} \)

    - \(x=0\)은 정의되지 않으므로 입력에서 제외해야 함.
    - \(1+x>0\) (즉 \(x>-1\)) 조건 필요.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = (1.0 + x) ** (1.0 / x)
    y[np.isinf(y) | (np.abs(y) > 1e308)] = np.nan
    return y


def _safe_g_of_n(n: np.ndarray) -> np.ndarray:
    r"""
    \( g(n) = \left(1+\frac{1}{n}\right)^n \)
    """
    n = n.astype(float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = (1.0 + 1.0 / n) ** n
    y[np.isinf(y) | (np.abs(y) > 1e308)] = np.nan
    return y


def _nice_step(span: float, target_ticks: int = 7) -> float:
    """
    y축 tick 폭발 방지용 '적당한 간격' (실수 간격)
    1,2,5,10 * 10^k 후보 중 선택.
    """
    if not np.isfinite(span) or span <= 0:
        return 1.0

    raw = span / max(target_ticks, 1)
    if raw <= 0:
        return 1.0

    k = 10 ** int(np.floor(np.log10(raw)))
    candidates = np.array([1, 2, 5, 10], dtype=float) * k
    return float(candidates[np.argmin(np.abs(candidates - raw))])


def _set_reasonable_ylim(ax, y: np.ndarray, pad_ratio: float = 0.08) -> None:
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        ax.set_ylim(0, 4)
        return

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))

    if y_min == y_max:
        pad = 0.2 if y_min == 0 else abs(y_min) * 0.1
        ax.set_ylim(y_min - pad, y_max + pad)
        return

    span = y_max - y_min
    pad = span * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)


def render(show_title: bool = True, key_prefix: str = "e_def") -> None:
    if show_title:
        st.title(TITLE)


    st.markdown("### 관찰 포인트")
    st.markdown(
        r"""
- $x$를 $0$에 가깝게 할수록 $(1+x)^{1/x}$ 값이 $e$에 가까워지는지 확인해보세요.
- $n$을 크게 할수록 $\left(1+\frac{1}{n}\right)^n$ 값이 $e$에 가까워지는지 확인해보세요.
"""
    )

    # ----------------------------
    # 입력 UI
    # ----------------------------
    with st.container(border=True):
        st.subheader("입력값 설정")

        col1, col2, col3 = st.columns([1.3, 1.0, 1.0])

        with col1:
            x0 = st.slider(
                "관찰할 x 값(0 제외)",
                min_value=-0.3,
                max_value=0.3,
                value=0.1,
                step=0.001,
                key=f"{key_prefix}_x0",
            )

            st.markdown(f"현재 선택된 값:  $x = {x0:.3f}$")
            
            if abs(x0) < 1e-6:
                st.info("x가 0에 너무 가까우면 계산이 불안정할 수 있어요. x를 조금만 더 떨어뜨려보세요.")

        with col2:
            n_max = st.slider(
                "수열 관찰 최대 n",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
                key=f"{key_prefix}_nmax",
            )


        show_hline = st.checkbox("y = e 기준선 표시", value=True, key=f"{key_prefix}_hline")

    e_val = float(np.e)

    # ----------------------------
    # 수치 계산(현재 값)
    # ----------------------------
    if abs(float(x0)) < 1e-12:
        fx0 = np.nan
    else:
        fx0 = float(_safe_f_of_x(np.array([float(x0)], dtype=float))[0])

    n_end = int(n_max)
    gx = float(_safe_g_of_n(np.array([n_end], dtype=int))[0])

    c1, c2, c3 = st.columns(3)
    c1.metric(r"$e$ (기준값)", f"{e_val:.10f}")
    c2.metric(r"$f(x)=(1+x)^{1/x}$", f"{fx0:.10f}" if np.isfinite(fx0) else "정의/계산 불가")
    c3.metric(
        rf"$g(n)=\left(1+\frac{{1}}{{n}}\right)^n$  (n={n_end})",
        f"{gx:.10f}" if np.isfinite(gx) else "계산 불가",
    )

    # ----------------------------
    # 그래프 영역: 좌(연속형) / 우(수열형)
    # ----------------------------
    left, right = st.columns(2)

    # ---- (A) 연속형
    with left:
        st.markdown(r"#### 연속형:  $f(x)=(1+x)^{1/x}$  $(x \to 0)$")

        SAMPLES = 120
        m = SAMPLES
        m_left = m // 2
        m_right = m - m_left
        
        eps = 1e-3
        half = 0.5

        xs_left = np.linspace(-half, -eps, m_left, dtype=float)
        xs_right = np.linspace(eps, half, m_right, dtype=float)
        xs = np.concatenate([xs_left, xs_right], axis=0)

        ys = _safe_f_of_x(xs)

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        ax.plot(xs, ys, marker="o", linestyle="None", markersize=3)
        ax.axvline(0, linewidth=1)
        if show_hline:
            ax.axhline(e_val, linewidth=1)

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")

        ax.xaxis.set_major_locator(MultipleLocator(0.1))

        finite_y = ys[np.isfinite(ys)]
        if finite_y.size > 0:
            step = _nice_step(float(np.max(finite_y) - np.min(finite_y)), target_ticks=6)
            ax.yaxis.set_major_locator(MultipleLocator(step))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(1.0))

        _set_reasonable_ylim(ax, ys)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

        st.pyplot(fig)
        plt.close(fig)

    # ---- (B) 수열형
    with right:
        st.markdown(r"#### 수열형:  $g(n)=\left(1+\frac{1}{n}\right)^n$  $(n \to \infty)$")

        ns = np.arange(1, int(n_max) + 1, dtype=int)
        ys2 = _safe_g_of_n(ns)

        fig2 = plt.figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)

        ax2.plot(ns, ys2, marker="o", linestyle="None", markersize=3)
        if show_hline:
            ax2.axhline(e_val, linewidth=1)

        ax2.set_xlabel("n")
        ax2.set_ylabel("g(n)")

        x_step = max(1, int(np.ceil(int(n_max) / 10)))
        ax2.xaxis.set_major_locator(MultipleLocator(x_step))

        finite_y2 = ys2[np.isfinite(ys2)]
        if finite_y2.size > 0:
            step2 = _nice_step(float(np.max(finite_y2) - np.min(finite_y2)), target_ticks=6)
            ax2.yaxis.set_major_locator(MultipleLocator(step2))
        else:
            ax2.yaxis.set_major_locator(MultipleLocator(1.0))

        _set_reasonable_ylim(ax2, ys2)
        ax2.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)

        st.pyplot(fig2)
        plt.close(fig2)


