# activities/calculus_geometric_sequence_limit.py
# 등비수열의 극한 시뮬레이션 (matplotlib 호환 + LaTeX 핵심정리 + 그래프 축소)

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

TITLE = "등비수열의 극한 (aₙ = a·r^(n-1))"


def _classify_behavior(a: float, r: float) -> tuple[str, str]:
    """(판정 라벨, 설명) 반환"""
    eps = 1e-12

    if abs(a) < eps:
        return "수렴 (항이 전부 0)", "a=0이므로 모든 항이 0입니다. 극한은 0."
    if abs(r) < eps:
        return "수렴 (유한 단계 후 0)", "r=0이면 a₁=a, a₂=0 이후 모든 항이 0. 극한은 0."

    ar = abs(r)

    if ar < 1 - 1e-9:
        return "수렴 (0으로)", r"$|r|<1$ 이므로 $r^{n-1}\to 0$, 따라서 $a_n\to 0$."
    if abs(ar - 1) <= 1e-9:
        if abs(r - 1) <= 1e-9:
            return "수렴 (상수 수열)", r"$r=1$ 이므로 $a_n=a$로 일정합니다. 극한은 $a$."
        return "수렴하지 않음 (진동)", r"$r=-1$ 이면 $a,-a,a,-a,\dots$ 로 진동하여 극한이 없습니다."

    # |r| > 1
    if r > 1:
        return "발산 (크기 증가)", r"$r>1$ 이면 $|a_n|$이 무한히 커집니다."
    if r < -1:
        return "발산 (진동하며 크기 증가)", r"$r<-1$ 이면 부호가 번갈아 바뀌면서 $|a_n|$이 무한히 커집니다."

    return "판정 보류", "입력값을 다시 확인해주세요."


def _safe_sequence(a: float, r: float, n_max: int) -> np.ndarray:
    """
    a_n = a * r^(n-1) 계산.
    너무 큰 값은 NaN으로 바꿔 시각화가 깨지지 않게 함.
    """
    n = np.arange(1, n_max + 1, dtype=float)

    if a == 0:
        return np.zeros_like(n)

    if r == 0:
        arr = np.zeros_like(n)
        arr[0] = a
        return arr

    with np.errstate(over="ignore", invalid="ignore"):
        arr = a * (r ** (n - 1))

    bad = np.isinf(arr) | (np.abs(arr) > 1e308)
    arr[bad] = np.nan
    return arr


def _stem(ax, x, y):
    """matplotlib 버전 차이에 안전한 stem(use_line_collection 제거)"""
    container = ax.stem(x, y)
    try:
        container.baseline.set_linewidth(1.0)
    except Exception:
        pass
    return container


def render():
    st.title(TITLE)

    st.markdown(
        r"""
이 페이지는 등비수열  
\[
a_n = a\,r^{\,n-1}
\]
의 **수렴/발산을 공비 \(r\)** 값에 따라 시각적으로 확인하는 시뮬레이션입니다.
"""
    )

    # --- Sidebar controls ---
    st.sidebar.header("입력값")
    a = st.sidebar.number_input("초항 a", value=2.0, step=0.5, format="%.6f")
    r = st.sidebar.number_input("공비 r", value=0.7, step=0.1, format="%.6f")
    n_max = st.sidebar.slider("표시할 항의 개수 (n_max)", min_value=5, max_value=200, value=60, step=1)

    st.sidebar.divider()
    show_stems = st.sidebar.checkbox("점+수직선(stem) 스타일로 보기", value=True)
    show_abs = st.sidebar.checkbox("|aₙ|도 함께 보기(보조 그래프)", value=False)
    use_log = st.sidebar.checkbox("y축 로그 스케일(절댓값 기준)", value=False)

    # --- 판정 표시 ---
    label, desc = _classify_behavior(float(a), float(r))
    if "수렴" in label and "하지 않음" not in label:
        st.success(f"판정: {label}")
    elif "수렴하지 않음" in label:
        st.warning(f"판정: {label}")
    else:
        st.error(f"판정: {label}")
    st.markdown(desc)

    # --- 수열 계산 ---
    n = np.arange(1, n_max + 1)
    a_n = _safe_sequence(float(a), float(r), int(n_max))

    # --- Plot 1: a_n (또는 log 모드에서는 |a_n|) ---
    fig = plt.figure(figsize=(6, 4))  # ✅ 그래프 크기 축소
    ax = fig.add_subplot(111)

    if use_log:
        # 로그축은 음수 불가 → |a_n|을 로그축으로 표시
        y = np.abs(a_n)

        if show_stems:
            _stem(ax, n, y)
        else:
            ax.plot(n, y, marker="o", linestyle="-")

        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel(r"$|a_n|$ (log scale)")
        ax.set_title(r"등비수열의 크기 $|a_n|$ (로그 스케일)")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    else:
        # 일반 모드에서는 a_n 자체를 표시
        if show_stems:
            _stem(ax, n, a_n)
        else:
            ax.plot(n, a_n, marker="o", linestyle="-")

        ax.axhline(0, linewidth=1)
        ax.set_xlabel("n")
        ax.set_ylabel(r"$a_n$")
        ax.set_title(r"등비수열 $a_n$의 변화")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)

    # --- Plot 2: |a_n| optional (일반 모드에서만) ---
    if show_abs and not use_log:
        fig2 = plt.figure(figsize=(6, 3.5))  # ✅ 보조 그래프도 축소
        ax2 = fig2.add_subplot(111)

        abs_vals = np.abs(a_n)
        ax2.plot(n, abs_vals, marker="o", linestyle="-")
        ax2.set_xlabel("n")
        ax2.set_ylabel(r"$|a_n|$")
        ax2.set_title(r"크기 $|a_n|$의 변화")
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        st.pyplot(fig2)

    # --- 핵심 정리: LaTeX로 깔끔하게 렌더링 ---
    st.markdown("### 핵심 정리(조건별)")
    st.markdown(r"""
- $$|r| < 1 \quad \Rightarrow \quad a_n \to 0$$
- $$r = 1 \quad \Rightarrow \quad a_n = a \;(\text{상수 수열})$$
- $$r = -1 \quad \Rightarrow \quad a, -a, a, -a, \dots \;(\text{진동, 극한 없음})$$
- $$|r| > 1 \quad \Rightarrow \quad |a_n| \to \infty \;(\text{발산; } r<0 \text{이면 진동하며 크기 증가})$$
""")

    st.caption("Tip: r을 0.99 → 1.01, -0.99 → -1.01로 바꿔 경계에서의 변화를 관찰해보세요.")
