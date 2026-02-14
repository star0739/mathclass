# calculus_geometric_sequence_limit.py
# 등비수열의 극한 시뮬레이션 (Streamlit)

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


TITLE = "등비수열의 극한 (aₙ = a·r^(n-1))"


def _classify_behavior(a: float, r: float) -> tuple[str, str]:
    """
    반환: (판정 라벨, 간단 설명)
    """
    eps = 1e-12

    if abs(a) < eps:
        return "수렴 (항이 전부 0)", "a=0이므로 모든 항이 0입니다. 극한은 0."
    if abs(r) < eps:
        return "수렴 (유한 단계 후 0)", "r=0이면 a₁=a, a₂=0 이후 모든 항이 0. 극한은 0."

    ar = abs(r)

    if ar < 1 - 1e-9:
        return "수렴 (0으로)", "|r|<1 이므로 r^(n-1) → 0, 따라서 aₙ → 0."
    if abs(ar - 1) <= 1e-9:
        # |r| = 1
        if abs(r - 1) <= 1e-9:
            return "수렴 (상수 수열)", "r=1 이므로 aₙ=a로 일정. 극한은 a."
        # r = -1
        return "수렴하지 않음 (진동)", "r=-1 이면 a, -a, a, -a, ...로 진동하여 극한이 없습니다."
    # |r| > 1
    if r > 1:
        return "발산 (±∞)", "r>1 이면 |aₙ|이 무한히 커집니다(크기 발산)."
    if r < -1:
        return "발산 (진동하며 크기 증가)", "r<-1 이면 부호가 번갈아 바뀌면서 |aₙ|이 무한히 커집니다."
    # 이론상 여기 올 일 없음
    return "판정 보류", "입력값을 다시 확인해주세요."


def _safe_sequence(a: float, r: float, n_max: int) -> np.ndarray:
    """
    overflow를 완화해서 시각화를 돕는 수열 계산.
    너무 큰 값은 NaN으로 처리해 그래프가 깨지지 않도록 함.
    """
    n = np.arange(1, n_max + 1, dtype=float)
    # a_n = a * r^(n-1)
    # 큰 거듭제곱에서 overflow가 날 수 있어 로그 기반으로 체크
    with np.errstate(over="ignore", invalid="ignore"):
        log_abs = np.log(abs(a)) + (n - 1) * np.log(abs(r)) if (a != 0 and r != 0) else None

    a_n = np.empty_like(n)

    if a == 0:
        a_n[:] = 0.0
        return a_n

    if r == 0:
        a_n[:] = 0.0
        a_n[0] = a
        return a_n

    # overflow 임계: exp(709) 근처가 float64 한계
    if log_abs is not None:
        too_big = log_abs > 700  # 보수적으로 컷
    else:
        too_big = np.zeros_like(n, dtype=bool)

    with np.errstate(over="ignore", invalid="ignore"):
        a_n[:] = a * (r ** (n - 1))

    a_n[too_big] = np.nan  # 너무 커지면 그래프에서 끊기게
    return a_n


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

    label, desc = _classify_behavior(a, r)
    if "수렴" in label and "하지 않음" not in label:
        st.success(f"판정: {label}")
    elif "수렴하지 않음" in label:
        st.warning(f"판정: {label}")
    else:
        st.error(f"판정: {label}")

    st.write(desc)

    # --- Compute sequence ---
    n = np.arange(1, n_max + 1)
    a_n = _safe_sequence(float(a), float(r), int(n_max))

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if show_stems:
        # stem plot (NaN 구간은 자동으로 끊김)
        markerline, stemlines, baseline = ax.stem(n, a_n, use_line_collection=True)  # type: ignore
        # baseline(0선) 강조
        baseline.set_linewidth(1.0)
    else:
        ax.plot(n, a_n, marker="o", linestyle="-")

    ax.axhline(0, linewidth=1)

    ax.set_xlabel("n")
    ax.set_ylabel("aₙ")
    ax.set_title("등비수열 aₙ의 변화")

    # 로그 스케일은 음수 불가 -> 절댓값으로 그리는 모드 안내
    if use_log:
        # 로그 모드에서는 a_n 자체는 음수 가능하니 y축 로그가 불가능
        # 대신 |a_n|을 같은 축에 다시 표시
        ax.clear()
        abs_vals = np.abs(a_n)
        if show_stems:
            ax.stem(n, abs_vals, use_line_collection=True)  # type: ignore
        else:
            ax.plot(n, abs_vals, marker="o", linestyle="-")
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("|aₙ| (log scale)")
        ax.set_title("등비수열의 크기 |aₙ| (로그 스케일)")
        ax.axhline(1, linewidth=1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    else:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    st.pyplot(fig)

    # --- Optional: abs plot (separate) ---
    if show_abs and not use_log:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        abs_vals = np.abs(a_n)
        ax2.plot(n, abs_vals, marker="o", linestyle="-")
        ax2.set_xlabel("n")
        ax2.set_ylabel("|aₙ|")
        ax2.set_title("크기 |aₙ|의 변화")
        ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        st.pyplot(fig2)

    # --- Key cases summary (compact) ---
    st.markdown(
        r"""
### 핵심 정리(조건별)
- \(|r|<1\)  → \(a_n \to 0\)
- \(r=1\)    → \(a_n=a\) (상수 수열)  
- \(r=-1\)   → \(a,-a,a,-a,\dots\) (진동, 극한 없음)
- \(|r|>1\)  → \(|a_n|\to \infty\) (발산; \(r<0\)이면 진동하며 크기 증가)
"""
    )

    st.caption("Tip: r을 0.99 → 1.01, -0.99 → -1.01로 바꿔 보면서 경계에서의 변화를 관찰해보세요.")
