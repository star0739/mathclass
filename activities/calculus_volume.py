# activities/calculus_volume.py
# 단면을 쌓아 만드는 부피(시각 중심)
# - 목적: "단면 넓이를 적분구간만큼 쌓으면 입체가 된다"를 직관적으로 시각화
# - 구분구적법(Σ, Δx) 강조는 최소화: 핵심 수식 2줄만 제시
# - 좌: f(x) 그래프 + 현재 단면 위치(x*) 표시
# - 우: 단면을 x방향으로 조금씩 이동해 겹쳐 그린(가짜 3D) 스택 시각화
# - 성능: 단면 개수 m ≤ 30, matplotlib 2D 패치만 사용

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

TITLE = "단면을 쌓아 만드는 부피"


# -----------------------------
# 함수/구간 예제
# -----------------------------
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
        "x": Case("x", lambda x: x, r"x", 0.0, 1.0, "0", "1"),
        "x^2": Case("x^2", lambda x: x**2, r"x^2", 0.0, 1.0, "0", "1"),
        "sqrt(x)": Case("sqrt(x)", lambda x: math.sqrt(x), r"\sqrt{x}", 0.0, 1.0, "0", "1"),
        "sin x": Case("sin x", lambda x: math.sin(x), r"\sin x", 0.0, math.pi, "0", r"\pi"),
    }


# -----------------------------
# 단면 넓이 S(x)
# -----------------------------
def _S_and_tex(shape: str, f_tex: str) -> Tuple[Callable[[float, Callable[[float], float]], float], str]:
    """
    shape:
      - "circle": 단면이 원(반지름 = f(x)) -> S(x)=π{f(x)}^2
      - "square": 단면이 정사각형(한 변 = f(x)) -> S(x)={f(x)}^2
    반환:
      S(x) 계산 함수(케이스의 f를 인자로 받는 형태), 표시용 LaTeX
    """
    if shape == "circle":
        return (lambda x, f: math.pi * (f(x) ** 2)), rf"S(x)=\pi\{{f(x)\}}^2=\pi\left({f_tex}\right)^2"
    return (lambda x, f: (f(x) ** 2)), rf"S(x)=\{{f(x)\}}^2=\left({f_tex}\right)^2"


def _safe_nonneg(v: float) -> float:
    # 시각화 목적: 반지름/변 길이는 0 이상으로 표시
    if not math.isfinite(v):
        return 0.0
    return max(0.0, float(v))


# -----------------------------
# 렌더
# -----------------------------
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
            m = st.slider(
                "쌓는 단면 개수 m",
                min_value=5,
                max_value=30,
                value=18,
                step=1,
                key=f"{key_prefix}_m",
            )

    case = cases[case_key]
    a, b = float(case.a), float(case.b)

    # 단면 한 장 위치(x*) 슬라이더 (구간 내부)
    x_star = st.slider(
        "현재 단면 위치 x*",
        min_value=float(a),
        max_value=float(b),
        value=float((a + b) / 2.0),
        step=float((b - a) / 200.0) if (b - a) > 0 else 0.01,
        key=f"{key_prefix}_xstar",
    )

    S_func, S_tex = _S_and_tex(shape, case.f_tex)

    # -----------------------------
    # 수식(최소)
    # -----------------------------
    st.markdown("### 핵심")
    st.latex(rf"f(x)={case.f_tex},\qquad \text{{적분구간 }}[{case.a_tex},{case.b_tex}]")
    st.latex(S_tex)
    st.latex(r"V=\int_a^b S(x)\,dx")

    # 현재 단면 정보(값)
    fx_star = _safe_nonneg(case.f(float(x_star)))
    Sx_star = _safe_nonneg(S_func(float(x_star), case.f))
    if shape == "circle":
        st.caption(f"현재 x*에서 반지름 r = f(x*) = {fx_star:.6f},  단면 넓이 S(x*) = {Sx_star:.6f}")
    else:
        st.caption(f"현재 x*에서 한 변 길이 s = f(x*) = {fx_star:.6f},  단면 넓이 S(x*) = {Sx_star:.6f}")

    # -----------------------------
    # 시각화: 좌(2D), 우(단면 스택)
    # -----------------------------
    left, right = st.columns([1, 1])

    # ---- 좌: f(x) 그래프 + x* 표시
    with left:
        st.markdown("### 1) 함수 그래프와 단면 위치")

        fig1 = plt.figure(figsize=(6.2, 4.0))
        ax1 = fig1.add_subplot(111)

        xs = np.linspace(a, b, 600)
        ys = np.array([case.f(float(x)) for x in xs], dtype=float)

        ax1.plot(xs, ys, linewidth=1.6)
        ax1.axhline(0, linewidth=1)

        # x* 위치 표시
        ax1.axvline(float(x_star), linewidth=1)
        ax1.plot([float(x_star)], [case.f(float(x_star))], marker="o")

        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        st.pyplot(fig1)

    # ---- 우: 단면을 조금씩 이동해 겹쳐 그리는 "쌓기" 시각화
    with right:
        st.markdown("### 2) 단면을 쌓아 입체가 되는 모습(시각화)")

        # 샘플 x들: [a,b]에서 m개
        xs_m = np.linspace(a, b, int(m))
        # 시각화에서 "겹쳐 쌓는 느낌"을 위한 화면상 이동(오프셋)
        # 너무 크게 이동하면 분리되고, 너무 작으면 안 보임
        offset_step = 0.08  # 화면상의 x방향(그림 좌표) 이동량
        base_shift = 0.0

        # 단면 크기(반지름/한 변)의 스케일을 화면에 맞추기
        f_vals = np.array([_safe_nonneg(case.f(float(x))) for x in xs_m], dtype=float)
        max_size = float(np.max(f_vals)) if f_vals.size else 1.0
        if max_size <= 0:
            max_size = 1.0

        # 단면 그림은 y-z 평면을 2D로 그린다고 생각하고,
        # "쌓기 방향"은 화면에서 오른쪽으로 조금씩 이동시키는 방식(가짜 3D)
        fig2 = plt.figure(figsize=(6.2, 4.0))
        ax2 = fig2.add_subplot(111)
        ax2.set_aspect("equal", adjustable="box")

        # 축 범위: 단면 크기 기준으로 고정(안정)
        # y-z 평면을 [-R, R] × [-R, R]에 두고, shift만 오른쪽으로
        R = max_size
        x_min = -R - 0.2
        x_max = (int(m) - 1) * offset_step + R + 0.2
        y_min = -R - 0.2
        y_max = R + 0.2
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

        # 축 라벨은 최소
        ax2.set_xticks([])
        ax2.set_yticks([])

        # m개 단면을 순서대로 그리기 (투명도 낮게)
        # 뒤에서 앞으로: 작은 shift부터 큰 shift로 그리면 겹침이 자연스러움
        for i, x in enumerate(xs_m):
            size = _safe_nonneg(case.f(float(x)))
            shift = base_shift + i * offset_step

            if shape == "circle":
                # 원: 중심 (shift, 0), 반지름=size
                patch = Circle((shift, 0.0), radius=size, fill=False, linewidth=1.0, alpha=0.35)
            else:
                # 정사각형: 중심 (shift,0), 한 변=size -> 좌하단 기준으로 그림
                patch = Rectangle(
                    (shift - size / 2.0, -size / 2.0),
                    width=size,
                    height=size,
                    fill=False,
                    linewidth=1.0,
                    alpha=0.35,
                )

            ax2.add_patch(patch)

        # 현재 x* 단면은 진하게 강조(가장 오른쪽 끝에 별도로 표시)
        # "쌓기" 그림 내부에서도 한 장을 강조하고 싶으면, x*에 해당하는 위치를 찾아 강조
        # 여기서는 x*를 가장 가까운 샘플로 매칭
        idx = int(np.argmin(np.abs(xs_m - float(x_star)))) if xs_m.size else 0
        shift_star = base_shift + idx * offset_step
        size_star = _safe_nonneg(case.f(float(xs_m[idx]))) if xs_m.size else 0.0

        if shape == "circle":
            patch_star = Circle((shift_star, 0.0), radius=size_star, fill=False, linewidth=2.0, alpha=0.9)
        else:
            patch_star = Rectangle(
                (shift_star - size_star / 2.0, -size_star / 2.0),
                width=size_star,
                height=size_star,
                fill=False,
                linewidth=2.0,
                alpha=0.9,
            )
        ax2.add_patch(patch_star)

        # 간단한 안내선(중심선)
        ax2.axhline(0, linewidth=0.8, alpha=0.4)

        st.pyplot(fig2)

    # 짧은 안내 문장
    st.caption("단면을 더 촘촘히(= m을 크게) 쌓을수록, 입체도형의 ‘연속적인’ 모습에 가까워집니다.")
