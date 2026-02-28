# activities/calculus_volume.py
# 단면을 쌓아 만드는 부피(3D 슬라이스 시각화, matplotlib mplot3d)
# - 좌: f(x) 그래프 + 현재 단면 위치 x*
# - 우: x축 방향으로 여러 단면(슬라이스)을 3D로 배치해 "쌓여서 입체가 되는 느낌" 시각화
# - 슬라이스 개수는 제한(M_MAX)하여 과부하 방지
# - 수식은 최소: S(x), V=∫_a^b S(x)dx

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

TITLE = "단면을 쌓아 만드는 부피"

# 과부하 방지: 실제로 그리는 슬라이스 최대 개수
M_MAX = 18
# 원 단면을 그릴 때 원 둘레 분할(너무 크면 느려짐)
CIRCLE_PTS = 60


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


def _S_tex(shape: str, f_tex: str) -> str:
    if shape == "circle":
        return rf"S(x)=\pi\{{f(x)\}}^2=\pi\left({f_tex}\right)^2"
    return rf"S(x)=\{{f(x)\}}^2=\left({f_tex}\right)^2"


def _safe_nonneg(v: float) -> float:
    if not math.isfinite(v):
        return 0.0
    return max(0.0, float(v))


def _circle_polygon(x: float, r: float) -> np.ndarray:
    # return (N,3) points on circle in yz-plane at given x
    th = np.linspace(0, 2 * math.pi, CIRCLE_PTS, endpoint=True)
    y = r * np.cos(th)
    z = r * np.sin(th)
    xs = np.full_like(y, x, dtype=float)
    return np.column_stack([xs, y, z])


def _square_polygon(x: float, s: float) -> np.ndarray:
    # square centered at origin in yz-plane
    h = s / 2.0
    pts = np.array(
        [
            [x, -h, -h],
            [x, +h, -h],
            [x, +h, +h],
            [x, -h, +h],
        ],
        dtype=float,
    )
    return pts


def render(show_title: bool = True, key_prefix: str = "cal_volume3d") -> None:
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
            # 사용자가 고르는 값은 넉넉히 두되, 실제 렌더는 M_MAX로 제한
            m_user = st.slider(
                "단면(슬라이스) 개수",
                min_value=5,
                max_value=M_MAX,
                value=min(14, M_MAX),
                step=1,
                key=f"{key_prefix}_m",
            )

    case = cases[case_key]
    a, b = float(case.a), float(case.b)

    # 현재 단면 위치 x*
    x_star = st.slider(
        "현재 단면 위치 x*",
        min_value=float(a),
        max_value=float(b),
        value=float((a + b) / 2.0),
        step=float((b - a) / 200.0) if (b - a) > 0 else 0.01,
        key=f"{key_prefix}_xstar",
    )

    # 수식(최소)
    st.markdown("### 핵심")
    st.latex(rf"f(x)={case.f_tex},\qquad \text{{적분구간 }}[{case.a_tex},{case.b_tex}]")
    st.latex(_S_tex(shape, case.f_tex))
    st.latex(r"V=\int_a^b S(x)\,dx")

    fx_star = _safe_nonneg(case.f(float(x_star)))
    if shape == "circle":
        st.caption(f"현재 x*에서 반지름 r = f(x*) = {fx_star:.6f}")
    else:
        st.caption(f"현재 x*에서 한 변 길이 s = f(x*) = {fx_star:.6f}")

    left, right = st.columns([1, 1])

    # -----------------------------
    # 좌: 2D 그래프
    # -----------------------------
    with left:
        st.markdown("### 1) 함수 그래프와 단면 위치")

        fig1 = plt.figure(figsize=(6.2, 4.0))
        ax1 = fig1.add_subplot(111)

        xs = np.linspace(a, b, 700)
        ys = np.array([case.f(float(x)) for x in xs], dtype=float)

        ax1.plot(xs, ys, linewidth=1.6)
        ax1.axhline(0, linewidth=1)
        ax1.axvline(float(x_star), linewidth=1)
        ax1.plot([float(x_star)], [case.f(float(x_star))], marker="o")

        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        st.pyplot(fig1)

    # -----------------------------
    # 우: 3D 슬라이스 시각화
    # -----------------------------
    with right:
        st.markdown("### 2) 단면을 쌓아 입체가 되는 모습(3D 슬라이스)")

        m = int(min(m_user, M_MAX))
        xs_m = np.linspace(a, b, m)

        # 크기 스케일(축 범위 안정)
        sizes = np.array([_safe_nonneg(case.f(float(x))) for x in xs_m], dtype=float)
        R = float(np.max(sizes)) if sizes.size else 1.0
        if R <= 0:
            R = 1.0

        fig2 = plt.figure(figsize=(6.2, 4.2))
        ax2 = fig2.add_subplot(111, projection="3d")

        # 보기 각도(교과서 느낌: 약간 위에서 사선으로)
        ax2.view_init(elev=18, azim=-55)

        # 슬라이스(반투명 면 + 테두리)
        polys = []
        for x, s in zip(xs_m, sizes):
            if shape == "circle":
                pts = _circle_polygon(float(x), float(s))
            else:
                pts = _square_polygon(float(x), float(s))
            polys.append(pts)

            # 테두리 라인
            ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.8, alpha=0.35)

        # 면(폴리곤) 반투명 채우기
        poly3d = [p.tolist() for p in polys]
        coll = Poly3DCollection(poly3d, alpha=0.12)
        ax2.add_collection3d(coll)

        # 현재 x*에 가장 가까운 슬라이스를 강조
        idx = int(np.argmin(np.abs(xs_m - float(x_star)))) if xs_m.size else 0
        xh = float(xs_m[idx])
        sh = float(sizes[idx])
        if shape == "circle":
            hi = _circle_polygon(xh, sh)
        else:
            hi = _square_polygon(xh, sh)
            # 사각형은 닫힌 선으로 보이게
            hi = np.vstack([hi, hi[0]])

        ax2.plot(hi[:, 0], hi[:, 1], hi[:, 2], linewidth=2.0, alpha=0.9)

        # 가이드: x축(적분방향)
        ax2.plot([a, b], [0, 0], [0, 0], linewidth=1.0, alpha=0.5)

        # 축 범위/표시 최소화
        ax2.set_xlim(a, b)
        ax2.set_ylim(-R, R)
        ax2.set_zlim(-R, R)
        ax2.set_xlabel("x")
        ax2.set_ylabel("")
        ax2.set_zlabel("")
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.grid(False)

        st.pyplot(fig2)

    st.caption("단면(슬라이스) 개수를 늘리면, 입체의 ‘연속적인’ 모습에 더 가까워집니다.")
