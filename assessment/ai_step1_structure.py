# assessment/ai_step1_structure.py

from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

from assessment.ai_loss import (
    make_loss_spec,
    E as E_loss,
    grad as grad_loss,
    latex_E,
    recommended_step_size,
    LOSS_CATALOG,
)

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0
GRID_N = 200


# --------------------------------------------------
# 등고선 생성
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def build_grid(a_min, a_max, b_min, b_max, n, loss_type, params):
    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    spec = make_loss_spec(loss_type, params)
    Z = E_loss(A, B, spec)
    return A, B, Z


# --------------------------------------------------
# 좌표축 번갈아 이동 (지그재그)
# --------------------------------------------------

def coord_descent_path(a0, b0, steps, step_size, loss_type, params):
    spec = make_loss_spec(loss_type, params)
    a, b = float(a0), float(b0)
    path = [(a, b)]

    for i in range(steps):
        dE_da, dE_db = grad_loss(a, b, spec)

        if i % 2 == 0:
            # a 방향만 이동
            a = a - step_size * dE_da
        else:
            # b 방향만 이동
            b = b - step_size * dE_db

        path.append((a, b))

    return np.array(path)


# --------------------------------------------------
# 백업 TXT 생성
# --------------------------------------------------

def build_backup_text(student_id, loss_spec):
    lines = []
    lines.append("=== 인공지능 수학 수행평가 1차시 백업 ===")
    lines.append(f"학번: {student_id}")
    lines.append(f"저장시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("[함수 설정]")
    lines.append(f"- loss_type: {loss_spec.type}")
    lines.append(f"- params: {loss_spec.params}")
    lines.append(f"- {latex_E(loss_spec)}")
    lines.append(f"- 관찰 범위: a∈[{A_MIN},{A_MAX}], b∈[{B_MIN},{B_MAX}]")
    return "\n".join(lines)


# --------------------------------------------------
# 메인
# --------------------------------------------------

def main():

    st.title("1차시: 손실함수 구조 관찰")

    # ----------------------------------------
    # 0) 함수 선택
    # ----------------------------------------
    st.subheader("0) 손실함수 선택")

    type_options = ["quad", "double_well", "banana"]
    type_labels = {t: LOSS_CATALOG[t]["label"] for t in type_options}

    loss_type = st.radio(
        "손실함수 유형",
        options=type_options,
        format_func=lambda t: type_labels[t],
    )

    meta = LOSS_CATALOG[loss_type]
    st.caption(meta["description"])

    param_key = meta["params"][0]
    lo, hi = meta["param_ranges"][param_key]
    default_val = meta["default_params"][param_key]

    param_val = st.slider(
        f"{param_key} 값 선택",
        min_value=float(lo),
        max_value=float(hi),
        value=float(default_val),
        step=0.5 if (hi - lo) > 5 else 0.1,
    )

    loss_spec = make_loss_spec(loss_type, {param_key: param_val})

    st.markdown("### 선택된 손실함수")
    st.latex(latex_E(loss_spec))

    step_hint = recommended_step_size(loss_spec)
    st.info(f"2차시 추천 step_size: {step_hint:.4f}")

    # ----------------------------------------
    # 1) 등고선 시각화
    # ----------------------------------------

    A, B, Z = build_grid(A_MIN, A_MAX, B_MIN, B_MAX, GRID_N,
                         loss_spec.type, loss_spec.params)

    fig, ax = plt.subplots()
    cs = ax.contour(A, B, Z, levels=20)
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("손실함수 등고선")

    # ----------------------------------------
    # 2) 시작점 + 지그재그 경로
    # ----------------------------------------

    st.subheader("1) 좌표축 번갈아 이동 관찰")

    col1, col2 = st.columns(2)
    with col1:
        a0 = st.number_input("초기 a", value=2.0)
    with col2:
        b0 = st.number_input("초기 b", value=2.0)

    steps = st.slider("이동 횟수", 2, 20, 8)
    step_size = st.number_input("step_size", value=float(step_hint))

    path = coord_descent_path(a0, b0, steps, step_size,
                              loss_spec.type, loss_spec.params)

    ax.plot(path[:, 0], path[:, 1], marker="o", color="red")
    st.pyplot(fig)

    # ----------------------------------------
    # 3) 백업 / 다음 단계
    # ----------------------------------------

    st.markdown("---")

    student_id = st.text_input("학번 입력")

    if student_id:
        backup_text = build_backup_text(student_id, loss_spec)

        st.download_button(
            "📄 1차시 백업 TXT 다운로드",
            data=backup_text.encode("utf-8-sig"),
            file_name=f"인공지능_수행평가_1차시_{student_id}.txt",
            mime="text/plain",
        )

        # 2차시로 넘길 세션 저장
        st.session_state["ai_loss_spec"] = {
            "type": loss_spec.type,
            "params": dict(loss_spec.params),
            "recommended_step": float(step_hint),
        }

        st.success("2차시로 이동 가능합니다.")


if __name__ == "__main__":
    main()
