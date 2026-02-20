# assessment/ai_step1_structure.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

from assessment.common import (
    init_assessment_session,
    require_student_id,
    set_save_status,
    render_save_status,
)

# ✅ NEW: loss registry
from assessment.ai_loss import (
    make_loss_spec,
    E as E_loss,
    grad as grad_loss,
    latex_E,
    recommended_step_size,
    LOSS_CATALOG,
)

TITLE = r"1차시: 구조(손실 지형) 관찰"

A_MIN, A_MAX = -3.0, 3.0
B_MIN, B_MAX = -3.0, 3.0

GRID_N = 121  # 고정 해상도(학생 선택 X) — 안정성 우선
DEFAULT_START_A = 2.2
DEFAULT_START_B = 2.2

COORD_STEPS = 18

# step_size는 함수에 따라 권장값이 다르므로, 기본값만 두고 실제 기본은 추천값으로 채움
STEP_SIZE_FALLBACK = 0.15

_BACKUP_STATE_KEY = "ai_step1_backup_payload"


@st.cache_data(show_spinner=False)
def build_grid(a_min: float, a_max: float, b_min: float, b_max: float, n: int, loss_type: str, params: dict):
    a = np.linspace(a_min, a_max, n)
    b = np.linspace(b_min, b_max, n)
    A, B = np.meshgrid(a, b)
    spec = make_loss_spec(loss_type, params)
    Z = E_loss(A, B, spec)
    return A, B, Z


def coord_descent_path(a0: float, b0: float, steps: int, step_size: float, loss_type: str, params: dict) -> np.ndarray:
    """
    한 번에 한 변수만 줄이는 이동(번갈아): 지그재그 관찰용
    - k 짝수: a만 이동
    - k 홀수: b만 이동
    """
    spec = make_loss_spec(loss_type, params)

    a, b = float(a0), float(b0)
    pts = [(a, b, float(E_loss(np.array(a), np.array(b), spec)))]

    for k in range(steps):
        da, db = grad_loss(a, b, spec)
        if k % 2 == 0:
            a = a - step_size * float(da)
        else:
            b = b - step_size * float(db)

        # 관찰 범위 안으로 클립 (안정성)
        a = float(np.clip(a, A_MIN, A_MAX))
        b = float(np.clip(b, B_MIN, B_MAX))
        pts.append((a, b, float(E_loss(np.array(a), np.array(b), spec))))

    return np.array(pts, dtype=float)


def build_backup_text(payload: dict) -> str:
    lines: list[str] = []
    lines.append("인공지능수학 수행평가 (1차시) 백업")
    lines.append("=" * 46)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")

    # ✅ loss_spec 출력(고정 alpha/beta 제거)
    loss_type = payload.get("loss_type", "")
    params = payload.get("loss_params", {}) or {}
    try:
        spec = make_loss_spec(loss_type, params) if loss_type else None
    except Exception:
        spec = None

    lines.append("[함수 설정]")
    if spec is not None:
        lines.append(f"- 난이도: Lv{LOSS_CATALOG[spec.type]['level']} ({LOSS_CATALOG[spec.type]['label']})")
        lines.append(f"- loss_type: {spec.type}")
        lines.append(f"- params: {dict(spec.params)}")
        lines.append(f"- {latex_E(spec)}")
    else:
        lines.append("- (함수 설정 정보를 불러오지 못했습니다)")

    lines.append(f"- 관찰 범위: a∈[{A_MIN:g},{A_MAX:g}], b∈[{B_MIN:g},{B_MAX:g}]")
    lines.append("")

    lines.append("[학생 입력(서술)]")
    lines.append("1) 전체 형태/최소점 관찰:")
    lines.append(payload.get("obs_shape", "").strip())
    lines.append("")
    lines.append("2) 민감도 큰 방향 + 근거(등고선/단면 등):")
    lines.append(payload.get("obs_sensitivity", "").strip())
    lines.append("")
    lines.append("3) 좌표축 방향 이동(지그재그) 관찰 + 이유:")
    lines.append(payload.get("obs_zigzag", "").strip())
    lines.append("")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title=TITLE, layout="wide")

    init_assessment_session()
    student_id = require_student_id()

    st.title(TITLE)

    # -------------------------
    # ✅ 학번 확인 이후: 손실함수 선택 UI (NEW)
    # -------------------------
    st.subheader("0) 손실함수 선택(난이도/계수)")

    prev = st.session_state.get("ai_loss_spec", {}) if isinstance(st.session_state.get("ai_loss_spec", {}), dict) else {}
    default_type = prev.get("type", "quad")

    type_options = ["quad", "double_well", "banana"]
    type_labels = {t: LOSS_CATALOG[t]["label"] for t in type_options}

    loss_type = st.radio(
        "손실함수 유형",
        options=type_options,
        format_func=lambda t: type_labels.get(t, t),
        index=type_options.index(default_type) if default_type in type_options else 0,
        horizontal=True,
    )

    meta = LOSS_CATALOG[loss_type]
    st.caption(meta["description"])

    param_key = meta["params"][0]  # 본 설계: 1개
    lo, hi = meta["param_ranges"][param_key]
    default_val = float(prev.get("params", {}).get(param_key, meta["default_params"][param_key]))

    param_val = st.slider(
        f"계수 선택 ({param_key})",
        min_value=float(lo),
        max_value=float(hi),
        value=float(np.clip(default_val, lo, hi)),
        step=0.5 if (hi - lo) >= 10 else 0.1,
    )

    loss_spec = make_loss_spec(loss_type, {param_key: param_val})
    step_hint = float(prev.get("recommended_step", recommended_step_size(loss_spec)))

    st.markdown("**선택된 손실함수:**")
    st.latex(latex_E(loss_spec))
    st.info(f"2차시 추천 step_size(참고): {recommended_step_size(loss_spec):.4f}")

    # 세션 저장(2차시에서 동일 함수로 진행)
    st.session_state["ai_loss_spec"] = {
        "type": loss_spec.type,
        "level": loss_spec.level,
        "label": loss_spec.label,
        "params": dict(loss_spec.params),
        "recommended_step": float(recommended_step_size(loss_spec)),
    }

    # -------------------------
    # ✅ 기존 안내문(고정 alpha/beta 제거하고 일반화)
    # -------------------------
    st.markdown(
        r"""
손실함수의 그래프 $z=E(a,b)$는 하나의 곡면이며, 이를 직관적으로 손실 지형(loss landscape)이라고 부릅니다.

위에서 선택한 손실함수 $E(a,b)$의 손실 지형을 관찰하며,
손실함수 값이 최소가 되는 지점과 그 방향적 특징을 분석해 봅시다.

관찰 포인트:
- 전역 최소점(global minimum) 또는 최소점(들)의 위치
- 좌표축에 대한 대칭성/비대칭성
- 방향에 따른 기울기 크기(가파름)
- 한 변수만 줄이는(좌표축 방향) 이동에서 나타나는 경로의 특징
"""
    )

    # -------------------------
    # 상단: 좌(①②) / 우(시각화)
    # -------------------------
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("① 현재 위치 선택")

        a0 = st.slider("$a$ 값", min_value=A_MIN, max_value=A_MAX, value=DEFAULT_START_A, step=0.05)
        b0 = st.slider("$b$ 값", min_value=B_MIN, max_value=B_MAX, value=DEFAULT_START_B, step=0.05)

        e0 = float(E_loss(np.array(a0), np.array(b0), loss_spec))
        st.metric("현재 손실", f"{e0:.6f}")

        st.markdown(
            r"""
- 해석의 기준: 등고선 간격이 **더 촘촘한 방향**일수록, 같은 거리 이동에서 손실 변화가 더 큽니다.
"""
        )

        st.divider()
        st.subheader("② 한 변수만 줄이는 이동")

        st.markdown(
            r"""
아래 버튼은 **$a$만, $b$만 번갈아** 이동하는 경로를 그립니다.  
이 경로의 특징을 ③에서 서술하세요.
"""
        )

        run_coord = st.button("▶ 좌표축 방향 이동(지그재그 관찰)", type="primary")

    with right:
        st.subheader("손실 지형 시각화")

        # ✅ grid가 선택 함수에 따라 바뀜
        A, B, Z = build_grid(A_MIN, A_MAX, B_MIN, B_MAX, GRID_N, loss_spec.type, dict(loss_spec.params))

        # ✅ path도 선택 함수에 따라 바뀜
        step_size_for_path = STEP_SIZE_FALLBACK
        path = (
            coord_descent_path(a0, b0, steps=COORD_STEPS, step_size=step_size_for_path, loss_type=loss_spec.type, params=dict(loss_spec.params))
            if run_coord
            else None
        )

        tab1, tab2 = st.tabs(["2D 등고선", "3D 손실곡면"])

        with tab1:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(
                    go.Contour(
                        x=np.linspace(A_MIN, A_MAX, GRID_N),
                        y=np.linspace(B_MIN, B_MAX, GRID_N),
                        z=Z,
                        contours=dict(showlabels=False),
                        line=dict(width=1),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[a0],
                        y=[b0],
                        mode="markers+text",
                        text=["현재"],
                        textposition="top center",
                        marker=dict(size=10),
                        name="현재 위치",
                    )
                )
                if path is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=path[:, 0],
                            y=path[:, 1],
                            mode="lines+markers",
                            marker=dict(size=5),
                            name="축만 번갈아 이동 경로",
                        )
                    )

                fig.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="a",
                    yaxis_title="b",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                cs = ax.contour(A, B, Z, levels=18)
                ax.clabel(cs, inline=True, fontsize=8)
                ax.scatter([a0], [b0], s=60)
                ax.text(a0, b0, "현재", fontsize=10)
                if path is not None:
                    ax.plot(path[:, 0], path[:, 1], marker="o")
                ax.set_xlabel("a")
                ax.set_ylabel("b")
                ax.set_title("Contour of E(a,b)")
                st.pyplot(fig, clear_figure=True)

            st.markdown(
                r"""
$$
\text{등고선 간격이 촘촘한 방향 } \Rightarrow \text{ 더 가파른 방향}
$$
"""
            )

        with tab2:
            if PLOTLY_AVAILABLE:
                surf = go.Surface(x=A, y=B, z=Z, showscale=False, opacity=0.95)
                fig3d = go.Figure(data=[surf])
                fig3d.add_trace(
                    go.Scatter3d(
                        x=[a0],
                        y=[b0],
                        z=[e0],
                        mode="markers+text",
                        text=["현재"],
                        textposition="top center",
                        marker=dict(size=5),
                        name="현재 위치",
                    )
                )
                fig3d.update_layout(
                    height=520,
                    margin=dict(l=10, r=10, t=10, b=10),
                    scene=dict(xaxis_title="a", yaxis_title="b", zaxis_title="E(a,b)"),
                )
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("3D 표면은 Plotly가 필요합니다. (현재 환경에서는 2D 등고선으로 충분합니다.)")

            st.markdown(
                r"""
$$
E(a,b) \text{ 는 손실 지형으로 해석할 수 있습니다. (선택한 함수에 따라 모양이 달라집니다.)}
$$
"""
            )

    # -------------------------
    # 하단(전체 폭): ③ 서술 + 백업 + 저장/이동 + 저장상태
    # -------------------------
    st.divider()
    st.subheader("③ 관찰 기록 서술")

    obs_shape = st.text_area(
        "1) 전역 최소점의 위치와 손실 지형의 전체적인 형태를 함께 설명하시오.",
        height=90,
        placeholder="예: 전역 최소점(또는 최소점들)의 좌표, 그 주변에서 함숫값이 어떻게 변하는지, 손실 지형의 전체적인 형태 서술",
        key="ai_step1_obs_shape",
    )

    obs_sensitivity = st.text_area(
        "2) 같은 거리만큼 이동했을 때 손실이 더 크게 변하는 방향은 어느 쪽인가? 등고선의 모양 또는 간격을 근거로 설명하시오.",
        height=110,
        placeholder="예: 어느 방향이 더 가파른지, 등고선 간격이나 모양이 어떤지 서술",
        key="ai_step1_obs_sensitivity",
    )

    obs_zigzag = st.text_area(
        "3) 좌표축 방향 이동했을 때 경로는 어떤 특징을 보이는가? 그 이유를 설명하시오.",
        height=120,
        placeholder="예: 경로의 모양 설명, 그렇게 되는 수학적 이유 서술",
        key="ai_step1_obs_zigzag",
    )

    st.caption("※ 구체적인 좌표, 방향, 등고선 근거를 포함하여 작성하세요.")

    # -------------------------
    # 저장/이동 처리
    # -------------------------
    def _validate_inputs() -> tuple[bool, str]:
        if not obs_shape.strip():
            return False, "서술 1) 전체 형태/최소점 관찰을 입력하세요."
        if not obs_sensitivity.strip():
            return False, "서술 2) 민감도 방향과 근거를 입력하세요."
        if not obs_zigzag.strip():
            return False, "서술 3) 지그재그 관찰과 이유를 입력하세요."
        return True, "OK"

    payload_for_backup = {
        "student_id": student_id,
        "loss_type": loss_spec.type,
        "loss_params": dict(loss_spec.params),
        "obs_shape": obs_shape,
        "obs_sensitivity": obs_sensitivity,
        "obs_zigzag": obs_zigzag,
    }

    saved_payload = st.session_state.get(_BACKUP_STATE_KEY) or None
    backup_payload = saved_payload if isinstance(saved_payload, dict) and saved_payload.get("student_id") == student_id else payload_for_backup
    backup_text = build_backup_text(backup_payload)

    cA, cB = st.columns([1, 1], gap="large")
    with cA:
        backup_make_clicked = st.button("⬇️ TXT 백업 만들기", use_container_width=True)
        st.download_button(
            label="📄 (다운로드) 1차시 백업 TXT",
            data=backup_text.encode("utf-8-sig"),
            file_name=f"인공지능_수행평가_1차시_{student_id}.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
        )

    with cB:
        btn1, btn2 = st.columns(2, gap="small")
        with btn1:
            save_clicked = st.button("✅ 제출/저장", use_container_width=True)
        with btn2:
            go_next = st.button("➡️ 2차시로 이동", use_container_width=True)

    if backup_make_clicked:
        ok, msg = _validate_inputs()
        if not ok:
            st.error(msg)
            st.stop()
        st.session_state[_BACKUP_STATE_KEY] = dict(payload_for_backup)
        st.rerun()

    render_save_status()

    if save_clicked or go_next:
        ok, msg = _validate_inputs()
        if not ok:
            st.error(msg)
            st.stop()

        # ✅ 세션 저장(2차시에서 참조)
        st.session_state["ai_step1_structure"] = {
            "student_id": student_id,
            "loss_spec": {
                "type": loss_spec.type,
                "level": loss_spec.level,
                "label": loss_spec.label,
                "params": dict(loss_spec.params),
                "recommended_step": float(recommended_step_size(loss_spec)),
            },
            "range": {"a": [A_MIN, A_MAX], "b": [B_MIN, B_MAX]},
            "start_point": {"a": float(a0), "b": float(b0)},
            "obs_shape": obs_shape.strip(),
            "obs_sensitivity": obs_sensitivity.strip(),
            "obs_zigzag": obs_zigzag.strip(),
            "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }

        # ✅ Google Sheet 저장(기존 함수명 유지하되, alpha/beta 대신 loss_type/param 저장)
        #    (google_sheets.py 컬럼이 아직 alpha/beta로만 되어 있다면, 그쪽도 함께 수정 필요)
        try:
            from assessment.google_sheets import append_ai_step1_row  # late import

            append_ai_step1_row(
                student_id=student_id,
                loss_type=loss_spec.type,
                loss_params=str(dict(loss_spec.params)),
                a0=float(a0),
                b0=float(b0),
                obs_shape=obs_shape.strip(),
                obs_sensitivity=obs_sensitivity.strip(),
                obs_zigzag=obs_zigzag.strip(),
            )
            set_save_status(True, "구글시트 저장 완료")
        except TypeError:
            # 기존 시트 함수가 alpha/beta만 받는 경우를 대비(최소한 깨지지 않게)
            try:
                append_ai_step1_row(
                    student_id=student_id,
                    alpha=float(loss_spec.params.get("alpha", 0.0)),
                    beta=1.0,
                    a0=float(a0),
                    b0=float(b0),
                    obs_shape=obs_shape.strip(),
                    obs_sensitivity=obs_sensitivity.strip(),
                    obs_zigzag=obs_zigzag.strip(),
                )
                set_save_status(True, "구글시트 저장(구형 포맷) 완료")
            except Exception as e:
                set_save_status(False, f"구글시트 저장 실패: {e}")
        except Exception as e:
            set_save_status(False, f"구글시트 저장 실패: {e}")

        if go_next:
            st.switch_page("assessment/ai_step2_path.py")
        else:
            st.rerun()


if __name__ == "__main__":
    main()
