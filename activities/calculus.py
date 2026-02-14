# activities/calculus.py
# 미적분 탐구활동 라우터 페이지 (단원=버튼, 활동=탭)
# - 단원명 반복 제거: 라우터에서만 표시
# - 활동명 과대 타이틀 제거: 활동 페이지는 "타이틀 숨김 모드"를 지원/가정
#   (각 활동 모듈에 render(show_title: bool = True) 형태로 맞추면 가장 깔끔)
#
# ⚠️ 현재 활동 모듈이 render()만 갖고 있다면,
#    아래 코드는 render(show_title=False)를 먼저 시도하고,
#    지원하지 않으면 render()로 폴백합니다.

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# --------------------------------------------------
# 1. 현재 폴더를 모듈 탐색 경로에 추가
# --------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# --------------------------------------------------
# 2. 탐구활동 모듈 import (activities/ 폴더 내)
# --------------------------------------------------
import calculus_geometric_sequence_limit as geom_seq_limit
import calculus_geometric_series_sum as geom_series_sum

# (Ⅱ. 미분법) 활동을 만들면 아래처럼 import 추가
# import calculus_derivative_limit_definition as deriv_def
# import calculus_tangent_slope as tangent_slope

# (Ⅲ. 적분법) 활동을 만들면 아래처럼 import 추가
# import calculus_riemann_sum_area as riemann_area
# import calculus_definite_integral_area as definite_area

# --------------------------------------------------
# 3. 단원별 활동 등록 (활동 수 2~4개 수준을 전제로 "탭" 사용)
#    각 항목: (탭 라벨, 모듈)
# --------------------------------------------------
UNIT_ACTIVITIES = {
    "Ⅰ. 수열의 극한": [
        ("등비수열", geom_seq_limit),
        ("등비급수", geom_series_sum),
    ],
    "Ⅱ. 미분법": [
        # ("미분계수의 정의", deriv_def),
        # ("접선의 기울기", tangent_slope),
    ],
    "Ⅲ. 적분법": [
        # ("리만합", riemann_area),
        # ("정적분과 넓이", definite_area),
    ],
}


def _init_state():
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = list(UNIT_ACTIVITIES.keys())[0]


def _render_activity(module):
    """
    활동 모듈 호출:
    - 가능하면 render(show_title=False)로 호출 (라우터에서 탭이 제목 역할)
    - 시그니처가 없으면 render()로 폴백
    """
    try:
        module.render(show_title=False)  # type: ignore[arg-type]
    except TypeError:
        module.render()


def main():
    st.set_page_config(page_title="미적분 탐구활동", layout="wide")
    _init_state()

    st.title("📘 미적분 탐구활동")

    # --------------------------------------------------
    # 단원 선택 (버튼식)
    # --------------------------------------------------
    st.markdown("#### 단원 선택")

    unit_names = list(UNIT_ACTIVITIES.keys())
    cols = st.columns(len(unit_names))

    for i, unit in enumerate(unit_names):
        is_selected = (st.session_state.selected_unit == unit)
        label = f"✅ {unit}" if is_selected else unit

        if cols[i].button(label, use_container_width=True):
            st.session_state.selected_unit = unit
            st.rerun()

    # 현재 선택 단원
    selected_unit = st.session_state.selected_unit
    st.markdown(f"**현재 단원:** {selected_unit}")

    activities = UNIT_ACTIVITIES[selected_unit]
    if not activities:
        st.info("이 단원에 연결된 탐구활동이 아직 없습니다. 활동 파일을 추가한 뒤 등록해주세요.")
        return

    # --------------------------------------------------
    # 활동 선택 (탭)
    # --------------------------------------------------
    tab_labels = [label for (label, _module) in activities]
    tabs = st.tabs(tab_labels)

    for tab, (_label, module) in zip(tabs, activities):
        with tab:
            _render_activity(module)


if __name__ == "__main__":
    main()
