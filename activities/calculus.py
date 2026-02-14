# activities/calculus.py
# 미적분 탐구활동 라우터 페이지
# - 단원: 버튼식
# - 활동: 탭
# - 단원 반복 표기 제거
# - 활동명은 TITLE 그대로 사용

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
# 2. 탐구활동 모듈 import
# --------------------------------------------------
import calculus_geometric_sequence_limit as geom_seq_limit
import calculus_geometric_series_sum as geom_series_sum

# (Ⅱ. 미분법) 추가 예정
# import calculus_derivative_limit_definition as deriv_def

# (Ⅲ. 적분법) 추가 예정
# import calculus_riemann_sum_area as riemann_area


# --------------------------------------------------
# 3. 단원별 활동 등록
# --------------------------------------------------
UNIT_ACTIVITIES = {
    "Ⅰ. 수열의 극한": [
        geom_seq_limit,
        geom_series_sum,
    ],
    "Ⅱ. 미분법": [
        # deriv_def,
    ],
    "Ⅲ. 적분법": [
        # riemann_area,
    ],
}


def _init_state():
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = list(UNIT_ACTIVITIES.keys())[0]


def _render_activity(module):
    """
    활동 모듈 호출
    - render(show_title=False) 지원 시 제목 숨김
    - 미지원 시 기본 render() 호출
    """
    try:
        module.render(show_title=False)  # type: ignore
    except TypeError:
        module.render()


def main():
    st.set_page_config(page_title="미적분 탐구활동", layout="wide")
    _init_state()

    st.title("📘 미적분 탐구활동")

    # --------------------------------------------------
    # 단원 선택 버튼
    # --------------------------------------------------
    unit_names = list(UNIT_ACTIVITIES.keys())
    cols = st.columns(len(unit_names))

    for i, unit in enumerate(unit_names):
        is_selected = (st.session_state.selected_unit == unit)
        label = f"✅ {unit}" if is_selected else unit

        if cols[i].button(label, use_container_width=True):
            st.session_state.selected_unit = unit
            st.rerun()

    selected_unit = st.session_state.selected_unit
    activities = UNIT_ACTIVITIES[selected_unit]

    if not activities:
        st.info("이 단원에 연결된 탐구활동이 아직 없습니다.")
        return

    # --------------------------------------------------
    # 활동 탭 (TITLE 그대로 사용)
    # --------------------------------------------------
    tab_labels = [module.TITLE for module in activities]
    tabs = st.tabs(tab_labels)

    for tab, module in zip(tabs, activities):
        with tab:
            _render_activity(module)


if __name__ == "__main__":
    main()
