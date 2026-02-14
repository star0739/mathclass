# activities/calculus.py
# 미적분 탐구활동 라우터 페이지 (단원 버튼 선택형: Ⅰ/Ⅱ/Ⅲ)

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
# 3. 단원별 활동 등록
# --------------------------------------------------
UNIT_SIMULATIONS = {
    "Ⅰ. 수열의 극한": {
        geom_seq_limit.TITLE: geom_seq_limit,
        geom_series_sum.TITLE: geom_series_sum,
    },
    "Ⅱ. 미분법": {
        # deriv_def.TITLE: deriv_def,
        # tangent_slope.TITLE: tangent_slope,
    },
    "Ⅲ. 적분법": {
        # riemann_area.TITLE: riemann_area,
        # definite_area.TITLE: definite_area,
    },
}


def _init_state():
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = list(UNIT_SIMULATIONS.keys())[0]


def main():
    st.set_page_config(page_title="미적분 탐구활동", layout="wide")
    _init_state()

    st.title("📖 미적분 탐구활동")
    st.divider()

    # --------------------------------------------------
    # 단원 선택 (버튼식)
    # --------------------------------------------------
    st.subheader("단원 선택")

    unit_names = list(UNIT_SIMULATIONS.keys())
    cols = st.columns(len(unit_names))

    for i, unit in enumerate(unit_names):
        is_selected = (st.session_state.selected_unit == unit)
        label = f"✅ {unit}" if is_selected else unit

        if cols[i].button(label, use_container_width=True):
            st.session_state.selected_unit = unit
            st.rerun()

    st.divider()

    selected_unit = st.session_state.selected_unit
    st.header(selected_unit)

    # --------------------------------------------------
    # 단원 내 활동 선택
    # --------------------------------------------------
    sims = UNIT_SIMULATIONS[selected_unit]

    if not sims:
        st.info("이 단원에 연결된 탐구활동이 아직 없습니다. ")
        return

    selected_title = st.selectbox("탐구활동을 선택하세요", list(sims.keys()))
    st.divider()

    sims[selected_title].render()


if __name__ == "__main__":
    main()
