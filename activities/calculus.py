# activities/calculus.py
# 미적분 탐구활동 라우터 페이지 (단원 선택형)

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
# 2. 시뮬레이션(탐구활동) 모듈 import
#    - 파일은 activities/ 폴더에 있어야 함
# --------------------------------------------------
import calculus_geometric_sequence_limit as geom_seq_limit
import calculus_geometric_series_sum as geom_series_sum

# (Ⅱ. 미분법) 활동을 만들면 아래처럼 import 추가
# import calculus_derivative_limit_definition as deriv_def
# import calculus_tangent_slope as tangent_slope

# --------------------------------------------------
# 3. 단원별 활동 등록
# --------------------------------------------------
UNIT_SIMULATIONS = {
    "Ⅰ. 수열의 극한": {
        geom_seq_limit.TITLE: geom_seq_limit,
        geom_series_sum.TITLE: geom_series_sum,
    },
    "Ⅱ. 미분법": {
        # 예시) 미분법 활동 파일을 만들면 아래처럼 추가
        # deriv_def.TITLE: deriv_def,
        # tangent_slope.TITLE: tangent_slope,
    },
}


def main():
    st.set_page_config(page_title="미적분 탐구활동", layout="wide")

    st.title("📘 미적분 탐구활동")
    st.divider()

    # --------------------------------------------------
    # 단원 선택
    # --------------------------------------------------
    unit_names = list(UNIT_SIMULATIONS.keys())
    selected_unit = st.radio("단원을 선택하세요", unit_names, horizontal=True)

    st.divider()
    st.header(selected_unit)

    # --------------------------------------------------
    # 단원 내 활동 선택
    # --------------------------------------------------
    sims = UNIT_SIMULATIONS[selected_unit]

    if not sims:
        st.info("이 단원에 연결된 탐구활동이 아직 없습니다. 활동 파일을 추가한 뒤 등록해주세요.")
        return

    selected_title = st.selectbox("탐구활동을 선택하세요", list(sims.keys()))
    st.divider()

    # 실행
    sims[selected_title].render()


if __name__ == "__main__":
    main()
