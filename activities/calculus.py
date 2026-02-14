# activities/calculus.py
# 미적분 시뮬레이션 라우터 페이지

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st


# --------------------------------------------------
# 1. 현재 폴더를 모듈 탐색 경로에 추가
#    (Streamlit Navigation 환경에서 ModuleNotFoundError 방지)
# --------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# --------------------------------------------------
# 2. 시뮬레이션 모듈 import
#    (같은 폴더에 있어야 함)
# --------------------------------------------------
import calculus_geometric_sequence_limit as geom_seq_limit


# --------------------------------------------------
# 3. 시뮬레이션 등록
# --------------------------------------------------
SIMULATIONS = {
    geom_seq_limit.TITLE: geom_seq_limit,
}


# --------------------------------------------------
# 4. 메인 라우터
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="미적분 시뮬레이션",
        layout="wide",
    )

    st.title("📘 미적분 시뮬레이션")
    st.markdown("단원을 선택하여 개념을 탐구하세요.")

    st.divider()

    # 단원 구분 (현재는 수열의 극한만 구성)
    st.header("수열의 극한")

    selected_title = st.selectbox(
        "실행할 시뮬레이션을 선택하세요",
        list(SIMULATIONS.keys()),
    )

    st.divider()

    # 선택된 시뮬레이션 실행
    selected_module = SIMULATIONS[selected_title]
    selected_module.render()


# --------------------------------------------------
# 5. 실행 진입점
# --------------------------------------------------
if __name__ == "__main__":
    main()
