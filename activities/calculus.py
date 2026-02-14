# activities/calculus.py
# 미적분 탐구활동 라우터 페이지

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
# 2. 탐구활동 모듈 import (한 곳에서만)
# --------------------------------------------------
import calculus_geometric_sequence_limit as geom_seq_limit
import calculus_geometric_series_sum as geom_series_sum

# --------------------------------------------------
# 3. 탐구활동 등록 (딱 한 번만)
# --------------------------------------------------
SIMULATIONS = {
    geom_seq_limit.TITLE: geom_seq_limit,
    geom_series_sum.TITLE: geom_series_sum,
}

# --------------------------------------------------
# 4. 메인 라우터
# --------------------------------------------------
def main():
    st.set_page_config(page_title="미적분 탐구활동", layout="wide")

    st.title("📘 미적분 탐구활동")
    st.divider()

    # 단원 구분
    st.header("Ⅰ. 수열의 극한")

    selected_title = st.selectbox(
        "실행할 탐구활동을 선택하세요",
        list(SIMULATIONS.keys()),
    )

    st.divider()

    # 선택된 탐구활동 실행
    SIMULATIONS[selected_title].render()

# --------------------------------------------------
# 5. 실행 진입점
# --------------------------------------------------
if __name__ == "__main__":
    main()
