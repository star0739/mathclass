# calculus.py
# 미적분 단원 시뮬레이션 라우터

import streamlit as st

# --- 개별 시뮬레이션 모듈 import ---
import calculus_geometric_sequence_limit as geom_seq_limit
# 이후 시뮬레이션 추가 시 여기 import 추가


# --- 시뮬레이션 등록 ---
SIMULATIONS = {
    geom_seq_limit.TITLE: geom_seq_limit,
    # "다른 시뮬레이션 제목": 모듈명,
}


def main():
    st.set_page_config(page_title="미적분 시뮬레이션", layout="wide")

    st.title("📘 미적분 시뮬레이션 페이지")
    st.markdown("단원을 선택하여 개념을 탐구하세요.")

    st.divider()

    # --- 단원 구분 (현재는 수열의 극한 단원만 구성) ---
    st.header("수열의 극한")

    selected_title = st.selectbox(
        "실행할 시뮬레이션을 선택하세요",
        list(SIMULATIONS.keys())
    )

    st.divider()

    # --- 선택된 시뮬레이션 실행 ---
    selected_module = SIMULATIONS[selected_title]
    selected_module.render()


if __name__ == "__main__":
    main()

