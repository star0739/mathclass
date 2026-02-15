# activities/calculus.py
# 미적분 탐구활동 라우터 페이지
# - 단원: 버튼식
# - 활동: 탭
# - 위젯 충돌 방지: key_prefix 전달
# - set_page_config 중복 호출/위치 문제 방지: 안전 처리

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# --------------------------------------------------
# 0. set_page_config는 가능한 한 "가장 먼저" 실행
#    (이미 다른 곳에서 호출된 경우 예외가 나므로 안전 처리)
# --------------------------------------------------
try:
    st.set_page_config(page_title="미적분 탐구활동", layout="wide")
except Exception:
    # home.py 또는 Navigation 프레임워크에서 이미 호출했을 수 있음
    pass

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

# Ⅱ. 미분법: e 정의(극한) 탐구활동
import calculus_e_definition_limit as e_def
import calculus_sin_x_over_x_area as sinx_over_x_area

# (Ⅱ. 미분법 추가 예정)
# import calculus_derivative_limit_definition as deriv_def

# (Ⅲ. 적분법 추가 예정)
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
        e_def,
        sinx_over_x_area,
        
    ],
    "Ⅲ. 적분법": [
        # riemann_area,
    ],
}

# --------------------------------------------------
# 4. 세션 상태 초기화
# --------------------------------------------------
def _init_state() -> None:
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = list(UNIT_ACTIVITIES.keys())[0]

# --------------------------------------------------
# 5. 활동 렌더링 (key_prefix 전달)
# --------------------------------------------------
def _render_activity(module) -> None:
    key_prefix = module.__name__  # 모듈명은 유니크하므로 prefix로 적합
    try:
        module.render(show_title=False, key_prefix=key_prefix)
    except TypeError:
        module.render()

# --------------------------------------------------
# 6. 메인
# --------------------------------------------------
def main() -> None:
    _init_state()

    st.title("📘 미적분 탐구활동")

    # 단원 선택 버튼
    unit_names = list(UNIT_ACTIVITIES.keys())
    cols = st.columns(len(unit_names))

    for i, unit in enumerate(unit_names):
        is_selected = (st.session_state.selected_unit == unit)
        label = f"✅ {unit}" if is_selected else unit

        # 버튼도 키를 주면 더 안전(라벨 변동 때문에)
        if cols[i].button(label, use_container_width=True, key=f"unit_btn_{i}"):
            st.session_state.selected_unit = unit
            st.rerun()

    selected_unit = st.session_state.selected_unit
    activities = UNIT_ACTIVITIES[selected_unit]

    if not activities:
        st.info("이 단원에 연결된 탐구활동이 아직 없습니다.")
        return

    # 활동 탭 (TITLE 그대로 사용)
    tab_labels = [module.TITLE for module in activities]
    tabs = st.tabs(tab_labels)

    for tab, module in zip(tabs, activities):
        with tab:
            _render_activity(module)

if __name__ == "__main__":
    main()
