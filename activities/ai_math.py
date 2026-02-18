
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

try:
    st.set_page_config(page_title="인공지능 수학 탐구활동", layout="wide")
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
try:
    import ai_mse as mse_activity  # activities 폴더를 sys.path에 넣었으면 이게 우선
except ModuleNotFoundError:
    # 패키지 경로로 재시도 (환경에 따라 이쪽이 필요할 때가 있음)
    from mathclass.activities import ai_mse as mse_activity


# 앞으로 추가될 활동 예시:
# import ai_something as something_activity

# --------------------------------------------------
# 3. 활동 등록
# --------------------------------------------------
ACTIVITIES = [
    mse_activity,
    # something_activity,
]

# --------------------------------------------------
# 4. 활동 렌더링 (key_prefix 전달)
# --------------------------------------------------
def _render_activity(module) -> None:
    key_prefix = module.__name__  # 모듈명은 유니크하므로 prefix로 적합
    try:
        module.render(show_title=False, key_prefix=key_prefix)
    except TypeError:
        # 구형 활동 모듈 호환(혹시 시그니처가 다를 때)
        module.render()

# --------------------------------------------------
# 5. 메인
# --------------------------------------------------
def main() -> None:
    st.title("🤖 인공지능 수학 탐구활동")

    if not ACTIVITIES:
        st.info("연결된 탐구활동이 아직 없습니다.")
        return

    # 활동 탭 (각 모듈의 TITLE 사용)
    tab_labels = [getattr(module, "TITLE", module.__name__) for module in ACTIVITIES]
    tabs = st.tabs(tab_labels)

    for tab, module in zip(tabs, ACTIVITIES):
        with tab:
            _render_activity(module)

if __name__ == "__main__":
    main()
