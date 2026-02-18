# activities/ai_math.py
from __future__ import annotations

import streamlit as st
from pathlib import Path
import importlib.util

# --------------------------------------------------
# 0. set_page_config (중복 호출 안전 처리)
# --------------------------------------------------
try:
    st.set_page_config(page_title="인공지능 수학 탐구활동", layout="wide")
except Exception:
    pass

# --------------------------------------------------
# 1. 현재 폴더 기준으로 활동 모듈 로드
# --------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent


def load_activity_module(py_filename: str, module_name: str):
    path = CURRENT_DIR / py_filename
    if not path.exists():
        raise FileNotFoundError(f"활동 파일을 찾을 수 없습니다: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로드 실패: {module_name} ({path})")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------
# 2. 활동 모듈 로드
# --------------------------------------------------

# 기존 활동 (있다면)
try:
    mse_activity = load_activity_module("ai_mse.py", "ai_mse")
except Exception:
    mse_activity = None

# 새 활동
try:
    ce_activity = load_activity_module("ai_cross_entropy.py", "ai_cross_entropy")
except Exception:
    ce_activity = None


# --------------------------------------------------
# 3. 활동 등록
# --------------------------------------------------
ACTIVITIES = []

if mse_activity is not None:
    ACTIVITIES.append(mse_activity)

if ce_activity is not None:
    ACTIVITIES.append(ce_activity)


# --------------------------------------------------
# 4. 렌더링
# --------------------------------------------------
def _render_activity(module) -> None:
    key_prefix = getattr(module, "__name__", "activity")
    try:
        module.render(show_title=False, key_prefix=key_prefix)
    except TypeError:
        module.render()


def main() -> None:
    st.title("인공지능 수학 탐구활동")

    if not ACTIVITIES:
        st.warning("연결된 탐구활동이 없습니다.")
        return

    tab_labels = [getattr(m, "TITLE", getattr(m, "__name__", "activity")) for m in ACTIVITIES]
    tabs = st.tabs(tab_labels)

    for tab, module in zip(tabs, ACTIVITIES):
        with tab:
            _render_activity(module)


if __name__ == "__main__":
    main()
