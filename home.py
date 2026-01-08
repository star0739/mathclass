import streamlit as st

st.set_page_config(
    page_title="수학 수업 연구실",
    page_icon="🏠",
    layout="wide",
)

# -----------------------------
# 상단 바(Top bar) + 홈으로 버튼
# -----------------------------
with st.container():
    col_left, col_right = st.columns([8, 2])
    with col_left:
        st.markdown("### ")  # 여백용 (원하시면 삭제 가능)
    with col_right:
        if st.button("🏠 홈으로", use_container_width=True):
            # 홈 화면 자체에서는 다시 홈으로 이동할 필요가 없으므로 rerun 처리
            st.rerun()

st.divider()

# -----------------------------
# 상단 바 하단 안내 문구
# -----------------------------
st.write(
    "이곳은 수학 수업에서 활용할 수 있는 시뮬레이션과 활동을 한 곳에 모은 연구실입니다. "
    "아래에서 교과를 고르고, 교과별 메인 페이지에서 구체 활동으로 들어가세요."
)

st.write("")  # 여백

# -----------------------------
# 좌측 사이드바 메뉴 구성
# -----------------------------
with st.sidebar:
    st.header("Home")

    st.markdown("---")
    st.subheader("📁 교과별 페이지")

    if st.button("미적분", use_container_width=True):
        st.switch_page("pages/calculus.py")

    if st.button("인공지능수학", use_container_width=True):
        st.switch_page("pages/ai_math.py")

    st.markdown("---")
    st.subheader("📁 도움 자료")

    if st.button("좌석 배정", use_container_width=True):
        st.switch_page("pages/seat.py")

# -----------------------------
# 홈 본문(선택) - 카드/안내 영역
# -----------------------------
st.markdown("#### 빠른 이동")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("미적분으로 이동", use_container_width=True):
        st.switch_page("pages/calculus.py")

with c2:
    if st.button("인공지능수학으로 이동", use_container_width=True):
        st.switch_page("pages/ai_math.py")

with c3:
    if st.button("좌석 배정으로 이동", use_container_width=True):
        st.switch_page("pages/seat.py")
