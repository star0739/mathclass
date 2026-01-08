import streamlit as st

st.set_page_config(
    page_title="수학 수업 연구실",
    page_icon="🏠",
    layout="wide",
)

# -----------------------------
# 페이지 등록 (현재 폴더 구조 기준)
# -----------------------------
def home_screen():
    st.write(
        "이곳은 수학 수업에서 활용할 수 있는 시뮬레이션과 활동을 한 곳에 모은 연구실입니다. "
        "아래에서 교과를 고르고, 교과별 메인 페이지에서 구체 활동으로 들어가세요."
    )
    st.write("")

    st.markdown("#### 빠른 이동")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("미적분으로 이동", use_container_width=True, key="quick_calculus"):
            st.switch_page(calculus_page)

    with c2:
        if st.button("인공지능수학으로 이동", use_container_width=True, key="quick_ai"):
            st.switch_page(ai_math_page)

    with c3:
        if st.button("좌석 배정으로 이동", use_container_width=True, key="quick_seat"):
            st.switch_page(seat_page)


home_page = st.Page(home_screen, title="Home", icon="🏠", default=True)

calculus_page = st.Page(
    "activities/calculus.py",
    title="미적분",
    icon="📘",
)

ai_math_page = st.Page(
    "activities/ai_math.py",
    title="인공지능수학",
    icon="🤖",
)

seat_page = st.Page(
    "sub/seat.py",
    title="좌석 배정",
    icon="🪑",
)

pages = {
    "Home": [home_page],
    "📁 교과별 페이지": [calculus_page, ai_math_page],
    "📁 도움 자료": [seat_page],
}

# 기본 네비게이션은 숨기고, 우리가 만든 사이드바로만 이동
pg = st.navigation(pages, position="hidden")

# -----------------------------
# 상단 바: 홈으로
# -----------------------------
col_left, col_right = st.columns([8, 2])
with col_right:
    if st.button("🏠 홈으로", use_container_width=True, key="top_home"):
        st.switch_page(home_page)

st.divider()

# -----------------------------
# 좌측 사이드바: 메뉴 구성
# -----------------------------
with st.sidebar:
    st.header("Home")

    st.markdown("---")
    st.subheader("📁 교과별 페이지")

    if st.button("미적분", use_container_width=True, key="sb_calculus"):
        st.switch_page(calculus_page)

    if st.button("인공지능수학", use_container_width=True, key="sb_ai"):
        st.switch_page(ai_math_page)

    st.markdown("---")
    st.subheader("📁 도움 자료")

    if st.button("좌석 배정", use_container_width=True, key="sb_seat"):
        st.switch_page(seat_page)

# -----------------------------
# 현재 선택된 페이지 실행
# -----------------------------
pg.run()
