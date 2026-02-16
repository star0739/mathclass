import streamlit as st

st.set_page_config(
    page_title="숭문고 수학 스튜디오",
    page_icon="✨",
    layout="wide",
)

# -----------------------------
# 페이지 등록 (현재 폴더 구조 기준)
# -----------------------------
def home_screen():
    st.markdown(
        """
        <h2 style="margin-bottom: 0.5em;">
            ✨ 숭문고 수학 스튜디오
        </h2>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        <p style="font-size: 1.1rem; line-height: 1.7;">
            '미적분'과 '인공지능 수학' 수업에서 직접 탐구하고 실험할 수 있는 탐구활동들을 모아두었습니다.<br>
            아래에서 교과를 고르고, 교과별 페이지에서 원하는 활동을 시작해 보세요.
        </p>
        """,
        unsafe_allow_html=True,
    )
    
    st.write("")

    st.markdown("#### 빠른 이동")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("미적분", use_container_width=True, key="quick_calculus"):
            st.switch_page(calculus_page)

    with c2:
        if st.button("인공지능 수학", use_container_width=True, key="quick_ai"):
            st.switch_page(ai_math_page)

    with c3:
        if st.button("좌석 확인", use_container_width=True, key="quick_finalseat"):
            st.switch_page(finalseat_page)


home_page = st.Page(home_screen, title="Home", icon="✨", default=True)

calculus_page = st.Page(
    "activities/calculus.py",
    title="미적분",
    icon="🧮",
)

ai_math_page = st.Page(
    "activities/ai_math.py",
    title="인공지능 수학",
    icon="🤖",
)

seat_page = st.Page(
    "sub/seat.py",
    title="선착순 배정",
    icon="🪑",
)

finalseat_page = st.Page(
    "sub/finalseat.py",
    title="좌석 확인",
    icon="✅",
)

pages = {
    "Home": [home_page],
    "📖 교과 학습": [calculus_page, ai_math_page],
    "🪑 좌석 관리": [seat_page, finalseat_page],
}

# ✅ (추가) Test 페이지 등록
test_page = st.Page(
    "sub/test.py",
    title="Test",
    icon="🧪",
)

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
    st.subheader("📖 교과 학습")

    if st.button("미적분", use_container_width=True, key="sb_calculus"):
        st.switch_page(calculus_page)

    if st.button("인공지능 수학", use_container_width=True, key="sb_ai"):
        st.switch_page(ai_math_page)

    st.markdown("---")
    st.subheader("🪑 좌석 관리")

    if st.button("선착순 배정", use_container_width=True, key="sb_seat"):
        st.switch_page(seat_page)

    if st.button("좌석 확인", use_container_width=True, key="sb_finalseat"):
        st.switch_page(finalseat_page)

    # ✅ (추가) Test 버튼
    if st.button("Test", use_container_width=True, key="sb_test"):
        st.switch_page(test_page)


# -----------------------------
# 현재 선택된 페이지 실행
# -----------------------------
pg.run()
