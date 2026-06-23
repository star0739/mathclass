import streamlit as st

st.set_page_config(
    page_title="숭문고 수학 스튜디오",
    page_icon="✨",
    layout="wide",
)

# -----------------------------
# 페이지 등록
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
        미적분 수행평가를 위한 공공데이터 분석 활동 공간입니다.\n\n
        데이터에 적절한 함수 모델을 세우고, 미분과 적분의 관점에서 모델을 비교·분석합니다.\n\n
        👉 **아래 메뉴에서 활동을 선택해 주세요.**
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown("#### 빠른 이동")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("함수 모델링", use_container_width=True, key="quick_data_modeling"):
            st.switch_page(data_modeling_page)

    with c2:
        if st.button("수치적분과 정적분", use_container_width=True, key="quick_data_integral"):
            st.switch_page(data_integral_page)


home_page = st.Page(
    home_screen,
    title="Home",
    icon="✨",
    default=True,
)

data_modeling_page = st.Page(
    "assessment/data_modeling.py",
    title="함수 모델링",
    icon="📈",
)

data_integral_page = st.Page(
    "assessment/data_integral.py",
    title="수치적분과 정적분",
    icon="📐",
)


pages = {
    "Home": [
        home_page,
    ],
    "📝 미적분 수행평가": [
        data_modeling_page,
        data_integral_page,
    ],
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
# 좌측 사이드바: 미적분 수행평가 메뉴만 표시
# -----------------------------
with st.sidebar:
    st.header("Home")

    if st.button("🏠 홈", use_container_width=True, key="sb_home"):
        st.switch_page(home_page)

    st.markdown("---")
    st.subheader("📝 미적분 수행평가")

    if st.button("함수 모델링", use_container_width=True, key="sb_data_modeling"):
        st.switch_page(data_modeling_page)

    if st.button("수치적분과 정적분", use_container_width=True, key="sb_data_integral"):
        st.switch_page(data_integral_page)


# -----------------------------
# 현재 선택된 페이지 실행
# -----------------------------
pg.run()
