import streamlit as st

st.set_page_config(
    page_title="수학 수업 연구실",
    page_icon="🏠",
    layout="wide",
)

# -----------------------------
# 1) 홈(메인) 화면 함수
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
        if st.button("미적분으로 이동", use_container_width=True, key="home_quick_calculus"):
            st.switch_page(calculus_page)

    with c2:
        if st.button("인공지능수학으로 이동", use_container_width=True, key="home_quick_ai"):
            st.switch_page(ai_math_page)

    with c3:
        if st.button("좌석 배정으로 이동", use_container_width=True, key="home_quick_seat"):
            st.switch_page(seat_page)


# -----------------------------
# 2) 페이지 등록 (폴더 내 파일로 연결)
# -----------------------------
home_page = st.Page(home_screen, title="Home", icon="🏠", default=True)

calculus_page = st.Page(
    "activities/calculus/calculus.py",
    title="미적분",
    icon="📘",
)

ai_math_page = st.Page(
    "activities/ai_math/ai_math.py",
    title="인공지능
