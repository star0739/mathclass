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
        수학적 사고가 실현되는 공간, 숭문고 수학 스튜디오에 오신 것을 환영합니다!\n\n
        '미적분'과 '인공지능 수학'의 원리를 직접 시각화하며 탐구할 수 있도록 설계되었습니다.\n\n
        교과별 탐구 활동부터 효율적인 수행평가 안내, 스마트한 좌석 관리까지 — 수업의 몰입도를 높이는 수학적 도구들을 지금 바로 경험해 보세요.\n\n
        👉 **아래 메뉴에서 탐구하고자 하는 교과를 선택해 주세요.**
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


assessment_step1 = st.Page(
    "assessment/step1_data.py",
    title="1차시: 데이터 탐색",
    icon="1️⃣",
)

assessment_step2 = st.Page(
    "assessment/step2_model.py",
    title="2차시: 함수 모델링",
    icon="2️⃣",
)

assessment_step3 = st.Page(
    "assessment/step3_integral.py",
    title="3차시: 누적량 해석",
    icon="3️⃣",
)

assessment_final = st.Page(
    "assessment/final_report.py",
    title="최종: 보고서 작성",
    icon="⭐",
)


ai_assessment_step1 = st.Page(
    "assessment/ai_step1_structure.py",
    title="1차시: 구조 관찰",
    icon="1️⃣",
)

ai_assessment_step2 = st.Page(
    "assessment/ai_step2_path.py",
    title="2차시: 경로 탐색",
    icon="2️⃣",
)

ai_assessment_final = st.Page(
    "assessment/ai_final_report.py",
    title="최종: 보고서 작성",
    icon="⭐",
)



pages = {
    "Home": [home_page],
    "📖 교과 학습": [calculus_page, ai_math_page],
    "🪑 좌석 관리": [seat_page, finalseat_page],
    "✏️ 공공데이터 분석 수행": [assessment_step1, assessment_step2, assessment_step3, assessment_final],
    "🤖 인공지능 수학 수행평가": [ai_assessment_step1, ai_assessment_step2, ai_assessment_final],
    
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
    st.subheader("📖 교과 학습")

    if st.button("미적분", use_container_width=True, key="sb_calculus"):
        st.switch_page(calculus_page)

    if st.button("인공지능 수학", use_container_width=True, key="sb_ai"):
        st.switch_page(ai_math_page)


    st.markdown("---")
    st.subheader("📝 미적분: 공공데이터 분석 수행")

    if st.button("1차시: 데이터 탐색", use_container_width=True, key="sb_assess_1"):
        st.switch_page("assessment/step1_data.py")

    if st.button("2차시: 함수 모델링", use_container_width=True, key="sb_assess_2"):
        st.switch_page("assessment/step2_model.py")

    if st.button("3차시: 누적량 해석", use_container_width=True, key="sb_assess_3"):
        st.switch_page("assessment/step3_integral.py")

    if st.button("최종: 보고서 작성", use_container_width=True, key="sb_final_report"):
        st.switch_page("assessment/final_report.py")


    st.markdown("---")
    st.subheader("📝 인공지능 수학: 경사하강법 수행")

    if st.button("1차시: 구조 관찰", use_container_width=True, key="sb_ai_assess_1"):
        st.switch_page("assessment/ai_step1_structure.py")

    if st.button("2차시: 경로 탐색", use_container_width=True, key="sb_ai_assess_2"):
        st.switch_page("assessment/ai_step2_path.py")

    if st.button("최종: 보고서 작성", use_container_width=True, key="sb_ai_final_report"):
        st.switch_page("assessment/ai_final_report.py")
    

    st.markdown("---")
    st.subheader("🪑 좌석 관리")

    if st.button("선착순 배정", use_container_width=True, key="sb_seat"):
        st.switch_page(seat_page)

    if st.button("좌석 확인", use_container_width=True, key="sb_finalseat"):
        st.switch_page(finalseat_page)


# -----------------------------
# 현재 선택된 페이지 실행
# -----------------------------
pg.run()
