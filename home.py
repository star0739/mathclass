import streamlit as st

st.set_page_config(
    page_title="Mathlab",
    page_icon="🧮",
    layout="wide",
)

# ---- (선택) 공통 스타일 약간 정리: 버튼/여백 ----
st.markdown(
    """
    <style>
      /* 메인 컨테이너 폭 */
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* 제목 위쪽 여백 줄이기 */
      h1 { margin-top: 0.2rem; }

      /* 큰 메뉴 버튼 높이 */
      div.stButton > button {
        height: 54px;
        width: 100%;
        border-radius: 10px;
        font-size: 16px;
      }

      /* 섹션 타이틀 */
      .section-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0.8rem 0 0.8rem 0;
      }

      /* 홈 상단 캡션 영역처럼 보이게 */
      .home-bar {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
        margin-bottom: 18px;
        font-size: 18px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- 상단 "홈으로" 바 ----
st.markdown('<div class="home-bar">🏠 홈으로</div>', unsafe_allow_html=True)

# ---- 좌측 사이드바 구성 ----
with st.sidebar:
    st.write("home")
    st.write("Dev Tree")
    st.divider()
    st.markdown("### 📁 교과별 페이지")

    # 사이드바에서도 페이지 이동 버튼 제공
    if st.button("좌석 배정", use_container_width=True):
        st.switch_page("pages/1_좌석_배정.py")
    if st.button("미적분", use_container_width=True):
        st.switch_page("pages/2_미적분.py")
    if st.button("인공지능수학", use_container_width=True):
        st.switch_page("pages/3_인공지능수학.py")

# ---- 메인 영역 ----
st.title("🧮 Mathlab")
st.write(
    "이곳은 수학 수업에서 활용할 수 있는 활동과 도구를 한 곳에 모은 페이지입니다.\n"
    "아래에서 원하는 메뉴를 선택하여 이동하세요."
)

st.markdown('<div class="section-title">메뉴로 이동</div>', unsafe_allow_html=True)

# 가운데 큰 버튼 3개 (스크린샷의 '교과로 이동' 버튼 느낌)
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    if st.button("좌석 배정 이동", use_container_width=True):
        st.switch_page("pages/1_좌석_배정.py")

with c2:
    if st.button("미적분 이동", use_container_width=True):
        st.switch_page("pages/2_미적분.py")

with c3:
    if st.button("인공지능수학 이동", use_container_width=True):
        st.switch_page("pages/3_인공지능수학.py")

st.info("모바일 사용 시 가로모드가 화면 확인에 유리합니다.")
