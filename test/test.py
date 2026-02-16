import streamlit as st
from datetime import datetime
import re

import gspread
from google.oauth2.service_account import Credentials


# -----------------------------
# 주의:
# - 멀티페이지 구조에서는 각 페이지에서 set_page_config를 또 호출하면 꼬일 수 있어
# - 그래서 여기서는 st.set_page_config()를 호출하지 않음
# -----------------------------

st.title("🧪 Test: Google Sheets 저장 테스트")
st.caption("학번/코드 + 메시지를 입력하고 제출하면 Google Sheets에 한 줄이 추가됩니다.")
st.divider()


# -----------------------------
# 설정값 (여기만 너 환경에 맞게 조정)
# -----------------------------
# 1) Google Sheet 이름 (드라이브에 있는 '스프레드시트 파일' 이름)
SHEET_NAME = "미적분_수행평가_제출"

# 2) 사용할 워크시트(탭) 이름
# - 기본은 첫 번째 탭(sheet1)
# - 탭 이름을 지정하고 싶으면 WORKSHEET_NAME에 탭 이름을 넣어줘.
WORKSHEET_NAME = None  # 예: "제출기록" / 없으면 None


# -----------------------------
# 유틸 함수
# -----------------------------
def get_worksheet():
    """
    Streamlit secrets의 서비스 계정 정보로 인증 후 워크시트를 반환.
    st.secrets["gcp_service_account"]는 Streamlit Cloud Secrets에 JSON 그대로 넣어둔 상태를 기대.
    """
    scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )
    gc = gspread.authorize(creds)

    sh = gc.open(SHEET_NAME)
    if WORKSHEET_NAME:
        return sh.worksheet(WORKSHEET_NAME)
    return sh.sheet1


def normalize_student_id(student_id: str) -> str:
    """
    학번/코드 정규화: 공백 제거 + 허용 문자 제한(영문/숫자/-/_)
    """
    s = student_id.strip()
    s = re.sub(r"\s+", "", s)
    # 허용 문자만 남김
    s = re.sub(r"[^0-9A-Za-z\-_]", "", s)
    return s


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def ensure_header(ws):
    """
    헤더가 없으면 생성.
    이미 1행에 헤더가 있다고 가정하는 운영도 가능하지만,
    테스트 단계에서는 안전하게 헤더 존재를 보장.
    """
    expected = ["timestamp", "student_id", "source", "message", "client_info"]
    try:
        first_row = ws.row_values(1)
    except Exception:
        first_row = []

    if first_row != expected:
        # 첫 행이 비어있거나 다르면 헤더를 1행에 넣는다.
        # (기존 데이터가 있는 상태에서 헤더가 다르면 덮어쓸 수 있으니 테스트용에서만 권장)
        if len(first_row) == 0:
            ws.insert_row(expected, index=1)
        # 첫 행이 다른 값(이미 운영 중인 시트)이라면 강제로 바꾸지 않고 안내만
        else:
            st.warning(
                "시트 1행 헤더가 예상과 다릅니다. "
                "운영 중인 시트라면 헤더 자동 수정은 건너뜁니다."
            )


def find_latest_row_by_student(ws, student_id: str):
    """
    동일 student_id의 마지막 제출 행 번호를 찾음.
    - 테스트 용도라 단순 검색(전체 가져오기).
    - 데이터가 많아지면 최적화 필요.
    """
    try:
        records = ws.get_all_records()  # 헤더를 기준으로 dict 리스트 반환
    except Exception:
        return None

    last_idx = None
    # get_all_records는 2행부터가 records[0]에 해당
    # 실제 시트 행 번호는 (index + 2)
    for i, rec in enumerate(records):
        if str(rec.get("student_id", "")).strip() == student_id:
            last_idx = i + 2
    return last_idx


def append_submission(ws, student_id: str, message: str, client_info: str):
    ts = datetime.now().isoformat(timespec="seconds")
    ws.append_row(
        [ts, student_id, "TEST_PAGE", message, client_info],
        value_input_option="USER_ENTERED",
    )
    return ts


def update_submission(ws, row_number: int, student_id: str, message: str, client_info: str):
    ts = datetime.now().isoformat(timespec="seconds")
    values = [[ts, student_id, "TEST_PAGE(UPDATE)", message, client_info]]
    # A:E 범위에 한 줄 업데이트
    ws.update(range_name=f"A{row_number}:E{row_number}", values=values)
    return ts


# -----------------------------
# 연결 테스트
# -----------------------------
with st.expander("🔧 연결 상태", expanded=True):
    try:
        ws = get_worksheet()
        st.success("✅ Google Sheets 연결 성공")
        st.write(f"- Spreadsheet: **{SHEET_NAME}**")
        st.write(f"- Worksheet: **{WORKSHEET_NAME or '첫 번째 탭(sheet1)'}**")
    except Exception as e:
        st.error("❌ Google Sheets 연결 실패")
        st.write("아래 항목을 확인하세요:")
        st.write("- Streamlit Cloud Secrets에 `gcp_service_account`가 JSON 형태로 정확히 들어있는지")
        st.write("- 서비스 계정 이메일이 해당 스프레드시트에 '편집자'로 공유되어 있는지")
        st.write("- 스프레드시트 이름이 정확한지(띄어쓰기 포함)")
        st.exception(e)
        st.stop()

st.divider()


# -----------------------------
# 입력 폼
# -----------------------------
st.subheader("제출 테스트")

col1, col2 = st.columns([2, 3])

with col1:
    raw_student_id = st.text_input("학번/식별 코드", placeholder="예: 30215")
    student_id = normalize_student_id(raw_student_id)
    st.caption(f"정규화 결과: `{student_id}`")

with col2:
    message = st.text_area(
        "테스트 메시지",
        placeholder="예: 구글 시트 저장 정상 동작 확인",
        height=120,
    )

st.markdown("#### 저장 방식")
mode = st.radio(
    "중복 제출 처리",
    options=["항상 새 행 추가(append)", "같은 학번이면 마지막 제출 행 덮어쓰기(update)"],
    index=0,
    horizontal=True,
)

client_info = st.text_input(
    "추가 정보(선택)",
    placeholder="예: 3학년 2반 / 기기:모바일 / 브라우저:크롬",
)

submit = st.button("✅ 제출", use_container_width=True)

if submit:
    if not student_id:
        st.warning("학번/식별 코드를 입력하세요. (영문/숫자/-, _ 만 사용 권장)")
        st.stop()

    if not message.strip():
        st.warning("테스트 메시지를 입력하세요.")
        st.stop()

    try:
        # 헤더 보장(테스트용)
        ensure_header(ws)

        if mode.startswith("항상"):
            ts = append_submission(ws, student_id, message.strip(), client_info.strip())
            st.success(f"🎉 제출 완료! (append) 저장 시각: {ts}")

        else:
            # 같은 학번이면 마지막 제출 행 찾고 있으면 update, 없으면 append
            row_num = find_latest_row_by_student(ws, student_id)
            if row_num:
                ts = update_submission(ws, row_num, student_id, message.strip(), client_info.strip())
                st.success(f"🎉 제출 완료! (update) 행 {row_num} 덮어쓰기, 저장 시각: {ts}")
            else:
                ts = append_submission(ws, student_id, message.strip(), client_info.strip())
                st.success(f"🎉 제출 완료! (append) 기존 기록 없음 → 새 행 추가, 저장 시각: {ts}")

    except Exception as e:
        st.error("❌ 제출 중 오류가 발생했습니다.")
        st.write("가능한 원인:")
        st.write("- 시트/탭 이름 불일치")
        st.write("- 서비스 계정에 편집 권한이 없음")
        st.write("- API 제한/일시적 네트워크 문제")
        st.exception(e)

st.divider()


# -----------------------------
# (선택) 최근 기록 일부 확인
# -----------------------------
st.subheader("최근 기록 확인(옵션)")

with st.expander("최근 10행 미리보기", expanded=False):
    try:
        # 마지막 10행만 뽑기(간단 버전)
        # 전체 레코드가 많아지면 get_all_values()는 무거울 수 있어 테스트 단계에서만 사용 추천
        values = ws.get_all_values()
        if len(values) <= 1:
            st.info("아직 데이터가 없습니다.")
        else:
            preview = values[-10:]
            st.table(preview)
    except Exception as e:
        st.warning("미리보기를 불러오지 못했습니다(권한/데이터 구조/네트워크).")
        st.exception(e)


st.markdown(
    """
### ✅ 테스트 체크리스트
- 제출 버튼을 누르면 Google Sheets에 **행이 추가**되는가?
- 새로고침해도 기존 기록이 **그대로 남는가?**
- 같은 학번으로 여러 번 제출했을 때  
  - append 모드: 행이 계속 추가되는가?  
  - update 모드: 마지막 제출이 덮어써지는가?

이 3개가 확인되면, 수행평가 페이지에 그대로 확장하면 됩니다.
"""
)
