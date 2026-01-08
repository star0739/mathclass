import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="좌석 배치도", layout="wide")

DB_PATH = "finalseat.db"
ROWS = 6
COLS = 5
TOTAL = ROWS * COLS

# secrets.toml 에 아래 중 하나로 비밀번호를 넣어두세요.
# 1) TEACHER_PASSWORD = "원하는비밀번호"
# 2) [auth]
#    password = "원하는비밀번호"
def get_teacher_password() -> str:
    if "TEACHER_PASSWORD" in st.secrets:
        return str(st.secrets["TEACHER_PASSWORD"])
    if "auth" in st.secrets and "password" in st.secrets["auth"]:
        return str(st.secrets["auth"]["password"])
    return ""


# ---------------------------
# DB utilities
# ---------------------------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seat_assignments (
                seat_no INTEGER PRIMARY KEY,
                student_name TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


def load_assignments() -> dict[int, str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT seat_no, student_name FROM seat_assignments ORDER BY seat_no"
        ).fetchall()
    return {int(seat_no): str(name) for seat_no, name in rows}


def save_assignments(assignments: dict[int, str]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            conn.execute("DELETE FROM seat_assignments;")
            for seat_no, name in assignments.items():
                if 1 <= seat_no <= TOTAL and name.strip():
                    conn.execute(
                        """
                        INSERT INTO seat_assignments (seat_no, student_name, updated_at)
                        VALUES (?, ?, ?);
                        """,
                        (seat_no, name.strip(), now),
                    )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def clear_assignments():
    with get_conn() as conn:
        conn.execute("DELETE FROM seat_assignments;")


# ---------------------------
# Parsing
# ---------------------------
LINE_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*([^\(\n\r]+?)\s*(?:\(|$)")

def parse_text_to_assignments(text: str) -> tuple[dict[int, str], list[str]]:
    """
    허용 입력 예:
      2: 김 (배정: 2026-01-08 08:31:39)
      1: 이
    규칙:
      - '번호: 이름'만 추출
      - (배정: ...) 같은 괄호 내용은 무시
      - 중복 번호는 '마지막에 나온 값'으로 덮어씀
    """
    assignments: dict[int, str] = {}
    errors: list[str] = []

    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        m = LINE_RE.match(line)
        if not m:
            errors.append(f"{idx}번째 줄 형식 오류: {raw}")
            continue

        seat_no = int(m.group(1))
        name = m.group(2).strip()

        if not (1 <= seat_no <= TOTAL):
            errors.append(f"{idx}번째 줄 좌석 번호 범위 오류(1~{TOTAL}): {seat_no}")
            continue

        if not name:
            errors.append(f"{idx}번째 줄 이름이 비어 있습니다: {raw}")
            continue

        assignments[seat_no] = name

    return assignments, errors


def assignments_to_text(assignments: dict[int, str]) -> str:
    lines = []
    for seat_no in range(1, TOTAL + 1):
        if seat_no in assignments:
            lines.append(f"{seat_no}: {assignments[seat_no]}")
    return "\n".join(lines)


# ---------------------------
# UI helpers
# ---------------------------
def seat_cell_html(seat_no: int, name: str) -> str:
    safe_name = (name or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <div style="
        border: 1px solid #d0d0d0;
        border-radius: 12px;
        padding: 10px 12px;
        min-height: 78px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="font-weight: 700; font-size: 14px;">{seat_no}번</div>
        <div style="font-size: 18px; font-weight: 700; line-height: 1.2; margin-top: 6px;">
            {safe_name if safe_name else "<span style='color:#999; font-weight:600;'>미배정</span>"}
        </div>
    </div>
    """


# ---------------------------
# App
# ---------------------------
init_db()

st.title("5×6 좌석 배치도 (1~30번)")

teacher_pw = get_teacher_password()
if not teacher_pw:
    st.warning("secrets.toml에 비밀번호가 설정되어 있지 않습니다. TEACHER_PASSWORD 또는 [auth].password를 설정하세요.")

assignments = load_assignments()

with st.sidebar:
    st.header("교사용 입력")
    pw_input = st.text_input("비밀번호", type="password")
    is_teacher = (teacher_pw != "") and (pw_input == teacher_pw)

    st.caption("입력 형식 예시")
    st.code("2: 김 (배정: 2026-01-08 08:31:39)\n1: 이", language="text")
    st.caption("※ (배정: …) 정보는 자동으로 무시됩니다.")

    default_text = assignments_to_text(assignments)
    text = st.text_area(
        "배정 현황 입력(여러 줄)",
        value=default_text,
        height=220,
        disabled=not is_teacher,
        placeholder="예) 1: 이\n2: 김",
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("저장", use_container_width=True, disabled=not is_teacher):
            new_assignments, errors = parse_text_to_assignments(text)
            if errors:
                st.error("입력 오류가 있습니다.")
                for e in errors:
                    st.write(f"- {e}")
            else:
                save_assignments(new_assignments)
                st.success("저장했습니다.")
                assignments = load_assignments()

    with col_b:
        if st.button("미리보기 반영", use_container_width=True, disabled=not is_teacher):
            new_assignments, errors = parse_text_to_assignments(text)
            if errors:
                st.error("입력 오류가 있습니다.")
                for e in errors:
                    st.write(f"- {e}")
            else:
                assignments = new_assignments
                st.success("미리보기에 반영했습니다. (저장은 아직)")

    with col_c:
        if st.button("전체 초기화", use_container_width=True, disabled=not is_teacher):
            clear_assignments()
            assignments = {}
            st.warning("모든 배정을 삭제했습니다.")

    if teacher_pw and not is_teacher:
        st.info("비밀번호를 입력하면 편집이 활성화됩니다.")


st.subheader("좌석 배치도")

# Grid render
for r in range(ROWS):
    cols = st.columns(COLS, gap="medium")
    for c in range(COLS):
        seat_no = r * COLS + c + 1
        name = assignments.get(seat_no, "")
        with cols[c]:
            st.markdown(seat_cell_html(seat_no, name), unsafe_allow_html=True)

st.caption("표시: 1행 1열부터 1번, 오른쪽으로 증가하며 다음 행으로 넘어갑니다.")
