import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="분반별 좌석 배치도", layout="wide")

DB_PATH = "finalseat.db"
ROWS = 6
COLS = 5
TOTAL = ROWS * COLS

CLASSES = ["A", "B", "C", "D"]  # 미적분 A~D


# ---------------------------
# Secrets (Teacher password)
# ---------------------------
# secrets.toml 예시
# TEACHER_PASSWORD = "원하는비밀번호"
# 또는
# [auth]
# password = "원하는비밀번호"
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
                class_id TEXT NOT NULL,
                seat_no INTEGER NOT NULL,
                student_name TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (class_id, seat_no)
            );
            """
        )


def load_assignments(class_id: str) -> dict[int, str]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT seat_no, student_name
            FROM seat_assignments
            WHERE class_id = ?
            ORDER BY seat_no;
            """,
            (class_id,),
        ).fetchall()
    return {int(seat_no): str(name) for seat_no, name in rows}


def save_assignments(class_id: str, assignments: dict[int, str]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            conn.execute("DELETE FROM seat_assignments WHERE class_id = ?;", (class_id,))
            for seat_no, name in assignments.items():
                if 1 <= seat_no <= TOTAL and name.strip():
                    conn.execute(
                        """
                        INSERT INTO seat_assignments (class_id, seat_no, student_name, updated_at)
                        VALUES (?, ?, ?, ?);
                        """,
                        (class_id, seat_no, name.strip(), now),
                    )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def clear_assignments(class_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM seat_assignments WHERE class_id = ?;", (class_id,))


# ---------------------------
# Parsing
# ---------------------------
# "2: 김 (배정: ...)" -> 좌석번호=2, 이름="김" 만 추출. 괄호 이후 무시.
LINE_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*([^\(\n\r]+?)\s*(?:\(|$)")


def parse_text_to_assignments(text: str) -> tuple[dict[int, str], list[str]]:
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

        # 중복 좌석번호는 마지막 입력으로 덮어씀
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


def render_grid(assignments: dict[int, str]):
    for r in range(ROWS):
        cols = st.columns(COLS, gap="medium")
        for c in range(COLS):
            seat_no = r * COLS + c + 1
            name = assignments.get(seat_no, "")
            with cols[c]:
                st.markdown(seat_cell_html(seat_no, name), unsafe_allow_html=True)

    st.caption("표시 규칙: 1행 1열부터 1번, 오른쪽으로 증가 후 다음 행으로 넘어갑니다.")


# ---------------------------
# App
# ---------------------------
init_db()

st.title("분반별 좌석 배치도 (미적분 A~D)")

teacher_pw = get_teacher_password()
if not teacher_pw:
    st.warning("secrets.toml에 비밀번호가 설정되어 있지 않습니다. TEACHER_PASSWORD 또는 [auth].password를 설정하세요.")

tabA, tabB, tabC, tabD, tabT = st.tabs(["미적분A", "미적분B", "미적분C", "미적분D", "교사"])

# ---- Student tabs (read-only display) ----
with tabA:
    st.subheader("미적분A 좌석 배치")
    render_grid(load_assignments("A"))

with tabB:
    st.subheader("미적분B 좌석 배치")
    render_grid(load_assignments("B"))

with tabC:
    st.subheader("미적분C 좌석 배치")
    render_grid(load_assignments("C"))

with tabD:
    st.subheader("미적분D 좌석 배치")
    render_grid(load_assignments("D"))

# ---- Teacher tab (password protected input per class) ----
with tabT:
    st.subheader("교사용 입력 (분반별 좌석 배정)")

    pw_input = st.text_input("비밀번호", type="password", key="teacher_pw_input")
    is_teacher = (teacher_pw != "") and (pw_input == teacher_pw)

    st.caption("입력 형식 예시 (괄호 내용은 무시됩니다)")
    st.code("2: 김 (배정: 2026-01-08 08:31:39)\n1: 이", language="text")

    if teacher_pw and not is_teacher:
        st.info("비밀번호를 입력하면 편집 및 저장이 활성화됩니다.")

    # 교사 탭 안에서도 분반별로 나누어 입력하도록 탭 구성
    tA, tB, tC, tD = st.tabs(["A 입력", "B 입력", "C 입력", "D 입력"])

    def teacher_editor(class_id: str):
        current = load_assignments(class_id)
        default_text = assignments_to_text(current)

        text_key = f"text_{class_id}"
        if text_key not in st.session_state:
            st.session_state[text_key] = default_text

        # DB가 바뀌었는데 이전 세션 텍스트가 남아있을 수 있어, "불러오기" 제공
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**미적분{class_id} 입력**")
        with col2:
            if st.button("현재 저장본 불러오기", key=f"reload_{class_id}", disabled=not is_teacher):
                st.session_state[text_key] = assignments_to_text(load_assignments(class_id))
                st.rerun()

        text = st.text_area(
            "배정 현황(여러 줄)",
            key=text_key,
            height=220,
            disabled=not is_teacher,
            placeholder="예) 1: 이\n2: 김",
        )

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("저장", key=f"save_{class_id}", use_container_width=True, disabled=not is_teacher):
                new_assignments, errors = parse_text_to_assignments(text)
                if errors:
                    st.error("입력 오류가 있습니다.")
                    for e in errors:
                        st.write(f"- {e}")
                else:
                    save_assignments(class_id, new_assignments)
                    st.success(f"미적분{class_id} 저장 완료. (미적분{class_id} 탭에 연동 표시됨)")
                    st.rerun()

        with b2:
            if st.button("미리보기", key=f"preview_{class_id}", use_container_width=True, disabled=not is_teacher):
                new_assignments, errors = parse_text_to_assignments(text)
                if errors:
                    st.error("입력 오류가 있습니다.")
                    for e in errors:
                        st.write(f"- {e}")
                else:
                    st.success("미리보기(저장 전)입니다.")
                    render_grid(new_assignments)

        with b3:
            if st.button("전체 초기화", key=f"clear_{class_id}", use_container_width=True, disabled=not is_teacher):
                clear_assignments(class_id)
                st.session_state[text_key] = ""
                st.warning(f"미적분{class_id} 배정을 모두 삭제했습니다.")
                st.rerun()

        st.divider()
        st.write("**현재 저장된 좌석 배치(읽기 전용)**")
        render_grid(load_assignments(class_id))

    with tA:
        teacher_editor("A")
    with tB:
        teacher_editor("B")
    with tC:
        teacher_editor("C")
    with tD:
        teacher_editor("D")
