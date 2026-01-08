import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="분반별 좌석 배치도", layout="wide")

ROWS = 6
COLS = 5
TOTAL = ROWS * COLS
CLASSES = ["A", "B", "C", "D"]  # 미적분 A~D

# DB를 "현재 파일(finalseat.py)과 같은 폴더"에 고정 생성
DB_PATH = str(Path(__file__).with_name("finalseat.db"))


# ---------------------------
# Secrets (Teacher password)
# ---------------------------
# .streamlit/secrets.toml 예시
# TEACHER_PASSWORD = "1234"
# 또는
# [auth]
# password = "1234"
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
    """
    최신 스키마:
      seat_assignments(class_id, seat_no, student_name, updated_at)
      PRIMARY KEY (class_id, seat_no)

    기존 DB에 seat_assignments 테이블이 다른 형태로 존재하면
    새 스키마로 자동 이관(마이그레이션)합니다.
    """
    with get_conn() as conn:
        # 테이블 존재 확인
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='seat_assignments';"
        ).fetchone()

        if row is None:
            conn.execute(
                """
                CREATE TABLE seat_assignments (
                    class_id TEXT NOT NULL,
                    seat_no INTEGER NOT NULL,
                    student_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (class_id, seat_no)
                );
                """
            )
            return

        # 컬럼 확인
        cols = conn.execute("PRAGMA table_info(seat_assignments);").fetchall()
        col_names = [c[1] for c in cols]

        is_new_schema = (
            "class_id" in col_names
            and "seat_no" in col_names
            and "student_name" in col_names
            and "updated_at" in col_names
        )
        if is_new_schema:
            return

        # 마이그레이션 수행
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("BEGIN;")
        try:
            conn.execute(
                """
                CREATE TABLE seat_assignments_new (
                    class_id TEXT NOT NULL,
                    seat_no INTEGER NOT NULL,
                    student_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (class_id, seat_no)
                );
                """
            )

            # 과거 스키마에서 가능한 정보 이관
            # (1) seat_no, student_name (+ optional updated_at)
            if "seat_no" in col_names and "student_name" in col_names:
                if "updated_at" in col_names:
                    conn.execute(
                        """
                        INSERT INTO seat_assignments_new (class_id, seat_no, student_name, updated_at)
                        SELECT 'A', seat_no, student_name, updated_at
                        FROM seat_assignments;
                        """
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO seat_assignments_new (class_id, seat_no, student_name, updated_at)
                        SELECT 'A', seat_no, student_name, ?
                        FROM seat_assignments;
                        """,
                        (now,),
                    )
            # 그 외 알 수 없는 형태면 데이터 이관 없이 스키마만 교체

            conn.execute("DROP TABLE seat_assignments;")
            conn.execute("ALTER TABLE seat_assignments_new RENAME TO seat_assignments;")
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


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
# 예: "2: 김 (배정: ...)" -> seat_no=2, name="김"만 추출. 괄호 이후 무시.
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

        # 중복 좌석번호는 마지막 값으로 덮어씀
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
    display = safe_name if safe_name else "<span style='color:#999; font-weight:600;'>미배정</span>"
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
            {display}
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

    if teacher_pw and not is_teacher:
        st.info("비밀번호를 입력하면 편집 및 저장이 활성화됩니다.")

    tA, tB, tC, tD = st.tabs(["A 입력", "B 입력", "C 입력", "D 입력"])

    def teacher_editor(class_id: str):
        current = load_assignments(class_id)
        text_key = f"text_{class_id}"

        # 최초 진입 시 DB 저장본을 세션에 로딩
        if text_key not in st.session_state:
            st.session_state[text_key] = assignments_to_text(current)

        top1, top2 = st.columns([1, 1])
        with top1:
            st.write(f"**미적분{class_id} 입력**")
        with top2:
            if st.button("현재 저장본 불러오기", key=f"reload_{class_id}", disabled=not is_teacher):
                st.session_state[text_key] = assignments_to_text(load_assignments(class_id))
                st.rerun()

        text = st.text_area(
            "배정 현황(여러 줄)",
            key=text_key,
            height=220,
            disabled=not is_teacher,
            placeholder="예) 1: 김숭문\n2: 이숭문문",
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
