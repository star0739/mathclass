import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="지정 좌석 확인", layout="wide")

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

    + class_meta(class_id -> last_date)
      분반별 마지막 변경일(표시용) 저장

    기존 DB에 seat_assignments 테이블이 다른 형태로 존재하면
    새 스키마로 자동 이관(마이그레이션)합니다.
    """
    with get_conn() as conn:
        # --- seat_assignments 존재 확인 ---
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
        else:
            cols = conn.execute("PRAGMA table_info(seat_assignments);").fetchall()
            col_names = [c[1] for c in cols]

            is_new_schema = (
                "class_id" in col_names
                and "seat_no" in col_names
                and "student_name" in col_names
                and "updated_at" in col_names
            )

            if not is_new_schema:
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

                    conn.execute("DROP TABLE seat_assignments;")
                    conn.execute("ALTER TABLE seat_assignments_new RENAME TO seat_assignments;")
                    conn.execute("COMMIT;")
                except Exception:
                    conn.execute("ROLLBACK;")
                    raise

        # --- class_meta 생성(표시용 마지막 변경일 저장) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS class_meta (
                class_id TEXT PRIMARY KEY,
                last_date TEXT NOT NULL
            );
            """
        )

        # 기본값(빈 문자열) 시드
        for cid in CLASSES:
            conn.execute(
                "INSERT OR IGNORE INTO class_meta(class_id, last_date) VALUES(?, ?);",
                (cid, ""),
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
            # ✅ 항상 먼저 삭제 -> 이후 입력된 것만 다시 INSERT (빈 입력이면 '초기화'와 동일)
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


def get_last_date(class_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT last_date FROM class_meta WHERE class_id = ?;",
            (class_id,),
        ).fetchone()
    return row[0] if row else ""


def set_last_date(class_id: str, last_date: str):
    with get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            cur = conn.execute(
                "UPDATE class_meta SET last_date = ? WHERE class_id = ?;",
                (last_date, class_id),
            )
            if cur.rowcount == 0:
                conn.execute(
                    "INSERT INTO class_meta(class_id, last_date) VALUES(?, ?);",
                    (class_id, last_date),
                )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


# ---------------------------
# Parsing
# ---------------------------
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
def render_front_bar():
    """좌석 배치 상단에 '칠판&교탁(앞)'을 가로로 길게 표시."""
    st.markdown(
        """
        <div style="
            width: 100%;
            border: 1px solid #cfcfcf;
            border-radius: 14px;
            padding: 14px 16px;
            margin: 6px 0 14px 0;
            background: #f7f7f7;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            text-align: center;
            font-weight: 800;
            font-size: 18px;
            letter-spacing: 0.5px;
        ">
            칠판 &amp; 교탁 (앞)
        </div>
        """,
        unsafe_allow_html=True,
    )


def seat_cell_html(seat_no: int, name: str) -> str:
    safe_name = (name or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    display = safe_name if safe_name else "&nbsp;"  # 미배정은 빈칸(레이아웃 유지)
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


# ---------------------------
# App
# ---------------------------
init_db()

st.title("지정 좌석 확인(미적분 A~D)")

teacher_pw = get_teacher_password()
if not teacher_pw:
    st.warning("secrets.toml에 비밀번호가 설정되어 있지 않습니다. TEACHER_PASSWORD 또는 [auth].password를 설정하세요.")

tabA, tabB, tabC, tabD, tabT = st.tabs(["미적분A", "미적분B", "미적분C", "미적분D", "교사"])


def render_class_tab(class_id: str, title: str):
    st.subheader(title)
    last_date = get_last_date(class_id)
    if last_date:
        st.caption(f"마지막 변경일: {last_date}")
    render_front_bar()
    render_grid(load_assignments(class_id))


# ---- Student tabs (read-only display) ----
with tabA:
    render_class_tab("A", "미적분A 좌석 배치")

with tabB:
    render_class_tab("B", "미적분B 좌석 배치")

with tabC:
    render_class_tab("C", "미적분C 좌석 배치")

with tabD:
    render_class_tab("D", "미적분D 좌석 배치")

# ---- Teacher tab ----
with tabT:
    st.subheader("교사용 입력 (분반별 좌석 배정)")

    pw_input = st.text_input("비밀번호", type="password", key="teacher_pw_input")
    is_teacher = (teacher_pw != "") and (pw_input == teacher_pw)

    if teacher_pw and not is_teacher:
        st.info("비밀번호를 입력하면 편집 및 저장이 활성화됩니다.")
    else:
        # ✅ (1) DB 백업 다운로드 (A~D 전체 포함)
        try:
            txt_data = export_db_to_text()
                st.download_button(
                    label="좌석 DB 백업 다운로드(A~D 전체)",
                    data=txt_data,
                    file_name="finalseat.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        except Exception:
            st.warning("DB 파일을 읽을 수 없어 백업 다운로드를 제공할 수 없습니다.")

        st.divider()
        st.caption("※ 좌석 입력을 모두 지운 뒤 저장하면 해당 분반 좌석이 초기화됩니다.")

        tA, tB, tC, tD = st.tabs(["A 입력", "B 입력", "C 입력", "D 입력"])

        def teacher_editor(class_id: str):
            text_key = f"text_{class_id}"
            date_key = f"last_date_{class_id}"

            def _reload():
                st.session_state[text_key] = assignments_to_text(load_assignments(class_id))
                st.session_state[date_key] = get_last_date(class_id)

            # 최초 진입 시 DB 저장본 로딩
            if text_key not in st.session_state:
                st.session_state[text_key] = assignments_to_text(load_assignments(class_id))
            if date_key not in st.session_state:
                st.session_state[date_key] = get_last_date(class_id)

            top1, top2 = st.columns([1, 1])
            with top1:
                st.write(f"**미적분{class_id} 입력**")
            with top2:
                st.button(
                    "현재 저장본 불러오기",
                    key=f"reload_{class_id}",
                    on_click=_reload,
                    use_container_width=True,
                )

            # ✅ (2) 마지막 변경일 입력
            last_date_text = st.text_input(
                "마지막 변경일 (예: 2026년 01월 08일)",
                key=date_key,
                placeholder="예) 2026년 01월 08일",
            )

            text = st.text_area(
                "배정 현황(여러 줄)",
                key=text_key,
                height=220,
                placeholder="예) 1: 홍길동\n2: 김철수",
            )

            if st.button("저장", key=f"save_{class_id}", use_container_width=True):
                new_assignments, errors = parse_text_to_assignments(text)
                if errors:
                    st.error("입력 오류가 있습니다.")
                    for e in errors:
                        st.write(f"- {e}")
                    return

                # 좌석 저장
                save_assignments(class_id, new_assignments)

                # 마지막 변경일 저장(빈 값도 허용: 표시 제거용)
                set_last_date(class_id, last_date_text.strip())

                st.success(f"미적분{class_id} 저장 완료. (분반 탭에 즉시 반영됩니다.)")
                st.rerun()

        with tA:
            teacher_editor("A")
        with tB:
            teacher_editor("B")
        with tC:
            teacher_editor("C")
        with tD:
            teacher_editor("D")
