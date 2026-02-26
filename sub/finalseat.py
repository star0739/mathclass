import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="좌석 확인", layout="wide")

# ✅ 5행 × 5열 = 25석
ROWS = 5
COLS = 5
TOTAL = ROWS * COLS

# ✅ 분반 변경 반영
# 미적분(A, C, D, J), AI수학(B, F, H)
CALC_CLASSES = ["A", "C", "D", "J"]
AI_MATH_CLASSES = ["B", "F", "H"]

# 화면 탭 순서(원하면 바꿔도 됨)
TAB_ORDER = [
    ("미적분", "A"),
    ("AI수학", "B"),
    ("미적분", "C"),
    ("미적분", "D"),
    ("AI수학", "F"),
    ("AI수학", "H"),
    ("미적분", "J"),
]

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


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl};")


def init_db():
    """
    최신 스키마:
      seat_assignments(track, class_id, seat_no, student_name, updated_at)
      PRIMARY KEY (track, class_id, seat_no)

    + class_meta(track, class_id -> last_date)
      과목/분반별 마지막 변경일(표시용) 저장

    기존 DB가 이전 스키마(class_id, seat_no, student_name, updated_at)로 존재하면
    기본 track='미적분' 으로 자동 이관(마이그레이션)합니다.
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
                    track TEXT NOT NULL,
                    class_id TEXT NOT NULL,
                    seat_no INTEGER NOT NULL,
                    student_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (track, class_id, seat_no)
                );
                """
            )
        else:
            cols = conn.execute("PRAGMA table_info(seat_assignments);").fetchall()
            col_names = [c[1] for c in cols]

            # 새 스키마 여부 판단
            is_new_schema = (
                "track" in col_names
                and "class_id" in col_names
                and "seat_no" in col_names
                and "student_name" in col_names
                and "updated_at" in col_names
            )

            if not is_new_schema:
                # 이전 스키마를 새 스키마로 이관: track 기본값='미적분'
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("BEGIN;")
                try:
                    conn.execute(
                        """
                        CREATE TABLE seat_assignments_new (
                            track TEXT NOT NULL,
                            class_id TEXT NOT NULL,
                            seat_no INTEGER NOT NULL,
                            student_name TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            PRIMARY KEY (track, class_id, seat_no)
                        );
                        """
                    )

                    if "class_id" in col_names and "seat_no" in col_names and "student_name" in col_names:
                        if "updated_at" in col_names:
                            conn.execute(
                                """
                                INSERT INTO seat_assignments_new (track, class_id, seat_no, student_name, updated_at)
                                SELECT '미적분', class_id, seat_no, student_name, updated_at
                                FROM seat_assignments;
                                """
                            )
                        else:
                            conn.execute(
                                """
                                INSERT INTO seat_assignments_new (track, class_id, seat_no, student_name, updated_at)
                                SELECT '미적분', class_id, seat_no, student_name, ?
                                FROM seat_assignments;
                                """,
                                (now,),
                            )
                    else:
                        # 정말 옛날 형태(예: A만 있던 형태)일 가능성 -> 최대한 복구
                        if "seat_no" in col_names and "student_name" in col_names:
                            if "updated_at" in col_names:
                                conn.execute(
                                    """
                                    INSERT INTO seat_assignments_new (track, class_id, seat_no, student_name, updated_at)
                                    SELECT '미적분', 'A', seat_no, student_name, updated_at
                                    FROM seat_assignments;
                                    """
                                )
                            else:
                                conn.execute(
                                    """
                                    INSERT INTO seat_assignments_new (track, class_id, seat_no, student_name, updated_at)
                                    SELECT '미적분', 'A', seat_no, student_name, ?
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
            else:
                # 혹시 track 컬럼이 없었다가 추가되는 경우를 대비(보수적으로 ensure)
                _ensure_column(conn, "seat_assignments", "track", "track TEXT")
                _ensure_column(conn, "seat_assignments", "class_id", "class_id TEXT")
                _ensure_column(conn, "seat_assignments", "seat_no", "seat_no INTEGER")
                _ensure_column(conn, "seat_assignments", "student_name", "student_name TEXT")
                _ensure_column(conn, "seat_assignments", "updated_at", "updated_at TEXT")

        # --- class_meta 생성(표시용 마지막 변경일 저장) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS class_meta (
                track TEXT NOT NULL,
                class_id TEXT NOT NULL,
                last_date TEXT NOT NULL,
                PRIMARY KEY (track, class_id)
            );
            """
        )

        # 기본값(빈 문자열) 시드: 현재 사용하는 과목/분반 조합만
        for tr, cid in TAB_ORDER:
            conn.execute(
                "INSERT OR IGNORE INTO class_meta(track, class_id, last_date) VALUES(?, ?, ?);",
                (tr, cid, ""),
            )


def load_assignments(track: str, class_id: str) -> dict[int, str]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT seat_no, student_name
            FROM seat_assignments
            WHERE track = ? AND class_id = ?
            ORDER BY seat_no;
            """,
            (track, class_id),
        ).fetchall()
    return {int(seat_no): str(name) for seat_no, name in rows}


def save_assignments(track: str, class_id: str, assignments: dict[int, str]):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            # ✅ 항상 먼저 삭제 -> 이후 입력된 것만 다시 INSERT (빈 입력이면 '초기화'와 동일)
            conn.execute(
                "DELETE FROM seat_assignments WHERE track = ? AND class_id = ?;",
                (track, class_id),
            )
            for seat_no, name in assignments.items():
                if 1 <= seat_no <= TOTAL and name.strip():
                    conn.execute(
                        """
                        INSERT INTO seat_assignments (track, class_id, seat_no, student_name, updated_at)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (track, class_id, seat_no, name.strip(), now),
                    )
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def get_last_date(track: str, class_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT last_date FROM class_meta WHERE track = ? AND class_id = ?;",
            (track, class_id),
        ).fetchone()
    return row[0] if row else ""


def set_last_date(track: str, class_id: str, last_date: str):
    with get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            cur = conn.execute(
                "UPDATE class_meta SET last_date = ? WHERE track = ? AND class_id = ?;",
                (last_date, track, class_id),
            )
            if cur.rowcount == 0:
                conn.execute(
                    "INSERT INTO class_meta(track, class_id, last_date) VALUES(?, ?, ?);",
                    (track, class_id, last_date),
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


def export_db_to_text() -> str:
    """
    전체 과목/분반 좌석 정보 + 마지막 변경일을 사람이 읽을 수 있는 TXT로 변환
    """
    lines = []
    with get_conn() as conn:
        for tr, cid in TAB_ORDER:
            # 마지막 변경일
            row = conn.execute(
                "SELECT last_date FROM class_meta WHERE track = ? AND class_id = ?;",
                (tr, cid),
            ).fetchone()
            last_date = row[0] if row else ""

            lines.append(f"[{tr}{cid}]")
            if last_date:
                lines.append(f"마지막 변경일: {last_date}")

            rows = conn.execute(
                """
                SELECT seat_no, student_name
                FROM seat_assignments
                WHERE track = ? AND class_id = ?
                ORDER BY seat_no;
                """,
                (tr, cid),
            ).fetchall()

            if not rows:
                lines.append("(배정 없음)")
            else:
                for seat_no, name in rows:
                    lines.append(f"{int(seat_no)}: {name}")

            lines.append("")  # 분반 사이 빈 줄

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

st.title("좌석 확인(미적분, AI수학)")

teacher_pw = get_teacher_password()
if not teacher_pw:
    st.warning("secrets.toml에 비밀번호가 설정되어 있지 않습니다. TEACHER_PASSWORD 또는 [auth].password를 설정하세요.")

# 탭 만들기
tab_labels = [f"{tr}{cid}" for tr, cid in TAB_ORDER] + ["교사"]
tabs = st.tabs(tab_labels)

# ---- Student tabs (read-only display) ----
def render_class_tab(track: str, class_id: str, title: str):
    st.subheader(title)
    last_date = get_last_date(track, class_id)
    if last_date:
        st.caption(f"마지막 변경일: {last_date}")
    render_front_bar()
    render_grid(load_assignments(track, class_id))


for i, (tr, cid) in enumerate(TAB_ORDER):
    with tabs[i]:
        render_class_tab(tr, cid, f"{tr}{cid} 좌석 배치")


# ---- Teacher tab ----
with tabs[-1]:
    st.subheader("교사용 입력 (과목/분반별 좌석 배정)")

    pw_input = st.text_input("비밀번호", type="password", key="teacher_pw_input")
    is_teacher = (teacher_pw != "") and (pw_input == teacher_pw)

    if teacher_pw and not is_teacher:
        st.info("비밀번호를 입력하면 편집 및 저장이 활성화됩니다.")
    else:
        # ✅ (1) DB 백업 다운로드 (전체 포함)
        try:
            txt_data = export_db_to_text()
            st.download_button(
                label="좌석 DB 백업 다운로드(전체)",
                data=txt_data,
                file_name="finalseat.txt",
                mime="text/plain",
                use_container_width=True,
            )
        except Exception:
            st.warning("DB 파일을 읽을 수 없어 백업 다운로드를 제공할 수 없습니다.")

        st.divider()
        st.caption("※ 좌석 입력을 모두 지운 뒤 저장하면 해당 과목/분반 좌석이 초기화됩니다.")

        teacher_tabs = st.tabs([f"{tr}{cid} 입력" for tr, cid in TAB_ORDER])

        def teacher_editor(track: str, class_id: str):
            text_key = f"text_{track}_{class_id}"
            date_key = f"last_date_{track}_{class_id}"

            def _reload():
                st.session_state[text_key] = assignments_to_text(load_assignments(track, class_id))
                st.session_state[date_key] = get_last_date(track, class_id)

            # 최초 진입 시 DB 저장본 로딩
            if text_key not in st.session_state:
                st.session_state[text_key] = assignments_to_text(load_assignments(track, class_id))
            if date_key not in st.session_state:
                st.session_state[date_key] = get_last_date(track, class_id)

            top1, top2 = st.columns([1, 1])
            with top1:
                st.write(f"**{track}{class_id} 입력**")
            with top2:
                st.button(
                    "현재 저장본 불러오기",
                    key=f"reload_{track}_{class_id}",
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

            if st.button("저장", key=f"save_{track}_{class_id}", use_container_width=True):
                new_assignments, errors = parse_text_to_assignments(text)
                if errors:
                    st.error("입력 오류가 있습니다.")
                    for e in errors:
                        st.write(f"- {e}")
                    return

                # 좌석 저장
                save_assignments(track, class_id, new_assignments)

                # 마지막 변경일 저장(빈 값도 허용: 표시 제거용)
                set_last_date(track, class_id, last_date_text.strip())

                st.success(f"{track}{class_id} 저장 완료. (분반 탭에 즉시 반영됩니다.)")
                st.rerun()

        for j, (tr, cid) in enumerate(TAB_ORDER):
            with teacher_tabs[j]:
                teacher_editor(tr, cid)
