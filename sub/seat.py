import sqlite3
import time
import uuid
from contextlib import contextmanager

import streamlit as st
from streamlit_autorefresh import st_autorefresh

DB_PATH = "seats.db"

# ✅ 5열 × 6행 = 30석
ROWS = 6
COLS = 5


# ---------------------------
# DB utilities
# ---------------------------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    yield conn
    conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
    """테이블에 컬럼이 없으면 추가(기존 DB 마이그레이션 용)."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl};")


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

        # ✅ student_id(학번) 컬럼 포함 (새로 만드는 경우)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                seat_id TEXT PRIMARY KEY,               -- seat unique ("1".."30")
                user_token TEXT UNIQUE NOT NULL,         -- user unique (one seat per user, per device)
                student_id TEXT NOT NULL,                -- 학번
                student_name TEXT NOT NULL,              -- 이름
                assigned_at REAL NOT NULL                -- server time (epoch)
            );
            """
        )

        # ✅ 기존 DB에 student_id 컬럼이 없을 수 있으니 자동 추가
        _ensure_column(conn, "assignments", "student_id", "student_id TEXT")
        _ensure_column(conn, "assignments", "student_name", "student_name TEXT")
        _ensure_column(conn, "assignments", "assigned_at", "assigned_at REAL")

        # ✅ (권장) 같은 학번이 여러 좌석을 갖는 것을 원천 차단: UNIQUE INDEX
        # 이미 중복 데이터가 있으면 생성이 실패할 수 있으므로 예외는 무시합니다.
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_assignments_student_id ON assignments(student_id);")
        except sqlite3.OperationalError:
            pass

        # defaults
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('is_open', '0');")
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('round_id', '1');")


def get_setting(key: str) -> str:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,)).fetchone()
    return row[0] if row else ""


def set_setting(key: str, value: str):
    """
    1) UPDATE 먼저 시도
    2) 없으면 INSERT
    """
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        cur = conn.execute("UPDATE settings SET value = ? WHERE key = ?;", (value, key))
        if cur.rowcount == 0:
            conn.execute("INSERT INTO settings(key, value) VALUES(?, ?);", (key, value))
        conn.execute("COMMIT;")


def list_assignments() -> dict:
    """Returns dict seat_id -> {user_token, student_id, student_name, assigned_at}"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT seat_id, user_token, student_id, student_name, assigned_at FROM assignments;"
        ).fetchall()
    out = {}
    for seat_id, user_token, student_id, student_name, assigned_at in rows:
        out[str(seat_id)] = {
            "user_token": user_token,
            "student_id": str(student_id),
            "student_name": student_name,
            "assigned_at": assigned_at,
        }
    return out


def get_user_assignment(user_token: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT seat_id, student_id, student_name, assigned_at FROM assignments WHERE user_token = ?;",
            (user_token,),
        ).fetchone()
    if not row:
        return None
    return {"seat_id": str(row[0]), "student_id": str(row[1]), "student_name": row[2], "assigned_at": row[3]}


def try_assign(seat_id: str, user_token: str, student_id: str, student_name: str):
    """
    Atomic seat assignment:
    - If seat is taken -> fail with 'seat_taken'
    - If user already has seat (user_token) -> fail with 'user_already_assigned'
    - If student_id already assigned -> fail with 'student_id_already_assigned'
    - Else success
    """
    now = time.time()
    try:
        with get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO assignments(seat_id, user_token, student_id, student_name, assigned_at)
                VALUES(?, ?, ?, ?, ?);
                """,
                (str(seat_id), user_token, str(student_id), student_name, now),
            )
            conn.execute("COMMIT;")
        return {"ok": True}
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "unique constraint failed: assignments.seat_id" in msg:
            return {"ok": False, "reason": "seat_taken"}
        if "unique constraint failed: assignments.user_token" in msg:
            return {"ok": False, "reason": "user_already_assigned"}
        if "unique constraint failed: assignments.student_id" in msg or "ux_assignments_student_id" in msg:
            return {"ok": False, "reason": "student_id_already_assigned"}
        return {"ok": False, "reason": "integrity_error"}
    except Exception:
        return {"ok": False, "reason": "unknown_error"}


def cancel_own_seat(user_token: str, seat_id: str):
    """Only cancels if (seat_id, user_token) matches."""
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        cur = conn.execute(
            "DELETE FROM assignments WHERE seat_id = ? AND user_token = ?;",
            (str(seat_id), user_token),
        )
        conn.execute("COMMIT;")
    return cur.rowcount == 1


def reset_round():
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute("DELETE FROM assignments;")

        row = conn.execute("SELECT value FROM settings WHERE key='round_id';").fetchone()
        rid = int(row[0]) if row and str(row[0]).isdigit() else 1
        new_rid = str(rid + 1)

        cur = conn.execute("UPDATE settings SET value=? WHERE key='round_id';", (new_rid,))
        if cur.rowcount == 0:
            conn.execute("INSERT INTO settings(key, value) VALUES('round_id', ?);", (new_rid,))

        cur = conn.execute("UPDATE settings SET value='0' WHERE key='is_open';")
        if cur.rowcount == 0:
            conn.execute("INSERT INTO settings(key, value) VALUES('is_open', '0');")

        conn.execute("COMMIT;")


def safe_seat_sort_key(seat_id: str):
    try:
        return (0, int(str(seat_id)))
    except Exception:
        return (1, str(seat_id))


# ---------------------------
# UI helpers
# ---------------------------
def render_front_bar():
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


# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="선착순 좌석 배정", layout="wide")
init_db()

if "user_token" not in st.session_state:
    st.session_state.user_token = str(uuid.uuid4())

st.title("선착순 좌석 배정")

tab_student, tab_teacher = st.tabs(["학생", "교사"])


with tab_student:
    st.subheader("학생 화면")

    # ✅ 학번 + 이름 입력
    student_id = st.text_input("학번을 입력하세요", value=st.session_state.get("student_id", ""))
    if student_id:
        st.session_state.student_id = student_id.strip()

    student_name = st.text_input("이름을 입력하세요", value=st.session_state.get("student_name", ""))
    if student_name:
        st.session_state.student_name = student_name.strip()

    is_open = get_setting("is_open") == "1"
    round_id = get_setting("round_id")

    if not st.session_state.get("student_id") or not st.session_state.get("student_name"):
        st.info("학번과 이름을 입력하면 대기 상태로 들어갑니다.")
    else:
        if not is_open:
            st.info("대기 중입니다. 교사가 시작하면 좌석 선택이 열립니다.")
            st_autorefresh(interval=1000, key="wait_refresh")

            my = get_user_assignment(st.session_state.user_token)
            if my:
                st.success(f"이미 배정됨: {my['seat_id']}번")
        else:
            st.success("좌석 선택이 열렸습니다. 원하는 좌석을 클릭하세요.")
            st_autorefresh(interval=1000, key="open_refresh")

            assignments = list_assignments()
            my = get_user_assignment(st.session_state.user_token)
            my_seat = my["seat_id"] if my else None

            render_front_bar()
            st.info("휴대폰 사용 시 가로모드로 해야 좌석이 잘 표시됩니다.")

            for r in range(1, ROWS + 1):
                cols = st.columns(COLS)
                for c in range(1, COLS + 1):
                    seat_num = (r - 1) * COLS + c  # 1..30
                    seat_id = str(seat_num)

                    taken = seat_id in assignments
                    taken_by_me = taken and assignments[seat_id]["user_token"] == st.session_state.user_token

                    label = f"{seat_num}"
                    if taken:
                        label = f"{seat_num}\n(사용중)"

                    disabled = False
                    if my_seat and seat_id != my_seat:
                        disabled = True

                    if taken_by_me:
                        disabled = False
                        label = f"{seat_num}\n(내 좌석·취소)"

                    if cols[c - 1].button(label, key=f"btn_{seat_id}_{round_id}", disabled=disabled):
                        if not my_seat:
                            result = try_assign(
                                seat_id=seat_id,
                                user_token=st.session_state.user_token,
                                student_id=st.session_state.student_id,
                                student_name=st.session_state.student_name,
                            )
                            if result["ok"]:
                                st.success(f"{seat_num}번 좌석이 배정되었습니다.")
                            else:
                                if result["reason"] == "seat_taken":
                                    st.error("이미 선택된 좌석입니다.")
                                elif result["reason"] == "student_id_already_assigned":
                                    st.warning("해당 학번은 이미 좌석이 배정되어 있습니다. (다른 기기/창에서 신청했을 수 있음)")
                                elif result["reason"] == "user_already_assigned":
                                    st.warning("이미 좌석이 배정되어 있습니다. 취소 후 다시 선택하세요.")
                                else:
                                    st.error("처리 중 오류가 발생했습니다. 다시 시도하세요.")
                        else:
                            if seat_id == my_seat:
                                ok = cancel_own_seat(st.session_state.user_token, seat_id)
                                if ok:
                                    st.success("좌석 배정이 취소되었습니다. 이제 다른 좌석을 선택할 수 있습니다.")
                                else:
                                    st.error("취소 권한이 없거나 이미 처리되었습니다.")
                            else:
                                st.warning("다른 좌석을 선택하려면 먼저 본인 좌석을 취소하세요.")

    st.markdown("---")
    st.markdown("### 신청 내역")
    my = get_user_assignment(st.session_state.user_token)
    if st.session_state.get("student_id") and st.session_state.get("student_name"):
        if my:
            st.write(f"- 학번 이름: {st.session_state.student_id} {st.session_state.student_name}")
            st.write(f"- 배정 좌석: **{my['seat_id']}번**")
            st.caption("취소하려면 본인 좌석 버튼을 다시 누르세요.")
        else:
            st.write(f"- 학번 이름: {st.session_state.student_id} {st.session_state.student_name}")
            st.write("- 배정 좌석: 없음")
    else:
        st.write("- 학번 이름: (미입력)")
        st.write("- 배정 좌석: 없음")


with tab_teacher:
    st.subheader("교사 화면")

    teacher_pass = st.text_input("교사용 비밀번호", type="password")
    REQUIRED = st.secrets.get("TEACHER_PASSWORD", "")

    if REQUIRED and teacher_pass != REQUIRED:
        st.info("비밀번호를 입력하세요.")
    else:
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("좌석 선택 시작"):
                set_setting("is_open", "1")
                st.success("좌석 선택을 오픈했습니다.")
        with colB:
            if st.button("좌석 선택 마감"):
                set_setting("is_open", "0")
                st.warning("좌석 선택을 마감했습니다.")
        with colC:
            if st.button("라운드 초기화"):
                reset_round()
                st.warning("초기화 완료")

    st.markdown("### 배정 현황")
    assignments = list_assignments()
    if not assignments:
        st.write("아직 배정된 좌석이 없습니다.")
    else:
        items = sorted(assignments.items(), key=lambda kv: safe_seat_sort_key(kv[0]))
        for seat_id, info in items:
            # ✅ (요청) "학번 이름" 형태로 표기
            st.write(f"- **{seat_id}** : {info['student_id']} {info['student_name']}")

    st.caption("이전 버전 데이터가 섞여 있으면, '라운드 초기화'로 한번 정리한 뒤 사용하면 가장 안정적입니다.")
