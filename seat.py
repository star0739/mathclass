import sqlite3
import time
import uuid
from contextlib import contextmanager

import streamlit as st

DB_PATH = "seats.db"

ROWS = 5
COLS = 6

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

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assignments (
                seat_id TEXT PRIMARY KEY,               -- seat unique
                user_token TEXT UNIQUE NOT NULL,         -- user unique (one seat per user)
                student_name TEXT NOT NULL,
                assigned_at REAL NOT NULL                -- server time (epoch)
            );
        """)
        # defaults
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('is_open', '0');")
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('round_id', '1');")

def get_setting(key: str) -> str:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,)).fetchone()
    return row[0] if row else ""

def set_setting(key: str, value: str):
    with get_conn() as conn:
        conn.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value;", (key, value))

def list_assignments() -> dict:
    """Returns dict seat_id -> {user_token, student_name, assigned_at}"""
    with get_conn() as conn:
        rows = conn.execute("SELECT seat_id, user_token, student_name, assigned_at FROM assignments;").fetchall()
    out = {}
    for seat_id, user_token, student_name, assigned_at in rows:
        out[seat_id] = {
            "user_token": user_token,
            "student_name": student_name,
            "assigned_at": assigned_at
        }
    return out

def get_user_assignment(user_token: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT seat_id, student_name, assigned_at FROM assignments WHERE user_token = ?;",
            (user_token,)
        ).fetchone()
    if not row:
        return None
    return {"seat_id": row[0], "student_name": row[1], "assigned_at": row[2]}

def try_assign(seat_id: str, user_token: str, student_name: str):
    """
    Atomic seat assignment:
    - If seat is taken -> fail with 'seat_taken'
    - If user already has seat -> fail with 'user_already_assigned'
    - Else success
    """
    now = time.time()
    try:
        with get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # Try insert; relies on UNIQUE constraints
            conn.execute(
                "INSERT INTO assignments(seat_id, user_token, student_name, assigned_at) VALUES(?, ?, ?, ?);",
                (seat_id, user_token, student_name, now)
            )
            conn.execute("COMMIT;")
        return {"ok": True}
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        # Identify which constraint likely failed
        if "assignments.seat_id" in msg or "unique constraint failed: assignments.seat_id" in msg:
            return {"ok": False, "reason": "seat_taken"}
        if "assignments.user_token" in msg or "unique constraint failed: assignments.user_token" in msg:
            return {"ok": False, "reason": "user_already_assigned"}
        return {"ok": False, "reason": "integrity_error"}
    except Exception:
        return {"ok": False, "reason": "unknown_error"}

def cancel_own_seat(user_token: str, seat_id: str):
    """
    Only cancels if (seat_id, user_token) matches.
    """
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        cur = conn.execute(
            "DELETE FROM assignments WHERE seat_id = ? AND user_token = ?;",
            (seat_id, user_token)
        )
        conn.execute("COMMIT;")
    return cur.rowcount == 1

def reset_round():
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute("DELETE FROM assignments;")
        # increment round_id
        rid = int(get_setting("round_id") or "1")
        set_setting("round_id", str(rid + 1))
        set_setting("is_open", "0")
        conn.execute("COMMIT;")

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="분반 좌석 선착순", layout="wide")
init_db()

# Session identity
if "user_token" not in st.session_state:
    st.session_state.user_token = str(uuid.uuid4())

st.title("분반 좌석 선착순 배정 (5x6)")

tab_student, tab_teacher = st.tabs(["학생", "교사"])

with tab_student:
    st.subheader("학생 화면")

    student_name = st.text_input("이름을 입력하세요", value=st.session_state.get("student_name", ""))
    if student_name:
        st.session_state.student_name = student_name.strip()

    is_open = get_setting("is_open") == "1"
    round_id = get_setting("round_id")

    if not st.session_state.get("student_name"):
        st.info("이름을 입력하면 대기 상태로 들어갑니다.")
        st.stop()

    st.caption(f"현재 라운드: {round_id}")
    if not is_open:
        st.info("대기 중입니다. 교사가 시작하면 좌석 선택이 열립니다.")
        st.autorefresh(interval=1000, key="wait_refresh")  # 1초마다 갱신
        # 현재 본인 배정 상태 표시(혹시 열리기 전에 배정된 경우 등)
        my = get_user_assignment(st.session_state.user_token)
        if my:
            st.success(f"이미 배정됨: {my['seat_id']} (취소하려면 해당 좌석을 다시 누르세요)")
        st.stop()

    # Open state
    st.success("좌석 선택이 열렸습니다. 원하는 좌석을 클릭하세요.")
    st.autorefresh(interval=1000, key="open_refresh")  # 실시간 반영용(선택현황 표시)

    assignments = list_assignments()
    my = get_user_assignment(st.session_state.user_token)
    my_seat = my["seat_id"] if my else None

    # Seat grid
    st.markdown("### 좌석 선택")
    for r in range(1, ROWS + 1):
        cols = st.columns(COLS)
        for c in range(1, COLS + 1):
            seat_id = f"R{r}C{c}"
            taken = seat_id in assignments
            taken_by_me = taken and assignments[seat_id]["user_token"] == st.session_state.user_token

            label = seat_id
            if taken:
                label = f"{seat_id}\n(사용중)"

            disabled = False
            # 규칙 4: 이미 다른 좌석을 가진 학생은 다른 좌석 선택 불가 (취소는 본인 좌석만)
            if my_seat and seat_id != my_seat:
                disabled = True

            # 본인 좌석은 다시 눌러 취소 가능
            if taken_by_me:
                disabled = False
                label = f"{seat_id}\n(내 좌석·취소)"

            if cols[c - 1].button(label, key=f"btn_{seat_id}_{round_id}", disabled=disabled):
                if not my_seat:
                    # assign attempt
                    result = try_assign(
                        seat_id=seat_id,
                        user_token=st.session_state.user_token,
                        student_name=st.session_state.student_name
                    )
                    if result["ok"]:
                        st.success(f"{seat_id} 좌석이 배정되었습니다.")
                    else:
                        if result["reason"] == "seat_taken":
                            st.error("이미 선택된 좌석입니다.")
                        elif result["reason"] == "user_already_assigned":
                            st.warning("이미 좌석이 배정되어 있습니다. 취소 후 다시 선택하세요.")
                        else:
                            st.error("처리 중 오류가 발생했습니다. 다시 시도하세요.")
                else:
                    # my_seat exists => only allow cancel by clicking my seat
                    if seat_id == my_seat:
                        ok = cancel_own_seat(st.session_state.user_token, seat_id)
                        if ok:
                            st.success("좌석 배정이 취소되었습니다. 이제 다른 좌석을 선택할 수 있습니다.")
                        else:
                            st.error("취소 권한이 없거나 이미 처리되었습니다.")
                    else:
                        st.warning("다른 좌석을 선택하려면 먼저 본인 좌석을 취소하세요.")

    st.markdown("---")
    st.markdown("### 내 상태")
    my = get_user_assignment(st.session_state.user_token)
    if my:
        st.write(f"- 이름: {st.session_state.student_name}")
        st.write(f"- 배정 좌석: **{my['seat_id']}**")
        st.write(f"- 배정 시각(서버): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(my['assigned_at']))}")
        st.caption("취소하려면 본인 좌석 버튼을 다시 누르세요.")
    else:
        st.write(f"- 이름: {st.session_state.student_name}")
        st.write("- 배정 좌석: 없음")

with tab_teacher:
    st.subheader("교사 화면")

    # 간단한 보호장치(운영 시 환경변수로 비밀번호 설정 권장)
    teacher_pass = st.text_input("교사용 비밀번호", type="password")
    REQUIRED = st.secrets.get("TEACHER_PASSWORD", "") if hasattr(st, "secrets") else ""
    if REQUIRED and teacher_pass != REQUIRED:
        st.info("비밀번호를 입력하세요.")
        st.stop()

    is_open = get_setting("is_open") == "1"
    round_id = get_setting("round_id")
    st.caption(f"현재 라운드: {round_id}")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("좌석 선택 시작(오픈)"):
            set_setting("is_open", "1")
            st.success("좌석 선택을 오픈했습니다.")
    with colB:
        if st.button("좌석 선택 마감(클로즈)"):
            set_setting("is_open", "0")
            st.warning("좌석 선택을 마감했습니다.")
    with colC:
        if st.button("라운드 초기화(전체 취소 + 새 라운드)"):
            reset_round()
            st.warning("초기화 완료. 새 라운드로 전환되었습니다.")

    st.markdown("### 배정 현황")
    assignments = list_assignments()
    if not assignments:
        st.write("아직 배정된 좌석이 없습니다.")
    else:
        # 정렬: assigned_at 빠른 순
        items = sorted(assignments.items(), key=lambda kv: kv[1]["assigned_at"])
        for seat_id, info in items:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info["assigned_at"]))
            st.write(f"- **{seat_id}** : {info['student_name']}  (배정: {ts})")
