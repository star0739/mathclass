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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assignments (
                seat_id TEXT PRIMARY KEY,               -- seat unique ("1".."30")
                user_token TEXT UNIQUE NOT NULL,         -- user unique (one seat per user)
                student_name TEXT NOT NULL,
                assigned_at REAL NOT NULL                -- server time (epoch)
            );
            """
        )
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('is_open', '0');")
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('round_id', '1');")


def get_setting(key: str) -> str:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?;", (key,)).fetchone()
    return row[0] if row else ""


def set_setting(key: str, value: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
            (key, value),
        )


def list_assignments() -> dict:
    """Returns dict seat_id -> {user_token, student_name, assigned_at}"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT seat_id, user_token, student_name, assigned_at FROM assignments;"
        ).fetchall()
    out = {}
    for seat_id, user_token, student_name, assigned_at in rows:
        out[str(seat_id)] = {
            "user_token": user_token,
            "student_name": student_name,
            "assigned_at": assigned_at,
        }
    return out


def get_user_assignment(user_token: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT seat_id, student_name, assigned_at FROM assignments WHERE user_token = ?;",
            (user_token,),
        ).fetchone()
    if not row:
        return None
    return {"seat_id": str(row[0]), "student_name": row[1], "assigned_at": row[2]}


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
            conn.execute(
                "INSERT INTO assignments(seat_id, user_token, student_name, assigned_at) VALUES(?, ?, ?, ?);",
                (str(seat_id), user_token, student_name, now),
            )
            conn.execute("COMMIT;")
        return {"ok": True}
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "unique constraint failed: assignments.seat_id" in msg:
            return {"ok": False, "reason": "seat_taken"}
        if "unique constraint failed: assignments.user_token" in msg:
            return {"ok": False, "reason": "user_already_assigned"}
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
        rid = int(get_setting("round_id") or "1")
        set_setting("round_id", str(rid + 1))
        set_setting("is_open", "0")
        conn.execute("COMMIT;")


def safe_seat_sort_key(seat_id: str):
    """
    기존 DB에 'R1C1' 같은 값이 남아있는 경우에도 터지지 않도록
    정렬 키를 안전하게 계산.
    - 숫자면 int로 정렬
    - 숫자가 아니면 뒤쪽으로 보내고(큰 값), 문자열로 2차 정렬
    """
    try:
        return (0, int(str(seat_id)))
    except Exception:
        return (1, str(seat_id))


# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="분반 좌석 선착순", layout="wide")
init_db()

# Session identity
if "user_token" not in st.session_state:
    st.session_state.user_token = str(uuid.uuid4())

st.title("3학년 미적분 좌석 지정")

tab_student, tab_teacher = st.tabs(["학생", "교사"])

with tab_student:
    st.subheader("학생 화면")

    student_name = st.text_input("이름을 입력하세요", value=st.session_state.get("student_name", ""))
    if student_name:
        st.session_state.student_name = student_name.strip()

    is_open = get_setting("is_open") == "1"
    round_id = get_setting("round_id")
    st.caption(f"현재 라운드: {round_id}")

    # ✅ NameError 방지: 기본값
    assignments = {}
    my = None
    my_seat = None

    if not st.session_state.get("student_name"):
        st.info("이름을 입력하면 대기 상태로 들어갑니다.")
    else:
        if not is_open:
            st.info("대기 중입니다. 교사가 시작하면 좌석 선택이 열립니다.")
            st_autorefresh(interval=1000, key="wait_refresh")

            my = get_user_assignment(st.session_state.user_token)
            if my:
                st.success(f"이미 배정됨: {my['seat_id']}번 (취소하려면 해당 좌석을 다시 누르세요)")

        else:
            st.success("좌석 선택이 열렸습니다. 원하는 좌석을 클릭하세요.")
            st_autorefresh(interval=1000, key="open_refresh")

            assignments = list_assignments()
            my = get_user_assignment(st.session_state.user_token)
            my_seat = my["seat_id"] if my else None

            st.markdown("## <칠판 & 교탁>")
            st.caption("아래 좌석을 번호로 선택하세요. (5열 × 6행)")

            st.markdown("### 좌석 선택")

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

                    # 규칙 4: 이미 다른 좌석을 가진 학생은 다른 좌석 선택 불가 (취소는 본인 좌석만)
                    if my_seat and seat_id != my_seat:
                        disabled = True

                    # 규칙 5: 본인 좌석은 다시 눌러 취소 가능
                    if taken_by_me:
                        disabled = False
                        label = f"{seat_num}\n(내 좌석·취소)"

                    if cols[c - 1].button(label, key=f"btn_{seat_id}_{round_id}", disabled=disabled):
                        if not my_seat:
                            result = try_assign(
                                seat_id=seat_id,
                                user_token=st.session_state.user_token,
                                student_name=st.session_state.student_name,
                            )
                            if result["ok"]:
                                st.success(f"{seat_num}번 좌석이 배정되었습니다.")
                            else:
                                if result["reason"] == "seat_taken":
                                    st.error("이미 선택된 좌석입니다.")
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
    st.markdown("### 내 상태")
    my = get_user_assignment(st.session_state.user_token)
    if st.session_state.get("student_name"):
        if my:
            st.write(f"- 이름: {st.session_state.student_name}")
            st.write(f"- 배정 좌석: **{my['seat_id']}번**")
            st.write(f"- 배정 시각(서버): {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(my['assigned_at']))}")
            st.caption("취소하려면 본인 좌석 버튼을 다시 누르세요.")
        else:
            st.write(f"- 이름: {st.session_state.student_name}")
            st.write("- 배정 좌석: 없음")
    else:
        st.write("- 이름: (미입력)")
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
        # ✅ 기존 DB에 숫자 아닌 seat_id가 남아도 터지지 않는 정렬
        items = sorted(assignments.items(), key=lambda kv: safe_seat_sort_key(kv[0]))
        for seat_id, info in items:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["assigned_at"]))
            # seat_id가 숫자가 아니어도 표시 가능
            st.write(f"- **{seat_id}** : {info['student_name']}  (배정: {ts})")

    st.caption("참고: 이전 버전(R1C1 형태) 데이터가 남아 있으면 '라운드 초기화'를 한 번 실행하는 것을 권장합니다.")
