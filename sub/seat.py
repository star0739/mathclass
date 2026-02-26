# ---------------------------
# 분반(과목) 설정
# ---------------------------
CALC_CLASSES = ["A", "C", "D", "J"]     # 미적분
AI_MATH_CLASSES = ["B", "F", "H"]      # AI수학

TRACK_OPTIONS = ["미적분", "AI수학"]

def normalize_class_letter(s: str) -> str:
    s = (s or "").strip().upper()
    # 사용자가 'A반', 'a', ' A '처럼 넣어도 처리
    if s.endswith("반"):
        s = s[:-1].strip().upper()
    return s


# ---------------------------
# DB init 수정: assignments에 track, class_letter 컬럼 추가
# ---------------------------
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
                seat_id TEXT PRIMARY KEY,
                user_token TEXT UNIQUE NOT NULL,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                class_letter TEXT,                      -- 분반 문자(A~J 등)
                track TEXT,                             -- 미적분 / AI수학
                assigned_at REAL NOT NULL
            );
            """
        )

        _ensure_column(conn, "assignments", "student_id", "student_id TEXT")
        _ensure_column(conn, "assignments", "student_name", "student_name TEXT")
        _ensure_column(conn, "assignments", "class_letter", "class_letter TEXT")
        _ensure_column(conn, "assignments", "track", "track TEXT")
        _ensure_column(conn, "assignments", "assigned_at", "assigned_at REAL")

        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_assignments_student_id ON assignments(student_id);")
        except sqlite3.OperationalError:
            pass

        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('is_open', '0');")
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES('round_id', '1');")
