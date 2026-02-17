
import re
import streamlit as st
import pandas as pd
import numpy as np

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

from assessment.common import (
    init_assessment_session,
    require_student_id,
    set_df,
    get_df,
    get_df_preview,
    set_xy,
    get_xy,
    get_step1_summary,
)

from assessment.google_sheets import append_step2_row

# -----------------------------
# 운영 기준
# -----------------------------
MIN_VALID_POINTS = 30


# -----------------------------
# 세션용 step2 저장(간단)
# -----------------------------
def _get_step2_state() -> dict:
    return st.session_state.get("assessment_step2", {})


def _set_step2_state(d: dict) -> None:
    st.session_state["assessment_step2"] = d


# -----------------------------
# CSV 로더 (1차시와 동일하게 관대)
# -----------------------------
def read_csv_kosis(file) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(
                file,
                encoding=enc,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError("CSV를 읽을 수 없습니다.")


def parse_year_month(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.+$", "", regex=True)
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)
    dt = pd.to_datetime(s, errors="coerce", format="%Y-%m")
    mask = dt.isna()
    if mask.any():
        digits = s[mask].str.replace(r"\D", "", regex=True)
        m6 = digits.str.fullmatch(r"\d{6}")
        if m6.any():
            dt2 = pd.to_datetime(digits[m6], errors="coerce", format="%Y%m")
            dt.loc[digits[m6].index] = dt2
    return dt


# -----------------------------
# 1차시 TXT 백업 파서(최소)
#  - step1 txt의 섹션 제목을 이용해 값 추출
# -----------------------------
def parse_step1_backup_txt(text: str) -> dict:
    # 매우 관대한 파서: 키워드 라인을 찾아 값 추출
    out = {}
    lines = [ln.strip() for ln in text.splitlines()]

    def find_value(prefix: str) -> str:
        for ln in lines:
            if ln.startswith(prefix):
                return ln.replace(prefix, "", 1).strip()
        return ""

    def find_section_text(start_key: str, end_key: str | None = None) -> str:
        # start_key 라인을 찾고, end_key(있으면) 전까지를 반환
        # end_key가 없으면 다음 섹션 헤더(대괄호 또는 "- "로 시작) 전까지 반환
        start_idx = None
        for i, ln in enumerate(lines):
            if ln == start_key:
                start_idx = i
                break
        if start_idx is None:
            return ""

        if end_key is not None:
            for j in range(start_idx + 1, len(lines)):
                if lines[j] == end_key:
                    return "\n".join(lines[start_idx + 1 : j]).strip()
            return "\n".join(lines[start_idx + 1 :]).strip()

        for j in range(start_idx + 1, len(lines)):
            ln = lines[j]
            if ln.startswith("[") and ln.endswith("]"):
                return "\n".join(lines[start_idx + 1 : j]).strip()
            if ln.startswith("- ") and lines[start_idx].startswith("- "):
                return "\n".join(lines[start_idx + 1 : j]).strip()
        return "\n".join(lines[start_idx + 1 :]).strip()

    out["student_id"] = find_value("학번:")
    out["data_source"] = find_value("- 데이터 출처:")
    out["x_col"] = ""
    out["y_col"] = ""
    for ln in lines:
        if ln.startswith("- X축:"):
            # "- X축: A  |  Y축: B"
            m = re.search(r"- X축:\s*(.*?)\s*\|\s*Y축:\s*(.*)$", ln)
            if m:
                out["x_col"] = m.group(1).strip()
                out["y_col"] = m.group(2).strip()
    out["x_mode"] = find_value("- X축 해석 방식:")
    out["valid_n"] = find_value("- 유효 데이터 점 개수:")
    out["features"] = ""

    sec = find_section_text("[그래프 관찰 특징]", "[모델링 가설]")
    if sec:
        out["features"] = sec

    out["model_primary"] = find_value("- 주된 모델:")

    sec2 = find_section_text("- 주된 모델 근거:", "[추가 메모]")
    out["model_primary_reason"] = sec2.strip() if sec2 else ""

    return out


# -----------------------------
# LaTeX 블록 추출/미리보기용
# -----------------------------
LATEX_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

def extract_latex_blocks(s: str) -> list[str]:
    if not s:
        return []
    return [m.group(1).strip() for m in LATEX_BLOCK.finditer(s)]


# -----------------------------
# 데이터 기반 근사 도함수(차분/gradient)
# -----------------------------
def compute_derivatives(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return dy, d2y


# -----------------------------
# TXT 백업 생성(2차시)
# -----------------------------
def build_step2_backup(payload: dict) -> bytes:
    lines = []
    lines.append("공공데이터 분석 수행 (2차시) 백업")
    lines.append("=" * 40)
    lines.append(f"저장시각: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"학번: {payload.get('student_id','')}")
    lines.append("")

    lines.append("[가설 재평가]")
    lines.append(f"- 1차시 가설 모델: {payload.get('model_hypothesis_step1','')}")
    lines.append(f"- 가설 판단: {payload.get('hypothesis_decision','')}")
    if payload.get("hypothesis_decision") == "가설 수정":
        lines.append(f"- 수정한 가설 모델: {payload.get('revised_model','')}")
    lines.append("")

    lines.append("[데이터 정보]")
    lines.append(f"- 데이터 출처: {payload.get('data_source','')}")
    lines.append(f"- X축: {payload.get('x_col','')} | Y축: {payload.get('y_col','')}")
    lines.append(f"- 유효 데이터 점: {payload.get('valid_n','')}")
    lines.append("")

    lines.append("[AI 프롬프트]")
    lines.append(payload.get("ai_prompt","").strip() or "(미입력)")
    lines.append("")

    lines.append("[AI 모델식/미분식(LaTeX)]")
    lines.append(payload.get("ai_model_latex","").strip() or "(미입력)")
    lines.append(payload.get("ai_derivative_latex","").strip() or "")
    lines.append(payload.get("ai_second_derivative_latex","").strip() or "")
    lines.append("")

    lines.append("[미분 관점의 모델 해석(학생 작성)]")
    lines.append(payload.get("student_analysis","").strip() or "(미입력)")
    lines.append("")

    lines.append("[추가 메모]")
    lines.append(payload.get("note","").strip() or "(없음)")
    lines.append("")
    lines.append("※ 수식은 $$...$$ 형태의 LaTeX로 유지하는 것이 안전합니다.")

    return "\n".join(lines).encode("utf-8-sig")


# -----------------------------
# AI 입력 수식으로 그래프 그리기(가능한 범위에서)
# -----------------------------
import ast

@st.cache_resource(show_spinner=False)
def _get_sympy_runtime():
    try:
        import sympy as sp
        try:
            from sympy.parsing.latex import parse_latex
        except Exception:
            parse_latex = None
        return sp, parse_latex
    except ModuleNotFoundError:
        return None, None
    except Exception:
        return None, None


def _latex_to_numpy_expr(expr_text: str) -> str | None:
    raw = (expr_text or "").strip()
    if not raw:
        return None

    blocks = extract_latex_blocks(raw)
    s = blocks[0] if blocks else raw

    if "=" in s:
        s = s.split("=", 1)[1].strip()

    s = s.replace("\\,", " ").replace("\\;", " ").replace("\n", " ").strip()

    # \left, \right 제거
    s = s.replace("\\left", "").replace("\\right", "")

    # (1) 상수/기호 치환
    s = s.replace("\\pi", "np.pi")

    # (2) 함수 치환: \cos(, \cos\(...), \cos\left(... 모두 커버
    #    일단 \cos, \sin 등을 np.cos, np.sin 으로 바꾼 뒤
    #    남는 역슬래시/공백을 정리한다.
    func_map = {
        "\\sin": "np.sin",
        "\\cos": "np.cos",
        "\\tan": "np.tan",
        "\\ln": "np.log",
        "\\log": "np.log",
        "\\exp": "np.exp",
        "\\sqrt": "np.sqrt",
    }
    for k, v in func_map.items():
        s = s.replace(k, v)

    # 곱셈 기호 치환
    s = s.replace("\\cdot", "*").replace("\\times", "*")

    # (3) \frac{a}{b} 처리 (중첩은 제한적)
    frac_pat = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    for _ in range(10):
        new_s = frac_pat.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s

    # (4) 지수 처리
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([0-9t])", r"**\1", s)

    # (5) 공백 제거
    s = re.sub(r"\s+", "", s)

    # (6) LaTeX에서 곱셈이 생략된 패턴들에 '*' 삽입
    # 6-1) 숫자 바로 뒤에 변수/np/괄호가 오면: 12t, 2np.pi, 3(...
    s = re.sub(r"(\d)(t)", r"\1*\2", s)
    s = re.sub(r"(\d)(np\.)", r"\1*\2", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)

    # 6-2) 변수/닫는괄호 뒤에 np.pi 등이 오면: tnp.pi, )np.pi
    s = re.sub(r"(t)(np\.)", r"\1*\2", s)
    s = re.sub(r"(\))(np\.)", r"\1*\2", s)

    # 6-3) t 바로 뒤에 '('가 오면: t( ... )  -> t*( ... )
    s = re.sub(r"(t)\(", r"\1*(", s)

    # (7) e^{...}를 np.exp로 처리하려면, "e**(...)" 패턴을 잡아 변환(선택)
    s = re.sub(r"\be\*\*\(([^)]+)\)", r"np.exp(\1)", s)

    if not s:
        return None
    return s



class _SafeExprChecker(ast.NodeVisitor):
    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.UAdd, ast.USub,
        ast.Call,
        ast.Load,
        ast.Name,
        ast.Constant,
        ast.Attribute,
    )

    ALLOWED_NAMES = {"t", "np"}

    ALLOWED_ATTRS = {
        "sin", "cos", "tan",
        "exp", "log",
        "sqrt",
        "pi",
    }

    def generic_visit(self, node):
        if not isinstance(node, self.ALLOWED_NODES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.ALLOWED_NAMES:
            raise ValueError(f"Disallowed name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute):
        if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
            raise ValueError("Only np.<func> attributes are allowed")
        if node.attr not in self.ALLOWED_ATTRS:
            raise ValueError(f"Disallowed np attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only calls to np.<func>(...) are allowed")
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        if node.keywords:
            raise ValueError("Keyword args are not allowed")


@st.cache_data(show_spinner=False)
def _latex_to_sympy_srepr(expr_text: str) -> str | None:
    sp, parse_latex = _get_sympy_runtime()
    if sp is None:
        return None

    raw = (expr_text or "").strip()
    if not raw:
        return None

    blocks = extract_latex_blocks(raw)
    s = blocks[0] if blocks else raw

    if "=" in s:
        s = s.split("=", 1)[1].strip()

    s = s.replace("\\,", " ").replace("\\;", " ").replace("\n", " ").strip()

    try:
        if parse_latex is not None:
            sym = parse_latex(s)
        else:
            sym = sp.sympify(s)
        return sp.srepr(sym)
    except Exception:
        return None


def _sympy_srepr_to_callable(srepr_text: str | None):
    if not srepr_text:
        return None
    sp, _ = _get_sympy_runtime()
    if sp is None:
        return None
    try:
        sym = sp.sympify(srepr_text)
        t = sp.Symbol("t", real=True)
        fn = sp.lambdify(t, sym, modules=["numpy"])
        return fn
    except Exception:
        return None


def _numpy_expr_to_callable(expr: str | None):
    if not expr:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        _SafeExprChecker().visit(tree)

        def _fn(t):
            return eval(compile(tree, "<expr>", "eval"), {"np": np, "t": t}, {})

        return _fn
    except Exception:
        return None


def _latex_to_callable(expr_text: str):
    srepr_text = _latex_to_sympy_srepr(expr_text or "")
    fn = _sympy_srepr_to_callable(srepr_text)
    if fn is not None:
        return fn

    expr = _latex_to_numpy_expr(expr_text or "")
    return _numpy_expr_to_callable(expr)


def _safe_eval(fn, t: np.ndarray) -> np.ndarray | None:
    if fn is None:
        return None
    try:
        y = fn(t)
        y = np.asarray(y, dtype=float)
        if y.shape != t.shape:
            y = np.broadcast_to(y, t.shape)
        y[~np.isfinite(y)] = np.nan
        return y
    except Exception:
        return None


def _plot_single_ai_function(xv, t, fn, title: str):
    y = _safe_eval(fn, t)
    if y is None:
        st.info("AI 출력식에서 LaTeX 수식을 해석하지 못했습니다. $$...$$ 형태로 다시 확인해 주세요.")
        return

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv, y=y, mode="lines", name=title))
        fig.update_layout(height=360, margin=dict(l=40, r=20, t=30, b=40), title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(xv, y, label=title)
        ax.set_title(title)
        ax.legend()
        st.pyplot(fig, use_container_width=True)


# ============================================================
# UI 시작
# ============================================================
init_assessment_session()
student_id = require_student_id("학번을 입력하세요.")

st.title("(2차시) AI 모델식 도출 & 미분 기반 분석")
st.caption("AI가 제안한 모델식을 입력하고, 미분 관점에서 모델을 해석합니다.")
st.divider()

# ============================================================
# 0) 1차시 기록 불러오기/복구
# ============================================================
st.subheader("0) 1차시 기록 불러오기")

step1 = get_step1_summary() or {}
step2_prev = _get_step2_state()

colA, colB = st.columns([1.2, 1])
with colA:
    st.markdown("**1차시 TXT 업로드로 복구**")
    txt_file = st.file_uploader("1차시 백업 TXT 업로드", type=["txt"], key="step2_txt_upload")

with colB:
    st.markdown("**CSV 다시 업로드(그래프/도함수 계산용)**")
    csv_file = st.file_uploader("CSV 업로드", type=["csv"], key="step2_csv_upload")

if txt_file is not None:
    try:
        raw = txt_file.getvalue().decode("utf-8", errors="replace")
        parsed = parse_step1_backup_txt(raw)
        step1.update({
            "student_id": parsed.get("student_id") or step1.get("student_id") or student_id,
            "data_source": parsed.get("data_source") or step1.get("data_source",""),
            "x_col": parsed.get("x_col") or step1.get("x_col",""),
            "y_col": parsed.get("y_col") or step1.get("y_col",""),
            "x_mode": parsed.get("x_mode") or step1.get("x_mode",""),
            "valid_n": parsed.get("valid_n") or step1.get("valid_n",""),
            "features": parsed.get("features") or step1.get("features",""),
            "model_primary": parsed.get("model_primary") or step1.get("model_primary",""),
            "model_primary_reason": parsed.get("model_primary_reason") or step1.get("model_primary_reason",""),
        })
        st.success("TXT에서 1차시 정보를 불러왔습니다.")
    except Exception as e:
        st.error("TXT를 읽는 중 오류가 발생했습니다.")
        st.exception(e)

if csv_file is not None:
    try:
        df = read_csv_kosis(csv_file)
        set_df(df)
        st.success(f"CSV 업로드 완료 ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
    except Exception as e:
        st.error("CSV를 읽지 못했습니다.")
        st.exception(e)

df = get_df()

st.divider()

# ============================================================
# 1) 데이터 시각화 + 데이터 기반 변화율 자동 계산
# ============================================================
st.subheader("1) 데이터 기반 변화율 확인")

st.session_state["step2_ai_xv"] = None
st.session_state["step2_ai_t"] = None

if df is None:
    st.info("CSV를 업로드하면 2차시에서 변화율 그래프를 자동으로 확인할 수 있습니다.")
else:
    cols = list(df.columns)
    x_prev, y_prev = get_xy()
    x_init = step1.get("x_col") if step1.get("x_col") in cols else (x_prev if x_prev in cols else cols[0])
    y_init = step1.get("y_col") if step1.get("y_col") in cols else (y_prev if y_prev in cols else (cols[1] if len(cols) > 1 else cols[0]))

    x_col = st.selectbox("X축", cols, index=cols.index(x_init), key="step2_x_col")
    y_col = st.selectbox("Y축", cols, index=cols.index(y_init), key="step2_y_col")
    set_xy(x_col, y_col)

    x_mode = st.radio(
        "X축 해석 방식",
        ["자동(권장)", "날짜(년월)", "숫자"],
        horizontal=True,
        key="step2_x_mode",
    )

    y = pd.to_numeric(df[y_col], errors="coerce")

    if x_mode == "숫자":
        x = pd.to_numeric(df[x_col], errors="coerce")
        x_type = "numeric"
    else:
        x_dt = parse_year_month(df[x_col])
        if x_mode == "자동(권장)" and x_dt.notna().mean() < 0.6:
            x = pd.to_numeric(df[x_col], errors="coerce")
            x_type = "numeric"
        else:
            x = x_dt
            x_type = "datetime"

    valid = x.notna() & y.notna()
    xv = x[valid]
    yv = y[valid]

    if len(xv) < MIN_VALID_POINTS:
        st.warning("유효 데이터가 부족하여 변화율 계산이 어렵습니다. (최소 30점 이상 권장)")
    else:
        order = np.argsort(xv.values) if x_type == "datetime" else np.argsort(xv.to_numpy())
        xv = xv.iloc[order]
        yv = yv.iloc[order]

        if x_type == "datetime":
            base = xv.iloc[0]
            t = ((xv.dt.year - base.year) * 12 + (xv.dt.month - base.month)).to_numpy(dtype=float)
        else:
            t = xv.to_numpy(dtype=float)

        y_arr = yv.to_numpy(dtype=float)

        tmp = pd.DataFrame({"t": t, "y": y_arr})
        tmp = tmp.groupby("t", as_index=False).mean(numeric_only=True)
        t = tmp["t"].to_numpy(dtype=float)
        y_arr = tmp["y"].to_numpy(dtype=float)

        if len(t) < MIN_VALID_POINTS:
            st.warning("유효 데이터가 부족하여 변화율 계산이 어렵습니다. (최소 30점 이상 권장)")
        else:
            dy, d2y = compute_derivatives(t, y_arr)
            valid_n = int(len(t))
            st.metric("유효 데이터 점 개수", valid_n)

            if x_type == "datetime":
                base_dt = pd.to_datetime(xv.iloc[0])
                xv_plot = pd.to_datetime(base_dt) + pd.to_timedelta(t * 30, unit="D")
            else:
                xv_plot = t

            st.session_state["step2_ai_xv"] = xv_plot
            st.session_state["step2_ai_t"] = t
            st.session_state["step2_valid_n"] = valid_n

            if PLOTLY_AVAILABLE:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=xv_plot, y=y_arr, mode="lines+markers", name="y"))
                fig1.update_layout(height=320, margin=dict(l=40, r=20, t=20, b=40),
                                   xaxis_title=str(x_col), yaxis_title=str(y_col))
                st.plotly_chart(fig1, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=xv_plot, y=dy, mode="lines+markers", name="dy/dt"))
                fig2.update_layout(height=320, margin=dict(l=40, r=20, t=20, b=40),
                                   xaxis_title=str(x_col), yaxis_title="변화율(Δy/Δt)")
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=xv_plot, y=d2y, mode="lines+markers", name="d2y/dt2"))
                fig3.update_layout(height=320, margin=dict(l=40, r=20, t=20, b=40),
                                   xaxis_title=str(x_col), yaxis_title="이계변화율(Δ²y/Δt²)")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                fig, ax = plt.subplots()
                ax.plot(xv_plot, y_arr, marker="o")
                ax.set_title("원자료 y")
                st.pyplot(fig, use_container_width=True)

                fig, ax = plt.subplots()
                ax.plot(xv_plot, dy, marker="o")
                ax.set_title("변화율 Δy/Δt")
                st.pyplot(fig, use_container_width=True)

                fig, ax = plt.subplots()
                ax.plot(xv_plot, d2y, marker="o")
                ax.set_title("이계변화율 Δ²y/Δt²")
                st.pyplot(fig, use_container_width=True)

st.divider()


# ============================================================
# 2) AI 프롬프트 자동 생성 (통합 템플릿)
# ============================================================
st.subheader("2) AI로 모델식(y=f(t)) 제안 받기")

st.info(
    "1차시에서 세운 가설 모델과 그 근거를 바탕으로,\n"
    "AI에게 모델식을 제안받습니다.\n\n"
    "⚠ 수식은 반드시 LaTeX 형식으로 출력하도록 지시하세요."
)

model_hypothesis = step1.get("model_primary", "")
model_reason = step1.get("model_primary_reason", "")

st.markdown("### 🔹 1차시 가설 확인")
st.write(f"**가설 모델:** {model_hypothesis or '(기록 없음)'}")
st.write(f"**가설 근거:** {model_reason or '(기록 없음)'}")

additional_context = st.text_area(
    "추가 설명(선택) — 1차시 이후 새롭게 생각한 점이 있다면 작성",
    height=40,
)

def build_unified_prompt(model_hypothesis, model_reason, additional_context):
    return f"""
너는 수학 모델링 조교다. 첨부한 데이터 파일을 토대로 아래 조건에 따라 구체적인 함수 모델식을 제안하라.

[중요 조건]
- 수식은 반드시 LaTeX 형식으로 출력하라.
- 모든 수식은 $$ ... $$ 로 감싸라.
- 유니코드 위첨자(², ³ 등)는 사용하지 말고 ^{{ }} 형태를 사용하라.
- 보고서처럼 길게 쓰지 말고, 식과 핵심 해석 위주로 작성하라.

[데이터 설명]
- t는 시간 인덱스(월 단위 또는 순차 인덱스)이다.
- (t, y) 데이터를 참고하여 모델을 제안하라.

[내가 세운 가설 모델]
- 모델 유형: {model_hypothesis}
- 그렇게 생각한 이유: {model_reason}

[추가 설명]
{additional_context}

[반드시 포함할 출력 항목]
1) 최종 모델식: $$f(t)=...$$
2) 도함수: $$f'(t)=...$$
3) 이계도함수: $$f''(t)=...$$
4) 모델의 한계를 하나의 문단으로 작성하고, 가설 모델의 수정 여부를 판단하라.
   (최소 두 가지 한계를 포함하고, 번호나 목록 형태로 나열하지 말 것)
""".strip()

if st.button("📌 프롬프트 자동 생성", use_container_width=True):
    generated_prompt = build_unified_prompt(
        model_hypothesis,
        model_reason,
        additional_context,
    )
    st.session_state["step2_ai_prompt"] = generated_prompt

ai_prompt = st.text_area(
    "AI에 입력할 프롬프트(자동 생성 후 필요하면 수정)",
    value=st.session_state.get("step2_ai_prompt", ""),
    height=260,
)


st.divider()

# ============================================================
# 3) AI 출력 결과 입력(LaTeX) + 미리보기 + 그래프
# (좌/우 컬럼 제거, 요청 순서대로 출력)
# ============================================================
st.subheader("3) AI 출력 식 입력 — LaTeX 그대로 붙여넣기")

ai_model_latex = st.text_area(
    "AI가 제안한 모델식 f(t) (LaTeX 포함)",
    value=step2_prev.get("ai_model_latex", ""),
    height=120,
    placeholder="예: $$ y = 3.2 e^{0.04 t} $$",
)

ai_derivative_latex = st.text_area(
    "AI가 제안한 도함수 f'(t) (LaTeX 포함)",
    value=step2_prev.get("ai_derivative_latex", ""),
    height=120,
    placeholder="예: $$ f'(t) = 0.128 e^{0.04 t} $$",
)

ai_second_derivative_latex = st.text_area(
    "AI가 제안한 이계도함수 f''(t) (LaTeX 포함)",
    value=step2_prev.get("ai_second_derivative_latex", ""),
    height=120,
    placeholder="예: $$ f''(t) = 0.00512 e^{0.04 t} $$",
)

ai_limitations = st.text_area(
    "AI가 제시한 모델의 한계 (문장 그대로 붙여넣기)",
    value=step2_prev.get("ai_limitations", ""),
    height=120,
    placeholder="AI가 제시한 '모델의 한계' 내용을 그대로 붙여넣으세요.",
)

with st.expander("LaTeX 미리보기(깨짐 확인)", expanded=True):
    blocks = (
        extract_latex_blocks(ai_model_latex)
        + extract_latex_blocks(ai_derivative_latex)
        + extract_latex_blocks(ai_second_derivative_latex)
    )
    if not blocks:
        st.caption("LaTeX 형식을 올바르게 입력하면 수식이 정상적으로 출력됩니다.")
    else:
        for b in blocks[:10]:
            try:
                st.latex(b)
            except Exception:
                st.code(b)

xv_plot = st.session_state.get("step2_ai_xv")
t_plot = st.session_state.get("step2_ai_t")

f_fn = _latex_to_callable(ai_model_latex)
fp_fn = _latex_to_callable(ai_derivative_latex)
fpp_fn = _latex_to_callable(ai_second_derivative_latex)

if xv_plot is None or t_plot is None:
    st.info("AI 함수 그래프는 1)에서 유효 데이터 30점 이상이 확보되면 자동으로 그릴 수 있습니다.")
else:
    st.markdown("### y=f(t) 그래프")
    _plot_single_ai_function(xv_plot, t_plot, f_fn, "y=f(t)")

    st.markdown("### y=f'(t) 그래프")
    _plot_single_ai_function(xv_plot, t_plot, fp_fn, "y=f'(t)")

    st.markdown("### y=f''(t) 그래프")
    _plot_single_ai_function(xv_plot, t_plot, fpp_fn, "y=f''(t)")

st.divider()

st.subheader("가설 재평가")

st.info(
    "AI가 제안한 모델과 한계점을 살펴보고, "
    "여러분이 1차시에 세운 가설 모델이 적절한지 판단해 봅시다."
)

hypothesis_decision = st.radio(
    "가설 판단",
    ["가설 유지", "가설 수정"],
    horizontal=True,
    key="hypothesis_decision",
)

revised_model = ""
if hypothesis_decision == "가설 수정":
    revised_model = st.text_input(
        "수정한 모델 유형을 작성하세요",
        placeholder="예: 다항함수",
        key="revised_model",
    )
    st.warning(
        "수정된 모델을 기준으로 AI에게 다시 분석을 요청하고, **항목 3)을 재작성 하세요.**"
    )

revised_model_safe = revised_model.strip() if hypothesis_decision == "가설 수정" else ""

# ============================================================
# 4) 학생 검증/비판(핵심 제출물)
# ============================================================
st.subheader("4) 미분 관점의 모델 해석")

st.info(
    "추론한 모델 식과 실제 데이터 그래프 사이의 상관관계를 바탕으로, 다음 항목들을 분석해 봅시다.\n\n"
    "🔹 변화율 비교\n\n"
    "데이터의 변화율($\\Delta y/\\Delta t$)에서 특징 두 가지를 찾고, "
    "AI가 제시한 도함수 $f'(t)$가 이를 얼마나 잘 설명하는지 서술하시오.\n\n"
    "🔹 곡선의 모양 분석\n\n"
    "데이터의 이계변화율($\\Delta^2 y/\\Delta t^2$)에 나타난 오목·볼록 상태를 "
    "AI의 이계도함수 $f''(t)$와 비교하여 분석하시오.\n\n"
    "🔹 모델의 한계\n\n"
    "실제 데이터와 모델 식의 차이가 큰 구간을 한 곳 제시하고, "
    "모델링 과정에서 누락되었을 가능성이 있는 변수나 환경적 요인을 추론하여 서술해 봅시다."
)

student_critical_review = st.text_area(
    "분석 내용(필수)",
    value=step2_prev.get("student_analysis", ""),
    height=220,
    placeholder=(
        "수식은 반드시 LaTeX 형식($$ ... $$)으로 입력하세요."
    ),
)

note = st.text_area("추가 메모(선택)", value=step2_prev.get("note", ""), height=100)

st.divider()

# ============================================================
# 5) 저장(구글시트) + TXT 백업 다운로드
# ============================================================
st.subheader("5) 저장 및 백업")

data_source = (step1.get("data_source") or "").strip()
model_hypothesis_step1 = (step1.get("model_primary") or "").strip()

x_col_now = st.session_state.get("step2_x_col", step1.get("x_col",""))
y_col_now = st.session_state.get("step2_y_col", step1.get("y_col",""))

valid_n_now = None
try:
    valid_n_now = int(st.session_state.get("step2_valid_n", ""))
except Exception:
    pass

payload = {
    "student_id": student_id,
    "data_source": data_source,
    "x_col": x_col_now,
    "y_col": y_col_now,
    "valid_n": valid_n_now,
    "model_hypothesis_step1": model_hypothesis_step1,
    "hypothesis_decision": hypothesis_decision,
    "revised_model": revised_model_safe,
    "ai_prompt": ai_prompt,
    "ai_model_latex": ai_model_latex,
    "ai_derivative_latex": ai_derivative_latex,
    "ai_second_derivative_latex": ai_second_derivative_latex,
    "student_analysis": student_critical_review,
    "note": note,
}

backup_bytes = build_step2_backup(payload)
st.download_button(
    label="📄 (다운로드) 2차시 백업 TXT",
    data=backup_bytes,
    file_name=f"미적분_수행평가_2차시_{student_id}.txt",
    mime="text/plain; charset=utf-8",
)

colS, colN = st.columns([1, 1])
save_clicked = colS.button("💾 저장(구글시트)", use_container_width=True)
go_next = colN.button("➡️ 3차시로 이동(추후)", use_container_width=True)


def _validate_step2() -> bool:
    if hypothesis_decision == "가설 수정" and not revised_model_safe:
        st.warning("가설을 수정했다면, 수정한 모델 유형을 입력하세요.")
        return False

    if not ai_prompt.strip():
        st.warning("AI 프롬프트(원문)를 입력하세요.")
        return False

    if not ai_model_latex.strip():
        st.warning("AI 모델식(LaTeX)을 입력하세요.")
        return False

    if not student_critical_review.strip():
        st.warning("분석 내용을 입력하세요.")
        return False

    return True


if save_clicked or go_next:
    if not _validate_step2():
        st.stop()

    _set_step2_state(payload)

    try:
        append_step2_row(
            student_id=payload["student_id"],
            data_source=payload["data_source"],
            x_col=payload["x_col"],
            y_col=payload["y_col"],
            valid_n=payload["valid_n"],
            model_hypothesis_step1=payload["model_hypothesis_step1"],
            hypothesis_decision=payload["hypothesis_decision"],
            revised_model=payload["revised_model"],
            ai_prompt=payload["ai_prompt"],
            ai_model_latex=payload["ai_model_latex"],
            ai_derivative_latex=payload["ai_derivative_latex"],
            ai_second_derivative_latex=payload["ai_second_derivative_latex"],
            student_analysis=payload["student_analysis"],
            note=payload["note"],
        )
        st.success("✅ 저장 완료! (Google Sheet에 기록되었습니다)")

    except Exception as e:
        st.error("⚠️ Google Sheet 저장 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()

    if go_next:
        st.info("3차시는 아직 페이지를 만들기 전이라 이동은 나중에 연결하면 됩니다.")
        # st.switch_page("assessment/step3_integral.py")
