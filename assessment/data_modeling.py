# ------------------------------------------------------------
# 공공데이터 함수 모델링 & 미분 기반 분석
# - CSV 업로드
# - X/Y 데이터 시각화
# - 함수식 f(t), 도함수 f'(t), 이계도함수 f''(t) 입력
# - 원데이터, 변화율, 곡률 비교 그래프 출력
# ------------------------------------------------------------

import ast
import streamlit as st
import pandas as pd
import numpy as np

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except Exception:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt


MIN_VALID_POINTS = 30


# ------------------------------------------------------------
# CSV 로더
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 년월 파싱
# 예: 2015.01, 2015-01, 2015/01, 201501
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 데이터 전처리
# - x_col, y_col을 선택한 뒤 유효값만 추출
# - 날짜형 X축이면 t를 시작 시점으로부터의 월 단위 인덱스로 변환
# - 숫자형 X축이면 t를 해당 숫자값으로 사용
# ------------------------------------------------------------
def prepare_xy_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_mode: str,
):
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

    if len(xv) >= 2:
        if x_type == "datetime":
            order = np.argsort(xv.values)
        else:
            order = np.argsort(xv.to_numpy())

        xv = xv.iloc[order]
        yv = yv.iloc[order]

    y_arr = yv.to_numpy(dtype=float)

    if x_type == "datetime":
        base = xv.iloc[0]
        t = ((xv.dt.year - base.year) * 12 + (xv.dt.month - base.month)).to_numpy(dtype=float)
        t_description = "t는 첫 번째 날짜를 0으로 두고 계산한 월 단위 시간 인덱스입니다."
    else:
        t = xv.to_numpy(dtype=float)
        t_description = "t는 선택한 X축 숫자값입니다."

    return xv, yv, t, y_arr, x_type, t_description


# ------------------------------------------------------------
# 데이터 기반 근사 도함수
# ------------------------------------------------------------
def compute_derivatives(t: np.ndarray, y: np.ndarray):
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return dy, d2y


# ------------------------------------------------------------
# 수식 안전 계산
# - 학생이 입력한 파이썬 수식을 t 배열에 대해 계산
# - np.exp, np.sin, np.cos, np.log 등 사용 가능
# ------------------------------------------------------------
ALLOWED_NAMES = {
    "t",
    "np",
    "exp",
    "sin",
    "cos",
    "tan",
    "log",
    "sqrt",
    "abs",
    "pi",
    "e",
}

ALLOWED_NP_ATTRS = {
    "exp",
    "sin",
    "cos",
    "tan",
    "log",
    "log10",
    "sqrt",
    "abs",
    "power",
    "pi",
    "e",
}


def validate_expression(expr: str) -> None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"수식 문법 오류: {e}")

    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Attribute,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Tuple,
        ast.List,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_node_types):
            raise ValueError("허용되지 않은 표현이 포함되어 있습니다.")

        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES:
                raise ValueError(f"허용되지 않은 이름입니다: {node.id}")

        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "np":
                raise ValueError("np.함수 형태만 사용할 수 있습니다.")
            if node.attr not in ALLOWED_NP_ATTRS:
                raise ValueError(f"허용되지 않은 numpy 함수입니다: np.{node.attr}")


def eval_model_expression(expr: str, t: np.ndarray) -> np.ndarray:
    expr = expr.strip()
    if not expr:
        raise ValueError("수식이 비어 있습니다.")

    validate_expression(expr)

    eval_env = {
        "__builtins__": {},
        "np": np,
        "t": t,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }

    result = eval(expr, eval_env)

    if np.isscalar(result):
        result = np.full_like(t, float(result), dtype=float)
    else:
        result = np.asarray(result, dtype=float)

    if result.shape != t.shape:
        raise ValueError(
            f"계산 결과의 길이가 데이터 길이와 다릅니다. "
            f"결과 shape={result.shape}, t shape={t.shape}"
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("계산 결과에 NaN 또는 무한대 값이 포함되어 있습니다.")

    return result


# ------------------------------------------------------------
# 그래프 함수
# ------------------------------------------------------------
def plot_raw_data(xv, y_arr, x_label: str, y_label: str):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=y_arr,
                mode="lines+markers",
                name="Data",
            )
        )
        fig.update_layout(
            height=520,
            title="원자료 그래프",
            xaxis_title=str(x_label),
            yaxis_title=str(y_label),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.plot(xv, y_arr, marker="o")
        ax.set_title("원자료 그래프")
        ax.set_xlabel(str(x_label))
        ax.set_ylabel(str(y_label))
        st.pyplot(fig, use_container_width=True)


def plot_model_compare(xv, y_arr, ai_y):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=y_arr,
                mode="markers",
                name="실제 데이터",
                marker=dict(color="gray", opacity=0.55),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=ai_y,
                mode="lines",
                name="AI 모델식",
                line=dict(color="red", width=2),
            )
        )
        fig.update_layout(
            height=360,
            title="원데이터 vs AI 모델 비교",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.scatter(xv, y_arr, label="실제 데이터")
        ax.plot(xv, ai_y, label="AI 모델식")
        ax.set_title("원데이터 vs AI 모델 비교")
        ax.legend()
        st.pyplot(fig, use_container_width=True)


def plot_derivative_compare(xv, dy, ai_dy):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=dy,
                mode="markers",
                name="데이터 변화율",
                marker=dict(color="gray", opacity=0.55),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=ai_dy,
                mode="lines",
                name="AI 도함수",
                line=dict(color="blue", width=2),
            )
        )
        fig.update_layout(
            height=360,
            title="변화율 비교 분석",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.scatter(xv, dy, label="데이터 변화율")
        ax.plot(xv, ai_dy, label="AI 도함수")
        ax.set_title("변화율 비교 분석")
        ax.legend()
        st.pyplot(fig, use_container_width=True)


def plot_second_derivative_compare(xv, d2y, ai_d2y):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=d2y,
                mode="markers",
                name="데이터 이계변화율",
                marker=dict(color="gray", opacity=0.55),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=ai_d2y,
                mode="lines",
                name="AI 이계도함수",
                line=dict(color="green", width=2),
            )
        )
        fig.update_layout(
            height=360,
            title="곡률(오목·볼록) 비교 분석",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots()
        ax.scatter(xv, d2y, label="데이터 이계변화율")
        ax.plot(xv, ai_d2y, label="AI 이계도함수")
        ax.set_title("곡률(오목·볼록) 비교 분석")
        ax.legend()
        st.pyplot(fig, use_container_width=True)


# ============================================================
# UI 시작
# ============================================================
st.set_page_config(page_title="함수 모델링 & 미분 기반 분석", layout="wide")

st.title("함수 모델링 & 미분 기반 분석")
st.caption("공공데이터를 업로드한 뒤, 함수식과 도함수·이계도함수를 입력하여 데이터와 비교합니다.")
st.divider()


# ============================================================
# 1) 공공데이터 선택
# ============================================================
st.subheader("1) 공공데이터 선택")
st.link_button("📊 KOSIS에서 데이터 다운로드", "https://kosis.kr")

st.markdown(
    """
- 공공데이터포털(data.go.kr), 서울 열린데이터 광장(data.seoul.go.kr) 등 다른 사이트도 가능
- **연도/월 등 시간에 따른 변화**를 분석할 수 있는 데이터를 선택하세요.
- 데이터는 **숫자 데이터**여야 합니다. (예: 인구 수, 비율, 농도, 금액 등)
- 다운로드 파일은 **CSV(UTF-8 권장)**
- 너무 짧은 데이터는 비선형 모델 비교가 어렵습니다. **유효 데이터 점 30개 이상 130개 이하 권장**
"""
)


# ============================================================
# 2) 업로드 전 전처리
# ============================================================
st.subheader("2) 업로드 전 전처리")

with st.expander("파일 규칙(권장)", expanded=True):
    st.markdown(
        """
- 파일 형식: **CSV(UTF-8 권장)**
- 첫 행: **열 이름(헤더)**
- **불필요한 행/열(주석, 합계, 공백 행 등)** 삭제
- X축, Y축으로 사용할 **2개의 열**이 포함되어 있어야 함
- X축이 **년월**인 경우 `2015.01`처럼 표기하면 됩니다.
"""
    )

st.divider()


# ============================================================
# 3) CSV 업로드
# ============================================================
st.subheader("3) CSV 업로드")
uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"], key="main_csv_upload")

if uploaded is None:
    st.info("CSV를 업로드하면 다음 단계로 진행할 수 있습니다.")
    st.stop()

try:
    df = read_csv_kosis(uploaded)
    st.success(f"업로드 완료: {uploaded.name}  ({df.shape[0]:,}행 × {df.shape[1]:,}열)")
except Exception as e:
    st.error("CSV 파일을 읽지 못했습니다. (구분자/형식 문제 가능)")
    st.exception(e)
    st.stop()

st.markdown("#### 참고: 데이터 미리보기")
st.dataframe(df.head(30), use_container_width=True)


# ============================================================
# 4) 시각화 (X/Y 선택)
# ============================================================
st.divider()
st.subheader("4) 시각화 (X/Y 선택)")

cols = list(df.columns)

if len(cols) < 2:
    st.error("열이 2개 이상이어야 합니다. CSV를 다시 확인하세요.")
    st.stop()

col_x, col_y = st.columns(2)

with col_x:
    x_col = st.selectbox(
        "X축(시간/연도/년월)",
        cols,
        index=0,
        key="x_col_select",
    )

with col_y:
    y_default_idx = 1 if len(cols) > 1 else 0
    if cols[y_default_idx] == x_col:
        y_default_idx = 0

    y_col = st.selectbox(
        "Y축(수치)",
        cols,
        index=y_default_idx,
        key="y_col_select",
    )

x_mode = st.radio(
    "X축 해석 방식",
    ["자동(권장)", "날짜(년월)", "숫자"],
    horizontal=True,
    key="x_mode_radio",
)

try:
    xv, yv, t, y_arr, x_type, t_description = prepare_xy_data(
        df=df,
        x_col=x_col,
        y_col=y_col,
        x_mode=x_mode,
    )
except Exception as e:
    st.error("X/Y 데이터를 처리하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

if len(t) < 2:
    st.warning("유효한 데이터가 부족하여 그래프를 그릴 수 없습니다. (X/Y 열 값 확인)")
    st.stop()

plot_raw_data(xv, y_arr, x_col, y_col)

st.caption(f"※ {t_description}")


# ============================================================
# 데이터 개수 점검
# ============================================================
st.divider()
st.subheader("✅ 데이터 개수 점검")

valid_n = int(len(t))
quality_ok = valid_n >= MIN_VALID_POINTS

st.metric("유효 데이터 점(숫자 쌍) 개수", valid_n)

if not quality_ok:
    st.warning(
        f"유효 데이터 점이 {MIN_VALID_POINTS}개 미만입니다. "
        "그래프 비교는 가능하지만, 변화율과 이계변화율 해석은 불안정할 수 있습니다."
    )

st.caption("※ 변화율 비교는 유효 데이터 점 30개 이상일 때 더 안정적으로 해석할 수 있습니다.")


# ============================================================
# 5) 파이썬 수식 입력
# ============================================================
st.divider()
st.subheader("5) 파이썬 수식 입력")

st.info(
    "아래 수식에서 변수는 반드시 **t**를 사용하세요. "
    "날짜형 자료의 경우 t는 첫 번째 날짜를 0으로 둔 월 단위 시간 인덱스입니다."
)

with st.expander("입력 예시 보기", expanded=False):
    st.markdown(
        """
예를 들어 다음과 같이 입력할 수 있습니다.

- 선형함수: `2.5 * t + 10`
- 이차함수: `0.03 * t**2 - 1.2 * t + 50`
- 지수함수: `3.2 * np.exp(0.04 * t)`
- 로그함수: `15 * np.log(t + 1) + 20`
- 삼각함수: `10 * np.sin(2 * np.pi * t / 12) + 50`

도함수와 이계도함수도 직접 계산하여 입력합니다.

예를 들어  
`f(t) = 0.03 * t**2 - 1.2 * t + 50` 이라면,

- `f'(t) = 0.06 * t - 1.2`
- `f''(t) = 0.06`
"""
    )

col1, col2 = st.columns(2)

with col1:
    st.markdown("**파이썬 수식 (그래프 시뮬레이션용)**")
    py_model = st.text_input(
        "모델식 f(t) 식",
        value=st.session_state.get("py_model", ""),
        placeholder="예: 3.2 * np.exp(0.04 * t)",
        key="py_model",
    )

with col2:
    st.markdown("**도함수와 이계도함수**")
    py_d1 = st.text_input(
        "도함수 f'(t) 식",
        value=st.session_state.get("py_d1", ""),
        placeholder="예: 0.128 * np.exp(0.04 * t)",
        key="py_d1",
    )
    py_d2 = st.text_input(
        "이계도함수 f''(t) 식",
        value=st.session_state.get("py_d2", ""),
        placeholder="예: 0.00512 * np.exp(0.04 * t)",
        key="py_d2",
    )


# ============================================================
# 6) 데이터 기반 변화율 및 AI 모델 비교
# ============================================================
st.divider()
st.subheader("6) 데이터 기반 변화율 및 AI 모델 비교")

if not py_model.strip():
    st.info("모델식 f(t)를 입력하면 그래프 비교가 표시됩니다.")
    st.stop()

dy, d2y = compute_derivatives(t, y_arr)

ai_y = None
ai_dy = None
ai_d2y = None

model_error = False

try:
    ai_y = eval_model_expression(py_model, t)
except Exception as e:
    model_error = True
    st.error(f"모델식 f(t) 계산 오류: {e}")

if py_d1.strip():
    try:
        ai_dy = eval_model_expression(py_d1, t)
    except Exception as e:
        model_error = True
        st.error(f"도함수 f'(t) 계산 오류: {e}")
else:
    st.warning("도함수 f'(t)를 입력하면 변화율 비교 분석 그래프가 표시됩니다.")

if py_d2.strip():
    try:
        ai_d2y = eval_model_expression(py_d2, t)
    except Exception as e:
        model_error = True
        st.error(f"이계도함수 f''(t) 계산 오류: {e}")
else:
    st.warning("이계도함수 f''(t)를 입력하면 곡률(오목·볼록) 비교 분석 그래프가 표시됩니다.")

if model_error:
    st.stop()


# 원데이터 vs 모델 비교
if ai_y is not None:
    plot_model_compare(xv, y_arr, ai_y)


# 변화율 비교 분석
if ai_dy is not None:
    plot_derivative_compare(xv, dy, ai_dy)


# 곡률 비교 분석
if ai_d2y is not None:
    plot_second_derivative_compare(xv, d2y, ai_d2y)


# ============================================================
# 참고 데이터 표
# ============================================================
st.divider()
st.subheader("참고: 계산에 사용된 t 값과 데이터")

result_df = pd.DataFrame(
    {
        "X축 값": xv.astype(str).to_numpy(),
        "t": t,
        "실제 데이터 y": y_arr,
        "데이터 변화율 Δy/Δt": dy,
        "데이터 이계변화율 Δ²y/Δt²": d2y,
    }
)

if ai_y is not None:
    result_df["모델식 f(t)"] = ai_y

if ai_dy is not None:
    result_df["도함수 f'(t)"] = ai_dy

if ai_d2y is not None:
    result_df["이계도함수 f''(t)"] = ai_d2y

st.dataframe(result_df, use_container_width=True)
