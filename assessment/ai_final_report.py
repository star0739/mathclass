
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, Optional, Tuple, List

import streamlit as st
import matplotlib.pyplot as plt

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
)

# ============================================================
# 기본 설정
# ============================================================
PAGE_TITLE = "인공지능수학 수행평가 최종 보고서"


# ============================================================
# TXT 읽기/파싱 유틸 (final_report.py 스타일로 안정화)
# ============================================================
def _read_uploaded_txt(file) -> str:
    raw = file.getvalue()
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _strip_bom(s: str) -> str:
    return (s or "").lstrip("\ufeff").lstrip("\ufffe").strip("\n")


def _find_line_value(lines: List[str], prefix: str) -> str:
    for ln in lines:
        t = ln.strip()
        if t.startswith(prefix):
            return t.replace(prefix, "", 1).strip()
    return ""


def _section_text(lines: List[str], header: str, next_headers: List[str]) -> str:
    """lines에서 header(정확일치) 찾고 다음 헤더 전까지 반환"""
    header = header.strip()
    idx = None
    for i, ln in enumerate(lines):
        if ln.strip() == header:
            idx = i
            break
    if idx is None:
        return ""

    end = len(lines)
    for nh in next_headers:
        nh = nh.strip()
        for j in range(idx + 1, len(lines)):
            if lines[j].strip() == nh:
                end = min(end, j)
                break

    body = "\n".join([ln.rstrip() for ln in lines[idx + 1 : end]]).strip()
    return body


def _parse_function_expr(text: str) -> str:
    # "- E(a,b) = 10 a^2 + 1 b^2" 형태를 라인에서 추출
    m = re.search(r"E\(a,b\)\s*=\s*(.+)$", text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "").strip()


def _parse_range(text: str) -> Tuple[str, str]:
    """
    예: "- 관찰 범위: a∈[-3,3], b∈[-3,3]" 에서
        a_range="[-3,3]", b_range="[-3,3]" 를 정확히 추출
    """
    m = re.search(r"관찰 범위\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return "", ""
    s = m.group(1)

    # 괄호로 둘러싸인 구간 전체를 캡처: [ ... ] 또는 ( ... )
    ma = re.search(r"a\s*[∈=]\s*([\[\(].*?[\]\)])", s)
    mb = re.search(r"b\s*[∈=]\s*([\[\(].*?[\]\)])", s)

    a_rng = ma.group(1).strip() if ma else ""
    b_rng = mb.group(1).strip() if mb else ""
    return a_rng, b_rng


def parse_ai_step1_backup_txt(text: str) -> Dict[str, str]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    out: Dict[str, str] = {}
    out["student_id"] = _find_line_value(lines, "학번:")
    out["saved_at"] = _find_line_value(lines, "저장시각:")
    out["function_expr"] = _parse_function_expr(text)

    a_rng, b_rng = _parse_range(text)
    out["a_range"] = a_rng
    out["b_range"] = b_rng

    # 섹션명은 백업 포맷에 맞춤
    narrative = _section_text(lines, "[학생 입력(서술)]", [])
    out["narrative_all"] = narrative.strip()

    # 1)2)3) 대략 분리
    def _q(n: int) -> str:
        m = re.search(
            rf"^{n}\)\s*(.*?)(?=^\d\)\s*|\Z)",
            narrative,
            flags=re.MULTILINE | re.DOTALL,
        )
        return (m.group(1).strip() if m else "").strip()

    out["q1"] = _q(1)
    out["q2"] = _q(2)
    out["q3"] = _q(3)
    return out


def parse_ai_step2_backup_txt(text: str) -> Dict[str, str]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    out: Dict[str, str] = {}
    out["student_id"] = _find_line_value(lines, "학번:")
    out["saved_at"] = _find_line_value(lines, "저장시각:")
    out["function_expr"] = _parse_function_expr(text)

    a_rng, b_rng = _parse_range(text)
    out["a_range"] = a_rng
    out["b_range"] = b_rng

    cond = _section_text(lines, "[함수/조건]", ["[시작점/결과]", "[학생 입력(서술)]"])
    m_step = re.search(r"step_size\s*=\s*([0-9]*\.?[0-9]+)", cond)
    out["step_size"] = (m_step.group(1) if m_step else "").strip()

    result = _section_text(lines, "[시작점/결과]", ["[학생 입력(서술)]"])
    out["start_point"] = (re.search(r"시작점\s*:\s*(\([^)]+\))", result).group(1).strip()
                          if re.search(r"시작점\s*:\s*(\([^)]+\))", result) else "")
    out["end_point"] = (re.search(r"최종점\s*:\s*(\([^)]+\))", result).group(1).strip()
                        if re.search(r"최종점\s*:\s*(\([^)]+\))", result) else "")
    out["steps"] = (re.search(r"사용 step 수\s*:\s*([0-9]+)", result).group(1).strip()
                    if re.search(r"사용 step 수\s*:\s*([0-9]+)", result) else "")
    out["final_E"] = (re.search(r"최종 손실 E\s*:\s*([0-9]*\.?[0-9]+)", result).group(1).strip()
                      if re.search(r"최종 손실 E\s*:\s*([0-9]*\.?[0-9]+)", result) else "")

    narrative = _section_text(lines, "[학생 입력(서술)]", [])
    out["narrative_all"] = narrative.strip()

    def _q(n: int) -> str:
        m = re.search(
            rf"^{n}\)\s*(.*?)(?=^\d\)\s*|\Z)",
            narrative,
            flags=re.MULTILINE | re.DOTALL,
        )
        return (m.group(1).strip() if m else "").strip()

    out["q1"] = _q(1)
    out["q2"] = _q(2)
    out["q3"] = _q(3)

    # 편미분 값은 1)에서 추출 시도
    q1 = out["q1"]
    m = re.search(r"∂E/∂a\s*=\s*([^\s]+)", q1)
    out["dE_da"] = (m.group(1).strip() if m else "").strip()
    m = re.search(r"∂E/∂b\s*=\s*([^\s]+)", q1)
    out["dE_db"] = (m.group(1).strip() if m else "").strip()

    return out


# ============================================================
# LaTeX 렌더링(이미지) - matplotlib mathtext
# ============================================================
def latex_to_png_bytes(latex: str, fontsize: int = 16) -> Optional[bytes]:
    latex = (latex or "").strip()
    if not latex:
        return None

    s = latex
    if not (s.startswith("$") and s.endswith("$")):
        s = f"${s}$"

    try:
        fig = plt.figure(figsize=(8, 0.9))
        fig.patch.set_alpha(0.0)
        fig.text(0.01, 0.5, s, fontsize=fontsize, va="center")
        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        return bio.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


# ============================================================
# 한글 폰트 등록(assets 폴더)
# ============================================================
def register_korean_fonts() -> Tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.normpath(os.path.join(here, "..", "assets"))

    regular_path = os.path.join(font_dir, "NanumGothic-Regular.ttf")
    bold_path = os.path.join(font_dir, "NanumGothic-Bold.ttf")

    if not os.path.exists(regular_path):
        raise FileNotFoundError(f"한글 폰트 파일이 없습니다: {regular_path}")
    if not os.path.exists(bold_path):
        bold_path = regular_path

    regular_name = "NanumGothic-Regular"
    bold_name = "NanumGothic-Bold"

    if regular_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(regular_name, regular_path))
    if bold_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(bold_name, bold_path))

    return regular_name, bold_name


# ============================================================
# PDF 생성(Platypus) - final_report.py 스타일(제목/메타/큰항목)
# ============================================================
def _img_scale_to_width(img_bytes: bytes, target_w_mm: float) -> RLImage:
    bio = BytesIO(img_bytes)
    img = RLImage(bio)
    target_w = target_w_mm * mm
    scale = target_w / float(img.imageWidth)
    img.drawWidth = target_w
    img.drawHeight = float(img.imageHeight) * scale
    return img


def _normalize_expr_for_latex(expr: str) -> str:
    s = (expr or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" 1 b^2", " b^2").replace("+ 1 b^2", "+ b^2").replace("1 b^2", "b^2")
    s = s.replace(" 1 a^2", " a^2").replace("+ 1 a^2", "+ a^2").replace("1 a^2", "a^2")
    s = s.replace(" a^2", "a^2").replace(" b^2", "b^2")
    return s


def build_ai_report_pdf(
    *,
    report_title: str,
    student_id: str,
    student_name: str,
    sections: Dict[str, str],
    latex_items: Dict[str, str],
    images: Dict[str, Optional[bytes]],
) -> bytes:
    bio = BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=report_title.strip() or "탐구 보고서",
        author=student_id.strip(),
    )

    regular_font, bold_font = register_korean_fonts()
    styles = getSampleStyleSheet()

    body = ParagraphStyle(
        "BODY",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceBefore=0,
        spaceAfter=0,
    )

    title_style = ParagraphStyle(
        "TITLE",
        parent=styles["Heading1"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=0,
    )

    meta_style = ParagraphStyle(
        "META",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=10.5,
        leading=14,
        alignment=TA_RIGHT,
        textColor=colors.black,
        spaceBefore=0,
        spaceAfter=0,
    )

    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=13.5,
        leading=18,
        alignment=TA_JUSTIFY,
        spaceBefore=6,
        spaceAfter=4,
    )

    caption = ParagraphStyle(
        "CAPTION",
        parent=styles["BodyText"],
        fontName=regular_font,
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceBefore=2,
        spaceAfter=6,
    )

    story: List[object] = []

    # 상단 헤더
    story.append(Paragraph(report_title.strip(), title_style))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"{student_id.strip()}  {student_name.strip()}", meta_style))
    story.append(Spacer(1, 10 * mm))
    story.append(Spacer(1, 10 * mm))

    # 수식(필요시 표시)
    # final_report는 본문 중간에 수식/그림이 나오므로, AI도 "서론 앞"에 간단히 배치
    fn_tex = (latex_items.get("fn") or "").strip()
    d1_tex = (latex_items.get("d1") or "").strip()
    d2_tex = (latex_items.get("d2") or "").strip()

    def _add_tex(tex: str, fontsize: int):
        png = latex_to_png_bytes(tex, fontsize=fontsize)
        if png:
            story.append(RLImage(BytesIO(png), width=170 * mm, height=18 * mm, hAlign="CENTER"))
            story.append(Spacer(1, 4 * mm))
        else:
            story.append(Paragraph(tex, body))
            story.append(Spacer(1, 4 * mm))

    if fn_tex or d1_tex or d2_tex:
        # 과도한 섹션 제목은 피하고, 간단히 수식만 배치
        if fn_tex:
            _add_tex(fn_tex, fontsize=18)
        if d1_tex:
            _add_tex(d1_tex, fontsize=16)
        if d2_tex:
            _add_tex(d2_tex, fontsize=16)

        story.append(Spacer(1, 6 * mm))

    # 1. 서론
    story.append(Paragraph("1. 서론", h1))
    if sections.get("intro", "").strip():
        story.append(Paragraph(sections["intro"].replace("\n", "<br/>"), body))
    story.append(Spacer(1, 10 * mm))
    story.append(Spacer(1, 10 * mm))

    # 2. 본론 (소문항 없이 자연스럽게 연결)
    story.append(Paragraph("2. 본론", h1))

    # 그림 1
    if images.get("fig1"):
        story.append(Paragraph("그림 1. 지그재그 관찰(1차시)", caption))
        story.append(_img_scale_to_width(images["fig1"], target_w_mm=170))
        story.append(Spacer(1, 8 * mm))

    # 본론 텍스트(1)
    if sections.get("body_main", "").strip():
        story.append(Paragraph(sections["body_main"].replace("\n", "<br/>"), body))
        story.append(Spacer(1, 6 * mm))

    # 그림 2
    if images.get("fig2"):
        story.append(Paragraph("그림 2. 이동 방향 비교/이동 결과(2차시)", caption))
        story.append(_img_scale_to_width(images["fig2"], target_w_mm=170))
        story.append(Spacer(1, 8 * mm))

    # 본론 텍스트(2) 결과/해석까지 연결
    if sections.get("body_result", "").strip():
        story.append(Paragraph(sections["body_result"].replace("\n", "<br/>"), body))

    story.append(Spacer(1, 10 * mm))
    story.append(Spacer(1, 10 * mm))

    # 3. 결론
    story.append(Paragraph("3. 결론", h1))
    if sections.get("conclusion", "").strip():
        story.append(Paragraph(sections["conclusion"].replace("\n", "<br/>"), body))

    doc.build(story)
    return bio.getvalue()


# ============================================================
# 초안 생성(업로드 값 기반) - final_report.py 스타일(세션키)
# ============================================================
K_INTRO = "ai_sec_intro"
K_BODY_MAIN = "ai_sec_body_main"
K_BODY_RESULT = "ai_sec_body_result"
K_CONC = "ai_sec_conclusion"

def _maybe_init_drafts(s1: Dict[str, str], s2: Dict[str, str]) -> Dict[str, str]:
    fn_expr = s2.get("function_expr") or s1.get("function_expr") or ""
    a_rng = s2.get("a_range") or s1.get("a_range") or ""
    b_rng = s2.get("b_range") or s1.get("b_range") or ""

    step_size = s2.get("step_size", "")
    start_pt = s2.get("start_point", "")
    end_pt = s2.get("end_point", "")
    steps = s2.get("steps", "")
    final_e = s2.get("final_E", "")

    dE_da = s2.get("dE_da", "")
    dE_db = s2.get("dE_db", "")

    fn_norm = _normalize_expr_for_latex(fn_expr)

    if K_INTRO not in st.session_state:
        st.session_state[K_INTRO] = (
            "본 보고서는 이변수 손실함수 $E(a,b)$의 등고선을 관찰하고, "
            "좌표축 방향 이동에서 나타나는 경로의 특징을 분석한 뒤, "
            "편미분 값을 이용해 손실을 줄이는 이동 방향을 판단하고 그 결과를 검증하는 것을 목적으로 한다.\n\n"
            f"본 활동에서 사용한 함수는 $E(a,b)={fn_norm}$ 이며, 관찰 범위는 a∈{a_rng}, b∈{b_rng}로 설정했다."
        ).strip()

    if K_BODY_MAIN not in st.session_state:
        s1_hint = (s1.get("q3") or s1.get("narrative_all") or "").strip()
        extra = f"\n\n(구조 관찰 서술)\n{s1_hint}" if s1_hint else ""
        st.session_state[K_BODY_MAIN] = (
            "등고선을 관찰한 결과 전역 최소점은 원점 부근에서 나타나며, 지형은 원점을 향해 내려가는 형태로 해석할 수 있었다. "
            "또한 등고선 간격을 보면 a방향의 변화가 더 민감하게 나타난다고 판단했다.\n\n"
            "좌표축 방향으로만 번갈아 이동시키면 경로가 한 번에 최소점으로 향하지 못하고 꺾이는 형태가 반복되는데, "
            "이 현상은 ( )와 같은 이유로 설명할 수 있다.\n\n"
            "이 관찰은 ‘어느 방향으로 움직여야 손실이 더 빠르게 줄어드는가?’라는 질문으로 이어진다."
            + extra
        ).strip()

    if K_BODY_RESULT not in st.session_state:
        s2_hint = (s2.get("q2") or s2.get("q3") or s2.get("narrative_all") or "").strip()
        extra = f"\n\n(경로 탐색 서술)\n{s2_hint}" if s2_hint else ""
        st.session_state[K_BODY_RESULT] = (
            f"시작점 {start_pt}에서 편미분을 계산하면 ∂E/∂a={dE_da}, "
            f"∂E/∂b={dE_db} 이므로, 손실을 줄이기 위한 이동 방향은 '기울기(편미분)의 부호와 반대'로 결정할 수 있다.\n\n"
            "따라서 a는 (증가/감소) 방향, b는 (증가/감소) 방향으로 움직여야 한다고 판단했다.\n\n"
            f"step_size={step_size}로 이동을 반복한 결과 {steps} step 후 최종점 {end_pt}에 도달했고, "
            f"최종 손실은 $E\\approx {final_e}$ 이었다. 그림 2를 근거로 이동 방향의 타당성과 한계를 서술한다."
            + extra
        ).strip()

    if K_CONC not in st.session_state:
        st.session_state[K_CONC] = (
            "등고선의 간격과 방향은 손실함수의 민감도(변화율)와 연결되며, 편미분은 각 축 방향에서 손실이 어떻게 변하는지를 정량적으로 알려준다.\n\n"
            "결론에서는 다음을 포함하여 정리한다.\n"
            "• 관찰(그림 1)에서 얻은 핵심 통찰\n"
            "• 편미분 기반 방향 판단과 결과(그림 2)의 연결\n"
            "• step_size/스케일 보정 등 개선 아이디어"
        ).strip()

    # LaTeX 아이템도 함께 반환(UI에서 expander로 확인/수정)
    latex_items = {
        "fn": (r"E(a,b) = " + fn_norm) if fn_norm else "",
        "d1": (r"\frac{\partial E}{\partial a} = " + dE_da) if dE_da else "",
        "d2": (r"\frac{\partial E}{\partial b} = " + dE_db) if dE_db else "",
    }
    return latex_items


# ============================================================
# Streamlit UI (final_report.py 스타일)
# ============================================================
st.title("최종: 보고서 작성 & PDF 생성")
st.caption("1~2차시 TXT 백업 + 그래프 이미지를 업로드하고, 서술형으로 편집한 뒤 PDF로 저장합니다.")
st.divider()

# 0) 기본 정보 입력
st.subheader("0) 기본 정보 입력")
col0, col1, col2 = st.columns([2, 1, 1])
with col0:
    report_title = st.text_input(
        "탐구 보고서 제목(필수)",
        value="",
        placeholder="예: 손실함수 등고선 관찰과 경사하강 기반 이동 분석",
    )
with col1:
    student_id_input = st.text_input("학번(필수)", value="", placeholder="예: 30901")
with col2:
    student_name = st.text_input("이름(필수)", value="", placeholder="예: 홍길동")

st.divider()

# 1) 자료 업로드
st.subheader("1) 자료 업로드")
colA, colB = st.columns([1, 1])

with colA:
    step1_txt_f = st.file_uploader("1차시 백업 TXT(필수)", type=["txt"], key="ai_final_step1")
    step2_txt_f = st.file_uploader("2차시 백업 TXT(필수)", type=["txt"], key="ai_final_step2")

with colB:
    st.markdown("**그래프 이미지 업로드(필수)**")
    img1 = st.file_uploader("지그재그 관찰(1차시)", type=["png", "jpg", "jpeg"], key="ai_img1")
    img2 = st.file_uploader("이동/방향 비교(2차시)", type=["png", "jpg", "jpeg"], key="ai_img2")

# 업로드/입력 게이트(final_report.py처럼 missing 리스트)
missing = []
if not report_title.strip():
    missing.append("제목")
if not student_id_input.strip():
    missing.append("학번")
if not student_name.strip():
    missing.append("이름")
if step1_txt_f is None or step2_txt_f is None:
    missing.append("TXT(1~2차시)")
if img1 is None or img2 is None:
    missing.append("그래프 이미지 2종")

if missing:
    st.info(f"입력/업로드가 필요합니다: {', '.join(missing)}")
    st.stop()

# 2) TXT 파싱
try:
    t1_raw = _read_uploaded_txt(step1_txt_f)
    t2_raw = _read_uploaded_txt(step2_txt_f)
    s1 = parse_ai_step1_backup_txt(t1_raw)
    s2 = parse_ai_step2_backup_txt(t2_raw)
except Exception as e:
    st.error("TXT를 읽거나 파싱하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# 학번 자동 채움(입력값이 비었을 때만) - final_report 패턴
if not student_id_input.strip() and (s1.get("student_id") or s2.get("student_id")):
    student_id_input = (s2.get("student_id") or s1.get("student_id") or "").strip()

# 3) 초안 생성 + 편집(세션키)
st.divider()
st.subheader("2) 보고서 본문 작성(서술형 편집)")

colx, coly = st.columns([1, 1])

with colx:
    if st.button("🧩 초안 자동 생성(세션에 없을 때만)", use_container_width=True):
        latex_items = _maybe_init_drafts(s1, s2)
        st.session_state["ai_latex_items"] = latex_items
        st.success("초안이 준비되었습니다. 아래에서 서술형으로 수정하세요.")

with coly:
    if st.button("🧹 초안 다시 만들기(덮어쓰기)", use_container_width=True):
        for k in [K_INTRO, K_BODY_MAIN, K_BODY_RESULT, K_CONC, "ai_latex_items"]:
            if k in st.session_state:
                del st.session_state[k]
        latex_items = _maybe_init_drafts(s1, s2)
        st.session_state["ai_latex_items"] = latex_items
        st.success("초안을 다시 생성했습니다.")

# 기본 1회 초기화
latex_items = st.session_state.get("ai_latex_items") or _maybe_init_drafts(s1, s2)
st.session_state["ai_latex_items"] = latex_items

st.markdown("### 1. 서론")
sec_intro = st.text_area("본문(서술형)", key=K_INTRO, height=220)

st.markdown("### 2. 본론")
sec_body_main = st.text_area("본문(서술형) — 관찰(그림 1)에서 판단으로 이어지도록 작성", key=K_BODY_MAIN, height=260)
sec_body_result = st.text_area("본문(서술형) — 편미분 기반 이동 및 결과(그림 2) 해석까지 연결", key=K_BODY_RESULT, height=260)

st.markdown("### 3. 결론")
sec_conclusion = st.text_area("본문(서술형)", key=K_CONC, height=240)

with st.expander("LaTeX(자동 생성) 확인/수정", expanded=False):
    latex_items["fn"] = st.text_input("함수(LaTeX)", value=latex_items.get("fn", ""))
    latex_items["d1"] = st.text_input("편미분 1(LaTeX)", value=latex_items.get("d1", ""))
    latex_items["d2"] = st.text_input("편미분 2(LaTeX)", value=latex_items.get("d2", ""))
    st.session_state["ai_latex_items"] = latex_items

st.divider()

# 4) PDF 생성/다운로드
st.subheader("3) PDF 저장")

def _validate() -> bool:
    if not report_title.strip():
        st.warning("제목을 입력하세요.")
        return False
    if not student_id_input.strip() or not student_name.strip():
        st.warning("학번과 이름을 입력하세요.")
        return False
    if not sec_conclusion.strip():
        st.warning("결론을 작성하세요.")
        return False
    return True


if st.button("📄 PDF 생성", use_container_width=True):
    if not _validate():
        st.stop()

    sections = {
        "intro": sec_intro.strip(),
        "body_main": sec_body_main.strip(),
        "body_result": sec_body_result.strip(),
        "conclusion": sec_conclusion.strip(),
    }

    images = {
        "fig1": img1.getvalue(),
        "fig2": img2.getvalue(),
    }

    try:
        pdf_bytes = build_ai_report_pdf(
            report_title=report_title,
            student_id=student_id_input,
            student_name=student_name,
            sections=sections,
            latex_items=latex_items,
            images=images,
        )

        fname = f"인공지능_수행평가_최종보고서_{student_id_input.strip()}.pdf"
        st.download_button(
            "⬇️ 최종 보고서 PDF 다운로드",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )
        st.success("PDF가 생성되었습니다. 다운로드 버튼을 눌러 저장하세요.")
    except Exception as e:
        st.error("PDF 생성 중 오류가 발생했습니다.")
        st.exception(e)
