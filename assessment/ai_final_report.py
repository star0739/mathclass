
from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, Optional, Tuple

import streamlit as st
import matplotlib.pyplot as plt

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
SS_KEY = "ai_final_report_draft"


# ============================================================
# TXT 읽기 유틸
# ============================================================
def _strip_bom(s: str) -> str:
    if not s:
        return s
    return s.lstrip("\ufeff").lstrip("\ufffe")


def _read_uploaded_txt(file) -> str:
    if file is None:
        return ""
    raw = file.read()
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return _strip_bom(raw.decode(enc))
        except Exception:
            continue
    # 최후: replace
    return _strip_bom(raw.decode("utf-8", errors="replace"))


def _section_text(txt: str, header: str) -> str:
    """
    [header] 섹션 내용을 다음 [ ... ] 등장 전까지 추출.
    """
    if not txt:
        return ""
    # header 예: "학생 입력(서술)" -> r"\[학생 입력\(서술\)\]"
    h = re.escape(header)
    pattern = rf"\[{h}\]\s*(.*?)(?=\n\[[^\]]+\]\s*|\Z)"
    m = re.search(pattern, txt, flags=re.DOTALL)
    return (m.group(1).strip() if m else "").strip()


def _find_line_value(txt: str, label: str) -> str:
    """
    "학번: 30901" 같은 라인에서 값 추출.
    """
    if not txt:
        return ""
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+?)\s*$", txt, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "").strip()


def _parse_function_expr(txt: str) -> str:
    # "- E(a,b) = 10 a^2 + 1 b^2" 형태
    m = re.search(r"E\(a,b\)\s*=\s*(.+)$", txt, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "").strip()


def _parse_range(txt: str) -> Tuple[str, str]:
    # "- 관찰 범위: a∈[-3,3], b∈[-3,3]" 형태
    # 실패 시 빈 문자열 반환
    m = re.search(r"관찰 범위\s*:\s*(.+)$", txt, flags=re.MULTILINE)
    if not m:
        return "", ""
    s = m.group(1)
    # a..., b... 분리
    ma = re.search(r"a\s*[∈=]\s*([^\s,]+)", s)
    mb = re.search(r"b\s*[∈=]\s*([^\s,]+)", s)
    a_rng = ma.group(1).strip() if ma else ""
    b_rng = mb.group(1).strip() if mb else ""
    return a_rng, b_rng


def parse_ai_step1_backup_txt(txt: str) -> Dict[str, str]:
    txt = _strip_bom(txt)
    fn_expr = _parse_function_expr(txt)
    a_rng, b_rng = _parse_range(txt)
    sid = _find_line_value(txt, "학번")
    saved = _find_line_value(txt, "저장시각")

    narrative = _section_text(txt, "학생 입력(서술)")
    # 1) 2) 3) 블록을 대략 분리(없어도 통째로)
    def _q(n: int) -> str:
        m = re.search(
            rf"^{n}\)\s*(.*?)(?=^\d\)\s*|\Z)",
            narrative,
            flags=re.MULTILINE | re.DOTALL,
        )
        return (m.group(1).strip() if m else "").strip()

    return {
        "student_id": sid,
        "saved_at": saved,
        "function_expr": fn_expr,
        "a_range": a_rng,
        "b_range": b_rng,
        "q1": _q(1),
        "q2": _q(2),
        "q3": _q(3),
        "narrative_all": narrative.strip(),
    }


def parse_ai_step2_backup_txt(txt: str) -> Dict[str, str]:
    txt = _strip_bom(txt)
    sid = _find_line_value(txt, "학번")
    saved = _find_line_value(txt, "저장시각")

    fn_expr = _parse_function_expr(txt)
    a_rng, b_rng = _parse_range(txt)

    cond = _section_text(txt, "함수/조건")
    m_step = re.search(r"step_size\s*=\s*([0-9]*\.?[0-9]+)", cond)
    step_size = (m_step.group(1) if m_step else "").strip()

    result = _section_text(txt, "시작점/결과")
    # "- 시작점: (-2.2000, 2.0000)" 등
    start_pt = ""
    end_pt = ""
    steps = ""
    final_e = ""

    m = re.search(r"시작점\s*:\s*(\([^)]+\))", result)
    start_pt = (m.group(1).strip() if m else "").strip()
    m = re.search(r"최종점\s*:\s*(\([^)]+\))", result)
    end_pt = (m.group(1).strip() if m else "").strip()
    m = re.search(r"사용 step 수\s*:\s*([0-9]+)", result)
    steps = (m.group(1).strip() if m else "").strip()
    m = re.search(r"최종 손실 E\s*:\s*([0-9]*\.?[0-9]+)", result)
    final_e = (m.group(1).strip() if m else "").strip()

    narrative = _section_text(txt, "학생 입력(서술)")

    def _q(n: int) -> str:
        m = re.search(
            rf"^{n}\)\s*(.*?)(?=^\d\)\s*|\Z)",
            narrative,
            flags=re.MULTILINE | re.DOTALL,
        )
        return (m.group(1).strip() if m else "").strip()

    # 편미분은 1)에서 따로 추출 시도
    q1 = _q(1)
    dEa = ""
    dEb = ""
    m = re.search(r"∂E/∂a\s*=\s*([^\s]+)", q1)
    dEa = (m.group(1).strip() if m else "").strip()
    m = re.search(r"∂E/∂b\s*=\s*([^\s]+)", q1)
    dEb = (m.group(1).strip() if m else "").strip()

    return {
        "student_id": sid,
        "saved_at": saved,
        "function_expr": fn_expr,
        "a_range": a_rng,
        "b_range": b_rng,
        "step_size": step_size,
        "start_point": start_pt,
        "end_point": end_pt,
        "steps": steps,
        "final_E": final_e,
        "dE_da": dEa,
        "dE_db": dEb,
        "q1": q1,
        "q2": _q(2),
        "q3": _q(3),
        "narrative_all": narrative.strip(),
    }


# ============================================================
# LaTeX -> PNG (matplotlib mathtext)
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
# 한글 폰트 등록(assets 폴더) - 미적분 final_report.py와 동일
# ============================================================
def register_korean_fonts() -> Tuple[str, str]:
    """
    assets/ 폴더의 TTF를 ReportLab에 등록하고 (regular, bold) 폰트명을 반환.
    파일명은 필요 시 여기서만 수정하면 됨.
    """
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
# PDF 생성(Platypus)
# ============================================================
def _img_scale_to_width(img_bytes: bytes, target_w_mm: float) -> RLImage:
    bio = BytesIO(img_bytes)
    img = RLImage(bio)
    target_w = target_w_mm * mm
    scale = target_w / float(img.imageWidth)
    img.drawWidth = target_w
    img.drawHeight = float(img.imageHeight) * scale
    return img


def build_ai_report_pdf(
    report_title: str,
    student_id: str,
    intro_text: str,
    body_text: str,
    analysis_text: str,
    result_text: str,
    conclusion_text: str,
    fn_latex: str,
    d1_latex: str,
    d2_latex: str,
    fig1_bytes: bytes,
    fig2_bytes: bytes,
) -> bytes:
    regular_font, bold_font = register_korean_fonts()

    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "base",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    h_style = ParagraphStyle(
        "h",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=13,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )
    caption = ParagraphStyle(
        "cap",
        parent=styles["Normal"],
        fontName=regular_font,
        fontSize=9.5,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    buff = BytesIO()
    doc = SimpleDocTemplate(
        buff,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=report_title,
        author=student_id or "",
    )

    story = []
    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(f"학번: {student_id or '-'}", base))
    story.append(Spacer(1, 6))

    # 핵심 수식(함수 + 편미분) - 깨짐 방지용 PNG 렌더링
    story.append(Paragraph("핵심 수식", h_style))

    def _add_latex(latex: str, fontsize: int = 15):
        png = latex_to_png_bytes(latex, fontsize=fontsize)
        if not png:
            story.append(Paragraph(latex, base))
            return
        img = _img_scale_to_width(png, target_w_mm=155)
        story.append(img)
        story.append(Spacer(1, 4))

    if fn_latex:
        _add_latex(fn_latex, fontsize=16)
    if d1_latex:
        _add_latex(d1_latex, fontsize=14)
    if d2_latex:
        _add_latex(d2_latex, fontsize=14)

    # 본문(자연스럽게 이어지게: 섹션은 편집 단락 단위)
    story.append(Paragraph("서론", h_style))
    for para in (intro_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, base))

    story.append(Paragraph("본문", h_style))
    # 그림 1(지그재그)
    if fig1_bytes:
        story.append(_img_scale_to_width(fig1_bytes, target_w_mm=165))
        story.append(Paragraph("그림 1. 좌표축 이동으로 나타난 지그재그 경로(등고선 관찰)", caption))
    for para in (body_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, base))

    # 편미분 기반 분석 단락(소제목은 주되, 번호형 소문항 느낌은 피함)
    story.append(Paragraph("방향 판단과 분석", h_style))
    for para in (analysis_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, base))

    # 그림 2(이동/비교)
    if fig2_bytes:
        story.append(_img_scale_to_width(fig2_bytes, target_w_mm=165))
        story.append(Paragraph("그림 2. 이동 방향 비교 및 누적 이동 경로(최대손실/추천 방향 포함)", caption))

    story.append(Paragraph("결과 해석", h_style))
    for para in (result_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, base))

    story.append(Paragraph("결론", h_style))
    for para in (conclusion_text or "").split("\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, base))

    doc.build(story)
    return buff.getvalue()


# ============================================================
# 초안 템플릿(업로드 값 기반 자동 채움)
# ============================================================
def _normalize_expr_for_latex(expr: str) -> str:
    """
    "10 a^2 + 1 b^2" -> "10a^2 + b^2" 정도로 정리(가벼운 정규화).
    """
    s = (expr or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" 1 b^2", " b^2").replace("+ 1 b^2", "+ b^2").replace("1 b^2", "b^2")
    s = s.replace(" 1 a^2", " a^2").replace("+ 1 a^2", "+ a^2").replace("1 a^2", "a^2")
    s = s.replace(" a^2", "a^2").replace(" b^2", "b^2")
    return s


def build_default_draft(s1: Dict[str, str], s2: Dict[str, str]) -> Dict[str, str]:
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

    # LaTeX(깨짐 방지 PNG 렌더링용)
    fn_latex = ""
    if fn_expr:
        fn_latex = r"E(a,b) = " + _normalize_expr_for_latex(fn_expr)
    d1_latex = r"\frac{\partial E}{\partial a} = " + dE_da if dE_da else ""
    d2_latex = r"\frac{\partial E}{\partial b} = " + dE_db if dE_db else ""

    intro = (
        "본 보고서는 이변수 손실함수 $E(a,b)$의 등고선을 관찰하고, "
        "좌표축 방향 이동에서 나타나는 지그재그 경로의 특징을 분석한 뒤, "
        "편미분 값을 이용해 손실을 줄이는 이동 방향을 판단하고 그 결과를 검증하는 것을 목적으로 한다.\n"
        f"본 활동에서 사용한 함수는 $E(a,b)= { _normalize_expr_for_latex(fn_expr) }$ 이며, "
        f"관찰 범위는 a∈{a_rng}, b∈{b_rng}로 설정했다."
    ).strip()

    # 본문(관찰 → 문제의식 전환문 포함)
    body = (
        "등고선을 관찰한 결과, 전역 최소점은 원점 부근에서 나타나며 전체 지형은 원점을 향해 내려가는 그릇 모양으로 해석할 수 있었다. "
        "또한 등고선 간격을 보면 a방향이 더 촘촘하여 같은 거리 이동에서 손실 변화가 더 크게 나타난다고 판단했다.\n"
        "좌표축 방향으로만 번갈아 이동시키면 경로가 한 번에 최소점으로 향하지 못하고 꺾이는 형태가 반복되는데, "
        "이 현상은 축별 기울기의 차이와 이동 규칙의 제약이 합쳐져 나타난 결과로 볼 수 있다.\n"
        "이 관찰은 ‘어느 방향으로 움직여야 손실이 가장 빠르게 줄어드는가?’라는 질문으로 이어진다."
    ).strip()

    # 방향 판단(편미분 부호/크기)
    analysis = (
        f"시작점 {start_pt}에서 각 축 방향 변화율인 편미분을 확인했다. "
        f"계산 결과 $\\partial E/\\partial a = {dE_da}$, $\\partial E/\\partial b = {dE_db}$ 이므로, "
        "손실을 줄이기 위한 이동 방향은 ‘기울기(편미분)의 부호와 반대’로 결정할 수 있다.\n"
        "따라서 a는 (증가/감소) 방향, b는 (증가/감소) 방향으로 움직여야 한다고 판단했다. "
        "또한 두 편미분의 절댓값을 비교하면 ( ) 방향의 영향이 더 크므로 해당 성분을 상대적으로 더 크게 반영하는 것이 효율적일 수 있다고 예상했다."
    ).strip()

    # 결과 해석(정량 요약 고정)
    result = (
        f"step_size = {step_size}로 이동을 반복한 결과, {steps} step 후 최종점 {end_pt}에 도달했으며 "
        f"최종 손실은 $E \\approx {final_e}$ 이었다.\n"
        "그림 2에서 ‘나의 방향’과 ‘추천(또는 최대손실) 방향’을 비교하면, "
        "(두 벡터의 방향 차이/성분 차이/이동 효과 차이)와 같은 특징을 확인할 수 있다.\n"
        "또한 1 step 이동 후 손실이 감소했는지 확인함으로써, ‘부호 판단 → 이동 → 손실 감소’의 연결이 타당했는지 검증할 수 있었다."
    ).strip()

    conclusion = (
        "등고선의 간격과 방향은 손실함수의 민감도(변화율)와 직접 연결되며, 편미분은 각 축 방향에서 손실이 어떻게 변하는지 정량적으로 알려준다. "
        "따라서 편미분의 부호를 이용해 손실을 줄이는 방향을 결정하고, 그 결정이 실제로 손실 감소로 이어지는지 결과로 검증하는 과정이 중요하다.\n"
        "추가로, step_size를 바꾸거나(너무 크면 발산/진동, 너무 작으면 수렴이 느림), "
        "a와 b의 스케일 차이를 보정하는 방법을 적용하면 더 안정적인 수렴 경로를 설계할 수 있을 것이라고 생각한다."
    ).strip()

    # 학생 원문을 “참고 자료”로 따로 보여주고 싶으면 UI에서 expander로 노출
    return {
        "fn_latex": fn_latex,
        "d1_latex": d1_latex,
        "d2_latex": d2_latex,
        "intro": intro,
        "body": body,
        "analysis": analysis,
        "result": result,
        "conclusion": conclusion,
    }


# ============================================================
# Streamlit UI
# ============================================================
def _get_draft() -> Dict[str, str]:
    return st.session_state.get(SS_KEY, {})


def _set_draft(d: Dict[str, str]) -> None:
    st.session_state[SS_KEY] = d


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title("🧾 인공지능수학 수행평가 최종 보고서 (1~2차시)")

    # --------------------------------------------------------
    # 기본 입력
    # --------------------------------------------------------
    col1, col2 = st.columns([2, 1])
    with col1:
        report_title = st.text_input("보고서 제목(필수)", value="손실함수 등고선 관찰과 이동 방향 분석")
    with col2:
        student_id = st.text_input("학번(필수)", value="")

    st.markdown("---")

    # --------------------------------------------------------
    # 업로드 게이트
    # --------------------------------------------------------
    st.subheader("1) 백업 파일 및 이미지 업로드")

    cA, cB = st.columns(2)
    with cA:
        up_s1 = st.file_uploader("1차시 백업 TXT 업로드(필수)", type=["txt"], key="ai_up_s1")
        up_img1 = st.file_uploader("그림 1 업로드(1차시: 지그재그 관찰)(필수)", type=["png", "jpg", "jpeg"], key="ai_up_img1")
    with cB:
        up_s2 = st.file_uploader("2차시 백업 TXT 업로드(필수)", type=["txt"], key="ai_up_s2")
        up_img2 = st.file_uploader("그림 2 업로드(2차시: 이동/방향 비교)(필수)", type=["png", "jpg", "jpeg"], key="ai_up_img2")

    if not (report_title and student_id):
        st.info("보고서 제목과 학번을 먼저 입력하세요.")
        st.stop()

    if not (up_s1 and up_s2 and up_img1 and up_img2):
        st.info("1차시/2차시 TXT 2개와 이미지 2개를 모두 업로드하면 다음 단계가 활성화됩니다.")
        st.stop()

    txt1 = _read_uploaded_txt(up_s1)
    txt2 = _read_uploaded_txt(up_s2)
    s1 = parse_ai_step1_backup_txt(txt1)
    s2 = parse_ai_step2_backup_txt(txt2)

    # 학번 자동 채움(빈칸일 때만)
    if not student_id:
        sid = s2.get("student_id") or s1.get("student_id")
        if sid:
            student_id = sid
            st.session_state["student_id_auto"] = sid

    img1_bytes = up_img1.read()
    img2_bytes = up_img2.read()

    # --------------------------------------------------------
    # 업로드 내용 미리보기(선택)
    # --------------------------------------------------------
    with st.expander("업로드에서 추출된 값 미리보기(참고)", expanded=False):
        st.markdown("**함수/범위**")
        st.write(f"- E(a,b) = {s2.get('function_expr') or s1.get('function_expr')}")
        st.write(f"- a 범위: {s2.get('a_range') or s1.get('a_range')}, b 범위: {s2.get('b_range') or s1.get('b_range')}")
        st.markdown("**2차시 결과 요약**")
        st.write(f"- 시작점: {s2.get('start_point')}, 최종점: {s2.get('end_point')}")
        st.write(f"- step_size: {s2.get('step_size')}, steps: {s2.get('steps')}, 최종 E: {s2.get('final_E')}")
        st.markdown("**학생 서술(원문)**")
        st.text_area("1차시 서술(원문)", value=s1.get("narrative_all", ""), height=180)
        st.text_area("2차시 서술(원문)", value=s2.get("narrative_all", ""), height=180)

    st.markdown("---")

    # --------------------------------------------------------
    # 초안 자동 채우기
    # --------------------------------------------------------
    st.subheader("2) 초안 자동 채우기 및 문장 수정")
    left, right = st.columns([1, 1])
    with left:
        fill = st.button("🪄 초안 자동 채우기(덮어쓰기)", use_container_width=True)
    with right:
        if st.button("🔄 업로드 값으로 수치만 재동기화", use_container_width=True):
            prev = _get_draft() or {}
            auto = build_default_draft(s1, s2)
            # 문장(학생 수정)은 유지하고 수식/수치가 포함된 항목만 갱신
            keep_keys = ["intro", "body", "analysis", "result", "conclusion"]
            merged = {**auto}
            for k in keep_keys:
                if prev.get(k):
                    merged[k] = prev[k]
            _set_draft(merged)
            st.success("수치/수식 정보를 업로드 값으로 재동기화했습니다(문장은 유지).")

    if fill or not _get_draft():
        _set_draft(build_default_draft(s1, s2))
        if fill:
            st.success("업로드 자료를 바탕으로 초안을 채웠습니다. 아래에서 문장을 자연스럽게 다듬어 완성하세요.")

    d = _get_draft()

    # --------------------------------------------------------
    # 편집 영역(본문을 자연스럽게 이어지게: 단락 단위)
    # --------------------------------------------------------
    st.markdown("아래 문장들은 자동으로 채워진 초안입니다. 괄호 ( ) 부분을 채우고 문장을 자연스럽게 수정해 완성하세요.")

    d["intro"] = st.text_area("서론(활동 목적·핵심 질문)", value=d.get("intro", ""), height=140)
    d["body"] = st.text_area("본문(등고선 관찰 → 지그재그 → 문제의식까지 자연스럽게)", value=d.get("body", ""), height=200)
    d["analysis"] = st.text_area("방향 판단과 분석(편미분·부호·성분 해석)", value=d.get("analysis", ""), height=200)
    d["result"] = st.text_area("결과 해석(정량 요약 + 그림 2 해석)", value=d.get("result", ""), height=200)
    d["conclusion"] = st.text_area("결론(핵심 정리 + 개선/확장 아이디어)", value=d.get("conclusion", ""), height=160)

    with st.expander("PDF에 들어갈 수식(자동) 확인/수정", expanded=False):
        d["fn_latex"] = st.text_input("함수 수식(LaTeX)", value=d.get("fn_latex", ""))
        d["d1_latex"] = st.text_input("편미분 수식 1(LaTeX)", value=d.get("d1_latex", ""))
        d["d2_latex"] = st.text_input("편미분 수식 2(LaTeX)", value=d.get("d2_latex", ""))

    _set_draft(d)

    st.markdown("---")

    # --------------------------------------------------------
    # PDF 생성
    # --------------------------------------------------------
    st.subheader("3) PDF 생성")

    gen = st.button("📄 PDF 생성", use_container_width=True)
    if gen:
        try:
            pdf_bytes = build_ai_report_pdf(
                report_title=report_title,
                student_id=student_id,
                intro_text=d.get("intro", ""),
                body_text=d.get("body", ""),
                analysis_text=d.get("analysis", ""),
                result_text=d.get("result", ""),
                conclusion_text=d.get("conclusion", ""),
                fn_latex=d.get("fn_latex", ""),
                d1_latex=d.get("d1_latex", ""),
                d2_latex=d.get("d2_latex", ""),
                fig1_bytes=img1_bytes,
                fig2_bytes=img2_bytes,
            )
            st.success("PDF를 생성했습니다. 아래에서 다운로드하세요.")
            safe_title = re.sub(r"[^0-9A-Za-z가-힣 _-]+", "", report_title).strip() or "ai_report"
            filename = f"{safe_title}_{student_id}.pdf"
            st.download_button(
                "⬇️ PDF 다운로드",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
