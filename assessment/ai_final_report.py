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
# TXT 읽기/파싱 유틸
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
    """lines에서 header(정확일치)를 찾고 다음 헤더 전까지 반환"""
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


def _split_backup_records(text: str) -> List[str]:
    """
    하나의 TXT 안에 여러 학생 백업이 이어 붙여진 경우를 대비해 기록 단위로 분리한다.
    각 기록은 '인공지능수학 수행평가'로 시작한다고 가정한다.
    """
    text = _strip_bom(text)
    if not text:
        return []

    pattern = r"(?=^인공지능수학\s+수행평가.*?백업\s*$)"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    records = [p.strip() for p in parts if p.strip()]
    return records or [text]


def _select_record_by_student_id(text: str, student_id: str = "") -> str:
    """
    여러 명의 백업이 들어 있는 파일에서 학번이 일치하는 기록만 선택한다.
    학번이 없거나 일치 기록이 없으면 첫 번째 기록을 사용한다.
    """
    records = _split_backup_records(text)
    sid = (student_id or "").strip()

    if sid:
        for rec in records:
            if re.search(rf"^학번:\s*{re.escape(sid)}\s*$", rec, flags=re.MULTILINE):
                return rec

    return records[0] if records else _strip_bom(text)


def _extract_student_narrative(text: str) -> str:
    """
    백업 TXT에서 [학생 입력(서술)] 뒤의 학생 작성 내용만 추출한다.
    이후 다른 [섹션]이나 다음 학생 백업 기록이 나오면 그 전까지만 가져온다.
    """
    text = _strip_bom(text)
    if not text:
        return ""

    m = re.search(
        r"\[학생 입력\(서술\)\]\s*(.*?)(?=\n\s*\[[^\]]+\]\s*|\n인공지능수학\s+수행평가.*?백업\s*$|\Z)",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )

    return (m.group(1).strip() if m else "").strip()


def _parse_function_expr(text: str) -> str:
    # '- E(a,b)=...' 또는 '- E(a,b) = ...' 형태 추출
    m = re.search(r"E\(a,b\)\s*=\s*(.+)$", text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "").strip()


def _parse_range(text: str) -> Tuple[str, str]:
    """
    예: '- 관찰 범위: a∈[-3,3], b∈[-3,3]'에서
        a_range='[-3,3]', b_range='[-3,3]' 추출
    """
    m = re.search(r"관찰 범위\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return "", ""
    s = m.group(1)

    ma = re.search(r"a\s*[∈=]\s*([\[\(].*?[\]\)])", s)
    mb = re.search(r"b\s*[∈=]\s*([\[\(].*?[\]\)])", s)

    a_rng = ma.group(1).strip() if ma else ""
    b_rng = mb.group(1).strip() if mb else ""
    return a_rng, b_rng


def _split_numbered_answers(narrative: str) -> Dict[str, str]:
    """[학생 입력(서술)] 안의 1), 2), 3) 답변을 분리한다."""
    out: Dict[str, str] = {}

    def _q(n: int) -> str:
        m = re.search(
            rf"^\s*{n}\)\s*(.*?)(?=^\s*\d\)\s*|\Z)",
            narrative,
            flags=re.MULTILINE | re.DOTALL,
        )
        return (m.group(1).strip() if m else "").strip()

    out["q1"] = _q(1)
    out["q2"] = _q(2)
    out["q3"] = _q(3)
    return out


def parse_ai_step1_backup_txt(text: str, student_id_hint: str = "") -> Dict[str, str]:
    text = _strip_bom(text)
    record = _select_record_by_student_id(text, student_id_hint)
    lines = [ln.rstrip("\n") for ln in record.splitlines()]

    out: Dict[str, str] = {}
    out["student_id"] = _find_line_value(lines, "학번:")
    out["saved_at"] = _find_line_value(lines, "저장시각:")
    out["function_expr"] = _parse_function_expr(record)

    a_rng, b_rng = _parse_range(record)
    out["a_range"] = a_rng
    out["b_range"] = b_rng

    narrative = _extract_student_narrative(record)
    out["narrative_all"] = narrative
    out.update(_split_numbered_answers(narrative))

    return out


def parse_ai_step2_backup_txt(text: str, student_id_hint: str = "") -> Dict[str, str]:
    text = _strip_bom(text)
    record = _select_record_by_student_id(text, student_id_hint)
    lines = [ln.rstrip("\n") for ln in record.splitlines()]

    out: Dict[str, str] = {}
    out["student_id"] = _find_line_value(lines, "학번:")
    out["saved_at"] = _find_line_value(lines, "저장시각:")
    out["function_expr"] = _parse_function_expr(record)

    a_rng, b_rng = _parse_range(record)
    out["a_range"] = a_rng
    out["b_range"] = b_rng

    cond = _section_text(lines, "[함수/조건]", ["[시작점/결과]", "[학생 입력(서술)]"])
    if not cond:
        cond = _section_text(lines, "[함수 설정]", ["[시작점/결과]", "[학생 입력(서술)]"])

    m_step = re.search(r"step_size\s*=\s*([0-9]*\.?[0-9]+)", cond)
    out["step_size"] = (m_step.group(1) if m_step else "").strip()

    result = _section_text(lines, "[시작점/결과]", ["[학생 입력(서술)]"])

    m = re.search(r"시작점\s*:\s*(\([^)]+\))", result)
    out["start_point"] = (m.group(1).strip() if m else "")

    m = re.search(r"최종점\s*:\s*(\([^)]+\))", result)
    out["end_point"] = (m.group(1).strip() if m else "")

    m = re.search(r"사용\s*step\s*수\s*:\s*([0-9]+)", result)
    out["steps"] = (m.group(1).strip() if m else "")

    m = re.search(r"최종\s*손실\s*E\s*:\s*([0-9]*\.?[0-9]+)", result)
    out["final_E"] = (m.group(1).strip() if m else "")

    narrative = _extract_student_narrative(record)
    out["narrative_all"] = narrative
    out.update(_split_numbered_answers(narrative))

    # 편미분 값은 학생 입력 전체에서 추출 시도
    m = re.search(r"∂E/∂a\s*=\s*([^\s,，]+)", narrative)
    out["dE_da"] = (m.group(1).strip() if m else "")

    m = re.search(r"∂E/∂b\s*=\s*([^\s,，]+)", narrative)
    out["dE_db"] = (m.group(1).strip() if m else "")

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
# PDF 생성
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
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\b1(?=[ab]\^)", "", s)
    s = re.sub(r"\^(\d+)", r"^{\1}", s)
    return s


def _safe_paragraph_text(s: str) -> str:
    """ReportLab Paragraph용 간단 이스케이프/줄바꿈 처리"""
    s = (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("\n", "<br/>")


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

    story.append(Paragraph(_safe_paragraph_text(report_title.strip()), title_style))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(_safe_paragraph_text(f"{student_id.strip()}  {student_name.strip()}"), meta_style))
    story.append(Spacer(1, 10 * mm))

    fn_tex = sections.get("function_expr", "").strip()
    if fn_tex:
        latex_str = r"E(a,b) = " + fn_tex
        png = latex_to_png_bytes(latex_str, fontsize=12)
        story.append(Spacer(1, 4 * mm))
        if png:
            img = _img_scale_to_width(png, target_w_mm=90)
            img.hAlign = "CENTER"
            story.append(img)
        else:
            story.append(Paragraph(_safe_paragraph_text(latex_str), body))

    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("1. 서론", h1))
    if sections.get("intro", "").strip():
        story.append(Paragraph(_safe_paragraph_text(sections["intro"]), body))
    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("2. 본론", h1))

    if images.get("fig1"):
        story.append(Paragraph("그림 1. 등고선 및 좌표축 방향 이동 관찰", caption))
        story.append(_img_scale_to_width(images["fig1"], target_w_mm=170))
        story.append(Spacer(1, 8 * mm))

    if sections.get("body_main", "").strip():
        story.append(Paragraph(_safe_paragraph_text(sections["body_main"]), body))
        story.append(Spacer(1, 6 * mm))

    if images.get("fig2"):
        story.append(Paragraph("그림 2. 편미분 기반 이동 방향 및 결과", caption))
        story.append(_img_scale_to_width(images["fig2"], target_w_mm=170))
        story.append(Spacer(1, 8 * mm))

    if sections.get("body_result", "").strip():
        story.append(Paragraph(_safe_paragraph_text(sections["body_result"]), body))

    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph("3. 결론", h1))
    if sections.get("conclusion", "").strip():
        story.append(Paragraph(_safe_paragraph_text(sections["conclusion"]), body))

    doc.build(story)
    return bio.getvalue()


# ============================================================
# 초안 생성(업로드 값 기반) - 학생 작성 유도형
# ============================================================
K_INTRO = "ai_sec_intro"
K_BODY_MAIN = "ai_sec_body_main"
K_BODY_RESULT = "ai_sec_body_result"
K_CONC = "ai_sec_conclusion"


def _format_backup_insert(title: str, text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return f"""
[{title} 백업 자료에서 가져온 학생 입력]
{text}
""".strip()


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

    s1_full_text = (s1.get("narrative_all") or "").strip()
    s2_full_text = (s2.get("narrative_all") or "").strip()

    s1_insert = _format_backup_insert("1차시", s1_full_text)
    s2_insert = _format_backup_insert("2차시", s2_full_text)

    if K_INTRO not in st.session_state:
        st.session_state[K_INTRO] = f"""
본 보고서는 지난 활동에서 저장한 백업 자료를 바탕으로 손실함수 $E(a,b)$의 등고선, 이동 경로, 편미분 기반 이동 방향을 정리하는 것이다.

활동에서 사용한 함수는 $E(a,b)={fn_norm}$ 이며, 관찰 범위는 a∈{a_rng}, b∈{b_rng}로 설정하였다.

이 보고서에서는 먼저 1차시에서 관찰한 손실 지형과 좌표축 방향 이동의 특징을 정리하고, 이어서 2차시에서 계산한 편미분 값과 이동 결과를 바탕으로 손실을 줄이는 방향을 해석한다.

아래 문장은 자동으로 생성된 초안이므로, 백업 자료의 내용과 그림을 확인하면서 자신의 표현으로 수정한다.
""".strip()

    if K_BODY_MAIN not in st.session_state:
        st.session_state[K_BODY_MAIN] = f"""
그림 1과 1차시 백업 자료를 바탕으로 손실함수의 전체 형태, 최소점의 위치, 민감도가 큰 방향, 좌표축 방향 이동 경로의 특징을 정리한다.

{s1_insert}

위 기록을 그대로 옮기는 데 그치지 말고, 다음 질문에 답하듯 문장을 다듬어 보고서 본문으로 완성한다.

1. 등고선이나 3D 손실 지형에서 손실값이 작아지는 위치는 어디로 보이는가?
2. a방향과 b방향 중 어느 방향에서 손실값 변화가 더 크게 나타나는가?
3. 그 근거를 등고선 간격, 곡면의 가파름, 함수식의 구조와 연결해 설명할 수 있는가?
4. 좌표축 방향 이동 경로가 직선이 아니라 꺾이거나 지그재그 형태로 나타난 이유는 무엇인가?
5. step_size가 너무 크거나 작을 때 경로가 어떻게 달라질 수 있는가?

이 내용을 바탕으로 1차시 활동에서 관찰한 핵심 특징을 하나의 자연스러운 문단으로 정리한다.
""".strip()

    if K_BODY_RESULT not in st.session_state:
        st.session_state[K_BODY_RESULT] = f"""
2차시에서는 시작점 {start_pt}에서 손실을 줄이기 위한 이동 방향을 편미분 값으로 판단하였다.

시작점에서 계산한 편미분 값은 다음과 같다.
∂E/∂a = {dE_da}
∂E/∂b = {dE_db}

편미분 값은 각 변수 방향으로 움직였을 때 손실값이 증가하거나 감소하는 경향을 알려준다. 따라서 손실을 줄이기 위해서는 편미분 값의 부호와 크기를 확인하고, 각 변수의 이동 방향을 판단해야 한다.

이후 step_size={step_size}로 이동을 반복한 결과, {steps} step 후 최종점 {end_pt}에 도달하였다. 이때 최종 손실은 $E\\approx {final_e}$ 이었다.

{s2_insert}

위 기록을 바탕으로 다음 질문에 답하듯 결과를 해석한다.

1. ∂E/∂a의 부호를 기준으로 a는 증가 방향과 감소 방향 중 어느 쪽으로 움직여야 하는가?
2. ∂E/∂b의 부호를 기준으로 b는 증가 방향과 감소 방향 중 어느 쪽으로 움직여야 하는가?
3. 실제 이동 경로는 손실이 작아지는 방향으로 진행되었는가?
4. 최종점과 최종 손실값을 보면 이동 결과는 적절했다고 볼 수 있는가?
5. step_size를 조절한다면 진동, 발산, 수렴 속도 중 어떤 점이 개선될 수 있는가?

이 내용을 바탕으로 편미분 기반 이동 방향 판단과 실제 이동 결과를 연결하여 서술한다.
""".strip()

    if K_CONC not in st.session_state:
        st.session_state[K_CONC] = """
이번 활동을 통해 등고선의 모양과 간격, 좌표축 방향 이동 경로, 편미분 값이 서로 어떻게 연결되는지 정리할 수 있었다.

결론에서는 다음 내용을 자신의 말로 정리한다.

1. 등고선을 관찰하면서 알게 된 손실함수의 특징
2. 좌표축 방향 이동 경로가 지그재그 또는 꺾인 형태로 나타난 이유
3. 편미분 값을 이용해 손실을 줄이는 방향을 판단한 과정
4. 실제 이동 결과와 최종 손실값에 대한 해석
5. step_size를 조절하면 경로가 어떻게 개선될 수 있는지에 대한 생각

마지막 문단에서는 손실을 줄이는 과정에서 이동 방향뿐 아니라 한 번에 얼마나 이동할지를 정하는 step_size도 중요하다는 점을 정리한다.
""".strip()

    latex_items = {
        "fn": (r"E(a,b) = " + fn_norm) if fn_norm else "",
        "d1": (r"\frac{\partial E}{\partial a} = " + dE_da) if dE_da else "",
        "d2": (r"\frac{\partial E}{\partial b} = " + dE_db) if dE_db else "",
    }

    return latex_items


# ============================================================
# Streamlit UI
# ============================================================
st.title("최종: 인공지능 수학 보고서 작성 & PDF 생성")
st.caption("1~2차시 TXT 백업 + 그래프 이미지를 업로드하고, 자동으로 불러온 학생 입력을 바탕으로 보고서를 완성합니다.")
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
    img1 = st.file_uploader("지그재그 관찰/등고선 이미지(1차시)", type=["png", "jpg", "jpeg"], key="ai_img1")
    img2 = st.file_uploader("편미분 기반 이동/방향 비교 이미지(2차시)", type=["png", "jpg", "jpeg"], key="ai_img2")

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
    s1 = parse_ai_step1_backup_txt(t1_raw, student_id_hint=student_id_input)
    s2 = parse_ai_step2_backup_txt(t2_raw, student_id_hint=student_id_input)
except Exception as e:
    st.error("TXT를 읽거나 파싱하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# 학번 확인 안내
sid1 = (s1.get("student_id") or "").strip()
sid2 = (s2.get("student_id") or "").strip()
if sid1 and sid1 != student_id_input.strip():
    st.warning(f"1차시 백업에서 선택된 학번({sid1})이 입력 학번({student_id_input.strip()})과 다릅니다. 파일 내용을 확인하세요.")
if sid2 and sid2 != student_id_input.strip():
    st.warning(f"2차시 백업에서 선택된 학번({sid2})이 입력 학번({student_id_input.strip()})과 다릅니다. 파일 내용을 확인하세요.")

with st.expander("백업 자료에서 불러온 학생 입력 확인", expanded=False):
    st.markdown("**1차시 학생 입력**")
    st.text_area("1차시 [학생 입력(서술)]", value=s1.get("narrative_all", ""), height=220, disabled=True)
    st.markdown("**2차시 학생 입력**")
    st.text_area("2차시 [학생 입력(서술)]", value=s2.get("narrative_all", ""), height=220, disabled=True)

# 3) 초안 생성 + 편집
st.divider()
st.subheader("2) 보고서 본문 작성(학생 작성 유도형 초안)")

colx, coly = st.columns([1, 1])

with colx:
    if st.button("🧩 초안 자동 생성(세션에 없을 때만)", use_container_width=True):
        latex_items = _maybe_init_drafts(s1, s2)
        st.session_state["ai_latex_items"] = latex_items
        st.success("초안이 준비되었습니다. 아래에서 자신의 말로 수정하세요.")

with coly:
    if st.button("🧹 초안 다시 만들기(덮어쓰기)", use_container_width=True):
        for k in [K_INTRO, K_BODY_MAIN, K_BODY_RESULT, K_CONC, "ai_latex_items"]:
            if k in st.session_state:
                del st.session_state[k]
        latex_items = _maybe_init_drafts(s1, s2)
        st.session_state["ai_latex_items"] = latex_items
        st.success("초안을 다시 생성했습니다.")

latex_items = st.session_state.get("ai_latex_items") or _maybe_init_drafts(s1, s2)
st.session_state["ai_latex_items"] = latex_items

st.markdown("### 1. 서론")
sec_intro = st.text_area("본문(서술형)", key=K_INTRO, height=220)

st.markdown("### 2. 본론")
sec_body_main = st.text_area("본문(서술형) — 1차시 관찰 자료를 바탕으로 완성", key=K_BODY_MAIN, height=360)
sec_body_result = st.text_area("본문(서술형) — 2차시 편미분/이동 결과를 바탕으로 완성", key=K_BODY_RESULT, height=380)

st.markdown("### 3. 결론")
sec_conclusion = st.text_area("본문(서술형)", key=K_CONC, height=260)

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
    if not sec_intro.strip() or not sec_body_main.strip() or not sec_body_result.strip() or not sec_conclusion.strip():
        st.warning("서론, 본론, 결론을 모두 작성하세요.")
        return False
    return True


if st.button("📄 PDF 생성", use_container_width=True):
    if not _validate():
        st.stop()

    sections = {
        "function_expr": _normalize_expr_for_latex(
            s2.get("function_expr") or s1.get("function_expr") or "",
        ),
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
