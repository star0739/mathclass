
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
    """
    - 공백 정리
    - 1a^2, 1b^2 → a^2, b^2
    - LaTeX용 ^{ } 형태로 변환
    """
    s = (expr or "").strip()

    # 공백 정리
    s = re.sub(r"\s+", "", s)

    # 1a^2, 1b^2 같은 계수 1 제거
    s = re.sub(r"\b1(?=[ab]\^)", "", s)

    # ^2 → ^{2} 로 변환 (LaTeX 안정성)
    s = re.sub(r"\^(\d+)", r"^{\1}", s)

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
            story.append(Paragraph(latex_str, body))

    story.append(Spacer(1, 8 * mm))

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
    """
    학생이 업로드한 1~2차시 백업 자료를 바탕으로 보고서 초안을 만든다.
    이 초안은 교사가 완성문을 대신 써주는 형태가 아니라,
    학생이 자신의 관찰과 결과를 확인하고 직접 수정·보완할 수 있도록 안내하는 작성 유도형 문장으로 구성한다.
    """
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

    # 학생 백업 서술: 보고서에 그대로 정답처럼 넣기보다,
    # 학생이 다시 읽고 정리할 수 있도록 "나의 기록" 형태로 안내한다.
    s1_observation = (
        s1.get("q3")
        or s1.get("q2")
        or s1.get("q1")
        or s1.get("narrative_all")
        or ""
    ).strip()

    s2_observation = (
        s2.get("q2")
        or s2.get("q3")
        or s2.get("q1")
        or s2.get("narrative_all")
        or ""
    ).strip()

    # 값이 비어 있을 때 문장이 어색해지는 것을 줄이기 위한 표시용 문자열
    fn_display = fn_norm if fn_norm else "[함수식을 확인하여 입력]"
    a_display = a_rng if a_rng else "[a의 관찰 범위]"
    b_display = b_rng if b_rng else "[b의 관찰 범위]"
    start_display = start_pt if start_pt else "[시작점]"
    end_display = end_pt if end_pt else "[최종점]"
    step_display = step_size if step_size else "[step_size]"
    steps_display = steps if steps else "[반복 횟수]"
    final_e_display = final_e if final_e else "[최종 손실값]"
    dE_da_display = dE_da if dE_da else "[∂E/∂a 값]"
    dE_db_display = dE_db if dE_db else "[∂E/∂b 값]"

    # --------------------------------------------------------
    # 1. 서론
    # --------------------------------------------------------
    if K_INTRO not in st.session_state:
        st.session_state[K_INTRO] = f"""
본 보고서는 내가 수행한 손실함수 탐구 활동을 정리하기 위한 것이다.
활동에서는 손실함수의 등고선, 좌표축 방향 이동 경로, 편미분을 이용한 이동 방향 판단을 차례로 살펴보았다.

내가 사용한 함수는 $E(a,b)={fn_display}$ 이며, 관찰 범위는 a∈{a_display}, b∈{b_display}로 설정하였다.

이 보고서에서는 먼저 등고선 그림을 통해 손실값이 작아지는 위치와 경로의 특징을 관찰한다.
그 다음 시작점에서의 편미분 값을 이용하여 손실을 줄이기 위한 이동 방향을 판단하고,
실제 이동 결과가 그 판단과 어떻게 연결되는지 정리한다.
마지막으로 step_size의 크기가 이동 경로와 손실 감소에 어떤 영향을 주었는지 생각해본다.
""".strip()

    # --------------------------------------------------------
    # 2. 본론 - 그림 1 관찰 및 좌표축 방향 이동 해석
    # --------------------------------------------------------
    if K_BODY_MAIN not in st.session_state:
        student_note = ""
        if s1_observation:
            student_note = f"""

[내가 백업한 1차시 관찰 기록]
{s1_observation}

위 기록을 그대로 두기보다, 그림 1을 다시 보면서 관찰한 내용과 그 이유가 자연스럽게 이어지도록 문장을 다듬는다.
""".strip()

        st.session_state[K_BODY_MAIN] = f"""
그림 1에서는 손실함수의 등고선과 좌표축 방향으로 이동한 경로를 확인할 수 있다.
등고선의 모양을 보면 손실값이 작아지는 위치가 어디인지 추측할 수 있고,
등고선의 간격을 비교하면 a방향과 b방향 중 어느 방향에서 손실값이 더 빠르게 변하는지도 판단할 수 있다.

좌표축 방향 이동은 한 번에 a 또는 b 중 한 방향만 바꾸는 방식이다.
따라서 이동 경로가 최소점으로 곧바로 향하기보다는 방향이 반복적으로 꺾이는 지그재그 형태로 나타날 수 있다.

이 부분에서는 다음 질문에 답하듯이 자신의 관찰을 보고서 문장으로 정리한다.

1. 그림 1에서 손실값이 작아지는 위치는 어디라고 볼 수 있는가?
2. 등고선의 간격을 보았을 때 a방향과 b방향 중 어느 쪽 변화가 더 크게 나타나는가?
3. 좌표축 방향 이동 경로가 지그재그 형태로 나타난 이유는 무엇인가?
4. step_size가 너무 크거나 작을 때 경로는 어떻게 달라졌는가?

{student_note}
""".strip()

    # --------------------------------------------------------
    # 3. 본론 - 편미분 기반 이동 방향 판단 및 결과 해석
    # --------------------------------------------------------
    if K_BODY_RESULT not in st.session_state:
        student_note = ""
        if s2_observation:
            student_note = f"""

[내가 백업한 2차시 경로 탐색 기록]
{s2_observation}

위 기록을 바탕으로, 편미분 값으로 판단한 이동 방향과 실제 이동 결과가 어떻게 연결되는지 설명한다.
""".strip()

        st.session_state[K_BODY_RESULT] = f"""
이제 시작점 {start_display}에서 손실을 줄이기 위한 이동 방향을 판단한다.
편미분 값은 각 축 방향으로 움직였을 때 손실값이 어떻게 변하는지를 알려준다.

시작점에서 계산한 편미분 값은 다음과 같다.

∂E/∂a = {dE_da_display}
∂E/∂b = {dE_db_display}

편미분 값이 양수이면 해당 변수의 값이 증가할 때 손실이 증가한다는 뜻이고,
편미분 값이 음수이면 해당 변수의 값이 증가할 때 손실이 감소한다는 뜻이다.
따라서 손실을 줄이기 위해서는 편미분 값의 부호를 확인한 뒤,
a와 b를 각각 증가시킬지 감소시킬지 판단해야 한다.

이 판단을 바탕으로 step_size={step_display}로 이동을 반복한 결과,
{steps_display} step 후 최종점 {end_display}에 도달하였다.
이때 최종 손실은 $E\approx {final_e_display}$ 이었다.

이 부분에서는 다음 질문에 답하듯이 자신의 결과를 해석한다.

1. ∂E/∂a의 부호를 보면 a는 증가 방향과 감소 방향 중 어디로 움직여야 하는가?
2. ∂E/∂b의 부호를 보면 b는 증가 방향과 감소 방향 중 어디로 움직여야 하는가?
3. 그림 2의 실제 이동 경로는 손실이 작아지는 방향으로 진행되었는가?
4. 최종점과 최종 손실값을 볼 때 이동 결과는 적절했다고 볼 수 있는가?
5. step_size를 조절한다면 경로가 어떻게 개선될 수 있을까?

{student_note}
""".strip()

    # --------------------------------------------------------
    # 4. 결론
    # --------------------------------------------------------
    if K_CONC not in st.session_state:
        st.session_state[K_CONC] = """
이번 활동을 통해 등고선, 좌표축 방향 이동, 편미분 기반 이동 사이의 관계를 정리할 수 있었다.

결론에서는 다음 내용을 자신의 말로 정리한다.

1. 등고선을 관찰하면서 알게 된 손실함수의 특징
2. 좌표축 방향 이동 경로가 지그재그 형태로 나타난 이유
3. 편미분 값을 이용해 이동 방향을 판단한 과정
4. 실제 이동 결과와 최종 손실값에 대한 해석
5. step_size를 조절하면 경로가 어떻게 달라질 수 있는지에 대한 생각

이를 바탕으로 손실을 줄이는 과정에서는 이동 방향뿐 아니라,
한 번에 얼마나 이동할지를 정하는 step_size도 중요하다는 점을 정리한다.
""".strip()

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
st.title("최종: 인공지능 수학 보고서 작성 & PDF 생성")
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
