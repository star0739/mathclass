# assessment/final_report.py
# ------------------------------------------------------------
# 최종 보고서 작성(서술형) + PDF 출력 페이지 (CSV 미사용)
# - 입력: 제목(필수), 학번(필수), 이름(필수)
# - 업로드: 1~3차시 TXT(필수) + 그래프 이미지 5종(필수)
#   * 원자료 / 변화율 / 이계변화율 / 적분(직사각형) / 적분(사다리꼴)
# - 본문: 1. 서론 / 2. 본론(1~3) / 3. 결론 (PageBreak 없이 이어쓰기)
# - 수식: Step2 LaTeX를 이미지로 렌더링하여 깨짐 방지
# - 한글 폰트: assets 폴더의 TTF 등록 후 전체 적용
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
from io import BytesIO
from typing import Dict, Optional, List, Tuple

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
# TXT 읽기/파싱 유틸
# ============================================================
def _read_uploaded_txt(file) -> str:
    raw = file.getvalue()
    try:
        return raw.decode("utf-8-sig")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff").strip("\n")


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


def _remove_notice_lines(text: str) -> str:
    """
    Step3 백업 하단에 붙는 '※ ...' 안내문 같은 문구 제거(포맷 변화에 강하게)
    - '※'로 시작하는 줄이 나오면 그 줄 포함 이후 전부 제거
    """
    out_lines = []
    for ln in (text or "").splitlines():
        if ln.strip().startswith("※"):
            break
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


# ============================================================
# Step1/2/3 TXT 파서 (백업 포맷 기반)
# ============================================================
def parse_step1_backup_txt(text: str) -> Dict[str, str]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    out: Dict[str, str] = {}
    out["student_id"] = _find_line_value(lines, "학번:")
    out["data_source"] = _find_line_value(lines, "- 데이터 출처:")

    out["x_col"] = ""
    out["y_col"] = ""
    for ln in lines:
        if ln.strip().startswith("- X축:"):
            m = re.search(r"- X축:\s*(.*?)\s*\|\s*Y축:\s*(.*)$", ln.strip())
            if m:
                out["x_col"] = m.group(1).strip()
                out["y_col"] = m.group(2).strip()
                break

    out["features"] = _section_text(lines, "[그래프 관찰 특징]", ["[모델링 가설]", "[추가 메모]"])
    out["model_primary"] = ""

    model_block = _section_text(lines, "[모델링 가설]", ["[추가 메모]"])
    if model_block:
        for ln in model_block.splitlines():
            if ln.strip().startswith("- 주된 모델:"):
                out["model_primary"] = ln.strip().replace("- 주된 모델:", "", 1).strip()

    return out


def parse_step2_backup_txt(text: str) -> Dict[str, str]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    out: Dict[str, str] = {}

    out["student_id"] = _find_line_value(lines, "학번:")

    # ✅ (추가) 2차시에서 '수정한 가설 모델' 파싱
    # - Step2 백업에는 보통 [가설 재평가] 섹션이 있고, 그 안에 "- 수정한 가설 모델:" 라인이 있음
    block = _section_text(lines, "[가설 재평가]", ["[데이터 정보]"])
    out["revised_model"] = _find_line_value(block.splitlines(), "- 수정한 가설 모델:") if block else ""

    # 기존 유지
    out["ai_latex_block"] = _section_text(
        lines,
        "[AI 모델식/미분식(LaTeX)]",
        ["[미분 관점의 모델 분석(학생 작성)]", "[추가 메모]"],
    )
    out["student_analysis"] = _section_text(
        lines,
        "[미분 관점의 모델 분석(학생 작성)]",
        ["[추가 메모]"],
    )
    return out


def parse_step3_backup_txt(text: str) -> Dict[str, str]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    out: Dict[str, str] = {}

    out["student_id"] = _find_line_value(lines, "학번:")

    review = _section_text(
        lines,
        "[4) 적분 관점의 모델 분석(학생 서술)]",
        [],
    ).strip()

    # ✅ 안내문 제거(※ ... 이후 삭제)
    out["student_critical_review2"] = _remove_notice_lines(review)
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
def build_report_pdf(
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

    # 본문(양측 정렬)
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

    # 제목(중앙)
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

    # 학번/이름(우측)
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

    # 큰 항목 제목(좌측, 굵게)
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

    # 소항목 제목(좌측, 굵게)
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading3"],
        fontName=bold_font,
        fontSize=11.5,
        leading=16,
        alignment=TA_JUSTIFY,
        spaceBefore=6,
        spaceAfter=3,
    )

    # 캡션(중앙)
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

    # -------------------------
    # 상단 헤더(표지 없음)
    # -------------------------
    story.append(Paragraph(report_title.strip(), title_style))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"{student_id.strip()}  {student_name.strip()}", meta_style))
    # 줄 간격 2칸 정도
    story.append(Spacer(1, 10 * mm))
    story.append(Spacer(1, 10 * mm))

    # -------------------------
    # 1. 서론
    # -------------------------
    story.append(Paragraph("1. 서론", h1))
    if sections.get("intro", "").strip():
        story.append(Paragraph(sections["intro"].replace("\n", "<br/>"), body))
    story.append(Spacer(1, 10 * mm))
    story.append(Spacer(1, 10 * mm))

    # -------------------------
    # 2. 본론
    # -------------------------
    story.append(Paragraph("2. 본론", h1))

    # 2-1 데이터(원자료 그래프 + 서술)
    story.append(Paragraph("1) 선택한 데이터", h2))
    if sections.get("body_data", "").strip():
        story.append(Paragraph(sections["body_data"].replace("\n", "<br/>"), body))
    story.append(Spacer(1, 6 * mm))

    if images.get("raw_graph"):
        story.append(Paragraph("그림 1. 원자료 그래프", caption))
        story.append(RLImage(BytesIO(images["raw_graph"]), width=170 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(Spacer(1, 10 * mm))

    # 2-2 미분(모델식 먼저, 그 다음 서술, 그 다음 도함수/이계도함수, 그래프)
    story.append(Paragraph("2) 미분 분석", h2))

    # 모델식(LaTeX) 먼저
    model_tex = (latex_items.get("model") or "").strip()
    if model_tex:
        png = latex_to_png_bytes(model_tex, fontsize=18)
        if png:
            story.append(RLImage(BytesIO(png), width=170 * mm, height=18 * mm, hAlign="CENTER"))
            story.append(Spacer(1, 4 * mm))
        else:
            story.append(Paragraph(model_tex, body))
            story.append(Spacer(1, 4 * mm))

    # 서술
    if sections.get("body_diff", "").strip():
        story.append(Paragraph(sections["body_diff"].replace("\n", "<br/>"), body))
        story.append(Spacer(1, 6 * mm))

    # 도함수/이계도함수(있으면)
    for key, label in [("d1", "도함수"), ("d2", "이계도함수")]:
        tex = (latex_items.get(key) or "").strip()
        if not tex:
            continue
        png = latex_to_png_bytes(tex, fontsize=16)
        if png:
            story.append(Paragraph(label, caption))
            story.append(RLImage(BytesIO(png), width=170 * mm, height=18 * mm, hAlign="CENTER"))
        else:
            story.append(Paragraph(f"{label}: {tex}", body))
        story.append(Spacer(1, 4 * mm))

    # 변화율 / 이계변화율 그래프
    if images.get("rate_graph"):
        story.append(Paragraph("그림 2. 변화율 그래프", caption))
        story.append(RLImage(BytesIO(images["rate_graph"]), width=170 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(Spacer(1, 8 * mm))

    if images.get("second_rate_graph"):
        story.append(Paragraph("그림 3. 이계변화율 그래프", caption))
        story.append(RLImage(BytesIO(images["second_rate_graph"]), width=170 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(Spacer(1, 10 * mm))

    # 2-3 적분(서술 + 직사각형/사다리꼴 도형 2장)
    story.append(Paragraph("3) 적분 분석", h2))
    if sections.get("body_integ", "").strip():
        story.append(Paragraph(sections["body_integ"].replace("\n", "<br/>"), body))
        story.append(Spacer(1, 6 * mm))

    if images.get("integral_rect"):
        story.append(Paragraph("그림 4. 적분 도형(직사각형)", caption))
        story.append(RLImage(BytesIO(images["integral_rect"]), width=170 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(Spacer(1, 8 * mm))

    if images.get("integral_trap"):
        story.append(Paragraph("그림 5. 적분 도형(사다리꼴)", caption))
        story.append(RLImage(BytesIO(images["integral_trap"]), width=170 * mm, height=90 * mm, hAlign="CENTER"))
        story.append(Spacer(1, 10 * mm))

    # 본론 끝나고 줄간격 2칸
    story.append(Spacer(1, 10 * mm))
    story.append(Spacer(1, 10 * mm))

    # -------------------------
    # 3. 결론
    # -------------------------
    story.append(Paragraph("3. 결론", h1))
    if sections.get("conclusion", "").strip():
        story.append(Paragraph(sections["conclusion"].replace("\n", "<br/>"), body))

    doc.build(story)
    return bio.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

st.title("최종: 보고서 작성 & PDF 생성")
st.caption("1~3차시 TXT 백업 + 그래프 이미지를 업로드하고, 서술형으로 편집한 뒤 PDF로 저장합니다.")

st.divider()

# ----------------------------
# 0) 기본 정보 입력
# ----------------------------
st.subheader("0) 기본 정보 입력")

col0, col1, col2 = st.columns([2, 1, 1])
with col0:
    report_title = st.text_input("탐구 보고서 제목(필수)", value="", placeholder="예: 공공데이터로 본 ○○의 변화와 미적분적 해석")
with col1:
    student_id_input = st.text_input("학번(필수)", value="", placeholder="예: 30901")
with col2:
    student_name = st.text_input("이름(필수)", value="", placeholder="예: 홍길동")

st.divider()

# ----------------------------
# 1) 업로드
# ----------------------------
st.subheader("1) 자료 업로드")

colA, colB = st.columns([1, 1])
with colA:
    step1_txt_f = st.file_uploader("1차시 백업 TXT(필수)", type=["txt"], key="final_step1")
    step2_txt_f = st.file_uploader("2차시 백업 TXT(필수)", type=["txt"], key="final_step2")
    step3_txt_f = st.file_uploader("3차시 백업 TXT(필수)", type=["txt"], key="final_step3")

with colB:
    st.markdown("**그래프 이미지 업로드(필수)**")
    img_raw = st.file_uploader("원자료 그래프", type=["png", "jpg", "jpeg"], key="img_raw")
    img_rate = st.file_uploader("변화율 그래프", type=["png", "jpg", "jpeg"], key="img_rate")
    img_second = st.file_uploader("이계변화율 그래프", type=["png", "jpg", "jpeg"], key="img_second")
    img_integ_rect = st.file_uploader("적분 도형(직사각형)", type=["png", "jpg", "jpeg"], key="img_integ_rect")
    img_integ_trap = st.file_uploader("적분 도형(사다리꼴)", type=["png", "jpg", "jpeg"], key="img_integ_trap")

missing = []
if not report_title.strip():
    missing.append("제목")
if not student_id_input.strip():
    missing.append("학번")
if not student_name.strip():
    missing.append("이름")
if step1_txt_f is None or step2_txt_f is None or step3_txt_f is None:
    missing.append("TXT(1~3차시)")
if any(x is None for x in [img_raw, img_rate, img_second, img_integ_rect, img_integ_trap]):
    missing.append("그래프 이미지 5종")

if missing:
    st.info(f"입력/업로드가 필요합니다: {', '.join(missing)}")
    st.stop()

# ----------------------------
# 2) TXT 파싱
# ----------------------------
try:
    t1_raw = _read_uploaded_txt(step1_txt_f)
    t2_raw = _read_uploaded_txt(step2_txt_f)
    t3_raw = _read_uploaded_txt(step3_txt_f)

    s1 = parse_step1_backup_txt(t1_raw)
    s2 = parse_step2_backup_txt(t2_raw)
    s3 = parse_step3_backup_txt(t3_raw)
except Exception as e:
    st.error("TXT를 읽거나 파싱하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# 학번 자동채움(입력값이 비었을 때만)
if not student_id_input.strip() and (s1.get("student_id") or s2.get("student_id") or s3.get("student_id")):
    student_id_input = (s1.get("student_id") or s2.get("student_id") or s3.get("student_id") or "").strip()

# ----------------------------
# 3) LaTeX 추출(모델/도함수/이계도함수)
# ----------------------------
latex_items = {"model": "", "d1": "", "d2": ""}
latex_block = (s2.get("ai_latex_block") or "").strip()

cands = [ln.strip() for ln in latex_block.splitlines() if ln.strip()]
filtered = [ln for ln in cands if ("=" in ln) or ("\\" in ln) or ("t" in ln)]
filtered = filtered if filtered else cands

if filtered:
    latex_items["model"] = filtered[0]
if len(filtered) >= 2:
    latex_items["d1"] = filtered[1]
if len(filtered) >= 3:
    latex_items["d2"] = filtered[2]

# ----------------------------
# 4) 초안 생성 + 편집(A안: 섹션별 텍스트 영역)
# ----------------------------
st.divider()
st.subheader("2) 보고서 본문 작성(서술형 편집)")

K_INTRO = "final_sec_intro"
K_BDATA = "final_sec_body_data"
K_BDIFF = "final_sec_body_diff"
K_BINT = "final_sec_body_integ"
K_CONC = "final_sec_conclusion"

def _maybe_init_drafts() -> None:
    if K_INTRO not in st.session_state:
        st.session_state[K_INTRO] = (
            "본 탐구는 공공데이터를 바탕으로 시간에 따른 변화 양상을 함수로 모델링하고, "
            "미분과 적분의 관점에서 그 의미를 해석하는 것을 목적으로 한다.\n\n"
            "데이터를 선택한 이유와, 해당 현상을 수학적으로 분석할 필요성을 서술한다."
        )

    if K_BDATA not in st.session_state:
        features = (s1.get("features") or "").strip()
        model_hint = (s2.get("revised_model") or s1.get("model_primary") or "").strip()
        extra = ""
        if features:
            extra += f"\n\n(그래프 관찰 특징)\n{features}"
        if model_hint:
            extra += f"\n\n(가설 모델)\n{model_hint}"

        st.session_state[K_BDATA] = (
            "본론에서는 먼저 원자료 그래프를 통해 전체적인 추세와 변동의 특징을 확인한다. "
            "특히 추세 변화 또는 주기적 변동 등 눈에 띄는 특징을 근거로 모델을 설정한다."
            + extra
        )

    if K_BDIFF not in st.session_state:
        st.session_state[K_BDIFF] = (
            "미분 관점에서는 변화율(Δy/Δt)과 이계변화율(Δ²y/Δt²)을 통해 증가·감소 및 오목·볼록의 변화를 해석한다.\n\n"
            "또한 모델식으로부터 얻은 도함수 f′(t), 이계도함수 f″(t)가 관찰된 특징을 얼마나 잘 설명하는지 분석한다.\n\n"
            + ((s2.get("student_analysis") or "").strip())
        ).strip()

    if K_BINT not in st.session_state:
        st.session_state[K_BINT] = (
            "적분 관점에서는 일정 구간에서의 누적량을 정적분으로 해석하고, "
            "직사각형/사다리꼴 도형을 이용한 수치적분이 모델의 정적분 값에 수렴하는 과정을 비교한다.\n\n"
            + ((s3.get("student_critical_review2") or "").strip())
        ).strip()

    if K_CONC not in st.session_state:
        st.session_state[K_CONC] = (
            "결론에서는 본론의 분석을 바탕으로 모델의 타당성을 정리한다.\n\n"
            "• 모델의 장점(근거 포함)\n"
            "• 모델의 한계(근거 포함)\n"
            "• 개선 방향 또는 추가 탐구 제안"
        )

# 버튼
colx, coly = st.columns([1, 1])
with colx:
    if st.button("🧩 초안 자동 생성(세션에 없을 때만)", use_container_width=True):
        _maybe_init_drafts()
        st.success("초안이 준비되었습니다. 아래에서 서술형으로 수정하세요.")
with coly:
    if st.button("🧹 초안 다시 만들기(덮어쓰기)", use_container_width=True):
        for k in [K_INTRO, K_BDATA, K_BDIFF, K_BINT, K_CONC]:
            if k in st.session_state:
                del st.session_state[k]
        _maybe_init_drafts()
        st.success("초안을 다시 생성했습니다.")

_maybe_init_drafts()

st.markdown("### 1. 서론")
sec_intro = st.text_area("본문(서술형)", key=K_INTRO, height=220)

st.markdown("### 2. 본론")
st.markdown("#### 1) 선택한 데이터")
sec_body_data = st.text_area("본문(서술형)", key=K_BDATA, height=230)

st.markdown("#### 2) 미분 분석")
sec_body_diff = st.text_area("본문(서술형)", key=K_BDIFF, height=260)

st.markdown("#### 3) 적분 분석")
sec_body_integ = st.text_area("본문(서술형)", key=K_BINT, height=260)

st.markdown("### 3. 결론")
sec_conclusion = st.text_area("본문(서술형)", key=K_CONC, height=240)

with st.expander("LaTeX(자동 추출) 미리보기", expanded=False):
    st.write(latex_items)

st.divider()

# ----------------------------
# 5) PDF 생성/다운로드
# ----------------------------
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
        "body_data": sec_body_data.strip(),
        "body_diff": sec_body_diff.strip(),
        "body_integ": sec_body_integ.strip(),
        "conclusion": sec_conclusion.strip(),
    }

    images = {
        "raw_graph": img_raw.getvalue(),
        "rate_graph": img_rate.getvalue(),
        "second_rate_graph": img_second.getvalue(),
        "integral_rect": img_integ_rect.getvalue(),
        "integral_trap": img_integ_trap.getvalue(),
    }

    try:
        pdf_bytes = build_report_pdf(
            report_title=report_title,
            student_id=student_id_input,
            student_name=student_name,
            sections=sections,
            latex_items=latex_items,
            images=images,
        )

        fname = f"미적분_수행평가_최종보고서_{student_id_input.strip()}.pdf"
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
