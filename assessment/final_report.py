# assessment/final_report.py
# ------------------------------------------------------------
# 최종 보고서 생성(서술형) + PDF 출력 페이지
#
# 요구사항(A안):
# - 학생이 CSV, 1~3차시 TXT 백업, 그래프 이미지(학생이 미리 저장한 것)를 업로드
# - 보고서 틀(Ⅰ/Ⅱ/Ⅲ + Ⅱ의 1)2)3))을 자동 생성하고,
#   각 섹션은 "서술형 문단"으로 학생이 편집 후 PDF로 저장
# - LaTeX는 PDF에서 깨지지 않도록 "이미지로 렌더링하여" 삽입
# - 그래프는 학생 업로드 이미지를 사용하여 보고서에 배치
# ------------------------------------------------------------

from __future__ import annotations

import re
from io import BytesIO
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image as RLImage,
    Table,
    TableStyle,
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
    """
    lines에서 'header' 정확 일치 줄을 찾고, 다음 헤더 전까지 본문 반환
    """
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


def _parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    is_pct = s.endswith("%")
    s2 = s.replace("%", "").replace(",", "").strip()
    try:
        v = float(s2)
        return v / 100.0 if is_pct else v
    except Exception:
        return None


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

    out["x_mode"] = _find_line_value(lines, "- X축 해석 방식:")
    out["valid_n"] = _find_line_value(lines, "- 유효 데이터 점 개수:")
    if not out["valid_n"]:
        out["valid_n"] = _find_line_value(lines, "- 유효 데이터 점:")

    out["features"] = _section_text(lines, "[그래프 관찰 특징]", ["[모델링 가설]", "[추가 메모]"])

    model_block = _section_text(lines, "[모델링 가설]", ["[추가 메모]"])
    out["model_primary"] = ""
    out["model_primary_reason"] = ""
    if model_block:
        for ln in model_block.splitlines():
            if ln.strip().startswith("- 주된 모델:"):
                out["model_primary"] = ln.strip().replace("- 주된 모델:", "", 1).strip()
        m = re.split(r"-\s*주된 모델 근거:\s*", model_block, maxsplit=1)
        if len(m) == 2:
            out["model_primary_reason"] = m[1].strip()

    out["note"] = _section_text(lines, "[추가 메모]", [])
    return out


def parse_step2_backup_txt(text: str) -> Dict[str, str]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    out: Dict[str, str] = {}

    out["student_id"] = _find_line_value(lines, "학번:")

    block = _section_text(lines, "[가설 재평가]", ["[데이터 정보]"])
    out["model_hypothesis_step1"] = _find_line_value(block.splitlines(), "- 1차시 가설 모델:") if block else ""
    out["hypothesis_decision"] = _find_line_value(block.splitlines(), "- 가설 판단:") if block else ""
    out["revised_model"] = _find_line_value(block.splitlines(), "- 수정한 가설 모델:") if block else ""

    info = _section_text(lines, "[데이터 정보]", ["[AI 프롬프트]", "[AI 모델식/미분식(LaTeX)]"])
    out["data_source"] = ""
    out["x_col"] = ""
    out["y_col"] = ""
    out["valid_n"] = ""
    if info:
        info_lines = [ln.strip() for ln in info.splitlines()]
        out["data_source"] = _find_line_value(info_lines, "- 데이터 출처:")
        for ln in info_lines:
            if ln.startswith("- X축:"):
                m = re.search(r"- X축:\s*(.*?)\s*\|\s*Y축:\s*(.*)$", ln)
                if m:
                    out["x_col"] = m.group(1).strip()
                    out["y_col"] = m.group(2).strip()
                    break
        out["valid_n"] = _find_line_value(info_lines, "- 유효 데이터 점:")

    out["ai_prompt"] = _section_text(
        lines,
        "[AI 프롬프트]",
        ["[AI 모델식/미분식(LaTeX)]", "[미분 관점의 모델 분석(학생 작성)]"],
    )
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
    out["note"] = _section_text(lines, "[추가 메모]", [])
    return out


def parse_step3_backup_txt(text: str) -> Dict[str, object]:
    text = _strip_bom(text)
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    out: Dict[str, object] = {}

    out["student_id"] = _find_line_value(lines, "학번:")

    info = _section_text(lines, "[데이터 정보]", ["[모델식 f(t) (py_model)]"])
    out["data_source"] = ""
    out["x_col"] = ""
    out["y_col"] = ""
    out["valid_n"] = ""
    out["i0"] = ""
    out["i1"] = ""
    if info:
        info_lines = [ln.strip() for ln in info.splitlines()]
        out["data_source"] = _find_line_value(info_lines, "- 데이터 출처:")
        out["x_col"] = _find_line_value(info_lines, "- X축(시간):")
        out["y_col"] = _find_line_value(info_lines, "- Y축(수치):")
        out["valid_n"] = _find_line_value(info_lines, "- 유효 데이터 점:")
        rng = _find_line_value(info_lines, "- 적분 구간(인덱스):")
        m = re.search(r"(\d+)\s*~\s*(\d+)", rng)
        if m:
            out["i0"] = m.group(1)
            out["i1"] = m.group(2)

    out["py_model"] = _section_text(
        lines,
        "[모델식 f(t) (py_model)]",
        ["[적분 비교 결과]", "[오차]"],
    ).strip()

    result_block = _section_text(lines, "[적분 비교 결과]", ["[오차]"])
    A_rect = A_trap = I_model = None
    if result_block:
        for ln in result_block.splitlines():
            ln = ln.strip()
            if ln.startswith("- 직사각형 값"):
                A_rect = _parse_number(ln.split(":")[-1].strip())
            elif ln.startswith("- 사다리꼴 값"):
                A_trap = _parse_number(ln.split(":")[-1].strip())
            elif ln.startswith("- 정적분 값"):
                I_model = _parse_number(ln.split(":")[-1].strip())

    err_block = _section_text(lines, "[오차]", ["[4) 적분 관점의 모델 분석(학생 서술)]"])
    err_rect = err_trap = rel_trap = None
    if err_block:
        for ln in err_block.splitlines():
            ln = ln.strip()
            if ln.startswith("- 직사각형 오차"):
                err_rect = _parse_number(ln.split(":")[-1].strip())
            elif ln.startswith("- 사다리꼴 오차"):
                err_trap = _parse_number(ln.split(":")[-1].strip())
            elif ln.startswith("- 사다리꼴 상대오차"):
                rel_trap = _parse_number(ln.split(":")[-1].strip())

    out["A_rect"] = A_rect
    out["A_trap"] = A_trap
    out["I_model"] = I_model
    out["err_rect"] = err_rect
    out["err_trap"] = err_trap
    out["rel_trap"] = rel_trap

    out["student_critical_review2"] = _section_text(
        lines,
        "[4) 적분 관점의 모델 분석(학생 서술)]",
        [],
    ).strip()

    return out


# ============================================================
# CSV 요약
# ============================================================
def read_csv_loose(file) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, sep=None, engine="python", on_bad_lines="skip")
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError("CSV를 읽을 수 없습니다.")


def summarize_csv(df: pd.DataFrame, max_head: int = 10) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["shape"] = df.shape
    out["head"] = df.head(max_head).copy()
    out["missing_total"] = int(df.isna().sum().sum())
    return out


# ============================================================
# LaTeX 렌더링(이미지)
# ============================================================
def latex_to_png_bytes(latex: str, fontsize: int = 16) -> Optional[bytes]:
    """
    latex 문자열을 matplotlib mathtext로 PNG 렌더링.
    - 실패하면 None 반환(텍스트 fallback)
    """
    latex = (latex or "").strip()
    if not latex:
        return None

    # 흔히 백업에는 이미 f(t)=... 형태가 들어올 수 있으니, $...$로 감싸기만 한다.
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
# PDF 생성(Platypus)
# ============================================================
def build_report_pdf(
    *,
    meta: Dict[str, str],
    csv_summary: Dict[str, object],
    sections: Dict[str, str],
    latex_items: Dict[str, str],
    images: Dict[str, Optional[bytes]],
    include_appendix_raw_txt: bool,
    raw_txts: Dict[str, str],
) -> bytes:
    bio = BytesIO()
    doc = SimpleDocTemplate(
        bio,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="미적분 수행평가 최종 보고서",
        author=meta.get("student_id", ""),
    )

    styles = getSampleStyleSheet()
    base = styles["BodyText"]
    base.fontName = "Helvetica"
    base.fontSize = 10
    base.leading = 14

    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
    )
    h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        spaceBefore=6,
        spaceAfter=3,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=base,
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceBefore=3,
        spaceAfter=8,
    )

    story = []

    # 표지/메타
    story.append(Paragraph("공공데이터 기반 함수 모델링과 미적분적 해석", ParagraphStyle(
        "CoverTitle", parent=h1, fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=10
    )))
    story.append(Paragraph("최종 탐구 보고서", ParagraphStyle(
        "CoverSub", parent=h2, alignment=TA_CENTER, spaceAfter=12
    )))

    meta_lines = [
        f"학번: {meta.get('student_id','')}",
        f"데이터 출처: {meta.get('data_source','')}",
        f"변수: X(시간)={meta.get('x_col','')} / Y={meta.get('y_col','')}",
        f"작성일: {datetime.now().strftime('%Y-%m-%d')}",
    ]
    story.append(Spacer(1, 4 * mm))
    for ln in meta_lines:
        story.append(Paragraph(ln, base))
    story.append(Spacer(1, 10 * mm))

    # CSV 요약 표(Ⅱ-1에서 다시 넣을 수도 있지만, 표지는 간단히)
    story.append(Paragraph("데이터 요약", h2))
    shape = csv_summary.get("shape", ("", ""))
    story.append(Paragraph(f"- 행 × 열: {shape[0]} × {shape[1]}", base))
    story.append(Paragraph(f"- 결측치 총합: {csv_summary.get('missing_total', '')}", base))
    story.append(Spacer(1, 6 * mm))

    story.append(PageBreak())

    # Ⅰ. 탐구 동기
    story.append(Paragraph("Ⅰ. 탐구 동기", h1))
    story.append(Paragraph(sections.get("I", "").replace("\n", "<br/>"), base))
    story.append(Spacer(1, 6 * mm))

    story.append(PageBreak())

    # Ⅱ. 탐구
    story.append(Paragraph("Ⅱ. 탐구", h1))

    # Ⅱ-1 선택한 데이터
    story.append(Paragraph("1) 선택한 데이터", h2))
    story.append(Paragraph(sections.get("II_1", "").replace("\n", "<br/>"), base))
    story.append(Spacer(1, 4 * mm))

    # CSV head 테이블
    head_df = csv_summary.get("head")
    if isinstance(head_df, pd.DataFrame) and head_df.shape[0] > 0:
        tbl_data = [list(head_df.columns)] + head_df.astype(str).values.tolist()
        t = Table(tbl_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (-1, -1), "LEFT"),
        ]))
        story.append(t)
        story.append(Spacer(1, 6 * mm))

    # 원자료 그래프 이미지
    fig_no = 1
    if images.get("raw_graph"):
        story.append(Paragraph(f"그림 {fig_no}. 원자료 그래프", caption))
        story.append(RLImage(BytesIO(images["raw_graph"]), width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 6 * mm))
        fig_no += 1

    story.append(Spacer(1, 6 * mm))

    # Ⅱ-2 미분 분석
    story.append(Paragraph("2) 미분 분석", h2))
    story.append(Paragraph(sections.get("II_2", "").replace("\n", "<br/>"), base))
    story.append(Spacer(1, 6 * mm))

    # LaTeX(모델식/도함수/이계도함수)
    # latex_items: {"model": "...", "d1": "...", "d2": "..."}
    latex_order = [
        ("model", "모델식"),
        ("d1", "도함수"),
        ("d2", "이계도함수"),
    ]
    for key, label in latex_order:
        tex = (latex_items.get(key) or "").strip()
        if not tex:
            continue
        story.append(Paragraph(f"{label}:", h3))
        png = latex_to_png_bytes(tex, fontsize=16)
        if png:
            story.append(RLImage(BytesIO(png), width=170 * mm, height=18 * mm))
        else:
            # fallback: 원문 텍스트
            story.append(Paragraph(f"<font color='grey'>{tex}</font>", base))
        story.append(Spacer(1, 4 * mm))

    # 변화율/이계변화율 그래프
    if images.get("rate_graph"):
        story.append(Paragraph(f"그림 {fig_no}. 변화율 그래프", caption))
        story.append(RLImage(BytesIO(images["rate_graph"]), width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 6 * mm))
        fig_no += 1

    if images.get("second_rate_graph"):
        story.append(Paragraph(f"그림 {fig_no}. 이계변화율 그래프", caption))
        story.append(RLImage(BytesIO(images["second_rate_graph"]), width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 6 * mm))
        fig_no += 1

    story.append(PageBreak())

    # Ⅱ-3 적분 분석
    story.append(Paragraph("3) 적분 분석", h2))
    story.append(Paragraph(sections.get("II_3", "").replace("\n", "<br/>"), base))
    story.append(Spacer(1, 6 * mm))

    # 적분 결과표(숫자)
    integ_tbl = []
    for k, label in [
        ("A_rect", "직사각형(데이터, 좌측)"),
        ("A_trap", "사다리꼴(데이터)"),
        ("I_model", "정적분(모델)"),
        ("err_rect", "직사각형 오차 |A-I|"),
        ("err_trap", "사다리꼴 오차 |A-I|"),
        ("rel_trap", "사다리꼴 상대오차"),
    ]:
        v = meta.get(k, "")
        if v != "":
            integ_tbl.append([label, str(v)])
    if integ_tbl:
        t2 = Table([["항목", "값"]] + integ_tbl, colWidths=[70 * mm, 90 * mm], repeatRows=1)
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t2)
        story.append(Spacer(1, 6 * mm))

    # 적분 도형/비교 그래프
    if images.get("integral_graph"):
        story.append(Paragraph(f"그림 {fig_no}. 적분(누적) 비교/도형 그래프", caption))
        story.append(RLImage(BytesIO(images["integral_graph"]), width=170 * mm, height=90 * mm))
        story.append(Spacer(1, 6 * mm))
        fig_no += 1

    story.append(PageBreak())

    # Ⅲ. 결론
    story.append(Paragraph("Ⅲ. 결론", h1))
    story.append(Paragraph(sections.get("III", "").replace("\n", "<br/>"), base))
    story.append(Spacer(1, 6 * mm))

    # 부록: 원문 TXT
    if include_appendix_raw_txt:
        story.append(PageBreak())
        story.append(Paragraph("부록. 백업 TXT 원문", h1))
        for key, title in [("step1", "1차시 TXT"), ("step2", "2차시 TXT"), ("step3", "3차시 TXT")]:
            raw = (raw_txts.get(key) or "").strip()
            if not raw:
                continue
            story.append(Paragraph(title, h2))
            # 원문은 길 수 있으니 폰트 작게
            story.append(Paragraph(raw.replace("\n", "<br/>"), ParagraphStyle(
                f"RAW_{key}", parent=base, fontSize=8.5, leading=11
            )))
            story.append(Spacer(1, 6 * mm))

    doc.build(story)
    return bio.getvalue()


# ============================================================
# Streamlit UI
# ============================================================
st.title("최종 보고서 작성 및 PDF 생성")
st.caption("CSV + 1~3차시 TXT + 그래프 이미지를 업로드하면 초안을 만들고, 서술형으로 편집한 뒤 PDF로 저장합니다.")
st.divider()

# 0) 업로드
st.subheader("0) 자료 업로드")

colA, colB = st.columns([1, 1])
with colA:
    csv_file = st.file_uploader("CSV 데이터 업로드(필수)", type=["csv"], key="final_csv")
    step1_txt_f = st.file_uploader("1차시 백업 TXT(필수)", type=["txt"], key="final_step1")
    step2_txt_f = st.file_uploader("2차시 백업 TXT(필수)", type=["txt"], key="final_step2")
    step3_txt_f = st.file_uploader("3차시 백업 TXT(필수)", type=["txt"], key="final_step3")
with colB:
    st.markdown("**그래프 이미지 업로드(학생이 1~3차시에서 저장한 그림 파일)**")
    img_raw = st.file_uploader("원자료 그래프(필수)", type=["png", "jpg", "jpeg"], key="img_raw")
    img_rate = st.file_uploader("변화율 그래프(필수)", type=["png", "jpg", "jpeg"], key="img_rate")
    img_second = st.file_uploader("이계변화율 그래프(필수)", type=["png", "jpg", "jpeg"], key="img_second")
    img_integral = st.file_uploader("적분 도형/비교 그래프(필수)", type=["png", "jpg", "jpeg"], key="img_integral")

include_appendix = st.checkbox("PDF에 부록(백업 TXT 원문) 포함", value=True)

missing = []
if csv_file is None:
    missing.append("CSV")
if step1_txt_f is None or step2_txt_f is None or step3_txt_f is None:
    missing.append("TXT(1~3차시)")
if any(x is None for x in [img_raw, img_rate, img_second, img_integral]):
    missing.append("그래프 이미지 4종")

if missing:
    st.info(f"업로드가 필요합니다: {', '.join(missing)}")
    st.stop()

# 1) 파싱/요약
try:
    df = read_csv_loose(csv_file)
    csv_sum = summarize_csv(df, max_head=10)

    t1_raw = _read_uploaded_txt(step1_txt_f)
    t2_raw = _read_uploaded_txt(step2_txt_f)
    t3_raw = _read_uploaded_txt(step3_txt_f)

    s1 = parse_step1_backup_txt(t1_raw)
    s2 = parse_step2_backup_txt(t2_raw)
    s3 = parse_step3_backup_txt(t3_raw)

except Exception as e:
    st.error("자료를 읽거나 파싱하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# 2) 메타/수치 정리
student_id = (s1.get("student_id") or s2.get("student_id") or str(s3.get("student_id") or "")).strip()
data_source = (s1.get("data_source") or s2.get("data_source") or str(s3.get("data_source") or "")).strip()
x_col = (s1.get("x_col") or s2.get("x_col") or str(s3.get("x_col") or "")).strip()
y_col = (s1.get("y_col") or s2.get("y_col") or str(s3.get("y_col") or "")).strip()

meta = {
    "student_id": student_id,
    "data_source": data_source,
    "x_col": x_col,
    "y_col": y_col,
}

# Step3 수치값(보고서 표에 사용)
def _fmt_num(v) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):,.6g}"
    except Exception:
        return str(v)

meta["A_rect"] = _fmt_num(s3.get("A_rect"))
meta["A_trap"] = _fmt_num(s3.get("A_trap"))
meta["I_model"] = _fmt_num(s3.get("I_model"))
meta["err_rect"] = _fmt_num(s3.get("err_rect"))
meta["err_trap"] = _fmt_num(s3.get("err_trap"))
rel = s3.get("rel_trap")
meta["rel_trap"] = f"{float(rel):.3%}" if isinstance(rel, (int, float)) else (_fmt_num(rel) if rel else "")

# 이미지 bytes
images = {
    "raw_graph": img_raw.getvalue() if img_raw else None,
    "rate_graph": img_rate.getvalue() if img_rate else None,
    "second_rate_graph": img_second.getvalue() if img_second else None,
    "integral_graph": img_integral.getvalue() if img_integral else None,
}

# LaTeX 블록에서 수식 3개 뽑기(가능한 범위)
# - 가장 안정적인 건 Step2 LaTeX 블록에서 "f(t)" / "f'(t)" / "f''(t)" 라인을 찾는 것
# - 실패해도 전체 블록을 그대로 "모델식"으로 두고, 나머지는 비워둔다.
latex_items = {"model": "", "d1": "", "d2": ""}
latex_block = (s2.get("ai_latex_block") or "").strip()

# 간단 파서: 가장 먼저 나타나는 수식 3개를 후보로 잡기
# (백업 포맷이 라인별로 모델/도함수/이계도함수를 나열하는 경우가 많음)
cands = [ln.strip() for ln in latex_block.splitlines() if ln.strip()]
# 너무 길면(설명 문장) 제외하기: '=' 또는 '\' 또는 't'가 포함된 라인 우선
filtered = [ln for ln in cands if ("=" in ln) or ("\\" in ln) or ("t" in ln)]
filtered = filtered if filtered else cands

# 매우 보수적: 첫 3개를 할당
if filtered:
    latex_items["model"] = filtered[0]
if len(filtered) >= 2:
    latex_items["d1"] = filtered[1]
if len(filtered) >= 3:
    latex_items["d2"] = filtered[2]

st.subheader("1) 파싱/요약 확인(검토용)")
with st.expander("CSV 요약", expanded=False):
    st.write(f"행×열: {csv_sum['shape'][0]}×{csv_sum['shape'][1]}, 결측치: {csv_sum['missing_total']}")
    st.dataframe(csv_sum["head"])
with st.expander("TXT 파싱 결과(요약)", expanded=False):
    st.json({
        "step1": {k: s1.get(k, "") for k in ["student_id", "data_source", "x_col", "y_col", "valid_n", "model_primary"]},
        "step2": {k: s2.get(k, "") for k in ["hypothesis_decision", "revised_model"]},
        "step3": {k: s3.get(k, "") for k in ["i0", "i1", "A_rect", "A_trap", "I_model", "rel_trap"]},
    })
with st.expander("LaTeX(자동 추출) 미리보기", expanded=False):
    st.write(latex_items)

st.divider()

# 3) 초안 생성 + 편집(A안: 섹션별 텍스트 영역)
st.subheader("2) 보고서 본문 작성(서술형 편집)")

# session_state keys
K_I = "final_sec_I"
K_II1 = "final_sec_II1"
K_II2 = "final_sec_II2"
K_II3 = "final_sec_II3"
K_III = "final_sec_III"

def _maybe_init_drafts():
    """
    세션에 초안이 없으면 자동 생성하여 채워 넣는다.
    """
    if K_I not in st.session_state:
        st.session_state[K_I] = (
            "본 탐구는 공공데이터를 활용하여 시간에 따른 변화 양상을 함수로 모델링하고, "
            "미분과 적분의 관점에서 그 타당성을 해석하는 것을 목적으로 한다.\n\n"
            f"선택한 데이터 출처는 '{data_source}'이며, 그래프 관찰 결과 다음과 같은 특징이 나타났다:\n"
            f"{(s1.get('features') or '').strip()}\n\n"
            "이러한 특징을 설명하기 위해 적절한 모델을 세우고, 이후 변화율(미분)과 누적량(적분) 관점에서 평가한다."
        )

    if K_II1 not in st.session_state:
        shape = csv_sum.get("shape", (0, 0))
        st.session_state[K_II1] = (
            f"본 탐구에서 사용한 데이터는 시간 변수(X) '{x_col}'와 수치 변수(Y) '{y_col}'로 구성되어 있다. "
            f"데이터는 총 {shape[0]}개 관측치로 이루어져 있으며, 결측치는 {csv_sum.get('missing_total', 0)}개이다.\n\n"
            "원자료 그래프(그림 1)를 통해 전체적인 추세와 변동의 특징을 확인하였다. "
            "특히 변동이 반복되는 구간/추세 변화가 관찰되며, 이는 모델 선택에 중요한 근거가 된다."
        )

    if K_II2 not in st.session_state:
        st.session_state[K_II2] = (
            "미분 관점에서는 데이터의 변화율(Δy/Δt)과 이계변화율(Δ²y/Δt²)을 이용하여 "
            "증가·감소 및 오목·볼록의 변화를 해석하였다.\n\n"
            "또한 모델식으로부터 도함수 f′(t), 이계도함수 f″(t)를 고려하여 "
            "그래프에서 나타난 변화율의 특징이 모델에 의해 얼마나 설명되는지 분석하였다.\n\n"
            f"{(s2.get('student_analysis') or '').strip()}"
        )

    if K_II3 not in st.session_state:
        i0 = str(s3.get("i0", "")).strip()
        i1 = str(s3.get("i1", "")).strip()
        st.session_state[K_II3] = (
            "적분 관점에서는 일정 구간에서의 누적량을 정적분으로 해석하였다. "
            "원본 데이터의 이산 점을 이용해 직사각형 합(좌측)과 사다리꼴 합으로 수치적분을 계산하고, "
            "이를 모델의 정적분 값과 비교하였다.\n\n"
            f"본 보고서에서 선택한 적분 구간은 인덱스 {i0} ~ {i1}이며, "
            "해당 구간에서의 값 비교 및 오차 분석 결과를 통해 모델의 누적 설명력을 평가하였다.\n\n"
            f"{(s3.get('student_critical_review2') or '').strip()}"
        )

    if K_III not in st.session_state:
        st.session_state[K_III] = (
            "본 탐구의 결과를 종합하면, 설정한 모델은 (여기에 핵심 결론을 1~2문장으로 요약).\n\n"
            "첫째, 장점은 (미분/적분 근거를 포함하여 2~4문장으로 서술).\n\n"
            "둘째, 한계는 (오차가 큰 구간/설명되지 않는 변동과 그 원인 추정).\n\n"
            "마지막으로 개선 및 추가 탐구로는 (변수 추가, 다른 모델 비교, 구간 재설정 등)을 제안한다."
        )

# 초안 생성 버튼(원하면 다시 만들기 가능)
colx, coly = st.columns([1, 1])
with colx:
    if st.button("🧩 초안 자동 생성(세션에 없을 때만)", use_container_width=True):
        _maybe_init_drafts()
        st.success("초안이 준비되었습니다. 아래에서 서술형으로 수정하세요.")
with coly:
    if st.button("🧹 초안 다시 만들기(덮어쓰기)", use_container_width=True):
        for k in [K_I, K_II1, K_II2, K_II3, K_III]:
            if k in st.session_state:
                del st.session_state[k]
        _maybe_init_drafts()
        st.success("초안을 다시 생성했습니다.")

# ensure initialized
_maybe_init_drafts()

st.markdown("### Ⅰ. 탐구 동기")
sec_I = st.text_area("본문(서술형)", key=K_I, height=220)

st.markdown("### Ⅱ. 탐구")
st.markdown("#### 1) 선택한 데이터")
sec_II1 = st.text_area("본문(서술형)", key=K_II1, height=220)

st.markdown("#### 2) 미분 분석")
sec_II2 = st.text_area("본문(서술형)", key=K_II2, height=260)

st.markdown("#### 3) 적분 분석")
sec_II3 = st.text_area("본문(서술형)", key=K_II3, height=260)

st.markdown("### Ⅲ. 결론")
sec_III = st.text_area("본문(서술형)", key=K_III, height=240)

st.divider()

# 4) PDF 생성/다운로드
st.subheader("3) PDF 저장")

def _validate() -> bool:
    # 최소한 결론은 필수로 받는 편이 안전(보고서 완성도)
    if not sec_III.strip():
        st.warning("Ⅲ. 결론을 작성하세요.")
        return False
    return True

if st.button("📄 PDF 생성", use_container_width=True):
    if not _validate():
        st.stop()

    sections = {
        "I": sec_I.strip(),
        "II_1": sec_II1.strip(),
        "II_2": sec_II2.strip(),
        "II_3": sec_II3.strip(),
        "III": sec_III.strip(),
    }

    try:
        pdf_bytes = build_report_pdf(
            meta=meta,
            csv_summary=csv_sum,
            sections=sections,
            latex_items=latex_items,
            images=images,
            include_appendix_raw_txt=include_appendix,
            raw_txts={"step1": t1_raw, "step2": t2_raw, "step3": t3_raw},
        )

        fname = f"미적분_수행평가_최종보고서_{student_id or 'unknown'}.pdf"
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
