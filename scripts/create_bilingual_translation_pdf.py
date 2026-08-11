#!/usr/bin/env python3
"""Create an original-plus-English research copy for paper 12029fe2805e."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


REPO = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/Users/baiyixin/Documents/Survey - design knowledge/"
    "expanded_pdf_corpus_459_20260811/12029fe2805e.pdf"
)
WORK = REPO / "tmp/pdfs/12029fe2805e_translation"
CHECKPOINT = WORK / "translations.json"
TRANSLATION_PDF = WORK / "12029fe2805e_english_translation.pdf"
OUTPUT = REPO / "output/pdf/12029fe2805e_original_and_english_translation.pdf"
MODEL = "llama3.1:8b"
HOST = "http://127.0.0.1:11434/api/generate"


def normalize(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Some local-model responses add a harmless conversational preface despite
    # the output-only instruction. Remove it from the research copy.
    text = re.sub(
        r"^Here (?:is|are) the (?:English )?translation(?:s)?(?: of (?:the )?source page(?: \d+)?)?:?\s*",
        "",
        text,
        flags=re.I,
    )
    return text.strip()


def translate_page(page_number: int, source_text: str) -> str:
    prompt = f"""You are translating a peer-reviewed design research article from
Brazilian Portuguese into academic English. Translate the supplied source page
faithfully and completely. Preserve headings, paragraph order, quotations,
citations, figure captions, names, URLs, and reference details. Keep the English
technical word 'pattern' wherever the source deliberately contrasts pattern,
standard, and default. Repair line-break hyphenation, but do not summarize, add
claims, invent missing text, or comment on the translation. Output only the
English translation. This is source page {page_number}.

SOURCE PAGE:
{source_text}
"""
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 20260811,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        HOST, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    return normalize(result["response"])


def get_font() -> str:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Helvetica.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ResearchSans", str(path)))
            return "ResearchSans"
    return "Helvetica"


def as_paragraphs(text: str) -> list[str]:
    values = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if block:
            block = (
                block.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            values.append(block)
    return values


def make_translation_pdf(translations: dict[str, str]) -> None:
    font = get_font()
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TranslationTitle",
        parent=styles["Title"],
        fontName=font,
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    heading = ParagraphStyle(
        "TranslationHeading",
        parent=styles["Heading1"],
        fontName=font,
        fontSize=13,
        leading=17,
        spaceAfter=10,
        textColor="#17365D",
    )
    body = ParagraphStyle(
        "TranslationBody",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9.5,
        leading=13.5,
        spaceAfter=8,
    )
    note = ParagraphStyle(
        "TranslationNote",
        parent=body,
        fontSize=9,
        leading=13,
        textColor="#555555",
        alignment=TA_CENTER,
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColorRGB(0.35, 0.35, 0.35)
        canvas.drawCentredString(letter[0] / 2, 0.42 * inch, f"English translation - {doc.page}")
        canvas.restoreState()

    TRANSLATION_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(TRANSLATION_PDF),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.65 * inch,
        title='English Translation - Considerations on "Pattern"',
        author="Andrea Graciano, Sergio Nesteriuck, Gilbertto Prado",
    )
    story = [
        Spacer(1, 1.25 * inch),
        Paragraph('Considerations on "Pattern"', title),
        Paragraph(
            "English research translation of <i>Consideracoes sobre \"pattern\"</i> "
            "(Graciano, Nesteriuck, and Prado, 2016)",
            note,
        ),
        Spacer(1, 0.35 * inch),
        Paragraph(
            "The original Portuguese PDF appears first in the combined file. "
            "This English rendering was produced with machine assistance for "
            "research screening; quotations and publication details should be "
            "checked against the original before formal citation.",
            note,
        ),
        PageBreak(),
    ]
    for page_number in range(1, len(translations) + 1):
        story.append(Paragraph(f"Translation of original page {page_number}", heading))
        for paragraph in as_paragraphs(translations[str(page_number)]):
            story.append(Paragraph(paragraph, body))
        if page_number != len(translations):
            story.append(PageBreak())
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(SOURCE))
    translations = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    translations = {key: normalize(value) for key, value in translations.items()}
    for index, page in enumerate(reader.pages, start=1):
        key = str(index)
        if key not in translations:
            source_text = normalize(page.extract_text() or "")
            translations[key] = translate_page(index, source_text)
            CHECKPOINT.write_text(
                json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Translated page {index}/{len(reader.pages)}", flush=True)

    make_translation_pdf(translations)
    writer = PdfWriter()
    writer.append(str(SOURCE))
    writer.append(str(TRANSLATION_PDF))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
