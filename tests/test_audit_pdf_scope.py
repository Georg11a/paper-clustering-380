import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_pdf_scope import (  # noqa: E402
    classify_scope,
    normalize_text,
    text_extraction_quality,
    title_similarity,
)


class PdfScopeAuditTests(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(
            normalize_text("Towards a Design Theory—of Blended Learning"),
            "towards a design theory of blended learning",
        )

    def test_title_similarity_accepts_case_and_punctuation(self) -> None:
        score = title_similarity(
            "Towards a Design Theory of Blended Learning Curriculum",
            "Towards a design theory of blended-learning curriculum",
        )
        self.assertGreaterEqual(score, 0.95)

    def test_large_pdf_with_outline_becomes_chapter_candidate(self) -> None:
        status, start, end, _ = classify_scope(
            page_count=485,
            outline_count=87,
            outline_score=1.0,
            outline_start=78,
            outline_end=90,
            title_page=78,
            title_score=1.0,
        )
        self.assertEqual(status, "chapter_candidate")
        self.assertEqual((start, end), (78, 90))

    def test_small_pdf_with_front_title_is_single_paper(self) -> None:
        status, start, end, _ = classify_scope(
            page_count=14,
            outline_count=0,
            outline_score=0.0,
            outline_start=None,
            outline_end=None,
            title_page=1,
            title_score=1.0,
        )
        self.assertEqual(status, "single_paper_likely")
        self.assertEqual((start, end), (1, 14))

    def test_article_section_bookmarks_do_not_imply_container(self) -> None:
        status, start, end, _ = classify_scope(
            page_count=20,
            outline_count=8,
            outline_score=0.25,
            outline_start=12,
            outline_end=12,
            title_page=1,
            title_score=1.0,
        )
        self.assertEqual(status, "single_paper_likely")
        self.assertEqual((start, end), (1, 20))

    def test_control_characters_flag_suspect_encoding(self) -> None:
        status, ratio = text_extraction_quality(["Title\x02\x13\x09 body text"])
        self.assertEqual(status, "suspect_encoding")
        self.assertGreater(ratio, 0)


if __name__ == "__main__":
    unittest.main()
