import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from export_springer_pdf_list import (  # noqa: E402
    cutting_priority,
    is_springer_book_chapter_doi,
    is_springer_family,
)


class SpringerPdfListTests(unittest.TestCase):
    def test_springer_family_doi_prefixes(self) -> None:
        self.assertTrue(is_springer_family("10.1007/s00163-021-00362-z", ""))
        self.assertTrue(is_springer_family("10.1038/s41586-020-2649-2", ""))
        self.assertFalse(is_springer_family("10.1145/123.456", ""))

    def test_book_chapter_doi(self) -> None:
        self.assertTrue(is_springer_book_chapter_doi("10.1007/978-3-540-85170-7_6"))
        self.assertFalse(is_springer_book_chapter_doi("10.1007/s00163-021-00362-z"))

    def test_book_chapter_gets_high_priority(self) -> None:
        row = pd.Series(
            {
                "doi": "10.1007/978-3-540-85170-7_6",
                "page_count": 13,
                "scope_status": "single_paper_likely",
            }
        )
        priority, action, _ = cutting_priority(row)
        self.assertEqual(priority, "medium")
        self.assertEqual(action, "verify_existing_chapter")

    def test_container_chapter_gets_cut_candidate_action(self) -> None:
        row = pd.Series(
            {
                "doi": "10.1007/978-3-540-85170-7_6",
                "page_count": 485,
                "scope_status": "chapter_candidate",
            }
        )
        priority, action, _ = cutting_priority(row)
        self.assertEqual(priority, "high")
        self.assertEqual(action, "cut_candidate")


if __name__ == "__main__":
    unittest.main()
