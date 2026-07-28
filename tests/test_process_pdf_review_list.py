import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from process_pdf_review_list import integer_page  # noqa: E402


class ProcessPdfReviewListTests(unittest.TestCase):
    def test_integer_page_accepts_csv_float(self) -> None:
        self.assertEqual(integer_page("40.0"), 40)

    def test_integer_page_rejects_empty_value(self) -> None:
        with self.assertRaises(ValueError):
            integer_page("")


if __name__ == "__main__":
    unittest.main()
