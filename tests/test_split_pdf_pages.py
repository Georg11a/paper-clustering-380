import sys
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader, PdfWriter


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from split_pdf_pages import split_pdf  # noqa: E402


class SplitPdfPagesTests(unittest.TestCase):
    def test_extracts_inclusive_one_based_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            output = root / "output.pdf"
            writer = PdfWriter()
            for _ in range(6):
                writer.add_blank_page(width=200, height=300)
            with source.open("wb") as stream:
                writer.write(stream)

            count = split_pdf(source, output, 2, 5)

            self.assertEqual(count, 4)
            self.assertEqual(len(PdfReader(str(output)).pages), 4)
            self.assertEqual(len(PdfReader(str(source)).pages), 6)

    def test_refuses_to_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pdf"
            output = root / "output.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=300)
            with source.open("wb") as stream:
                writer.write(stream)
            with output.open("wb") as stream:
                writer.write(stream)

            with self.assertRaises(FileExistsError):
                split_pdf(source, output, 1, 1)


if __name__ == "__main__":
    unittest.main()
