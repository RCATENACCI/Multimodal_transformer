import fitz


class PDFExtractor:

    @staticmethod
    def extract_text(path: str) -> str:

        doc = fitz.open(path)
        text = " ".join(page.get_text() for page in doc)

        return text