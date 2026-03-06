from PIL import Image
import pytesseract


class ImageExtractor:

    @staticmethod
    def extract_text(path: str) -> str:

        image = Image.open(path)

        return pytesseract.image_to_string(image)