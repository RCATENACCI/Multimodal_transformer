import mimetypes

from multimodal_transformer.extractors.pdf_extractor import PDFExtractor
from multimodal_transformer.extractors.audio_extractor import AudioExtractor
from multimodal_transformer.extractors.image_extractor import ImageExtractor
from multimodal_transformer.extractors.video_extractor import VideoExtractor
from multimodal_transformer.extractors.text_extractor import TextExtractor

from multimodal_transformer.embeddings.text_embedding import TextEmbedder
from multimodal_transformer.embeddings.image_embedding import ImageEmbedder
from multimodal_transformer.sentiment.market_sentiment import MarketSentiment


class TransformService:


    @staticmethod
    def detect_type(path):

        mime, _ = mimetypes.guess_type(path)

        if mime is None:
            raise ValueError("Unknown file type")

        if "pdf" in mime:
            return "pdf"

        if "audio" in mime:
            return "audio"

        if "image" in mime:
            return "image"

        if "video" in mime:
            return "video"
        
        if "text" in mime:
            return "text"

        raise ValueError("Unsupported type")


    def process(self, path):

        file_type = self.detect_type(path)

        if file_type == "pdf":

            text = PDFExtractor.extract_text(path)
            embedding = TextEmbedder.embed(text)
            sentiment = MarketSentiment.compute(text)

        elif file_type == "text":

            text = TextExtractor.extract_text(path)
            embedding = TextEmbedder.embed(text)
            sentiment = MarketSentiment.compute(text)

        elif file_type == "audio":

            text = AudioExtractor.extract_text(path)
            embedding = TextEmbedder.embed(text)
            sentiment = MarketSentiment.compute(text)

        elif file_type == "image":

            text = ImageExtractor.extract_text(path)
            embedding = TextEmbedder.embed(text)
            sentiment = MarketSentiment.compute(text)

        elif file_type == "video":

            embedding = VideoExtractor.extract_embedding(path)
            text = ""
            sentiment = None

        return {
            "file_type": file_type,
            "embedding": embedding,
            "sentiment": sentiment
        }