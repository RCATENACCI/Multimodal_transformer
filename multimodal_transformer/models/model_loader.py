import whisper
from transformers import CLIPModel, CLIPProcessor, pipeline
from sentence_transformers import SentenceTransformer

from multimodal_transformer.config import DEVICE, TEXT_EMBED_MODEL, CLIP_MODEL, SENTIMENT_MODEL, WHISPER_MODEL


class ModelRegistry:

    _text_embedder = None
    _clip_model = None
    _clip_processor = None
    _whisper = None
    _sentiment = None

    @classmethod
    def text_embedder(cls):

        if cls._text_embedder is None:
            cls._text_embedder = SentenceTransformer(TEXT_EMBED_MODEL, device=DEVICE)

        return cls._text_embedder


    @classmethod
    def clip(cls):

        if cls._clip_model is None:

            cls._clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
            cls._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

        return cls._clip_model, cls._clip_processor


    @classmethod
    def whisper(cls):

        if cls._whisper is None:
            cls._whisper = whisper.load_model(WHISPER_MODEL)

        return cls._whisper


    @classmethod
    def sentiment(cls):

        if cls._sentiment is None:

            cls._sentiment = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL,
                device=0 if DEVICE == "cuda" else -1
            )

        return cls._sentiment