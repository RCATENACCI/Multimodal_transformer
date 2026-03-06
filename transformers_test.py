import torch
import fitz
import cv2
import whisper
import numpy as np
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import pytesseract

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# MODELS INITIALIZATION
# =========================

text_embedder = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

whisper_model = whisper.load_model("base")


# =========================
# TEXT EXTRACTION
# =========================

def extract_pdf_text(path):
    doc = fitz.open(path)
    return " ".join([page.get_text() for page in doc])


def extract_audio_text(path):
    result = whisper_model.transcribe(path)
    return result["text"]


def extract_video_text(path):
    cap = cv2.VideoCapture(path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return ""

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    image_features = clip_model.get_image_features(**inputs)

    return image_features.detach().cpu().numpy()


def extract_image_text(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    return text


# =========================
# EMBEDDING PIPELINE
# =========================

def embed_text(text):
    return text_embedder.encode(text)


def embed_image(path):
    image = Image.open(path)
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    features = clip_model.get_image_features(**inputs)
    return features.detach().cpu().numpy().flatten()


# =========================
# SENTIMENT PROJECTION
# =========================

def compute_market_sentiment(text):
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=0 if DEVICE == "cuda" else -1
    )

    result = sentiment_pipe(text[:512])[0]

    label = result["label"]
    score = result["score"]

    mapping = {
        "positive": score,
        "negative": -score,
        "neutral": 0
    }

    return mapping.get(label.lower(), 0)


# =========================
# UNIFIED PIPELINE
# =========================

def process_input(path, input_type):

    if input_type == "pdf":
        text = extract_pdf_text(path)
        embedding = embed_text(text)
        sentiment = compute_market_sentiment(text)

    elif input_type == "audio":
        text = extract_audio_text(path)
        embedding = embed_text(text)
        sentiment = compute_market_sentiment(text)

    elif input_type == "image":
        text = extract_image_text(path)
        embedding = embed_text(text)
        sentiment = compute_market_sentiment(text)

    elif input_type == "video":
        embedding = extract_video_text(path)
        text = ""
        sentiment = None

    else:
        raise ValueError("Unsupported type")

    return {
        "embedding": embedding,
        "sentiment_score": sentiment
    }


# =========================
# EXAMPLE
# =========================

if __name__ == "__main__":

    result = process_input("input/sample.pdf", "pdf")

    print("Embedding shape:", np.array(result["embedding"]).shape)
    print("Market sentiment score:", result["sentiment_score"])