import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_EMBED_MODEL = "all-mpnet-base-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
SENTIMENT_MODEL = "ProsusAI/finbert"
WHISPER_MODEL = "base"