from PIL import Image
from multimodal_transformer.models.model_loader import ModelRegistry
from multimodal_transformer.config import DEVICE


class ImageEmbedder:

    @staticmethod
    def embed(path):

        clip_model, clip_processor = ModelRegistry.clip()

        image = Image.open(path)

        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

        features = clip_model.get_image_features(**inputs)

        return features.detach().cpu().numpy().flatten()