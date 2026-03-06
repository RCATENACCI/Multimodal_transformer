import cv2
from PIL import Image

from multimodal_transformer.models.model_loader import ModelRegistry
from multimodal_transformer.config import DEVICE


class VideoExtractor:

    @staticmethod
    def extract_embedding(path):

        clip_model, clip_processor = ModelRegistry.clip()

        cap = cv2.VideoCapture(path)
        success, frame = cap.read()
        cap.release()

        if not success:
            return None

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

        features = clip_model.get_image_features(**inputs)

        return features.detach().cpu().numpy().flatten()