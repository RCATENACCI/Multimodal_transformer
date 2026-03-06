from multimodal_transformer.models.model_loader import ModelRegistry


class MarketSentiment:

    @staticmethod
    def compute(text):

        pipe = ModelRegistry.sentiment()

        result = pipe(text[:512])[0]

        label = result["label"].lower()
        score = result["score"]

        mapping = {
            "positive": score,
            "negative": -score,
            "neutral": 0
        }

        return mapping.get(label, 0)