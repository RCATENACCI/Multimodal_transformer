from multimodal_transformer.models.model_loader import ModelRegistry


class TextEmbedder:

    @staticmethod
    def embed(text):

        model = ModelRegistry.text_embedder()

        return model.encode(text)