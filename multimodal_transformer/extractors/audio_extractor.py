from multimodal_transformer.models.model_loader import ModelRegistry


class AudioExtractor:

    @staticmethod
    def extract_text(path: str) -> str:

        whisper_model = ModelRegistry.whisper()

        result = whisper_model.transcribe(path)

        return result["text"]