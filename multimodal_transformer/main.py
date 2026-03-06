import numpy as np

from service.transform_service import TransformService


if __name__ == "__main__":

    service = TransformService()

    result = service.process("input/sample.pdf")

    print("Detected type:", result["file_type"])
    print("Embedding shape:", np.array(result["embedding"]).shape)
    print("Sentiment score:", result["sentiment"])