from sentence_transformers import SentenceTransformer
import numpy


class EmbeddingManager:
    def __init__(self, model_name):
        self.Model = SentenceTransformer(model_name)

    def Generate(self, tokens: list[str]) -> numpy.ndarray:
        try:
            return self.Model.encode(tokens)
        except:
            raise

    def __call__(self, tokens: list[str]) -> numpy.ndarray:
        return self.Generate(tokens)
