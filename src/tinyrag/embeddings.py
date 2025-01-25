from sentence_transformers import SentenceTransformer
from tinytune.llmcontext import LLMContext

import numpy

from openai import OpenAI


class EmbeddingContext(LLMContext):
    def OnRun(self, *args, **kwargs):
        tokens: list | None = kwargs.get("tokens")

        if not (tokens):
            raise Exception("Input tokens are missing")

        pass

    def __init__(self):
        return

    def __call__(self, *args, **kwargs):
        return


class OllamaEmbeddings:
    def __init__(self, url: str, model: str):
        self.Model: str = model
        self.Client = OpenAI(base_url=url, api_key="ollama")

    def Run(self, tokens: list) -> numpy.ndarray | None:
        return numpy.array(
            self.Client.embeddings.create(input=tokens, model=self.Model)
        )

    def __call__(self, tokens: list) -> numpy.ndarray | None:
        return self.Run(tokens)


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
