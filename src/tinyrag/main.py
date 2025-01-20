from embeddings import EmbeddingManager
from pprint import pprint

pprint(
    type(
        EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")(
            ["Hello mother fucker, is this shit fast enough"]
        )
    )
)
