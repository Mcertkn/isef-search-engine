import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearcher:

    def __init__(
        self,
        model_name: str,
        embeddings_path: str,
        data_path: str
    ):
        self.model = SentenceTransformer(model_name)

        self.embeddings = np.load(embeddings_path)

        self.data = pd.read_csv(data_path)

    def search(self, query: str, top_k: int = 10):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float64)

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = self.data.iloc[top_indices].copy()

        results["similarity"] = similarities[top_indices]

        return results[
            [
                "id",
                "title",
                "category",
                "year",
                "country",
                "similarity"
            ]
        ]