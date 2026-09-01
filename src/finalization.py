import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from data_loader import DataLoader



loader = DataLoader()

df = loader.load_data(
    "data/processed/temiz_veri.csv"
)

text_df = df[df["abstract"].notna()].copy()

text_df["text"] = (
    text_df["title"].fillna("")
    + " "
    + text_df["abstract"]
)



vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

X_tfidf = vectorizer.fit_transform(
    text_df["text"]
)

tfidf_kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto"
)

tfidf_labels = tfidf_kmeans.fit_predict(
    X_tfidf
)

tfidf_silhouette = silhouette_score(
    X_tfidf,
    tfidf_labels
)

print("TF-IDF")
print("-" * 40)
print("Shape:", X_tfidf.shape)
print("Silhouette:", tfidf_silhouette)



model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

X_embeddings = model.encode(
    text_df["text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

# Avoid numerical issues in sklearn operations
X_embeddings = X_embeddings.astype(np.float64)

print()
print("Embeddings")
print("-" * 40)
print("Shape:", X_embeddings.shape)



embedding_kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto"
)

embedding_labels = embedding_kmeans.fit_predict(
    X_embeddings
)

embedding_silhouette = silhouette_score(
    X_embeddings,
    embedding_labels
)

print("Embedding Silhouette:", embedding_silhouette)



text_df["cluster"] = embedding_labels

print()
print("Cluster sizes:")
print(
    text_df["cluster"]
    .value_counts()
    .sort_index()
)


pca = PCA(
    n_components=2,
    random_state=42
)

X_2d = pca.fit_transform(
    X_embeddings
)

plt.figure(figsize=(10, 7))

plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=embedding_labels,
    s=5,
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("ISEF Projects - Semantic Clusters")

plt.show()



def semantic_search(
    query: str,
    model,
    embeddings,
    dataframe,
    top_k: int = 10
):
    """
    Find projects semantically similar to a query.
    """

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float64)

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(
        similarities
    )[-top_k:][::-1]

    results = dataframe.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarities[
        top_indices
    ]

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


query = "machine learning for detecting cancer"

results = semantic_search(
    query,
    model,
    X_embeddings,
    text_df,
    top_k=10
)

print()
print("Semantic Search Results")
print("-" * 40)

print(results.to_string(index=False))

X_embeddings = model.encode(
    text_df["text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

X_embeddings = X_embeddings.astype(np.float64)

np.save("models/embeddings.npy", X_embeddings)
text_df.to_csv("data/processed/search_data.csv", index=False)