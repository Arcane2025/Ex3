"""
Implementing K-means clustering on documents with cosine similarity.

- We generate 15 documents represented as tf-idf vectors over 10 sports-related terms.
- If a term appears in a document, it gets a random tf-idf weight in [2, 6]; otherwise 0.
- We then run K-means with K=2 using cosine similarity (via L2-normalized vectors).
- Finally, we print the 5 most significant features (terms) per cluster prototype.
"""

import numpy as np
import pandas as pd


def generate_documents(n_docs=15, random_seed=42):
    terms = [
        "team", "coach", "hockey", "baseball", "soccer",
        "penalty", "score", "win", "loss", "season"
    ]
    rng = np.random.default_rng(random_seed)
    n_terms = len(terms)

    # Random 0/1 presence matrix
    presence = (rng.random((n_docs, n_terms)) < 0.45).astype(float)

    # Ensure each document has at least one term
    for i in range(n_docs):
        if presence[i].sum() == 0:
            j = rng.integers(0, n_terms)
            presence[i, j] = 1.0

    # Ensure each term appears in at least one document
    for j in range(n_terms):
        if presence[:, j].sum() == 0:
            i = rng.integers(0, n_docs)
            presence[i, j] = 1.0

    # For each non-zero cell, assign a tf-idf weight between 2 and 6
    weights = rng.uniform(2.0, 6.0, size=(n_docs, n_terms))
    tfidf = presence * weights

    docs = [f"D{i+1}" for i in range(n_docs)]
    return docs, terms, tfidf


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def kmeans_cosine(X, k=2, max_iter=100, tol=1e-4, random_seed=123):
    """
    K-means with cosine similarity:
    - We normalize rows of X to unit length.
    - Similarity is dot product between normalized docs and centroids.
    """
    rng = np.random.default_rng(random_seed)
    X_norm = l2_normalize_rows(X)
    n_docs, n_features = X_norm.shape

    # Initialize centroids as random distinct documents
    init_indices = rng.choice(n_docs, size=k, replace=False)
    centroids = X_norm[init_indices].copy()

    for it in range(max_iter):
        # Cosine similarity = dot product of normalized vectors
        sims = X_norm @ centroids.T  # shape: (n_docs, k)
        labels = np.argmax(sims, axis=1)

        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = X_norm[labels == c]
            if len(members) == 0:
                # Reinitialize empty cluster to a random doc
                idx = rng.integers(0, n_docs)
                new_centroids[c] = X_norm[idx]
            else:
                centroid = members.mean(axis=0)
                # Normalize centroid
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                new_centroids[c] = centroid

        shift = np.linalg.norm(centroids - new_centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids


def main():
    # A. Input preparation
    docs, terms, tfidf = generate_documents()
    df_input = pd.DataFrame(tfidf, index=docs, columns=terms)
    print("===== INPUT: TF-IDF matrix (Bag-of-Words over sports terms) =====")
    print(df_input)

    # B. K-means clustering with K=2 and cosine similarity
    labels, centroids = kmeans_cosine(tfidf, k=2)

    print("\n===== CLUSTER ASSIGNMENT (K=2) =====")
    for doc, lab in zip(docs, labels):
        print(f"{doc} -> Cluster {lab}")

    # C. Display 5 most significant features in each prototype
    print("\n===== CLUSTER PROTOTYPES: TOP 5 FEATURES PER CLUSTER =====")
    for c_idx, centroid in enumerate(centroids):
        # Get indices of top 5 weights in centroid
        top_idx = np.argsort(centroid)[::-1][:5]
        top_terms = [(terms[i], centroid[i]) for i in top_idx]
        print(f"Cluster {c_idx}:")
        for term, weight in top_terms:
            print(f"  {term:8s}  weight={weight:.4f}")
        print()


if __name__ == "__main__":
    main()
