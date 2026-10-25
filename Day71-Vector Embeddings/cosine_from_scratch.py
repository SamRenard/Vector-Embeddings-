"""Day 71 - Cosine similarity from scratch (NumPy), then verified with sentence-transformers."""
import numpy as np


def cosine_similarity(a, b):
    """cos(theta) = (a . b) / (|a| * |b|), range -1 to 1."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm_a, norm_b = np.sqrt(np.sum(a * a)), np.sqrt(np.sum(b * b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.sum(a * b) / (norm_a * norm_b))


def cosine_matrix(queries, corpus):
    """All pairwise cosine similarities: shape (n_queries, n_corpus)."""
    q = np.atleast_2d(queries).astype(float)
    c = np.atleast_2d(corpus).astype(float)
    q_norm = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    c_norm = np.sqrt(np.sum(c * c, axis=1, keepdims=True))
    if (q_norm == 0).any() or (c_norm == 0).any():
        raise ValueError("cosine similarity is undefined for a zero vector")
    # After normalizing, cosine similarity is just a dot product.
    return (q / q_norm) @ (c / c_norm).T


def run_self_tests():
    assert abs(cosine_similarity([1, 2, 3], [1, 2, 3]) - 1) < 1e-9      # identical
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9                # orthogonal
    assert abs(cosine_similarity([1, 2], [-1, -2]) + 1) < 1e-9          # opposite
    assert abs(cosine_similarity([1, 2, 3], [2, 4, 6]) - 1) < 1e-9      # length does not matter
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3, 8)), rng.normal(size=(4, 8))
    matrix = cosine_matrix(a, b)
    for i in range(3):
        for j in range(4):
            assert abs(matrix[i, j] - cosine_similarity(a[i], b[j])) < 1e-9
    try:
        cosine_similarity([0, 0], [1, 1])
        raise AssertionError("zero vector should raise")
    except ValueError:
        pass
    print("Self-tests passed.")


def verify_with_sentence_transformers():
    from sentence_transformers import SentenceTransformer, util

    sentences = [
        "A cat is sleeping on the sofa.",
        "A kitten naps on the couch.",
        "The stock market fell sharply today.",
        "Shares dropped on Wall Street this afternoon.",
        "How do I reset my password?",
        "I forgot my login credentials.",
    ]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    print(f"Embedding shape: {embeddings.shape}")

    ours = cosine_matrix(embeddings, embeddings)
    reference = util.cos_sim(embeddings, embeddings).numpy()
    print("Matches sentence-transformers cos_sim:", np.allclose(ours, reference, atol=1e-5))

    query = "Equities declined at the market"
    scores = cosine_matrix(model.encode([query]), embeddings)[0]
    print(f'\nQuery: "{query}"')
    for index in np.argsort(-scores):
        print(f"  {scores[index]:+.3f}  {sentences[index]}")


if __name__ == "__main__":
    run_self_tests()
    try:
        verify_with_sentence_transformers()
    except ImportError:
        print("sentence-transformers not installed; skipping verification "
              "(pip install -r requirements.txt).")
