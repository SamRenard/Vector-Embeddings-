# Day 71 - Vector Embeddings

Cosine similarity written from scratch with NumPy, with self-tests, then
verified against `sentence-transformers` (`all-MiniLM-L6-v2`).

```bash
pip install -r requirements.txt
python cosine_from_scratch.py
```

The math: `cos(theta) = (a . b) / (|a| * |b|)`. It ranges from -1 to 1 and
compares direction, not length. If vectors are normalized to length 1, cosine
similarity is just the dot product.

Key terms: embedding, semantic space, cosine similarity, dot product, norm,
normalization.
