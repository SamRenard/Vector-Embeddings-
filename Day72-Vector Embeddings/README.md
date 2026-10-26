# Day 72 - Embedding Models and a Similarity Heatmap

Embeds a 100-sentence corpus (5 topics x 20 sentences), builds the cosine
similarity matrix, saves it as a heatmap, and compares embedding models by how
well they separate the topics.

```bash
pip install -r requirements.txt
python similarity_heatmap.py
python similarity_heatmap.py --models all-MiniLM-L6-v2 paraphrase-MiniLM-L3-v2 --show
```

You should see 5 bright blocks along the diagonal (sentences of the same topic
are similar). Models download from the Hugging Face Hub on the first run.

Key terms: sentence-transformers, embedding model, embedding dimension,
cosine similarity matrix, normalization, within/across-topic similarity.
