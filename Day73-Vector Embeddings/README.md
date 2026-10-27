# Day 73 - Semantic Search vs Keyword Search

A mini search engine: documents are embedded once, the query is embedded at
search time, and the top-k results are found with a NumPy dot product
(`np.argpartition` picks the top k without sorting everything). A simple
keyword search is included for comparison.

```bash
pip install -r requirements.txt
python mini_search_engine.py
python mini_search_engine.py --query "how do I cancel my plan" --k 3
```

When to use which:

| | Keyword search | Semantic search |
|---|---|---|
| Good at | exact IDs, error codes, names, rare terms | paraphrases, synonyms, natural-language questions |
| Weak at | different wording for the same meaning | exact strings, codes, unusual terms |
| Other | fast, easy to explain | needs an embedding model |

In practice the best systems combine both (hybrid search).

Key terms: semantic search, keyword search, top-k retrieval, cosine similarity,
argpartition, hybrid search.
