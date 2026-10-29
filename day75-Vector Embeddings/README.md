# Day 75 - Retrieval Metrics (recall@k, MRR)

Measures how good a search engine is, instead of guessing. A test set of 20
questions (each with the document(s) that should be found) is run through
keyword search and semantic search, and both are scored.

```bash
pip install -r requirements.txt
python evaluate_retrieval.py
python evaluate_retrieval.py --docs my_docs.json --testset my_testset.json --k 5
```

Metrics (per question, then averaged):

- **recall@k**: relevant documents found in the top k / all relevant documents
- **precision@k**: relevant documents found in the top k / k
- **MRR**: mean of 1 / (rank of the first relevant document), 0 if not found

Example: relevant = {5}, retrieved = [2, 5, 7]. recall@1 = 0, recall@3 = 1,
precision@3 = 1/3, reciprocal rank = 1/2.

Your own search engine: put your documents in a JSON list of strings and write
20+ questions in the format below. `relevant` holds document indexes (position
in the list, starting at 0).

```json
[{"question": "How do I recover my password?", "relevant": [0]}]
```

Tips: write questions the way real users would (do not copy words from the
document), include some questions with several relevant documents, and read the
"missed questions" list to understand failures.

Key terms: recall@k, precision@k, MRR, relevance judgments, test set, ground truth.
