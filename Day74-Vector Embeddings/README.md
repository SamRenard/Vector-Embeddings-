# Day 74 - Chunking Strategies

Splits the same document three ways and compares retrieval quality:

- **fixed-size**: cut every N characters (with overlap), ignores meaning
- **recursive**: split on paragraph, then line, then sentence, then word boundaries
- **semantic**: embed sentences and start a new chunk where the topic shifts

```bash
pip install -r requirements.txt
python compare_chunking.py                                  # built-in demo document
python compare_chunking.py --pdf my.pdf --questions q.json  # your own PDF
```

`q.json` is a list of questions with a short exact phrase from the PDF that
answers each one:

```json
[{"question": "How long is the warranty?", "answer": "two-year limited warranty"}]
```

A chunk "hits" if it contains that phrase. Try `--chunk-size`, `--overlap`,
`--k` and `--percentile` to see how results change. Ten questions is a small
sample, so treat results as a rough signal.

Key terms: chunking, overlap, recursive splitting, semantic chunking,
hit@k, MRR, context size.
