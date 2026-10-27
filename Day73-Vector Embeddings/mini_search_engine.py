"""Day 73 - Mini semantic search engine (NumPy top-k) compared with keyword search."""
import argparse
import re

import numpy as np

DOCS = [
    "To reset your password, click 'Forgot password' on the sign-in page and follow the email link.",
    "Refunds are issued within 14 days of purchase if the item arrives damaged or defective.",
    "Standard shipping takes 3 to 5 business days; express shipping arrives the next day.",
    "Error ERR_4021 means the payment token has expired; request a new checkout session.",
    "Our store is open Monday to Saturday from 9 AM to 6 PM.",
    "You can change your subscription plan at any time from the billing settings.",
    "Two-factor authentication adds a second verification step when you sign in.",
    "To delete your account permanently, contact support and confirm your identity.",
    "The mobile app supports offline mode, syncing your changes when you reconnect.",
    "Gift cards never expire and can be used for any product in the store.",
    "Error ERR_5003 indicates that the server could not reach the database.",
    "Invoices are emailed on the first day of every month and can be downloaded as PDF.",
]

DEMO_QUERIES = [
    "I can't log into my account",          # paraphrase: few shared words
    "money back for a broken product",      # synonyms
    "how long until my package arrives",    # natural-language question
    "ERR_4021",                             # exact identifier
]

STOPWORDS = {"a", "an", "and", "are", "as", "can", "for", "i", "in", "into", "is", "it",
             "my", "of", "on", "or", "the", "to", "until", "your", "how", "with"}


def tokenize(text):
    return [word for word in re.findall(r"[a-z0-9_]+", text.lower()) if word not in STOPWORDS]


class KeywordSearch:
    """Score = number of distinct query words that appear in the document."""

    def __init__(self, docs):
        self.docs = docs
        self.doc_words = [set(tokenize(doc)) for doc in docs]

    def search(self, query, k=3):
        query_words = set(tokenize(query))
        scored = [(i, float(len(query_words & words))) for i, words in enumerate(self.doc_words)]
        scored = [item for item in scored if item[1] > 0]
        return sorted(scored, key=lambda item: -item[1])[:k]


class SemanticSearch:
    """Embeds documents once, then ranks them by cosine similarity to the query."""

    def __init__(self, docs, embed_fn):
        self.docs = docs
        self.embed_fn = embed_fn
        self.vectors = self._normalize(np.asarray(embed_fn(docs), dtype=float))

    @staticmethod
    def _normalize(vectors):
        norms = np.sqrt(np.sum(vectors * vectors, axis=-1, keepdims=True))
        return vectors / np.maximum(norms, 1e-12)

    def search(self, query, k=3):
        if k < 1:
            raise ValueError("k must be at least 1")
        query_vector = self._normalize(np.asarray(self.embed_fn([query]), dtype=float))[0]
        scores = self.vectors @ query_vector          # cosine similarity (vectors are normalized)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]     # top-k without sorting everything
        top = top[np.argsort(-scores[top])]           # sort only those k
        return [(int(i), float(scores[i])) for i in top]


def sentence_transformer_embed_fn(name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    return lambda texts: model.encode(list(texts))


def show(label, results, docs):
    print(f"  {label}")
    if not results:
        print("    (no results)")
    for index, score in results:
        print(f"    {score:+.3f}  {docs[index][:70]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Run one query instead of the demo queries")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    keyword = KeywordSearch(DOCS)
    semantic = SemanticSearch(DOCS, sentence_transformer_embed_fn())

    for query in [args.query] if args.query else DEMO_QUERIES:
        print(f'\nQuery: "{query}"')
        show("keyword :", keyword.search(query, args.k), DOCS)
        show("semantic:", semantic.search(query, args.k), DOCS)


if __name__ == "__main__":
    main()
