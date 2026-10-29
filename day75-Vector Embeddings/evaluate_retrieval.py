"""Day 75 - Evaluate a search engine with a 20-question test set: recall@k, precision@k, MRR.

Compares keyword search and semantic search on the same test set, then lists
the questions each one missed so you can see WHY it failed.
"""
import argparse
import json
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

# "relevant" = indexes of the documents that correctly answer the question.
TESTSET = [
    {"question": "How do I recover my password?", "relevant": [0]},
    {"question": "Can I get my money back for a defective item?", "relevant": [1]},
    {"question": "How many days do I have to ask for a refund?", "relevant": [1]},
    {"question": "How long does delivery take?", "relevant": [2]},
    {"question": "Is there next-day delivery?", "relevant": [2]},
    {"question": "What does error ERR_4021 mean?", "relevant": [3]},
    {"question": "Why was my payment rejected because of an expired token?", "relevant": [3]},
    {"question": "When is the store open?", "relevant": [4]},
    {"question": "Are you open on Sundays?", "relevant": [4]},
    {"question": "How do I switch to a different subscription?", "relevant": [5]},
    {"question": "Where can I find billing settings?", "relevant": [5]},
    {"question": "How do I add extra security when signing in?", "relevant": [6]},
    {"question": "How can I permanently remove my account?", "relevant": [7]},
    {"question": "Does the app work without internet?", "relevant": [8]},
    {"question": "Do gift cards have an expiration date?", "relevant": [9]},
    {"question": "The server can't connect to the database, what error is that?", "relevant": [10]},
    {"question": "Where do I download my invoices?", "relevant": [11]},
    {"question": "Which error codes exist?", "relevant": [3, 10]},
    {"question": "How do I manage billing and invoices?", "relevant": [5, 11]},
    {"question": "How can I protect my account from unauthorized access?", "relevant": [6]},
]

STOPWORDS = {"a", "an", "and", "are", "as", "can", "do", "does", "for", "how", "i", "in", "is",
             "it", "my", "of", "on", "or", "the", "to", "what", "when", "where", "which", "you", "your"}


# ----------------------------------------------------------------- metrics --

def metrics_for_query(retrieved, relevant, ks):
    """
    retrieved: ranked list of doc ids (best first)
    relevant:  doc ids that are correct answers

    recall@k    = relevant docs found in top k / all relevant docs
    precision@k = relevant docs found in top k / k
    hit@k       = 1 if at least one relevant doc is in the top k
    rr          = 1 / rank of the first relevant doc (0 if none was retrieved)
    """
    relevant = set(relevant)
    if not relevant:
        raise ValueError("every question needs at least one relevant document")
    result = {}
    for k in ks:
        found = len(relevant & set(retrieved[:k]))
        result[f"recall@{k}"] = found / len(relevant)
        result[f"precision@{k}"] = found / k
        result[f"hit@{k}"] = float(found > 0)
    rank = next((r for r, doc in enumerate(retrieved, 1) if doc in relevant), None)
    result["rr"] = 1 / rank if rank else 0.0
    return result


def evaluate(search_fn, testset, ks=(1, 3, 5)):
    """Run every question through search_fn(query, k) -> ranked doc ids; average the metrics."""
    per_query = []
    for item in testset:
        retrieved = search_fn(item["question"], max(ks))
        per_query.append((item, retrieved, metrics_for_query(retrieved, item["relevant"], ks)))
    names = per_query[0][2].keys()
    summary = {name: float(np.mean([m[name] for _, _, m in per_query])) for name in names}
    summary["mrr"] = summary.pop("rr")
    return summary, per_query


# ----------------------------------------------------------- search engines --

def tokenize(text):
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS}


def keyword_search_fn(docs):
    doc_words = [tokenize(doc) for doc in docs]

    def search(query, k):
        words = tokenize(query)
        scored = [(i, len(words & dw)) for i, dw in enumerate(doc_words)]
        scored = [s for s in scored if s[1] > 0]  # only docs sharing at least one word
        return [i for i, _ in sorted(scored, key=lambda s: -s[1])[:k]]

    return search


def semantic_search_fn(docs, embed_fn):
    def normalize(v):
        v = np.asarray(v, dtype=float)
        return v / np.maximum(np.sqrt(np.sum(v * v, axis=-1, keepdims=True)), 1e-12)

    vectors = normalize(embed_fn(docs))

    def search(query, k):
        scores = vectors @ normalize(embed_fn([query]))[0]
        return [int(i) for i in np.argsort(-scores)[:k]]

    return search


def sentence_transformer_embed_fn(name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    return lambda texts: model.encode(list(texts))


# ------------------------------------------------------------------ report --

def print_table(results, ks):
    columns = [f"recall@{k}" for k in ks] + [f"precision@{ks[-1]}", "mrr"]
    print(f"\n{'engine':<12}" + "".join(f"{c:>13}" for c in columns))
    print("-" * (12 + 13 * len(columns)))
    for name, summary in results.items():
        print(f"{name:<12}" + "".join(f"{summary[c]:>13.2f}" for c in columns))


def print_misses(name, per_query, docs, k):
    misses = [(item, retrieved) for item, retrieved, m in per_query if m[f"recall@{k}"] < 1]
    print(f"\n{name}: {len(misses)} of {len(per_query)} questions with recall@{k} < 1")
    for item, retrieved in misses:
        print(f'  Q: {item["question"]}')
        print(f'     expected {item["relevant"]}, got {retrieved[:k]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", help="JSON list of document strings (default: built-in docs)")
    parser.add_argument("--testset", help='JSON list: [{"question": "...", "relevant": [doc indexes]}]')
    parser.add_argument("--k", type=int, default=5, help="Largest k to evaluate (default 5)")
    args = parser.parse_args()

    docs, testset = DOCS, TESTSET
    if args.docs:
        with open(args.docs, encoding="utf-8") as handle:
            docs = json.load(handle)
    if args.testset:
        with open(args.testset, encoding="utf-8") as handle:
            testset = json.load(handle)

    ks = sorted({1, min(3, args.k), args.k})
    print(f"{len(docs)} documents, {len(testset)} test questions")

    engines = {
        "keyword": keyword_search_fn(docs),
        "semantic": semantic_search_fn(docs, sentence_transformer_embed_fn()),
    }
    results, details = {}, {}
    for name, search_fn in engines.items():
        results[name], details[name] = evaluate(search_fn, testset, ks)

    print_table(results, ks)
    for name in engines:
        print_misses(name, details[name], docs, args.k)


if __name__ == "__main__":
    main()
