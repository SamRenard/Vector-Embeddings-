"""Day 74 - Chunk the same document 3 ways (fixed-size, recursive, semantic) and compare retrieval.

Metrics per method:
    hit@k  share of questions whose answer text appears in at least one of the top-k chunks
    MRR    mean reciprocal rank of the first chunk containing the answer
    ctx    average characters retrieved per question (less is cheaper for the LLM)
"""
import argparse
import json
import re

import numpy as np

DEMO_TEXT = "\n\n".join([
    "The Nimbus X1 is a dual-band Wi-Fi router designed for small homes and apartments. It supports speeds up to 1200 Mbps and covers roughly 100 square meters.",
    "To set up the router, connect the power adapter and plug an Ethernet cable from your modem into the blue WAN port. Wait two minutes until the status light turns solid green.",
    "Open a browser and visit 192.168.8.1 to reach the admin page. The default username is admin and the default password is printed on the label under the device. Change the password during the first login.",
    "The router has a physical reset button next to the power port. To restore factory settings, hold the button for ten seconds until all lights blink once. This erases your Wi-Fi name, password and every custom setting.",
    "Firmware updates are released every three months. Open the Maintenance tab, choose Firmware Update and click Check Now. Do not unplug the router during an update, because this can damage the device.",
    "A solid red status light means the router has no internet connection. Restart your modem first, then restart the router. If the light stays red, check that the Ethernet cable is firmly connected to the WAN port.",
    "A blinking orange light means the router is installing an update or starting up. This usually lasts less than three minutes and no action is needed.",
    "The guest network lets visitors use the internet without access to your files and devices. Enable it in the Wireless tab and set a separate password. Guest access can be limited to a fixed number of hours.",
    "Parental controls allow you to block websites by category and set daily internet schedules for each device. Schedules are configured in the Family tab using the device's MAC address.",
    "For the best signal, place the router in a central, elevated position away from microwaves, cordless phones and thick concrete walls. Metal shelves and mirrors also reduce the range.",
    "The Nimbus X1 comes with a two-year limited warranty that covers manufacturing defects. Damage caused by water, power surges or unauthorized repairs is not covered. Keep your receipt as proof of purchase.",
    "To contact support, open a ticket on the support portal and include the serial number printed on the label. Support is available on weekdays from 9:00 to 18:00 and replies within one business day.",
])

DEMO_QUESTIONS = [
    {"question": "How long do I hold the reset button to restore factory settings?", "answer": "hold the button for ten seconds"},
    {"question": "What does a solid red light mean?", "answer": "no internet connection"},
    {"question": "What is the address of the admin page?", "answer": "192.168.8.1"},
    {"question": "How long is the warranty?", "answer": "two-year limited warranty"},
    {"question": "How often are firmware updates released?", "answer": "every three months"},
    {"question": "How quickly does support reply?", "answer": "within one business day"},
    {"question": "Why is the light blinking orange?", "answer": "installing an update"},
    {"question": "How can I block websites for my kids?", "answer": "block websites by category"},
    {"question": "Where should I place the router for the best signal?", "answer": "central, elevated position"},
    {"question": "How do I let visitors use Wi-Fi safely?", "answer": "without access to your files"},
]

SEPARATORS = ("\n\n", "\n", ". ", " ")


# ---------------------------------------------------------------- chunkers --

def fixed_size_chunks(text, size=500, overlap=50):
    """Cut every `size` characters, overlapping by `overlap`. Ignores meaning completely."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        if start + size >= len(text):
            break
        start += size - overlap
    return [chunk for chunk in chunks if chunk]


def _merge(pieces, separator, size):
    """Greedily glue small pieces back together up to `size` characters."""
    chunks, current = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def recursive_chunks(text, size=500, separators=SEPARATORS):
    """Split on the biggest natural boundary (paragraph, line, sentence, word) that fits."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    for index, separator in enumerate(separators):
        if separator in text:
            pieces = []
            for part in text.split(separator):
                if len(part) <= size:
                    pieces.append(part)
                else:
                    pieces.extend(recursive_chunks(part, size, separators[index + 1:]))
            return _merge(pieces, separator, size)
    return fixed_size_chunks(text, size, 0)  # no separator left: hard cut


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.sqrt(np.sum(vectors * vectors, axis=-1, keepdims=True))
    return vectors / np.maximum(norms, 1e-12)


def semantic_chunks(text, embed_fn, breakpoint_percentile=25, max_chars=800):
    """Start a new chunk where adjacent sentences are least similar (topic shift)."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return sentences
    vectors = normalize(embed_fn(sentences))
    similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(similarities, breakpoint_percentile)

    chunks, current = [], [sentences[0]]
    for index, similarity in enumerate(similarities):
        next_sentence = sentences[index + 1]
        too_long = len(" ".join(current + [next_sentence])) > max_chars
        if similarity < threshold or too_long:
            chunks.append(" ".join(current))
            current = [next_sentence]
        else:
            current.append(next_sentence)
    chunks.append(" ".join(current))
    return chunks


# -------------------------------------------------------------- evaluation --

def squash(text):
    return " ".join(text.lower().split())


def evaluate(chunks, questions, embed_fn, k=3):
    vectors = normalize(embed_fn(chunks))
    hits, reciprocal_ranks, context_sizes = 0, [], []
    for item in questions:
        query = normalize(embed_fn([item["question"]]))[0]
        top = np.argsort(-(vectors @ query))[:k]
        context_sizes.append(sum(len(chunks[i]) for i in top))
        answer = squash(item["answer"])
        rank = next((r for r, i in enumerate(top, 1) if answer in squash(chunks[i])), None)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
    return {
        "chunks": len(chunks),
        "avg_chars": float(np.mean([len(c) for c in chunks])),
        "hit": hits / len(questions),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ctx": float(np.mean(context_sizes)),
    }


def run_comparison(text, questions, embed_fn, size=300, overlap=30, k=3, percentile=25):
    methods = {
        "fixed-size": fixed_size_chunks(text, size, overlap),
        "recursive": recursive_chunks(text, size),
        "semantic": semantic_chunks(text, embed_fn, percentile, max_chars=size * 2),
    }
    return {name: evaluate(chunks, questions, embed_fn, k) for name, chunks in methods.items()}


def print_table(results, k):
    print(f"\n{'method':<12}{'chunks':>8}{'avg chars':>11}{f'hit@{k}':>9}{'MRR':>7}{'ctx chars':>11}")
    print("-" * 58)
    for name, r in results.items():
        print(f"{name:<12}{r['chunks']:>8}{r['avg_chars']:>11.0f}{r['hit']:>9.0%}{r['mrr']:>7.2f}{r['ctx']:>11.0f}")


def read_pdf(path):
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Your own PDF (needs --questions)")
    parser.add_argument("--questions", help='JSON list: [{"question": "...", "answer": "exact text from the PDF"}]')
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--percentile", type=int, default=25, help="Semantic split sensitivity (higher = more splits)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    if args.pdf:
        if not args.questions:
            parser.error("--pdf needs --questions")
        text = read_pdf(args.pdf)
        with open(args.questions, encoding="utf-8") as handle:
            questions = json.load(handle)
    else:
        text, questions = DEMO_TEXT, DEMO_QUESTIONS
        print("No --pdf given: using the built-in demo document.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    embed_fn = lambda texts: model.encode(list(texts))

    results = run_comparison(text, questions, embed_fn, args.chunk_size, args.overlap, args.k, args.percentile)
    print_table(results, args.k)
    print("\nhit@k: answer found in the top-k chunks | ctx: avg characters sent to the LLM per question")


if __name__ == "__main__":
    main()
