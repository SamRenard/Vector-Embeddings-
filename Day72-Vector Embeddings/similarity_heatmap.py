"""Day 72 - Embed a 100-sentence corpus and visualize the similarity matrix as a heatmap.

Compares embedding models: for each model it saves a heatmap and reports how
well the model separates the 5 topics (within-topic vs. across-topic similarity).
"""
import argparse
import time

import numpy as np

CORPUS = {
    "animals": [
        "The dog barked at the mailman.",
        "A cat curled up on the windowsill.",
        "Elephants use their trunks to drink water.",
        "The eagle soared above the mountain.",
        "Dolphins communicate with clicks and whistles.",
        "A rabbit hopped across the meadow.",
        "Bees pollinate flowers and make honey.",
        "The lion rested in the shade after the hunt.",
        "Penguins huddle together to stay warm.",
        "A horse galloped across the open field.",
        "Sharks have been swimming in oceans for millions of years.",
        "The parrot repeated every word it heard.",
        "Wolves hunt in coordinated packs.",
        "Turtles can live for more than a hundred years.",
        "A fox slipped quietly into the forest.",
        "Giraffes eat leaves from tall trees.",
        "The goldfish circled its small bowl.",
        "Bats navigate in the dark using echolocation.",
        "A herd of cows grazed on the hillside.",
        "The owl hooted softly during the night.",
    ],
    "cooking": [
        "Simmer the tomato sauce for twenty minutes.",
        "Knead the dough until it is smooth and elastic.",
        "Chop the onions and fry them in olive oil.",
        "Preheat the oven to 180 degrees before baking.",
        "Whisk the eggs with a pinch of salt.",
        "The soup needs more garlic and black pepper.",
        "Roast the vegetables until they are golden brown.",
        "Let the steak rest for five minutes after grilling.",
        "Add a spoonful of sugar to balance the acidity.",
        "Fresh basil makes the pasta taste wonderful.",
        "Marinate the chicken in lemon juice and herbs.",
        "Boil the rice until all the water is absorbed.",
        "The cake came out of the oven perfectly fluffy.",
        "Sprinkle grated cheese over the hot lasagna.",
        "Slice the cucumbers thinly for the salad.",
        "Melt the butter slowly over low heat.",
        "Stir the risotto constantly while adding broth.",
        "The bread crust should be crisp and golden.",
        "Season the fish with salt, pepper and lemon.",
        "Pour the pancake batter onto a hot pan.",
    ],
    "space": [
        "The telescope revealed a distant galaxy.",
        "Astronauts trained for months before the launch.",
        "Mars has the tallest volcano in the solar system.",
        "A black hole bends light with its gravity.",
        "The rocket reached orbit in under ten minutes.",
        "Saturn is famous for its bright rings.",
        "Comets grow tails as they approach the sun.",
        "The moon controls the ocean tides.",
        "Light from the nearest star takes over four years to reach us.",
        "The space station circles Earth every ninety minutes.",
        "A meteor streaked across the night sky.",
        "Jupiter's great red spot is a giant storm.",
        "Scientists search for water on other planets.",
        "The sun is a star at the center of our solar system.",
        "A lunar eclipse turned the moon deep red.",
        "Satellites orbit Earth to relay signals and images.",
        "Nebulae are clouds where new stars are born.",
        "The probe sent back photos from the edge of the solar system.",
        "Gravity holds the planets in their orbits.",
        "Cosmic radiation is a danger for long space missions.",
    ],
    "finance": [
        "The central bank raised interest rates again.",
        "Stock prices fell sharply after the earnings report.",
        "She opened a savings account with a high yield.",
        "Inflation reduced the purchasing power of households.",
        "The company announced a dividend for shareholders.",
        "Investors moved their money into government bonds.",
        "The startup raised ten million dollars in funding.",
        "Mortgage payments rose along with market rates.",
        "A diversified portfolio lowers investment risk.",
        "The currency lost value against the dollar.",
        "Quarterly revenue exceeded analyst expectations.",
        "He paid off the credit card debt early.",
        "The merger created one of the largest banks in the country.",
        "Oil prices influence the whole economy.",
        "Taxes are due at the end of the fiscal year.",
        "The budget deficit grew for the third year in a row.",
        "Retirement funds are invested for the long term.",
        "The auditor found errors in the balance sheet.",
        "Exports increased as the economy recovered.",
        "Crypto markets swung wildly over the weekend.",
    ],
    "programming": [
        "The function returns a list of sorted integers.",
        "A null pointer caused the application to crash.",
        "Write unit tests before you refactor the code.",
        "The server responded with a 404 error.",
        "Use a dictionary for fast key lookups.",
        "The loop runs until the queue is empty.",
        "Merge the feature branch into main after review.",
        "Python uses indentation to define code blocks.",
        "The database query was slow without an index.",
        "Docker containers package the app with its dependencies.",
        "A recursive function calls itself with a smaller input.",
        "The compiler reported a syntax error on line twelve.",
        "Deploy the API behind a load balancer.",
        "Variables should have clear and descriptive names.",
        "The memory leak slowly consumed all available RAM.",
        "Git keeps a complete history of every change.",
        "Asynchronous code avoids blocking the main thread.",
        "The regex matches every email address in the text.",
        "Cache the results to avoid repeated computation.",
        "A race condition appeared when two threads wrote together.",
    ],
}


def build_corpus():
    sentences, labels = [], []
    for index, items in enumerate(CORPUS.values()):
        sentences += items
        labels += [index] * len(items)
    return sentences, np.array(labels)


def separation_score(similarity, labels):
    """Mean similarity within topics vs. across topics (the diagonal is ignored)."""
    same_topic = labels[:, None] == labels[None, :]
    not_diagonal = ~np.eye(len(labels), dtype=bool)
    within = similarity[same_topic & not_diagonal].mean()
    across = similarity[~same_topic].mean()
    return float(within), float(across)


def plot_heatmap(similarity, labels, topics, title, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(similarity, cmap="viridis")

    sizes = np.bincount(labels)
    edges = np.cumsum(sizes)
    for edge in edges[:-1]:
        ax.axhline(edge - 0.5, color="white", linewidth=0.8)
        ax.axvline(edge - 0.5, color="white", linewidth=0.8)
    centers = edges - sizes / 2 - 0.5
    ax.set_xticks(centers)
    ax.set_xticklabels(topics, rotation=45, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(topics)

    fig.colorbar(image, ax=ax, label="cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all-MiniLM-L6-v2", "all-mpnet-base-v2"])
    parser.add_argument("--show", action="store_true", help="Also open the heatmap window")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    sentences, labels = build_corpus()
    topics = list(CORPUS)
    print(f"Corpus: {len(sentences)} sentences, {len(topics)} topics\n")

    print(f"{'model':<24}{'dim':>5}{'time (s)':>10}{'within':>9}{'across':>9}{'gap':>8}")
    print("-" * 65)
    for name in args.models:
        try:
            model = SentenceTransformer(name)
        except Exception as error:
            print(f"{name:<24} could not load: {error}")
            continue
        start = time.perf_counter()
        # Normalized vectors: the dot product equals cosine similarity.
        embeddings = model.encode(sentences, normalize_embeddings=True)
        seconds = time.perf_counter() - start

        similarity = embeddings @ embeddings.T
        within, across = separation_score(similarity, labels)
        print(f"{name:<24}{embeddings.shape[1]:>5}{seconds:>10.1f}{within:>9.3f}{across:>9.3f}{within - across:>8.3f}")

        path = f"heatmap_{name.replace('/', '_')}.png"
        plot_heatmap(similarity, labels, topics, f"{name} - cosine similarity", path, args.show)
        print(f"  saved {path}")

    print("\nA larger gap (within - across) means topics are separated more clearly.")


if __name__ == "__main__":
    main()
