from .lexical import Hit

K = 60

def rrf(rankings: list[list[Hit]], k: int = 5, constant: int = K) -> list[Hit]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for hit in ranking:
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (constant + hit.rank)
    order = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(i, round(s, 6), rank) for rank, (i, s) in enumerate(order[:k], 1)]