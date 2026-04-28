"""Clustering helpers using embedding similarity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cluster:
    cluster_id: int
    indices: list[int]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cluster_embeddings(embeddings: list[list[float]], threshold: float = 0.35) -> list[Cluster]:
    """Greedy clustering by cosine similarity threshold."""
    clusters: list[Cluster] = []

    for idx, emb in enumerate(embeddings):
        assigned = False

        for cluster in clusters:
            centroid = [0.0] * len(emb)
            for member_idx in cluster.indices:
                member = embeddings[member_idx]
                for i, value in enumerate(member):
                    centroid[i] += value
            size = len(cluster.indices)
            centroid = [value / size for value in centroid]

            if cosine_similarity(emb, centroid) >= threshold:
                cluster.indices.append(idx)
                assigned = True
                break

        if not assigned:
            clusters.append(Cluster(cluster_id=len(clusters), indices=[idx]))

    return clusters
