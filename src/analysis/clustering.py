"""Clustering helpers using UMAP dimensionality reduction and HDBSCAN.

Improvement #2: Replaces the greedy single-pass clustering with the established
Embeddings → UMAP → HDBSCAN pipeline for scientifically sound clustering that
auto-determines the number of clusters and handles noise.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Cluster:
    cluster_id: int
    indices: list[int]


def cluster_embeddings(
    embeddings: list[list[float]],
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    umap_components: int = 15,
) -> list[Cluster]:
    """Cluster embeddings using UMAP dimensionality reduction + HDBSCAN.

    Args:
        embeddings: List of embedding vectors from TextEmbedder.
        min_cluster_size: Minimum number of posts to form a cluster.
            Defaults to env var HDBSCAN_MIN_CLUSTER_SIZE or 5.
        min_samples: Controls clustering conservativeness.
            Defaults to env var HDBSCAN_MIN_SAMPLES or 3.
        umap_components: Number of dimensions for UMAP reduction.

    Returns:
        List of Cluster objects. Posts that HDBSCAN labels as noise (-1)
        are grouped into a final "unclustered" cluster.
    """
    import hdbscan
    import umap

    if not embeddings:
        return []

    if min_cluster_size is None:
        min_cluster_size = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "5"))
    if min_samples is None:
        min_samples = int(os.getenv("HDBSCAN_MIN_SAMPLES", "3"))

    emb_array = np.array(embeddings, dtype=np.float32)
    n_samples = emb_array.shape[0]

    # For small datasets, UMAP spectral initialization can fail
    # (scipy eigsh error when k >= N). Fall back to greedy clustering.
    if n_samples < 20:
        LOGGER.info(
            "Only %d posts — too few for UMAP+HDBSCAN, using greedy fallback", n_samples
        )
        return cluster_embeddings_greedy(embeddings)

    # Adjust UMAP components if we have fewer features or samples
    effective_components = min(umap_components, n_samples - 1, emb_array.shape[1])
    effective_components = max(2, effective_components)

    LOGGER.info(
        "Running UMAP (%d -> %d dims) on %d embeddings...",
        emb_array.shape[1], effective_components, n_samples,
    )
    reducer = umap.UMAP(
        n_components=effective_components,
        metric="cosine",
        random_state=42,
        n_neighbors=min(15, n_samples - 1),
    )
    reduced = reducer.fit_transform(emb_array)

    LOGGER.info(
        "Running HDBSCAN (min_cluster_size=%d, min_samples=%d)...",
        min_cluster_size, min_samples,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # Build clusters from labels
    cluster_map: dict[int, list[int]] = {}
    noise_indices: list[int] = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            cluster_map.setdefault(int(label), []).append(idx)

    clusters: list[Cluster] = []
    for cluster_id in sorted(cluster_map.keys()):
        clusters.append(Cluster(cluster_id=cluster_id, indices=cluster_map[cluster_id]))

    # Group noise points into an "unclustered" bucket if any exist
    if noise_indices:
        noise_id = max(cluster_map.keys(), default=-1) + 1
        clusters.append(Cluster(cluster_id=noise_id, indices=noise_indices))
        LOGGER.info(
            "HDBSCAN found %d clusters + %d noise points (grouped as cluster %d)",
            len(clusters) - 1, len(noise_indices), noise_id,
        )
    else:
        LOGGER.info("HDBSCAN found %d clusters (no noise)", len(clusters))

    return clusters


# --- Legacy greedy clustering (kept for comparison) ---


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cluster_embeddings_greedy(embeddings: list[list[float]], threshold: float = 0.35) -> list[Cluster]:
    """Greedy clustering by cosine similarity threshold.

    This is the legacy approach kept for comparison. It is order-dependent
    and has no noise detection. Prefer cluster_embeddings() for production use.
    """
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
