"""Clustering helpers using UMAP dimensionality reduction and HDBSCAN.

Improvement #2: Replaces the greedy single-pass clustering with the established
Embeddings → UMAP → HDBSCAN pipeline for scientifically sound clustering that
auto-determines the number of clusters and handles noise.

Tuning update: Parameters optimised for social-listening use cases where posts
share a common search term and need stable, inspectable subtopic separation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Cluster:
    cluster_id: int
    indices: list[int]


@dataclass
class ClusterDiagnostics:
    """Diagnostics about the clustering result."""
    cluster_count: int
    cluster_sizes: list[int]
    largest_cluster_ratio: float
    noise_count: int
    total_posts: int
    warning: str | None = None
    umap_coordinates: list[list[float]] = field(default_factory=list)
    coordinate_method: str = "umap_2d"


def cluster_embeddings(
    embeddings: list[list[float]],
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    umap_components: int = 5,
) -> tuple[list[Cluster], ClusterDiagnostics]:
    """Cluster embeddings using UMAP dimensionality reduction + HDBSCAN.

    Args:
        embeddings: List of embedding vectors from TextEmbedder.
        min_cluster_size: Minimum number of posts to form a cluster.
            Defaults to env var HDBSCAN_MIN_CLUSTER_SIZE or 6.
        min_samples: Controls clustering conservativeness.
            Defaults to env var HDBSCAN_MIN_SAMPLES or 2.
        umap_components: Number of dimensions for UMAP reduction.

    Returns:
        Tuple of (clusters, diagnostics).
        Posts that HDBSCAN labels as noise (-1) are grouped into a final
        "unclustered" cluster.
    """
    import hdbscan
    import umap

    if not embeddings:
        diag = ClusterDiagnostics(0, [], 0.0, 0, 0)
        return [], diag

    if min_cluster_size is None:
        min_cluster_size = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "6"))
    if min_samples is None:
        min_samples = int(os.getenv("HDBSCAN_MIN_SAMPLES", "2"))

    emb_array = np.array(embeddings, dtype=np.float32)
    n_samples = emb_array.shape[0]

    # For small datasets, UMAP spectral initialization can fail
    # (scipy eigsh error when k >= N). Fall back to greedy clustering.
    if n_samples < 20:
        LOGGER.info(
            "Only %d posts — too few for UMAP+HDBSCAN, using greedy fallback", n_samples
        )
        clusters = cluster_embeddings_greedy(embeddings)
        sizes = [len(c.indices) for c in clusters]
        max_ratio = max(sizes) / n_samples if sizes else 0.0
        diag = ClusterDiagnostics(
            cluster_count=len(clusters),
            cluster_sizes=sizes,
            largest_cluster_ratio=max_ratio,
            noise_count=0,
            total_posts=n_samples,
            umap_coordinates=_fallback_embedding_coordinates(emb_array),
            coordinate_method="embedding_projection_fallback",
        )
        return clusters, diag

    # Adjust UMAP components if we have fewer features or samples
    effective_components = min(umap_components, n_samples - 1, emb_array.shape[1])
    effective_components = max(2, effective_components)

    # Use tighter n_neighbors for better local structure preservation
    n_neighbors = min(10, n_samples - 1)

    LOGGER.info(
        "Running UMAP (%d -> %d dims, n_neighbors=%d) on %d embeddings...",
        emb_array.shape[1], effective_components, n_neighbors, n_samples,
    )
    reducer = umap.UMAP(
        n_components=effective_components,
        metric="cosine",
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=0.0,  # Allow tighter clusters in UMAP space
    )
    reduced = reducer.fit_transform(emb_array)
    visualization_coordinates = _build_umap_visualization_coordinates(
        emb_array,
        reduced,
        n_neighbors=n_neighbors,
    )

    cluster_selection_method = os.getenv("HDBSCAN_CLUSTER_SELECTION_METHOD", "eom").strip().lower()
    if cluster_selection_method not in {"eom", "leaf"}:
        LOGGER.warning(
            "Invalid HDBSCAN_CLUSTER_SELECTION_METHOD=%s; using eom",
            cluster_selection_method,
        )
        cluster_selection_method = "eom"

    LOGGER.info(
        "Running HDBSCAN (min_cluster_size=%d, min_samples=%d, method=%s)...",
        min_cluster_size, min_samples, cluster_selection_method,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
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

    # Build diagnostics
    sizes = [len(c.indices) for c in clusters]
    max_ratio = max(sizes) / n_samples if sizes else 0.0
    warning = None
    if max_ratio > 0.60:
        warning = (
            f"WARNING: Largest cluster contains {max_ratio:.0%} of all posts "
            f"({max(sizes)}/{n_samples}). The data may be too homogeneous for "
            f"meaningful subtopic separation, or the search term was not fully "
            f"stripped from embedding input."
        )
        LOGGER.warning(warning)

    diag = ClusterDiagnostics(
        cluster_count=len(clusters),
        cluster_sizes=sizes,
        largest_cluster_ratio=max_ratio,
        noise_count=len(noise_indices),
        total_posts=n_samples,
        warning=warning,
        umap_coordinates=visualization_coordinates,
        coordinate_method="umap_2d",
    )

    LOGGER.info(
        "Clustering diagnostics: %d clusters, sizes=%s, largest_ratio=%.2f",
        diag.cluster_count, diag.cluster_sizes, diag.largest_cluster_ratio,
    )

    return clusters, diag


def _build_umap_visualization_coordinates(
    emb_array: np.ndarray,
    reduced: np.ndarray,
    *,
    n_neighbors: int,
) -> list[list[float]]:
    """Return 2D coordinates for visualization, aligned with input embeddings."""
    if reduced.shape[1] == 2:
        return _round_coordinates(reduced)

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.0,
        )
        return _round_coordinates(reducer.fit_transform(emb_array))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not build 2D UMAP coordinates, using first reduced dimensions: %s", exc)
        return _round_coordinates(reduced[:, :2])


def _fallback_embedding_coordinates(emb_array: np.ndarray) -> list[list[float]]:
    """Small-dataset visualization fallback when UMAP is intentionally skipped."""
    if emb_array.size == 0:
        return []
    if emb_array.shape[1] == 1:
        coords = np.column_stack([emb_array[:, 0], np.zeros(emb_array.shape[0])])
    else:
        coords = emb_array[:, :2]
    return _round_coordinates(coords)


def _round_coordinates(coords: np.ndarray) -> list[list[float]]:
    return [
        [round(float(row[0]), 6), round(float(row[1]), 6)]
        for row in coords
    ]


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
