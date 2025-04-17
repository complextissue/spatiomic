"""Get a nearest neighbor finder."""

from typing import Callable, Literal, Union
from warnings import warn

from scipy.spatial.distance import correlation, cosine
from sklearn.neighbors import NearestNeighbors as NearestNeighborsSk


def get_neighbor_finder(
    neighbor_count: int,
    channel_count: int,
    distance_metric: Literal["euclidean", "manhattan", "correlation", "cosine"] = "euclidean",
    accuracy: Literal["fast", "accurate"] = "accurate",
    job_count: int = -1,
    use_gpu: bool = True,
) -> NearestNeighborsSk:
    """Get the nearest neighbor finder.

    Args:
        neighbor_count (int): The neighbor count for the neighborhood graph.
        channel_count (int): The number of channels in the data.
        distance_metric (Literal["euclidean", "manhattan", "correlation", "cosine"], optional): The distance metric
            to be used for nearest neighbor calculation. Defaults to "euclidean".
        accuracy (Literal["fast", "accurate"], optional): The accuracy of the nearest neighbor calculation. Used
            only when method is `simple` or `batch_balanced`. Defaults to "accurate".
        job_count (int, optional): Parallelization core count when method is `simple`. Defaults to -1.
        use_gpu (bool, optional): Whether to use the GPU for nearest neighbor calculation if possible.
            Defaults to True.

    Returns:
        Union[NearestNeighborsSk, Callable]: The nearest neighbor finder.
    """
    if use_gpu:
        try:
            from cuml.neighbors import NearestNeighbors  # type: ignore
        except Exception:
            warn(
                "Could not import cuml implementation. Switching to non-GPU implementation.", UserWarning, stacklevel=2
            )
            use_gpu = False

    if use_gpu:
        if distance_metric != "euclidean" and accuracy == "fast" and channel_count <= 3:
            warn(
                "The fast accuracy setting is not available for the specified distance metric and data shape."
                + " Falling back to the accurate setting.",
                UserWarning,
                stacklevel=2,
            )

        neighbor_finder = NearestNeighbors(
            n_neighbors=neighbor_count,
            algorithm=(
                "rbc" if distance_metric == "euclidean" and accuracy == "fast" and channel_count <= 3 else "brute"
            ),
            metric=distance_metric,
            output_type="numpy",
        )
    else:
        metric: Union[str, Callable] = distance_metric

        if distance_metric == "correlation":
            metric = correlation
        elif distance_metric == "cosine":
            metric = cosine

        # neighbor fitting
        neighbor_finder = NearestNeighborsSk(
            n_neighbors=neighbor_count,
            algorithm=("ball_tree" if accuracy == "fast" else "brute"),
            metric=metric,
            n_jobs=job_count,
        )

    return neighbor_finder
