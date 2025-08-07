from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from spatiomic._internal._anndata_method import anndata_method
from spatiomic._internal._import_package import import_package
from spatiomic._internal._seed_method import seed_method

pixel_level_queen_neighborhood_offset = np.array(
    [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)]
)


@anndata_method(input_obs="cluster", output_uns="vicinity_composition")
@seed_method
def vicinity_composition(
    data: NDArray,
    neighborhood_offset: NDArray = pixel_level_queen_neighborhood_offset,
    ignore_identities: bool = True,
    ignore_repeated: bool = False,
    permutations: Optional[int] = None,
    use_gpu: bool = False,
    seed: int = 0,
    n_jobs: int = -1,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Get the counts of neighboring clusters for each cluster in an image.

    Args:
        data (np.ndarray): The image of clusters.
        neighborhood_offset (np.ndarray, optional): The neighborhood offset array. Can be created using the
            `neighborhood_offset` function. Defaults to `pixel_level_queen_neighborhood_offset`.
        ignore_identities (bool, optional): Whether to ignore pixels from the same cluster in the vicinity.
            Defaults to True.
        ignore_repeated (bool, optional): Whether to just count pixels that are in the vicinity of multiple pixels
            from the same cluster once. Defaults to False.
        permutations (int, optional): The number of permutations to use for p-value calculation, 0 if no simulation is
            to be performed. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU for calculations. Defaults to False.
        seed (int, optional): The random seed to use for the permutations. Defaults to 0.
        n_jobs (int, optional): Number of parallel jobs to use for permutation calculations. Defaults to -1.

    Returns:
        Union[pd.DataFrame, [pd.DataFrame, pd.DataFrame]]: A DataFrame containing the counts of neighbors from each
            cluster for each cluster. If permutations is not None, a tuple of two DataFrames containing the counts of
            neighbors from each cluster for each cluster and the p-values for each cluster.
    """
    if TYPE_CHECKING or not use_gpu:
        xp = np
    else:
        xp: "np" = import_package("cupy", alternative=np)

    # check that the data is integer type and has no negative values
    assert data.dtype in [
        int,
        np.int8,
        np.int16,
        np.uint8,
        np.uint16,
    ], "Clusters in data_reference must be integer type."
    assert bool(np.any(data < 0)) is False, "data_reference contains negative values. Clusters cannot be negative."

    padding_size = np.max(np.abs(neighborhood_offset))
    data = xp.array(data).astype(int)
    m, n = data.shape

    def calculate_vicinity_composition(
        data: NDArray,
        neighborhood_offset: NDArray,
        ignore_identities: bool,
        ignore_repeated: bool,
        padding_size: int,
    ) -> pd.DataFrame:
        # pad to include boundaries
        img_padded = xp.pad(data, (padding_size, padding_size), mode="constant", constant_values=-1)

        results = {}
        clusters = sorted(xp.unique(data))

        # get values of neighbouring pixels for all offsets
        neighbors_list = [
            img_padded[padding_size + i : m + padding_size + i, padding_size + j : n + padding_size + j].flatten()
            for i, j in neighborhood_offset
        ]
        neighbors = xp.stack(neighbors_list, axis=-1)  # noqa: PD013

        # get values of current cluster pixels for all offsets
        cluster_pixels_stack = xp.repeat(
            xp.expand_dims(
                data.flatten(),
                axis=-1,
            ),
            len(neighborhood_offset),
            axis=-1,
        )
        cluster_pixels = cluster_pixels_stack.flatten()

        for cluster in clusters:
            cluster_neighbors = neighbors.copy()

            if ignore_repeated:
                repeated = xp.zeros(cluster_pixels_stack.shape, dtype=xp.uint8)
                repeated[(cluster_pixels_stack == cluster)] = 1
                cluster_neighbors[xp.sum(repeated, axis=-1) >= 2, 1:] = -1

            # get neighbors for the current cluster, exclude self, boundaries and repeated (-1)
            cluster_neighbors = cluster_neighbors.flatten()
            conditions = (cluster_pixels == cluster) & (cluster_neighbors != -1)

            if ignore_identities:
                conditions &= cluster_neighbors != cluster

            # count unique neighbors
            unique_neighbors, counts = xp.unique(cluster_neighbors[conditions], return_counts=True)

            if xp.__name__ == "cupy":
                unique_neighbors = xp.asnumpy(unique_neighbors)  # type: ignore
                counts = xp.asnumpy(counts)  # type: ignore

            result = dict(zip(unique_neighbors, counts, strict=True))
            results[int(cluster)] = result

        return pd.DataFrame(results).fillna(0).sort_index(axis=0).sort_index(axis=1).astype(int)

    df_vicinity_composition = (
        pd.DataFrame(
            calculate_vicinity_composition(
                data,
                neighborhood_offset,
                ignore_identities,
                ignore_repeated,
                padding_size,
            )
        )
        .fillna(0)
        .sort_index(axis=0)
        .sort_index(axis=1)
        .astype(int)
    )

    if permutations is None:
        return df_vicinity_composition

    # calculate p-values
    assert permutations > 0, "Permutations must be a positive integer."

    def process_single_permutation(permutation_seed: int) -> pd.DataFrame:
        """Process a single permutation of the data.

        Args:
            permutation_seed (int): The seed for the permutation.

        Returns:
            pd.DataFrame: The DataFrame containing the counts of neighbors from each cluster for each cluster after
                permutation.
        """
        # Set a unique seed for each permutation based on the main seed
        np.random.seed(seed + permutation_seed)
        if xp.__name__ == "cupy":
            xp.random.seed(seed + permutation_seed)

        # Generate permutation and calculate vicinity composition
        return calculate_vicinity_composition(
            xp.random.permutation(data),
            neighborhood_offset,
            ignore_identities,
            ignore_repeated,
            padding_size,
        )

    if n_jobs != 1:
        # Parallel processing of permutations
        df_permutations = Parallel(n_jobs=n_jobs)(delayed(process_single_permutation)(i) for i in range(permutations))
    else:
        # Sequential processing if n_jobs=1
        df_permutations = [process_single_permutation(i) for i in range(permutations)]

    permutation_values = [df_permutation.to_numpy() for df_permutation in df_permutations]
    higher_or_equal = xp.sum(
        [permutation_values[i] >= df_vicinity_composition.to_numpy() for i in range(permutations)], axis=0
    )
    p_values = (higher_or_equal + 1) / (permutations + 1)

    df_p_values = pd.DataFrame(p_values, index=df_vicinity_composition.index, columns=df_vicinity_composition.columns)

    if ignore_identities:
        np.fill_diagonal(df_p_values.values, np.nan)

    return df_vicinity_composition, df_p_values
