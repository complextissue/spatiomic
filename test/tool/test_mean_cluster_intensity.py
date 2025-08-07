"""Test the mean_cluster_intensity function."""

import numpy as np
import pandas as pd

import spatiomic as so


def test_mean_cluster_intensity() -> None:
    """Test the mean_cluster_intensity function."""
    channel_names_collection = [
        None,
        [
            "channel_1",
            "channel_2",
            "channel_3",
        ],
        [
            3,
            4,
            5,
        ],
    ]

    # generate random data with 3 channels
    data = np.random.default_rng().integers(low=0, high=10, size=(10, 10, 3))
    labels = np.random.default_rng().integers(low=0, high=10, size=100)
    labels_unique = np.sort(np.unique(labels.ravel()))

    for channel_names in channel_names_collection:
        mean_intensity_df = so.tool.mean_cluster_intensity(
            data=data,
            clusters=labels,
            channel_names=channel_names,
        )

        # check that the column names were set correctly
        if channel_names is not None:
            pd.testing.assert_index_equal(
                mean_intensity_df.columns,
                pd.Index(channel_names),
            )

        pd.testing.assert_index_equal(
            mean_intensity_df.index,
            pd.Index(labels_unique),
        )
