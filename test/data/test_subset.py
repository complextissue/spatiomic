"""Test the channel_subset function."""

import numpy as np

import spatiomic as so


def test_subset() -> None:
    """Test the channel_subset function."""
    # generate random data with 3 channels
    data = np.random.default_rng().integers(low=0, high=10, size=(10, 10, 3))
    channel_names_data = ["channel_1", "channel_2", "channel_3"]

    # subset the data while not changing the order, only removing the second channel
    channel_names_subset = ["channel_1", "channel_3"]
    data_subset = so.data.subset(
        data=data,
        channel_names_data=channel_names_data,
        channel_names_subset=channel_names_subset,
    )

    # check that the data was subsetted correctly
    assert data_subset.shape == (10, 10, 2)
    assert np.array_equal(data_subset[..., 0], data[..., 0])
    assert np.array_equal(data_subset[..., 1], data[..., 2])

    # subset the data while changing the order
    channel_names_subset = ["channel_3", "channel_1", "channel_2"]
    data_subset = so.data.subset(
        data=data,
        channel_names_data=channel_names_data,
        channel_names_subset=channel_names_subset,
    )

    # check that the data was subsetted correctly
    assert data_subset.shape == (10, 10, 3)
    assert np.array_equal(data_subset[..., 0], data[..., 2])
    assert np.array_equal(data_subset[..., 1], data[..., 0])
    assert np.array_equal(data_subset[..., 2], data[..., 1])

    # check that the function raises an error if a subset channel name is not in the data channel names
    channel_names_subset = ["channel_1", "channel_4"]
    with np.testing.assert_raises(AssertionError):
        so.data.subset(
            data=data,
            channel_names_data=channel_names_data,
            channel_names_subset=channel_names_subset,
        )
