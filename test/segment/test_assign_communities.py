"""Test the assign_communities function."""

import numpy as np
import pytest

from spatiomic.segment import assign_communities


@pytest.mark.cpu
def test_assign_communities_offset() -> None:
    """Community k maps to pixel value k + 1; background stays 0."""
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    communities = [0, 1]  # region 1 -> community 0, region 2 -> community 1

    result = assign_communities(masks, communities)

    np.testing.assert_array_equal(result, np.array([[0, 1, 1], [0, 2, 2]]))


@pytest.mark.cpu
def test_assign_communities_offset_is_fixed_across_images() -> None:
    """The community -> pixel-value mapping must not depend on whether community 0 is present."""
    masks = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)

    result_a = assign_communities(masks, [0, 1])  # communities include 0
    result_b = assign_communities(masks, [1, 2])  # 1-indexed, no 0

    # community k -> pixel value k + 1 in BOTH images (previously the offset was data-dependent)
    assert result_a[0, 1] == 1  # community 0 -> 1
    assert result_a[1, 1] == 2  # community 1 -> 2
    assert result_b[0, 1] == 2  # community 1 -> 2 (same mapping as image A)
    assert result_b[1, 1] == 3  # community 2 -> 3


@pytest.mark.cpu
def test_assign_communities_length_mismatch_raises() -> None:
    """A community count that does not match the number of mask regions must raise."""
    masks = np.array([[0, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="must match"):
        assign_communities(masks, [0])  # only one community for two regions
