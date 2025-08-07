"""Tests for the Quench class and its methods."""

# from typing import List, Tuple

# import numpy as np
# import pytest

# from spatiomic.process import quench as Quench
# from spatiomic.process import quench_channel as QuenchChannel


# # Sample test image data (adjust shapes, values for your tests)
# def generate_test_image(shape: Tuple[int, int], n_channels: int) -> np.ndarray:
#     """Generate a random test image with given shape and number of channels."""
#     return np.random.randint(0, 256, size=(*shape, n_channels), dtype=np.uint16)


# @pytest.fixture
# def background_channels() -> List[QuenchChannel]:
#     """Fixture providing consistent background channel data."""
#     return [
#         QuenchChannel(data=generate_test_image((100, 100), 1), name="Background1", wavelength=488, cycle_index=1),
#         QuenchChannel(data=generate_test_image((100, 100), 1), name="Background1", wavelength=488, cycle_index=2),
#         QuenchChannel(data=generate_test_image((100, 100), 1), name="Background2", wavelength=561, cycle_index=3),
#     ]


# @pytest.fixture
# def quencher(background_channels: List[QuenchChannel]) -> Quench:
#     """Fixture providing a Quench instance with background channels and method."""
#     return Quench(background_channels=background_channels, method="interpolate")


# def test_transform_interpolation(quencher: Quench, background_channels: List[QuenchChannel]) -> None:
#     """Test the 'transform' method with interpolation mode."""
#     return
#     data = generate_test_image((3, 100, 100), 3)
#     wavelengths = [488, 561, 640]
#     cycle_indices = [1, 2, 3]

#     result = quencher.transform(data, wavelengths, cycle_indices, first_cycle_index=1)

#     # Basic shape check
#     assert result.shape == data.shape

#     # Check that the result has lower values than the original data
#     assert np.all(result[result > 0] < data[result > 0])


# def test_transform_nearest(quencher: Quench) -> None:
#     """Test the 'transform' method with nearest mode."""
#     # Similar to above, but test 'nearest' mode. Adjust expected behavior accordingly.
#     pass


# def test_transform_first(quencher: Quench) -> None:
#     """Test the 'transform' method with first mode."""
#     # Test the 'first' background removal mode
#     pass


# def test_transform_error_mismatched_wavelengths(quencher: Quench) -> None:
#     """Test the 'transform' method with mismatched wavelengths."""
#     data = generate_test_image((100, 100), 3)
#     wavelengths = [405, 561, 640]  # 405 doesn't exist in background channels
#     with pytest.raises(ValueError):
#         quencher.transform(data, wavelengths)


# def test_transform_error_missing_cycle_indices(quencher: Quench) -> None:
#     """Test the 'transform' method with missing cycle indices."""
#     data = generate_test_image((100, 100), 3)
#     wavelengths = [488, 561, 640]

#     with pytest.raises(ValueError):
#         # Required for 'interpolate' or 'nearest' modes
#         quencher.transform(data, wavelengths)
