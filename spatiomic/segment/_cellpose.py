"""Cellpose segmentation implementation for spatiomic."""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt


class Cellpose:
    """Class that exposes the Cellpose segmentation algorithm.

    This class provides a simplified wrapper around the Cellpose segmentation model
    for cell segmentation in microscopy images. Uses Cellpose 4 with SAM-based models.

    Args:
        use_gpu: Whether to use GPU acceleration. Defaults to True.
        pretrained_model: The type of Cellpose model to use. Defaults to 'cpsam'.
        **kwargs: Additional parameters passed to the CellposeModel().

    Raises:
        ImportError: If cellpose is not installed.

    Example:
        Basic usage for cell segmentation:

        ```python
        import spatiomic as so
        import numpy as np

        # Create a Cellpose segmentation instance
        cellpose = so.segment.cellpose(use_gpu=True)

        # Perform segmentation on 2D grayscale image
        masks_2d = cellpose.predict(image_2d)

        # Perform segmentation on 3D RGB image (first 3 channels used automatically)
        masks_3d = cellpose.predict(image_3d)

        # masks contains the segmented cell masks with unique integer labels
        ```
    """

    def __init__(
        self,
        pretrained_model: str = "cpsam",
        use_gpu: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Cellpose segmentation class.

        Args:
            pretrained_model: The type of Cellpose model to use. Defaults to 'cpsam'.
            use_gpu: Whether to use GPU acceleration. Defaults to True.
            **kwargs: Additional parameters passed to the CellposeModel().

        Raises:
            ImportError: If cellpose is not installed.
        """
        try:
            from cellpose import models
        except ImportError as e:
            msg = (
                "The cellpose package is required for this functionality. "
                "Please install it with: uv add spatiomic --extra cellpose"
            )
            raise ImportError(msg) from e

        if use_gpu:
            try:
                import torch

                gpu_available = torch.cuda.is_available() or (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
            except ImportError:
                gpu_available = False
            self.use_gpu = gpu_available
        else:
            self.use_gpu = False

        self.pretrained_model = pretrained_model
        self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model=self.pretrained_model, **kwargs)

    def predict(
        self,
        pixels: npt.NDArray[np.floating[Any]],
        diameter: Optional[float] = None,
        **kwargs: Any,
    ) -> tuple[npt.NDArray[np.integer[Any]], list[Any], npt.NDArray[np.floating[Any]]]:
        """Predict cell segmentation masks using Cellpose.

        Args:
            pixels: Input image array. Can be 2D (H x W) for grayscale or 3D (H x W x C) for color.
                     For best results with 3D arrays, use RGB format where the first 3 channels
                     will be automatically used by Cellpose v4.0.1+.
            diameter: Expected diameter of cells in pixels. If None, will be estimated.
            **kwargs: Additional parameters passed to the Cellpose model.eval() method.

        Returns:
            A tuple containing (masks, flows, styles):
            - masks: Array of segmented cell masks with unique integer labels (0 = background).
            - flows: List containing flow data [flow_hsv, flows_xy, cellprob, final_locations].
            - styles: Style vector (zeros for compatibility with Cellpose v3).

        Raises:
            ValueError: If input image has invalid dimensions.
        """
        if pixels.ndim not in [2, 3]:
            msg = f"Input image must be 2D or 3D, got {pixels.ndim}D"
            raise ValueError(msg)

        masks, flows, styles = self.model.eval(pixels, diameter=diameter, **kwargs)

        return masks, flows, styles  # type: ignore[no-any-return]
