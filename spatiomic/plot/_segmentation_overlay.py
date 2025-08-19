"""Segmentation visualization functions for spatiomic."""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from skimage.segmentation import find_boundaries

from ._colormap import colormap as create_colormap


def segmentation_overlay(
    image: NDArray,
    masks: Optional[NDArray] = None,
    boundary_color: str = "#FF0000",
    mask_colormap: Optional[ListedColormap] = None,
    mask_alpha: float = 0.5,
    show_boundaries: bool = True,
    show_masks: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
) -> Union[plt.Figure, plt.Axes]:
    """Display an image with optional mask boundaries and/or colored masks overlaid.

    Args:
        image: The base image to display. Can be 2D grayscale or 3D RGB.
        masks: Segmentation masks where each region has a unique positive integer ID.
            Background should be 0. Optional.
        boundary_color: Color for mask boundaries. Defaults to "#FF0000" (red).
        mask_colormap: Custom colormap for displaying masks. If None, uses default colormap.
        mask_alpha: Transparency of mask overlay (0=transparent, 1=opaque). Defaults to 0.5.
        show_boundaries: Whether to show mask boundaries. Defaults to True.
        show_masks: Whether to show colored mask overlay. Defaults to False.
        ax: Existing axes to plot on. If None, creates new figure. Defaults to None.
        figsize: Figure size as (width, height). Only used if ax is None. Defaults to None.

    Returns:
        plt.Figure if ax is None, otherwise the provided plt.Axes.

    Raises:
        ValueError: If neither show_boundaries nor show_masks is True.
    """
    if masks is not None and not show_boundaries and not show_masks:
        raise ValueError("At least one of show_boundaries or show_masks must be True when masks are provided")

    # Setup plotting
    if ax is None:
        sns.set_theme(style="white", font="Arial")
        sns.set_context("paper")
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        return_fig = False

    # Display base image
    if len(image.shape) == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)

    # Add mask visualization if requested
    if masks is not None:
        if show_masks:
            # Create colored mask overlay
            mask_count = np.max(masks) + 1
            if mask_colormap is None:
                mask_colormap = create_colormap(color_count=mask_count, color_override={0: "#000000"})

            ax.imshow(masks, cmap=mask_colormap, alpha=mask_alpha)

        if show_boundaries:
            # Find and display boundaries
            boundaries = find_boundaries(masks, mode="thick")

            # Create boundary overlay
            boundary_overlay = np.zeros((*boundaries.shape, 4))
            if boundary_color.startswith("#"):
                # Convert hex to RGB
                hex_color = boundary_color.lstrip("#")
                rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
            else:
                # Assume named color, convert via matplotlib
                from matplotlib.colors import to_rgba

                rgb = to_rgba(boundary_color)[:3]

            boundary_overlay[boundaries, :3] = rgb
            boundary_overlay[boundaries, 3] = 1.0  # Full opacity for boundaries

            ax.imshow(boundary_overlay)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])

    if return_fig:
        plt.tight_layout()
        return fig
    else:
        return ax


def save_segmentation_overlay(
    image: NDArray,
    save_path: str,
    masks: Optional[NDArray] = None,
    boundary_color: str = "#FF0000",
    mask_colormap: Optional[ListedColormap] = None,
    mask_alpha: float = 0.5,
    show_boundaries: bool = True,
    show_masks: bool = False,
    dpi: int = 300,
) -> None:
    """Save segmentation overlay to file without borders.

    Args:
        image: The base image to display.
        save_path: Path where to save the image.
        masks: Segmentation masks. Optional.
        boundary_color: Color for mask boundaries. Defaults to "#FF0000".
        mask_colormap: Custom colormap for masks. Defaults to None.
        mask_alpha: Transparency of mask overlay. Defaults to 0.5.
        show_boundaries: Whether to show mask boundaries. Defaults to True.
        show_masks: Whether to show colored mask overlay. Defaults to False.
        dpi: Resolution for saved image. Defaults to 300.
    """
    # Create figure with exact image dimensions
    height, width = image.shape[:2]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width / dpi, height / dpi)

    ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.add_axes(ax)

    # Use the overlay function to create the visualization
    segmentation_overlay(
        image=image,
        masks=masks,
        boundary_color=boundary_color,
        mask_colormap=mask_colormap,
        mask_alpha=mask_alpha,
        show_boundaries=show_boundaries,
        show_masks=show_masks,
        ax=ax,
    )

    # Save without borders
    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
