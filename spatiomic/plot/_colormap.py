"""Provide colormaps for the plot submodule."""

import ast
import colorsys
import os
from typing import Dict, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
from matplotlib.colors import ListedColormap


def get_random_colors(
    color_count: int = 5,
    seed: Optional[int] = None,
) -> List[Tuple[float, float, float]]:
    """Generate a random color from HLS space.

    Colors will cover all hues but have a restricted luminosity range and a minimum saturation to separate it from
    backgrounds in black and create colors that fit well together.

    Args:
        color_count (int, optional): The number of colors needed. Defaults to 5.

    Returns:
        List[Tuple[float, float, float]]: List of RGB color values.
    """
    if seed is not None:
        np.random.seed(seed)

    hues = np.random.randint(0, 360, size=color_count) / 360
    luminosities = np.random.randint(30, 85, size=color_count) / 100
    saturations = np.random.randint(30, 100, size=color_count) / 100

    return [
        colorsys.hls_to_rgb(hue, luminosity, saturation)
        for hue, luminosity, saturation in zip(hues, luminosities, saturations, strict=True)
    ]


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert a hex color to RGB.

    Args:
        hex_color (str): The hex color.

    Returns:
        Tuple[float, float, float]: The RGB color.
    """
    hex_color = hex_color.lstrip("#")
    return tuple(float.fromhex(hex_color[i : i + 2]) / 255 for i in (0, 2, 4))  # type: ignore


def colormap(
    color_count: int,
    flavor: Literal["random_hls", "glasbey"] = "random_hls",
    color_override: Optional[Dict[int, Union[str, Tuple[float, float, float]]]] = None,
    use_system_background_list: bool = True,
    seed: int = 0,
) -> ListedColormap:
    """Return a colormap with the provided number of colors.

    Args:
        color_count (int, optional): The number of colors needed.
        flavor (Literal["random_hls", "random_hls"], optional): The method for choosing colors. Either generates
            random HLS values with luminosity between 30 and 85 and saturation between 30 and 100 or uses the glasbey
            light color palette from colorcet (up to 100 colors).
            Defaults to "random_hls".
        color_override (Optional[Dict[int, str]], optional): A dictionary of cluster indices and colors to override
            the default colors. Defaults to None.
        use_system_background_list (bool, optional): Whether to adjust the colors of background clusters based on
            the environment variables `SPATIOMIC_BG_CLUSTER_LIST` and `SPATIOMIC_BG_COLOR`.
            Defaults to True.
        seed (int, optional): The seed for the random color generation. Defaults to 0.

    Returns:
        ListedColormap: Colormap of the first color_count colors from the spatiomic colormap.
    """
    if flavor == "glasbey" and color_count > 100:
        raise ValueError(
            "The glasbey flavor is only compatible with color_count <= 100, please change the color_count"
            " or choose the `random_hls` flavor."
        )

    if flavor == "glasbey":
        from colorcet import glasbey_light

        warn(
            "The glasbey color palette is part of colorcet and distributed under the Creative Commons Attribution"
            " 4.0 International Public License (CC-BY).",
            UserWarning,
            stacklevel=2,
        )

        colormap_raw = glasbey_light[:color_count]
    else:
        colormap_raw = get_random_colors(color_count=color_count, seed=seed)

    # allow system level setting of clusters to interpret as background and color differently
    if use_system_background_list and os.getenv("SPATIOMIC_BG_CLUSTER_LIST", False):
        background_clusters = list(ast.literal_eval(os.getenv("SPATIOMIC_BG_CLUSTER_LIST", "")))
        background_color = os.getenv("SPATIOMIC_BG_COLOR", "#000000")

        for background_cluster in background_clusters:
            colormap_raw[background_cluster] = (
                hex_to_rgb(background_color) if not isinstance(background_color, tuple) else background_color
            )

    # override the colors with the provided colors
    if color_override is not None:
        for cluster, color in color_override.items():
            colormap_raw[cluster] = hex_to_rgb(color) if not isinstance(color, tuple) else color

    cmap = colormap_raw[:color_count]

    return ListedColormap(cmap)
