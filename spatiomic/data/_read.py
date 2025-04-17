import os
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tifffile import imread, imwrite


class Read:
    """A class to read in microscopy files."""

    @staticmethod
    def read_lif(
        file_path: str,
        image_idx: int = 0,
    ) -> NDArray:
        """Read a single lif file on a given path and returns it.

        Args:
            file_path (str): Location of the file to be read.

        Returns:
            NDArray: An array containing the channels of the .lif file
        """
        assert "." in file_path and file_path.rsplit(".", 1)[1].lower() == "lif", "File has to have a .lif extension."

        assert os.path.exists(file_path), "Path to .lif image does not exist."

        try:
            from readlif.reader import LifFile  # type: ignore
        except ImportError as excp:
            raise ImportError("The readlif package is required to read .lif files. Please install it.") from excp

        lif_file = LifFile(Path(file_path))

        image = lif_file.get_image(image_idx)

        # iterate through all channels, channels being Pillow objects
        channel_list = np.array([np.array(i) for i in image.get_iter_c(t=0, z=0)], dtype=np.float32)

        return channel_list

    @staticmethod
    def split_multi_image_lif(
        file_path: str,
        save_folder: str,
        save_prefix: str,
    ) -> None:
        """Read a lif file with multiple images and save every image in a defined folder.

        Args:
            file_path (str): [description]
            save_folder (str): [description]
            save_prefix (str): [description]
        """
        assert "." in file_path and file_path.rsplit(".", 1)[1].lower() == "lif", "File has to have a .lif extension."
        assert os.path.exists(file_path), "Path to .lif image does not exist."
        assert os.path.isdir(save_folder), "The path to the save folder is incorrect."

        try:
            from readlif.reader import LifFile  # type: ignore
        except ImportError as excp:
            raise ImportError("The readlif package is required to read .lif files. Please install it.") from excp

        temp_lif_path = Path(file_path)
        lif_file = LifFile(temp_lif_path)

        for lif_image in lif_file.get_iter_image():
            channel_list = np.array(
                [np.array(channel) for channel in lif_image.get_iter_c(t=0, z=0)],
                dtype=np.uint16,
            )

            imwrite(
                f"{save_folder}/{save_prefix}-{lif_image.name.replace('/', '')}.tiff",
                channel_list,
            )

    @staticmethod
    def get_czi_image_channels(
        image: NDArray,
        image_shape: List[Tuple[str, int]],
        ubyte: bool = True,
    ) -> NDArray:
        """Get the channels of an .czi image.

        Args:
            image (NDArray): The read CziFile of an image.
            image_shape (List[Tuple[str, int]]): The shape of the CziFile.
                Example format of the image shape:
                [('B', 1), ('H', 1), ('T', 1), ('C', 3), ('Z', 1), ('Y', 2048), ('X', 2048)]
            ubyte (bool, optional): Whether to interpret the data as np.uint8.. Defaults to True.

        Returns:
            NDArray: An array containing the channels of the .czi file
        """
        image_max_x = image_shape[-1][1]
        image_max_y = image_shape[-2][1]

        channel_count = image_shape[-4][1]

        channels: Union[List, NDArray] = []

        if isinstance(channels, list):
            for i in range(0, channel_count):
                channels.append(image[0, 0, 0, i, 0, 0:image_max_y, 0:image_max_x])

        channels = np.array(channels)

        if ubyte:
            channels = channels.astype(np.uint8)

        return channels

    @classmethod
    def read_czi(
        cls,
        file_path: str,
        input_dimension_order: str = "XYC",
    ) -> NDArray:
        """Read a single czi file on a given path and returns it.

        Args:
            file_path (str): Location of the file to be read.

        Returns:
            NDArray: An array containing the channels of the .czi file
        """
        assert "." in file_path and file_path.rsplit(".", 1)[1].lower() == "czi", "File has to have a .czi extension."

        assert os.path.exists(file_path), "Path to .czi image does not exist."

        try:
            from aicspylibczi import CziFile  # type: ignore
        except ImportError as excp:
            raise ImportError("The aicspylibczi package is required to read .czi files. Please install it.") from excp

        czi_file = CziFile(Path(file_path))

        image, image_shape = czi_file.read_image()
        image_data = cls.get_czi_image_channels(
            image=image,
            image_shape=list(image_shape),
        )

        transpose_dimension_order = [
            input_dimension_order.index("X"),
            input_dimension_order.index("Y"),
            input_dimension_order.index("C"),
        ]

        image_data = np.transpose(
            image_data,
            transpose_dimension_order,
        )

        return image_data

    @staticmethod
    def read_tiff(
        file_path: Union[str, List[str]],
        input_dimension_order: str = "XYC",
        precision: Literal["float32", "float64"] = "float32",
    ) -> NDArray:
        """Read a single tiff file on a given path and returns it.

        Args:
            file_path (str): Location of the file to be read.
            input_dimension_order (str, optional): The dimension order of the channels in the tiff file.
                Defaults to "XYC".
            precision (Literal["float32", "float64"], optional): The precision of the data in the tiff file.
                Defaults to "float32".

        Returns:
            NDArray: An array containing the channels of the .tiff file in XYC dimension order.
        """
        image_data = []
        file_paths = file_path if isinstance(file_path, list) else [file_path]

        for file_path in file_paths:
            assert "." in file_path and file_path.rsplit(".", 1)[1].lower() in [
                "tiff",
                "tif",
            ], "File has to have a .tiff extension."
            assert os.path.exists(file_path), "Path to .tiff image does not exist."

            transpose_dimension_order = [
                input_dimension_order.index("X"),
                input_dimension_order.index("Y"),
                input_dimension_order.index("C"),
            ]

            image_data.append(
                np.transpose(
                    np.array(imread(file_path), dtype=(np.float32 if precision == "float32" else np.float64)),
                    transpose_dimension_order,
                )
            )

        # if only one image was read, return it as an array
        if len(file_paths) == 1:
            return np.array(image_data[0])

        return np.array(image_data)
