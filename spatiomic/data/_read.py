import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from tifffile import TiffFile, imwrite

if TYPE_CHECKING:
    from aicspylibczi import CziFile  # noqa: F401
    from qptifffile import QPTiffFile  # noqa: F401
    from readlif.reader import LifFile  # noqa: F401


class Read:
    """A class to read in microscopy files."""

    @classmethod
    def _get_transpose_order(cls, input_dimension_order: str) -> List[int]:
        """Get transpose order for converting to XYC format with optional T and Z dimensions.

        Args:
            input_dimension_order: String specifying dimension order (e.g., "TCYX", "XYC").

        Returns:
            List of indices for numpy transpose operation.
        """
        target_dims = ["X", "Y", "C"]

        if "T" in input_dimension_order:
            target_dims.insert(0, "T")
        if "Z" in input_dimension_order:
            target_dims.insert(-1, "Z")

        return [input_dimension_order.index(dim) for dim in target_dims]

    @staticmethod
    def read_lif(
        file_path: Union[str, Path],
        image_idx: int = 0,
        dtype: Optional[DTypeLike] = None,
    ) -> NDArray:
        """Read a single lif file on a given path and returns it.

        Args:
            file_path (str): Location of the file to be read.
            image_idx (int, optional): The index of the image in the lif file to be read. Defaults to 0.
            dtype (Optional[DTypeLike], optional): The dtype of the data in the lif file.
                If None, keeps original dtype. Defaults to None.

        Returns:
            NDArray: An array containing the channels of the .lif file
        """
        file_path = str(file_path)
        assert "." in file_path and file_path.rsplit(".", 1)[1].lower() == "lif", "File has to have a .lif extension."

        assert os.path.exists(file_path), "Path to .lif image does not exist."

        try:
            from readlif.reader import LifFile  # type: ignore
        except ImportError as excp:
            raise ImportError("The readlif package is required to read .lif files. Please install it.") from excp

        lif_file = LifFile(Path(file_path))

        image = lif_file.get_image(image_idx)

        # iterate through all channels, channels being Pillow objects
        channel_list = np.array([np.array(i) for i in image.get_iter_c(t=0, z=0)])

        if dtype is not None:
            channel_list = channel_list.astype(dtype)

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
        file_path: Union[str, Path],
        input_dimension_order: str = "XYC",
        dtype: Optional[DTypeLike] = None,
    ) -> NDArray:
        """Read a single czi file on a given path and returns it.

        Args:
            file_path (str): Location of the file to be read.
            input_dimension_order (str, optional): The dimension order of the channels in the czi file.
                Defaults to "XYC".
            dtype (Optional[DTypeLike], optional): The dtype of the data in the czi file.
                If None, keeps original dtype. Defaults to None.

        Returns:
            NDArray: An array containing the channels of the .czi file
        """
        file_path = str(file_path)
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
        if dtype is not None and image_data.dtype != dtype:
            image_data = image_data.astype(dtype)

        transpose_dimension_order = cls._get_transpose_order(input_dimension_order)

        image_data = np.transpose(
            image_data,
            transpose_dimension_order,
        )

        return image_data

    @classmethod
    def read_tiff(
        cls,
        file_path: Union[str, Path, List[Union[str, Path]]],
        input_dimension_order: str = "XYC",
        dtype: Optional[DTypeLike] = None,
        return_channels: bool = False,
    ) -> Union[NDArray, Tuple[NDArray, List[str]]]:
        """Read a single tiff file on a given path and returns it.

        Args:
            file_path (str): Location of the file to be read.
            input_dimension_order (str, optional): The dimension order of the channels in the tiff file.
                Defaults to "XYC".
            dtype (Optional[DTypeLike], optional): The dtype of the data in the tiff file.
                If None, keeps original dtype. Defaults to None.
            return_channels (bool, optional): Whether to return the channel names. Only works on imagej tiffs.
                Defaults to False.

        Returns:
            Union[NDArray, Tuple[NDArray, List[str]]]: An array containing the channels of the .tiff file in XYC dimension order.
                If return_channels is True and ImageJ metadata is found, returns tuple of (array, channel_names).
        """
        image_data = []
        channels: Optional[List[str]] = None
        file_paths = file_path if isinstance(file_path, list) else [file_path]

        for file_path in file_paths:
            file_path = str(file_path)
            assert "." in file_path and file_path.rsplit(".", 1)[1].lower() in [
                "tiff",
                "tif",
            ], "File has to have a .tiff extension."
            assert os.path.exists(file_path), "Path to .tiff image does not exist."

            transpose_dimension_order = cls._get_transpose_order(input_dimension_order)

            # Use TiffFile to check for ImageJ format and read metadata
            with TiffFile(file_path) as tif:
                is_imagej = tif.is_imagej
                img_array = tif.asarray()

                # Extract channel names from ImageJ metadata if available
                if is_imagej and return_channels and tif.imagej_metadata is not None:
                    imagej_meta = tif.imagej_metadata
                    if "Labels" in imagej_meta:
                        file_channels = imagej_meta["Labels"]
                        if channels is None:
                            channels = file_channels
                        else:
                            assert channels == file_channels, "All images must have the same channel names."

            # Apply dtype conversion only if specified
            if dtype is not None:
                img_array = img_array.astype(dtype)

            image_data.append(
                np.transpose(
                    img_array,
                    transpose_dimension_order,
                )
            )

        # if only one image was read, return it as an array
        result_array = np.array(image_data[0]) if len(file_paths) == 1 else np.array(image_data)

        if return_channels and channels is not None:
            return result_array, channels

        return result_array

    @staticmethod
    def read_qptiff(
        file_path: Union[str, Path, List[Union[str, Path]]],
        level: int = 0,
        dtype: Optional[DTypeLike] = None,
        return_channels: bool = True,
    ) -> Union[NDArray, Tuple[NDArray, List[str]]]:
        """Read a single qupath tiff file on a given path and returns it.

        Args:
            file_path (str): Location of the file or files to be read.
            level (int, optional): The level of the qupath tiff file to be read. Defaults to 0.
            dtype (Optional[DTypeLike], optional): The dtype of the data in the qptiff file.
                If None, keeps original dtype. Defaults to None.
            return_channels (bool, optional): Whether to return the channel names. Defaults to True.

        Returns:
            Union[NDArray, Tuple[NDArray, List[str]]]: An array containing the channels of the qupath tiff file.
        """
        try:
            from qptifffile import QPTiffFile  # type: ignore
        except ImportError as excp:
            raise ImportError(
                "The qptifffile package is required to read qupath tiff files. Please install it."
            ) from excp

        image_data = []
        channels: Optional[List[str]] = None
        file_paths = file_path if isinstance(file_path, list) else [file_path]

        for file_path in file_paths:
            file_path = str(file_path)
            assert "." in file_path and file_path.rsplit(".", 1)[1].lower() in [
                "qptiff",
                "qptif",
            ], "File has to have a .qptiff extension."
            assert os.path.exists(file_path), "Path to .qptiff image does not exist."

            with QPTiffFile(file_path) as qptiff:
                image_channels = qptiff.get_biomarkers()

                assert len(image_channels) > 0, f"Image {file_path} contained no channels"

                if channels is None:
                    channels = image_channels
                else:
                    assert set(channels) == set(image_channels), "All images have to contain the same channels."

                img_array = qptiff.read_region(channels, level=level)
                if dtype is not None:
                    img_array = img_array.astype(dtype)
                image_data.append(img_array)

        if return_channels:
            return np.array(image_data), cast(List[str], channels)

        return np.array(image_data)
