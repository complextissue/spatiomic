from typing import Dict, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as translate_scipy
from skimage.registration import optical_flow_tvl1
from skimage.registration import phase_cross_correlation as phase_cross_correlation_sk
from skimage.transform import warp


class Register:
    """Expose registration methods."""

    @staticmethod
    def get_ssim(
        pixels: NDArray,
        reference_pixels: NDArray,
    ) -> float:
        """Calculate the structural similarity index measure.

        Args:
            pixels (NDArray): A 2D array of pixels.
            reference_pixels (NDArray): The 2D reference array for calculation of the structural similarity.

        Returns:
            float: The structural similarity index measure.
        """
        from skimage.metrics import structural_similarity

        ssim = structural_similarity(
            pixels,
            reference_pixels,
            full=False,
            data_range=np.max(pixels) - np.min(pixels),
        )

        return float(ssim)

    @classmethod
    def get_shift(
        cls,
        pixels: NDArray,
        reference_pixels: NDArray,
        blur: bool = False,
        match_histogram: bool = False,
        threshold: bool = False,
        threshold_percentile: Union[int, float] = 70,
        method: Literal["chi2_shift", "phase_correlation"] = "phase_correlation",
        upsample_factor: int = 1,
        use_gpu: bool = True,
    ) -> Tuple[float, float]:
        """Calculate the y- and the x-offset between two images.

        Args:
            pixels (NDArray): The pixels for which a shift is to be detected.
            reference_pixels (NDArray): The reference pixels to calculate the shift with.
            blur (bool, optional): Whether to blur the reference and the pixels before calculating the shift.
                Defaults to False.
            match_histogram (bool, optional): Whether to match the histograms between the reference and the pixels.
                Defaults to False.
            threshold (bool, optional): Whether to threshold the reference and the pixels to everything >= the 70th
                percentile of the reference. Defaults to False.
            threshold_percentile (Union[int, float], optional): The percentile to use for thresholding.
            method (Literal["chi2_shift", "phase_correlation"], optional): The method to use for shift detection.
                Defaults to "phase_correlation".
            upsample_factor (int, optional): The upsample factor to use for the phase correlation method.
                Defaults to 1.
            use_gpu (bool, optional):  Whether to use the cucim phase_correlation gpu implementation.
                Defaults to True.

        Returns:
            Tuple[float, float]: The offset on the y- and the x-axis.
        """
        if blur:
            from skimage.filters import gaussian

            pixels = gaussian(pixels)
            reference_pixels = gaussian(reference_pixels)

        if match_histogram:
            from skimage.exposure import match_histograms

            pixels = match_histograms(pixels, reference_pixels)

        if threshold:
            threshold_limit = np.percentile(reference_pixels, threshold_percentile)
            reference_pixels[reference_pixels < threshold_limit] = 0
            pixels[pixels < threshold_limit] = 0

        if method == "chi2_shift":
            # requires image_registration and typing_extensions>=3.10.0.1 and fftw for best performance
            from image_registration.chi2_shifts import chi2_shift  # type: ignore

            shift = chi2_shift(
                reference_pixels,
                pixels,
                return_error=False,
                zeromean=False,
                upsample_factor=upsample_factor,
            )

            # note that here the first offset is the y-offset
            (offset_y, offset_x) = (-shift[0], -shift[1])
        elif method == "phase_correlation":
            (offset_y, offset_x) = cls.get_phase_shift(
                pixels=pixels,
                reference_pixels=reference_pixels,
                blur=blur,
                match_histogram=match_histogram,
                threshold=threshold,
                use_gpu=use_gpu,
                upsample_factor=upsample_factor,
            )

        return (offset_y, offset_x)

    @staticmethod
    def get_phase_shift(
        pixels: NDArray,
        reference_pixels: NDArray,
        blur: bool = False,
        match_histogram: bool = False,
        threshold: bool = False,
        upsample_factor: int = 1,
        use_gpu: bool = True,
    ) -> Tuple[float, float]:
        """Calculate the y- and the x-offset between two images.

        Args:
            pixels (NDArray): The pixels for which a shift is to be detected.
            reference_pixels (NDArray): The reference pixels to calculate the shift with.
            blur (bool, optional): Whether to blur the reference and the pixels before calculating the shift.
                Defaults to False.
            match_histogram (bool, optional): Whether to match the histograms between the reference and the pixels.
                Defaults to False.
            threshold (bool, optional): Whether to threshold the reference and the pixels to everything >= the 70th
                percentile of the reference. Defaults to False.
            upsample_factor (int, optional): The upsample factor to use for the phase correlation method.
                Defaults to 1.
            use_gpu (bool, optional):  Whether to use the cucim phase_correlation gpu implementation instead of chi2
                shift. Defaults to True.

        Returns:
            Tuple[float, float]: The offset on the y- and the x-axis.
        """
        if blur:
            from skimage.filters import gaussian

            pixels = gaussian(pixels)
            reference_pixels = gaussian(reference_pixels)

        if match_histogram:
            from skimage.exposure import match_histograms

            pixels = match_histograms(pixels, reference_pixels)

        if threshold:
            threshold_limit = np.percentile(reference_pixels, 70)  # type: ignore
            reference_pixels[reference_pixels < threshold_limit] = 0
            pixels[pixels < threshold_limit] = 0

        if use_gpu:
            try:
                import cupy as cp  # type: ignore
                from cucim.skimage.registration import (  # type: ignore
                    phase_cross_correlation,
                )

                # cucim expects cupy arrays
                pixels = cp.array(pixels)
                reference_pixels = cp.array(reference_pixels)
            except Exception:
                phase_cross_correlation = phase_cross_correlation_sk
                use_gpu = False
        else:
            phase_cross_correlation = phase_cross_correlation_sk

        # older skimage versions return fewer parameters, thus we catch all and then select the first for compatibility
        result = phase_cross_correlation(
            reference_pixels,
            pixels,
            upsample_factor=upsample_factor,
        )
        shift = result[0]

        if use_gpu and shift is not None and hasattr(shift, "get"):
            shift = shift.get()  # type: ignore

        # note that here the first offset is the y-offset
        (offset_y, offset_x) = (shift[0], shift[1])

        return (offset_y, offset_x)

    @staticmethod
    def apply_shift(
        pixels: NDArray,
        shift: Union[Tuple[float, float], NDArray],
        precision: Literal["float32", "float64"] = "float32",
        use_gpu: bool = True,
    ) -> NDArray:
        """Shift the pixels of an image by a given offset.

        Args:
            pixels (NDArray): The 2D image to be shifted.
            shift (Union[Tuple[float, float], NDArray[Float, Float]]): The y- and the x-offset to translate the image
                by.
            precision (Literal["float32", "float64"], optional): The precision of the pixels. Defaults to "float32".
            use_gpu (bool, optional): Whether to use cupyx for translation on the GPU. Defaults to True.

        Returns:
            NDArray: The shifted image.

        Raises:
            ValueError: If the pixels do not have 2 or 3 dimensions.
        """
        if use_gpu:
            try:
                import cupy as cp  # type: ignore
                from cupyx.scipy.ndimage import shift as translate  # type: ignore

                pixels = cp.array(pixels)
            except Exception:
                translate = translate_scipy
                use_gpu = False
        else:
            translate = translate_scipy

        if len(pixels.shape) == 3:
            for channel_idx in range(0, pixels.shape[-1]):
                pixels[:, :, channel_idx] = translate(pixels[:, :, channel_idx], shift=shift, mode="constant")
        elif len(pixels.shape) == 2:
            pixels = translate(pixels, shift=shift, mode="constant")
        else:
            raise ValueError("Shift can only be applied to image in XY or XYC format.")

        # check that cupy exists and that pixels is a cupy array
        if use_gpu and "cp" in globals() and isinstance(pixels, cp.NDArray):  # type: ignore
            pixels = pixels.get()  # type: ignore

        return pixels.astype(np.float32 if precision == "float32" else np.float64)

    @staticmethod
    def get_optical_flow(
        pixels: NDArray,
        reference_pixels: NDArray,
        normalize: bool = True,
    ) -> NDArray:
        """Calculate the optical flow between two images.

        Args:
            pixels (NDArray): The pixels for which a shift is to be detected.
            reference_pixels (NDArray): The reference pixels to calculate the shift with.
            normalize (bool, optional): Whether to normalize the optical flow. Defaults to True.

        Returns:
            NDArray: The optical flow.
        """
        if normalize:
            from ._normalize import Normalize

            pixels = Normalize(use_gpu=False).fit_transform(pixels)
            reference_pixels = Normalize(use_gpu=False).fit_transform(reference_pixels)

        v, u = optical_flow_tvl1(reference_pixels, pixels)
        row_count, column_count = reference_pixels.shape

        row_coords, col_coords = np.meshgrid(
            np.arange(row_count),
            np.arange(column_count),
            indexing="ij",
        )

        row_coords = row_coords + v
        col_coords = col_coords + u

        return np.array([row_coords + v, col_coords + u])

    @staticmethod
    def apply_optical_flow(
        pixels: NDArray,
        flow: NDArray,
    ) -> NDArray:
        """Apply the optical flow to the pixels.

        Args:
            pixels (NDArray): The pixels to be warped.

        Returns:
            NDArray: The warped pixels.
        """
        warped: NDArray = warp(
            pixels,
            flow,
            mode="edge",
        )
        return warped

    @staticmethod
    def get_homography(
        pixels: NDArray,
        reference_pixels: NDArray,
        keypoint_distance_multiplier: float = 0.7,
        knn_matcher: Literal["bf", "flann"] = "flann",
        ransac_threshold: float = 5.0,
    ) -> NDArray:
        """Calculate the homography between two images.

        Args:
            pixels (NDArray): The pixels for which a shift is to be detected.
            reference_pixels (NDArray): The reference pixels to calculate the shift with.
            keypoint_distance_multiplier (float, optional): The multiplier for the keypoint distance used in Lowe's
                ratio test. Defaults to 0.7.
            knn_matcher (Literal["bf", "flann"], optional): The keypoint matcher to use. Defaults to "flann".
            ransac_threshold (float, optional): The threshold for the RANSAC algorithm. Defaults to 5.0.

        Returns:
            NDArray: The homography.
        """
        try:
            import cv2  # type: ignore
        except ImportError as excp:
            raise ImportError(
                "The homography method is an optional feature that requires the `opencv-python-headless`"
                " package to be installed."
            ) from excp

        # create keypoint detector
        detector: cv2.Feature2D = cv2.SIFT_create()  # type: ignore

        # if image is going to be source that we will want to fit to the pas reference image
        keypoints_src, ref_src = detector.detectAndCompute(pixels, None)  # type: ignore
        keypoints_dst, ref_dst = detector.detectAndCompute(reference_pixels, None)  # type: ignore

        # brute force match keypoints with k-nearest neighbor
        matcher: cv2.DescriptorMatcher

        if knn_matcher == "bf":
            matcher = cv2.BFMatcher()
        elif knn_matcher == "flann":
            index_params = {"algorithm": 1, "trees": 5}
            search_params: Dict[str, Union[str, bool, int, float]] = {"checks": 50}
            matcher = cv2.FlannBasedMatcher(index_params, search_params)  # type: ignore

        # get the best and second best matches
        matches = matcher.knnMatch(ref_dst, ref_src, k=2)

        # get good keypoint matches by Lowe's ratio test
        good_keypoint_matches = []
        for m, n in matches:
            if m.distance < keypoint_distance_multiplier * n.distance:
                good_keypoint_matches.append([m])

        # get matched keypoints for each image
        matches_src = np.float32([keypoints_src[m[0].trainIdx].pt for m in good_keypoint_matches])  # type: ignore
        matches_dst = np.float32([keypoints_dst[m[0].queryIdx].pt for m in good_keypoint_matches])  # type: ignore

        # Compute homography, only keep good matches thanks to RANSAC
        homography, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, ransac_threshold)  # type: ignore

        return homography  # type: ignore

    @staticmethod
    def apply_homography(
        pixels: NDArray,
        homography: NDArray,
    ) -> NDArray:
        """Apply the homography to the pixels.

        Args:
            pixels (NDArray): The pixels to be warped.

        Returns:
            NDArray: The warped pixels.
        """
        try:
            import cv2  # type: ignore
        except ImportError as excp:
            raise ImportError(
                "The homography method is an optional feature that requires the `opencv-python-headless`"
                " package to be installed."
            ) from excp
        warped: NDArray = cv2.warpPerspective(pixels, homography, (pixels.shape[1], pixels.shape[0]))
        return warped
