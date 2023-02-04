"""
This module contains functions for image processing.
"""

import numpy as np


def read_dicom_with_windowing(dcm_file: str) -> np.ndarray:
    """
    Read a DICOM file and apply windowing.
    source: https://www.kaggle.com/code/davidbroberts/mammography-apply-windowing/

    Args:
        dcm_file: Path to the DICOM file

    Returns:
        The image as a numpy array
    """
    # pylint: disable=import-outside-toplevel
    import pydicom
    from pydicom.pixel_data_handlers import apply_windowing

    image = pydicom.dcmread(dcm_file)
    data = image.pixel_array
    data = apply_windowing(data, image)

    if image.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    else:
        data = data - np.min(data)

    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def pad_image_to_square(image: np.ndarray, keep_image_in_center=True) -> np.ndarray:
    """
    Pad an image to make it square.

    Args:
        image: The image to pad

    Returns:
        The padded image
    """
    height, width = image.shape[:2]
    if height == width:
        return image

    if height > width:
        padding = ((0, 0), ((height - width) // 2, (height - width) // 2), (0, 0))
    else:
        padding = (((width - height) // 2, (width - height) // 2), (0, 0), (0, 0))

    if keep_image_in_center:
        padded_image = np.pad(image, padding, mode="constant", constant_values=0)

    else:
        padded_image = np.pad(image, padding, mode="edge")

    return padded_image
