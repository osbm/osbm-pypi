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
