# import pytest

# import imaging
import osbm
import numpy as np


# testing osbm.imaging.pad_image_to_square
def test_pad_image_to_square():
    """
    Test the pad_image_to_square function.
    Tests:
        - Square image should not be padded
        - Tall image should be padded to be square
        - Wide image should be padded to be square
    """
    square_image = np.zeros((100, 100, 3))
    padded_image = osbm.imaging.pad_image_to_square(square_image)
    assert (
        padded_image.shape == (100, 100, 3) == square_image.shape
    ), "Square image should not be padded"

    tall_image = np.zeros((200, 100, 3))
    padded_image = osbm.imaging.pad_image_to_square(tall_image)
    assert padded_image.shape == (
        200,
        200,
        3,
    ), "Tall image should be padded to be square"

    wide_image = np.zeros((100, 200, 3))
    padded_image = osbm.imaging.pad_image_to_square(wide_image)
    assert padded_image.shape == (
        200,
        200,
        3,
    ), "Wide image should be padded to be square"
