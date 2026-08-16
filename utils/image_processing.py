"""
image_processing.py
-------------------
Image preprocessing utilities for Intelligent Traffic Control System.

Author : Rachana
Project : Intelligent Traffic Control System Using Vehicle Detection
"""

import cv2

from config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ENABLE_PREPROCESSING
)


class ImageProcessor:
    """
    Handles all image preprocessing operations.
    """

    def __init__(self):
        pass

    def resize(self, image):
        """
        Resize image to the required input size.
        """

        return cv2.resize(
            image,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

    def clahe(self, image):
        """
        Enhance image contrast using CLAHE.
        """

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

    def denoise(self, image):
        """
        Reduce image noise.
        """

        return cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    def preprocess(self, image):
        """
        Complete preprocessing pipeline.
        """

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        image = self.resize(image)

        if ENABLE_PREPROCESSING:

            image = self.clahe(image)

            image = self.denoise(image)

        return image