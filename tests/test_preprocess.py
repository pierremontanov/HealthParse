# tests/test_preprocess.py

import cv2
import numpy as np
import pytest
from src.pipeline.preprocess import preprocess_image

def generate_dummy_image():
    """Creates a simple dummy color image for preprocessing tests."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(image, "Test", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image

def test_preprocess_image_shape():
    """Ensure output is 2D (grayscale) and same size as input."""
    image = generate_dummy_image()
    processed = preprocess_image(image)
    assert processed.ndim == 2, "Processed image should be grayscale"
    assert processed.shape == image.shape[:2], "Image dimensions should be preserved"

def test_preprocess_image_thresholding():
    """Ensure that thresholding results in only two unique pixel values."""
    image = generate_dummy_image()
    processed = preprocess_image(image)
    unique_vals = np.unique(processed)
    assert all(val in [0, 255] for val in unique_vals), "Image should be binary (0 and 255 only)"
