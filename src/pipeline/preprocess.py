import cv2

def preprocess_image(image):
    """
    Convert image to grayscale and apply thresholding for better OCR accuracy.

    Args:
        image (numpy.ndarray): Original image in BGR format.

    Returns:
        numpy.ndarray: Preprocessed image ready for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return thresh
