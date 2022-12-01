from typing import List
import imutils
import numpy as np
import cv2
import pytesseract


def generate_ocr(tesseract_path: str) -> None:
    """
    If the Tesseract executable isn't in the system path, this method must be run to specify the directory containing
    the Tesseract executable so that pytesseract can find it.

    Doesn't actually generate anything, just given this name to match the signatures of the methods in the other
    "predict" modules for easier reuse.

    :param tesseract_path: file path of the directory containing the Tesseract executable
    """
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def _create_image(image_file: str) -> np.ndarray:
    image = cv2.imread(image_file)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb


def _sharpen_image(image) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(image, -1, kernel)
    return image_sharp


def _generate_single_prediction_set(image) -> List[str]:
    full_prediction = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    conf = full_prediction["conf"]
    text = full_prediction["text"]
    predictions = []
    for i, confidence in enumerate(conf):
        if confidence > 0 and len(text[i]) > 0:
            predictions.append(text[i])
    return predictions


def generate_predictions(ocr, image_file: str, rotate=False) -> List[List[str]]:
    """
    Generates predictions from the specified image file, optionally rotating it by 90, 180, and 270 degrees (useful for
    testing accuracy against test images).

    Note that the returned object will NOT be an empty List if no text is recognized.  It will instead contain one or
    more empty Lists, each from a permutation of the supplied image.

    :param ocr: ignored parameter, just used to make the method signatures identical across the predict modules, so
                they can be more easily plugged into the shared test_utils
    :param image_file: path of the image file to check for text
    :param rotate: True if predictions should also be generated for the image at 90, 180, and 270 degrees, intended
                   for use with batch testing of stock images that may be rotated, defaults to False
    :return: List containing prediction groups for each image permutation, where each prediction group is itself a
             List of text strings.  Note that if no text is found, this will NOT be an empty List, but will instead
             contain multiple empty Lists
    """
    base_image = _create_image(image_file)
    image_sharp = _sharpen_image(base_image)
    images = [base_image, image_sharp]
    all_predictions = []
    for image in images:
        # Rotate the image across 0, 90, 180, and 270, in case the pill is rotated, to improve readability
        if rotate:
            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rotated = image
                else:
                    rotated = imutils.rotate_bound(image, angle)
                all_predictions.append(_generate_single_prediction_set(rotated))
        else:
            all_predictions.append(_generate_single_prediction_set(image))

    return all_predictions
