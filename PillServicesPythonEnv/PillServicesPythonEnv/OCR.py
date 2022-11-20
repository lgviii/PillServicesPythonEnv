import os
from typing import List
import imutils as imutils
import numpy as np
import cv2
import keras_ocr

os.environ["KERAS_OCR_CACHE_DIR"] = "./models"

def pill_imprint_prediction(file_path):
    # Try running prediction on sample image
    
    # temp pill run

    #test_file_path = "C:\Users\lgvii\Desktop\pills\ocr_test.jpg"
    predictions = output_predictions(file_path)
    print(predictions)

    return "THIS IS A TEMP OCR PREDICTION for file: " + predictions

#!wget https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc1/images/!_4!DFN2-E8ODAOMKCG28FS3M8OQLX.JPG -O pill.jpg


def create_pipeline():
    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    # Set environment variable KERAS_OCR_CACHE_DIR to specify where the weights get downloaded
    pipeline = keras_ocr.pipeline.Pipeline()
    return pipeline


def _create_image(image_file: str) -> np.ndarray:
    image = keras_ocr.tools.read(image_file)
    return image


def _sharpen_image(image) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(image, -1, kernel)
    return image_sharp


def _generate_single_prediction_set(pipeline, image) -> List[str]:
    prediction_groups = pipeline.recognize([image])
    predictions = [prediction[0] for prediction in prediction_groups[0]]
    return predictions


def generate_predictions(pipeline, image_file: str, rotate: bool = False) -> List[List[str]]:
    """
    Generates predictions from the specified image file, optionally rotating it by 90, 180, and 270 degrees (useful for
    testing accuracy against test images).

    Note that the keras-ocr pipeline object is assumed to be created by the caller.  This is to facilitate batch
    testing, so the pipeline doesn't get reinitialized for each image tested.

    Note that the returned object will NOT be an empty List if no text is recognized.  It will instead contain one or
    more empty Lists, each from a permutation of the supplied image.

    :param pipeline: keras-ocr pipeline, created/initialized outside this method so that batch testing doesn't have to
                     recreate it for each image tested
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
                all_predictions.append(_generate_single_prediction_set(pipeline, rotated))
        else:
            all_predictions.append(_generate_single_prediction_set(pipeline, image))

    return all_predictions

def output_predictions(image_file: str, output_file: str = None, rotate: bool = False, delimiter: str = ";") -> str:
    """
    Creates a keras-ocr pipeline and generates text predictions on the specified image file, optionally writing the
    output to a specified output file.

    Output will consist of multiple lines, each terminated with "\n", where each line contains all predictions from a
    single permutation of the image (original, sharpened, rotated, etc) concatenated together using the specified
    delimiter. (The default delimiter is chosen specifically for use with pill imprints, as the imprint text itself
    uses the same delimiter.)  If no text was detected, output will contain one or more empty lines.

    :param image_file: path to the image on which OCR predictions should be generated
    :param output_file: path of an output text file to which the generated OCR predictions should be written, will
                        overwrite any previous content of the file, defaults to None
    :param rotate: True if additional rotational permutations of the image should be generated when predicting text,
                   False if only the image (and any sharpening or other operation permutations) should be used,
                   defaults to False
    :param delimiter: delimiter used to separate the text groups predicted from a single image permutation,
                      defaults to ";"
    :return: string containing predictions from all permutations, with the text from each permutation prediction
             concatenated to a single line using the specified delimiter, and each permutation line separated by "\n"
    """
    pipeline = create_pipeline()
    all_predictions = generate_predictions(pipeline, image_file, rotate)

    result = ""
    for prediction_group in all_predictions:
        output = delimiter.join(prediction_group)
        result = result + output + "\n"

    if output_file:
        with open(output_file, "w") as file:
            file.write(f"{result}\n")

    return result
