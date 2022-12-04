import os
import csv
import re
from typing import List, Tuple
import time
import Levenshtein
import pandas as pd


def read_labels_file_to_dataframe(labels_filepath: str, parent_image_dir: str) -> pd.DataFrame :
    """Read a labels file and return a pandas DataFrame with a new "file_path" column containing the full path to the
    image file in that entry.

    Labels file assumed to be in the format:
     "(Image file name)", (Imprint rating), (Imprint type), "(Imprint text)", (NDC11), (Part)
    Imprint text expected to list all imprint items separated by semicolons, if present, or be null if not present

    DataFrame columns:
    ["image_file", "imprint_rating", "imprint_type", "imprint", "ndc11", "part", "file_path"]

    :param labels_filepath: Path to labels file
    :param parent_image_dir: Path to parent folder containing the C3PI data in the original C3PI directory structure
    :return: Pandas DataFrame containing the elements from the original CSV file, plus a new "file_path" column
             containing the full path to the image file in each entry
    """
    labels = pd.read_csv(labels_filepath,
                         names=["image_file", "imprint_rating", "imprint_type", "imprint", "ndc11", "part"],
                         quotechar='"',
                         dtype="string")

    labels["file_path"] = \
        labels.apply(lambda row: os.path.join(parent_image_dir, row["image_file"]),
                     axis=1)
    labels = labels.astype("string")
    # Fill in just the imprint_rating column missing values with "Empty" for accuracy tracking
    labels["imprint_rating"] = labels["imprint_rating"].fillna("Empty")
    return labels


def read_c3pi_to_dataframe(labels_filepath: str, parent_image_dir: str) -> pd.DataFrame:
    """
    Read a labels file and return a pandas DataFrame with a new "file_path" column containing the full path to the
    image file in that entry.

    Labels file assumed to be in the format
    "(C3PI directory)", "(Image file name)", (Imprint rating), (Imprint type), "(Imprint text)", (NDC11), (Part)
    Imprint text expected to list all imprint items separated by semicolons.

    DataFrame columns:
    ["image_dir", "image_file", "imprint_rating", "imprint_type", "imprint", "ndc11", "part", "file_path"]

    :param labels_filepath: Path to labels file
    :param parent_image_dir: Path to parent folder containing the C3PI data in the original C3PI directory structure
    :return: Pandas DataFrame containing the elements from the original CSV file, plus a new "file_path" column
             containing the full path to the image file in each entry
    """
    labels = pd.read_csv(labels_filepath,
                         names=["image_dir", "image_file", "imprint_rating", "imprint_type", "imprint",
                                "ndc11", "part"],
                         quotechar='"',
                         dtype="string")

    labels["file_path"] = \
        labels.apply(lambda row: os.path.join(parent_image_dir, row["image_dir"], "images", row["image_file"]),
                     axis=1)
    labels = labels.astype("string")
    # Fill in just the imprint_rating column missing values with "Empty" for accuracy tracking
    labels["imprint_rating"] = labels["imprint_rating"].fillna("Empty")
    return labels


def sample_images(labels: pd.DataFrame, head: bool = False, sample: bool = False, n: int = 100,
                  start_index: int = None) -> (str, pd.DataFrame):
    """
    Narrows down the images that will be used for testing based on the specified options, if any, returning a tuple
    containing a display string to be appended to the output file name and the final DataFrame containing the images
    to be tested.

    Options are preferentially used in the order of the arguments.  If head is True, sample and start_index are ignored.
    If head is False but sample is True, start_index is ignored.

    :param labels: pandas DataFrame to be reduced as specified
    :param head: True if only the first n entries should be included in the returned DataFrame, False if some other
                 selection mechanism (if any) should be used, defaults to False
    :param sample: True if only a random sample of n entries should be included in the returned DataFrame, False if
                   some other selection mechanism (if any) should be used, ignored if head is True, defaults to False
    :param n: number of entries to include in the returned DataFrame, only used if either head or sample are True,
              ignored otherwise, defaults to 100
    :param start_index: index of the first entry to be included in the returned DataFrame, skipping all previous
                        entries, ignored if either head or sample are True, defaults to None
    :return: tuple containing a display string to be appended to the output file name based on the specified options
             plus the final DataFrame containing the images to be tested based
    """
    sample_display = ""
    if head:
        labels = labels.head(n=n)
        sample_display = f"_{n}"
    elif sample:
        labels = labels.sample(n=n)
        sample_display = f"_{n}_random"
    elif start_index is not None:
        labels = labels[start_index:]
        sample_display = f"_start{start_index}"
    return sample_display, labels


def create_accuracy_tracking_dict():
    return {
        "Clear": {
            "total": 0.0,
            "count": 0
        },
        "Partial": {
            "total": 0.0,
            "count": 0
        },
        "Empty": {
            "total": 0.0,
            "count": 0
        }
    }


def find_strict_match(prediction: str, imprint_sections: List[str]) -> (int, float):
    """
    Find the imprint section that is identical to the predicted text.

    :param prediction: Predicted text, should not be null/empty
    :param imprint_sections: List of all the imprint sections
    :return: Tuple containing the index of the imprint section that is identical to the prediction as the first element
             and 1.0 (the accuracy factor) as the second element, or (-1, 0.0) if none of the imprint sections are
             identical to the prediction
    """
    for i, imprint in enumerate(imprint_sections):
        if prediction == imprint:
            return i, 1.0
    return -1, 0.0


def find_distance(prediction: str, imprint_sections: List[str]) -> (int, float):
    """
    Find the imprint section that best matches the specified prediction "word", based on Levenshtein ration, with a
    cutoff of 0.5 minimum accuracy.

    Returns a tuple containing the index of the matching imprint section and the calculated accuracy of the match, or
    (-1, 0.0) if none of the imprint sections match at better than 0.5

    :param prediction: Predicted text, should not be null/empty
    :param imprint_sections: List of all the imprint sections
    :return: Tuple containing the index of the imprint section that is the closest match to the prediction (if any)
             and the Levenshtein ratio accuracy, or (-1, 0.0) if none of the imprint sections match the prediction
             within 0.5 accuracy
    """
    # Try using the Levenshtein ratio to match, with a cutoff of 0.5
    # No need to first check exact match, since that will have a distance of 1.0
    # and end up being the best distance
    index = -1
    best_distance = 0.0

    for i, section in enumerate(imprint_sections):
        # The ratio function returns a normalized similarity in range [0, 1], and is 1 - normalized distance
        # So perfect match = 1.0, failed match = 0.0
        distance = Levenshtein.ratio(prediction, section, score_cutoff=0.5)
        if distance > best_distance:
            index = i
            best_distance = distance

    return index, best_distance


def calc_accuracy(predictions: List[str], imprint: str, match_imprint) -> float:
    """
    Calculates the overall accuracy for the specified imprint predictions from a single image permutation, using the
    specified match_imprint function to find matching imprint sections and calculate individual "word" match accuracy.

    Iterates across each predicted "word" to find a matching imprint section.  If found, removes that imprint section
    from future iterations to avoid incorrect duplicate matches.  Adds together the total accuracy for each predicted
    word, and divides by the total number of imprint sections to generate the overall accuracy of the predictions.

    :param predictions: List of predicted "words" from a single image permutation
    :param imprint: actual imprint for the pill, containing sections separated by a semicolon
    :param match_imprint: function used to match a predicted "word" against an imprint section
    :return: overall accuracy in matching the predicted "words" against the specified imprint
    """
    if pd.isnull(imprint):
        empty_prediction = len(predictions) == 0 or "".join(predictions).strip() == ""
        return 1.0 if empty_prediction else 0.0
    else:
        # Convert the imprint to lower case to avoid case issues and break it into separate portions using the
        # semicolon delimiter
        imprint_sections = imprint.lower().split(";")
        # Convert predictions into lower case as well to make matching easier
        predictions = [prediction.lower() for prediction in predictions]

        sections_matched_accuracy = 0.0
        sections = imprint_sections.copy()
        if len(predictions) > 0:
            for prediction in predictions:
                if len(prediction.strip()):
                    matching_index, factor = match_imprint(prediction, sections)
                    # If a match is found, remove that element from the list of imprint sections in case the prediction has
                    # duplicates but the imprints don't - that way only the first one will be counted as a successful match
                    if matching_index > -1:
                        print(f"Prediction: {prediction}, match: {sections[matching_index]}, factor: {factor}")
                        sections_matched_accuracy = sections_matched_accuracy + factor
                        del sections[matching_index]

        return sections_matched_accuracy / len(imprint_sections)


def test_image(ocr, df_row, output_file: str, do_ocr, match_imprint) -> float:
    image_file = df_row.file_path
    imprint = df_row.imprint
    print(image_file)
    start = time.time()
    all_predictions = do_ocr(ocr, image_file, rotate=True)
    stop = time.time()

    prediction_outputs = []
    highest_accuracy = 0.0

    for prediction_group in all_predictions:
        accuracy = calc_accuracy(prediction_group, imprint, match_imprint)
        prediction_outputs.append(";".join(prediction_group))
        prediction_outputs.append(accuracy)
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy

    with open(output_file, "a", newline="") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([image_file, df_row.ndc11, df_row.part, df_row.imprint_rating, df_row.imprint_type,
                         stop - start, imprint, highest_accuracy] + prediction_outputs)

    return highest_accuracy


def run_test(ocr, labels: pd.DataFrame, output_file: str, do_ocr, calc_match):
    total_accuracy = 0.0
    accuracy_tracking = create_accuracy_tracking_dict()

    for row in labels.itertuples():
        image_file = row.file_path
        # Only test accuracy if the image file exists - it may not
        if os.path.isfile(image_file):
            accuracy = test_image(ocr, row, output_file, do_ocr, calc_match)
            total_accuracy = total_accuracy + accuracy
            accuracy_sub = accuracy_tracking[row.imprint_rating]
            accuracy_sub["total"] = accuracy_sub["total"] + accuracy
            accuracy_sub["count"] = accuracy_sub["count"] + 1

    accuracy_output = f"Overall accuracy: {total_accuracy / len(labels)}"
    for rating in accuracy_tracking:
        rating_data = accuracy_tracking[rating]
        if rating_data["count"] > 0:
            rating_accuracy = rating_data["total"] / rating_data["count"]
        else:
            rating_accuracy = "No values"
        accuracy_output = accuracy_output + f"  {rating} accuracy: {rating_accuracy}"

    print(accuracy_output)
    with open(output_file, "a") as file:
        file.write(accuracy_output + "\n")
