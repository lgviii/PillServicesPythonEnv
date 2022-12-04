import os

import ocr_accuracy_test_utils as utils
import keras_predict as predict


# Modify to set model storage directory
os.environ["KERAS_OCR_CACHE_DIR"] = r"E:\NoBackup\DGMD_E-14_FinalProject\ocr\models2"


def run_tests(parent_dir: str, labels_file_name: str, image_parent_dir: str, generate_labels,
              head: bool = False, sample: bool = False, n_entries: int = None, start_index: int = None) -> None:
    """
    Runs OCR imprint prediction accuracy testing for the keras-ocr library.

    Dumps all accuracy output to a csv file in an "outputs" subdirectory in the specified parent folder, where the CSV
    file is named based on the specified labels file name combined with element selection options.

    Expected format of the labels file depends on the method used to read it - see ocr_accuracy_test_utils methods
    read_labels_file_to_dataframe() and read_c3pi_to_dataframe() for more information.

    :param parent_dir: directory containing the labels file with the images and imprint metadata, plus an outputs
                       directory in which the accuracy output file will be created
    :param labels_file_name: name of the CSV file containing image files mapped to imprint and associated metadata
    :param image_parent_dir: parent directory containing all the image files to be run through the imprint detection.
                             For C3PI images, the directory structure is expected to match the structure on the
                             download website, otherwise all images are expected to be in this directory
    :param generate_labels: function used to build a pandas DataFrame from the specified labels file, should be either
                            read_labels_file_to_dataframe() or read_c3pi_to_dataframe() from ocr_accuracy_test_utils
                            module
    :param head: True if only the first n_entries images from the labels file should be tested, False if either a
                 random sample of images should be tested, or all images should be tested, defaults to False
    :param sample: True if a random sample of n_entries images from the labels file should be tested, False if entries
                   should be tested in order, defaults to False, ignored if head is set to True
    :param n_entries: number of images to test, only used if either head or sample is set to True, defaults to None
    :param start_index: index of the image at which testing should start, skipping all previous entries, defaults to
                        None, ignored if head or sample is set to True
    """
    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    # Set environment variable KERAS_OCR_CACHE_DIR to specify where the weights get downloaded
    ocr = predict.generate_ocr()

    print(f"Loading labels from: {labels_file_name}")
    text_file = os.path.join(parent_dir, labels_file_name)

    labels = generate_labels(text_file, image_parent_dir)

    sample_display, labels = utils.sample_images(labels, head=head, sample=sample, n=n_entries, start_index=start_index)

    output_file = os.path.join(parent_dir, "output",
                               f"keras_accuracy_{labels_file_name[0:-4]}{sample_display}_distance.csv")
    print(f"Output file: {output_file}")

    utils.run_test(ocr, labels, output_file, predict.generate_predictions, utils.find_distance)


if __name__ == "__main__":
    # Parent directory where everything else is located
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"
    # Run accuracy test on the "Clear" rate C3PI_Test images
    run_tests(parent_folder + r"\ocr", "pill_labels_full_clear.csv",
              image_parent_dir=parent_folder + r"\images\C3PI full data", generate_labels=utils.read_c3pi_to_dataframe)
    # Run accuracy test on the "Challenge" images
    run_tests(parent_folder + r"\ocr", "pill_labels_challenge.csv",
              image_parent_dir=parent_folder + r"\images\C3PI full data", generate_labels=utils.read_c3pi_to_dataframe)
    # Run accuracy test on the Split_SPL square images (front and back separately, so the predictions can be combined)
    run_tests(parent_folder + r"\ocr", "pill_labels_spl_front.csv",
              image_parent_dir=parent_folder + r"\images\split_spl_images\all_square",
              generate_labels=utils.read_labels_file_to_dataframe)
    run_tests(parent_folder + r"\ocr", "pill_labels_spl_back.csv",
              image_parent_dir=parent_folder + r"\images\split_spl_images\all_square",
              generate_labels=utils.read_labels_file_to_dataframe)