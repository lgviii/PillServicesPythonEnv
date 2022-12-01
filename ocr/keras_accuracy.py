import os

import ocr_accuracy_test_utils as utils
import keras_predict as predict


def run_test_c3pi_original(parent_folder: str, text_file_name: str, head: bool = False, sample: bool = False,
                           n_entries: int = None, start_index: int = None) -> None:

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    # Set environment variable KERAS_OCR_CACHE_DIR to specify where the weights get downloaded
    pipeline = predict.generate_ocr()

    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_c3pi_to_dataframe(text_file, parent_folder + r"\images\C3PI full data")

    sample_display, labels = utils.sample_images(labels, head=head, sample=sample, n=n_entries, start_index=start_index)
    print(labels["image_file"].head())

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"keras_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(f"Output file: {output_file}")

    utils.run_test(pipeline, labels, output_file, predict.generate_predictions, utils.find_distance)


def run_test_spl_front_square(parent_folder: str, text_file_name: str, image_parent_dir: str,
                              sample: bool = False, head: bool = False,
                              n_entries: int = None, start_index: int = None) -> None:

    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    # Set environment variable KERAS_OCR_CACHE_DIR to specify where the weights get downloaded
    ocr = predict.generate_ocr()

    print(f"Loading labels from: {text_file_name}")
    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_labels_file_to_dataframe(text_file, image_parent_dir)

    sample_display, labels = utils.sample_images(labels, head=head, sample=sample, n=n_entries, start_index=start_index)
    print(labels["image_file"].head())

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"keras_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(f"Output file: {output_file}")

    utils.run_test(ocr, labels, output_file, predict.generate_predictions, utils.find_distance)


if __name__ == "__main__":
    # Parent directory where everything else is located
    parent_dir = r"E:\NoBackup\DGMD_E-14_FinalProject"
    run_test_spl_front_square(parent_dir, "pill_labels_spl_back.txt",
                              parent_dir + r"\images\split_spl_images\all_square")
