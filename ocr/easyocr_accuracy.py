import os

import ocr__accuracy_test_utils as utils
import easyocr_predict as predict


def run_test_c3pi_original(parent_folder: str, text_file_name: str, sample: bool = False, head: bool = False,
                           n_entries: int = None, start_index: int = None) -> None:
    # Set gpu False to use CPU only
    # Set model_path with directory where model files should be downloaded
    ocr = predict.generate_ocr()

    print(f"Loading labels from: {text_file_name}")
    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_c3pi_to_dataframe(text_file, parent_folder + r"\images\C3PI full data")

    sample_display, labels = utils.sample_images(labels, head=head, sample=sample, n=n_entries, start_index=start_index)

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"easyocr_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(f"Output file: {output_file}")

    utils.run_test(ocr, labels, output_file, predict.generate_predictions, utils.find_distance)


def run_test_spl_front_square(parent_folder: str, text_file_name: str, image_parent_dir: str,
                              sample: bool = False, head: bool = False,
                              n_entries: int = None, start_index: int = None) -> None:
    # Set gpu False to use CPU only
    # Set model_path with directory where model files should be downloaded
    ocr = predict.generate_ocr()

    print(f"Loading labels from: {text_file_name}")
    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_labels_file_to_dataframe(text_file, image_parent_dir)

    sample_display, labels = utils.sample_images(labels, head=head, sample=sample, n=n_entries, start_index=start_index)

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"easyocr_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(f"Output file: {output_file}")

    utils.run_test(ocr, labels, output_file, predict.generate_predictions, utils.find_distance)


if __name__ == "__main__":
    # Parent directory where everything else is located
    parent_dir = r"E:\NoBackup\DGMD_E-14_FinalProject"
    # run_test_spl_front_square(parent_dir, "pill_labels_spl_back.txt",
    #                           parent_dir + r"\images\split_spl_images\all_square")
    run_test_c3pi_original(parent_dir, "pill_labels_full_clear.txt", start_index=2599)
