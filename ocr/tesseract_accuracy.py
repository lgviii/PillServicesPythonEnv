import os

import test_utils as utils
import tesseract_predict as predict


def run_test_c3pi_original(parent_folder: str, text_file_name: str, sample: bool = False, head: bool = False,
                           n_entries: int = None, start_index: int = None) -> None:
    print(f"Loading labels from: {text_file_name}")
    text_file = os.path.join(parent_folder, "ocr", text_file_name)

    labels = utils.read_c3pi_to_dataframe(text_file, parent_folder + r"\images\C3PI full data")

    sample_display, labels = utils.sample_images(labels, head=head, sample=sample, n=n_entries, start_index=start_index)

    output_file = os.path.join(parent_folder, "ocr", "output",
                               f"tesseract_accuracy_{text_file_name[0:-4]}{sample_display}_distance.csv")
    print(f"Output file: {output_file}")

    utils.run_test(None, labels, output_file, predict.generate_predictions, utils.find_distance)


if __name__ == "__main__":
    predict.generate_ocr(r"C:\Main\Tesseract-OCR\tesseract")

    # Parent directory where everything else is located
    parent_dir = r"E:\NoBackup\DGMD_E-14_FinalProject"
    run_test_c3pi_original(parent_dir, "pill_labels_full_clear.txt", head=True, n_entries=1)
