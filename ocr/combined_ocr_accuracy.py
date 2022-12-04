import os
import pandas as pd
import ocr_accuracy_test_utils as utils


def read_labels_file_to_dataframe(labels_filepath: str, latin1: bool = False) -> pd.DataFrame:
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
    labels = None
    if latin1:
        labels = pd.read_csv(labels_filepath,
                             names=["image_file", "ndc11", "part", "imprint_rating", "imprint_type", "time",
                                    "imprint", "best_accuracy", "pred0", "acc0", "pred90", "acc90", "pred180", "acc180",
                                    "pred270", "acc270", "pred_s0", "acc_s0", "pred_s90", "acc_s90", "pred_s180",
                                    "acc_s180", "pred_s270", "acc_s270"],
                             quotechar='"', encoding="latin1")
    else:
        labels = pd.read_csv(labels_filepath,
                             names=["image_file", "ndc11", "part", "imprint_rating", "imprint_type", "time",
                                    "imprint", "best_accuracy", "pred0", "acc0", "pred90", "acc90", "pred180", "acc180",
                                    "pred270", "acc270", "pred_s0", "acc_s0", "pred_s90", "acc_s90", "pred_s180",
                                    "acc_s180", "pred_s270", "acc_s270"],
                             quotechar='"')

    # Extract just the original source file name (minus SF_ or SB_) from the full path
    labels["file_name"] = \
        labels.apply(lambda row: os.path.basename(row["image_file"])[3:], axis=1)

    # Convert all the strings to string type - leave NDC11 as an int, even though it's actually a string it's fine that
    # way for this purpose
    labels = labels.astype({
        "image_file": "string",
        "imprint_rating": "string",
        "imprint_type": "string",
        "imprint": "string",
        "pred0": "string",
        "pred90": "string",
        "pred180": "string",
        "pred270": "string",
        "pred_s0": "string",
        "pred_s90": "string",
        "pred_s180": "string",
        "pred_s270": "string",
        "file_name": "string"
    })
    # Don't bother including all the accuracy calcs, we won't use them
    return labels[["image_file", "ndc11", "part", "imprint_rating", "imprint_type", "time",
                   "imprint", "pred0", "pred90", "pred180", "pred270", "pred_s0",
                   "pred_s90", "pred_s180", "pred_s270", "file_name"]]


def calc_combined_accuracy(parent_dir: str, model: str, latin1: bool = False):
    front = read_labels_file_to_dataframe(os.path.join(parent_dir,
                                                       f"{model}_accuracy_pill_labels_spl_front_distance_no_total.csv"),
                                          latin1=latin1)
    back = read_labels_file_to_dataframe(os.path.join(parent_dir,
                                                      f"{model}_accuracy_pill_labels_spl_back_distance_no_total.csv"),
                                         latin1=latin1)
    # Only merge the image file and predictions from the "back" data set, the other data will be the same
    combined = pd.merge(front, back[["image_file", "time", "pred0", "pred90", "pred180", "pred270", "pred_s0",
                                     "pred_s90", "pred_s180", "pred_s270", "file_name"]],
                        on="file_name", suffixes=("_f", "_b"))
    print(combined.info())
    combined["time"] = combined["time_f"] + combined["time_b"]
    for degree in [0, 90, 180, 270]:
        for image in ["", "_s"]:
            combined[f"pred{image}{degree}"] = \
                combined[[f"pred{image}{degree}_f", f"pred{image}{degree}_b"]].stack().groupby(level=0).agg(";".join)

            # Now that the combined predictions have been generated, replace all N/A values with ""
            # (didn't want to do this earlier to avoid having ";" prediction strings)
            combined[f"pred{image}{degree}"] = combined[f"pred{image}{degree}"].fillna("")
    print(combined[["pred0_f", "pred0_b", "pred0", "pred_s0"]].head(30))
    print(combined[["time_f", "time_b", "time"]].head())

    # Now calculate the accuracy using the combined predictions
    for degree in [0, 90, 180, 270]:
        for image in ["", "_s"]:
            combined[f"acc{image}{degree}"] = \
                combined.apply(lambda row: utils.calc_accuracy(row[f"pred{image}{degree}"].split(";"),
                                                               row["imprint"], utils.find_distance),
                               axis=1)
    combined["acc_max"] = combined.filter(regex=r"acc(_s)?\d+").stack().groupby(level=0).max()
    combined.to_csv(os.path.join(parent_dir, f"{model}_combined_accuracy_distance.csv"))


if __name__ == "__main__":
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject\ocr\output"
    calc_combined_accuracy(parent_folder, "keras")
    calc_combined_accuracy(parent_folder, "easyocr", latin1=True)
