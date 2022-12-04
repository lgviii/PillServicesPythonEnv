import os
import shutil
import pandas as pd


def read_c3pi_to_dataframe(labels_filepath: str, parent_image_dir: str, category: str) -> pd.DataFrame:
    """
    Read a labels file and return a pandas DataFrame with a new "file_path" column containing the full path to the
    image file in that entry.

    Labels file assumed to be in the format (C3PI directory), (Image file name), "(category value)"
    Imprint text expected to list all imprint items separated by semicolons

    :param labels_filepath: path to labels file
    :param parent_image_dir: path to parent folder containing the C3PI data in the original C3PI directory structure
    :param category: name of the column for the category values
    :return: Pandas DataFrame containing the elements from the original CSV file labeled as "image_dir", "image_file",
             and the specified category name, plus a new "file_path" column containing the full path to the image file
             in each entry
    """
    labels = pd.read_csv(labels_filepath,
                         names=["image_dir", "image_file", category],
                         quotechar='"',
                         dtype="string")

    labels["file_path"] = \
        labels.apply(lambda row: os.path.join(parent_image_dir, row["image_dir"], "images", row["image_file"]),
                     axis=1)
    labels = labels.astype("string")
    labels = labels.fillna("Empty")
    return labels


def split_original_size_images(parent_folder: str, textfile: str, category: str, image_dest_parent: str,
                               get_category_value_from_row):
    image_src_parent = parent_folder + r"\images\C3PI full data"

    df = read_c3pi_to_dataframe(textfile, image_src_parent, category)

    total_image_count = 0
    category_count = {}
    for row in df.itertuples():
        category_value = get_category_value_from_row(row)
        img_src = row.file_path
        # Check if the image source file exists in case there are more entries in the text file than exist in the
        # image folder
        # Only try to copy if the source file exists
        if os.path.isfile(img_src):
            # Add to the color count
            if category_value in category_count:
                category_count[category_value] = category_count[category_value] + 1
            else:
                category_count[category_value] = 1

            image_folder = image_dest_parent + f"\\{category_value}"
            # Create the shape directory if it doesn't already exist
            os.makedirs(image_folder, exist_ok=True)

            image_dest = os.path.join(image_folder, row.image_file)
            shutil.copyfile(img_src, image_dest)
            total_image_count = total_image_count + 1

    print(f"Total images copied: {total_image_count}")
    print(category_count)
