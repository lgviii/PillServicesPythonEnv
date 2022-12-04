import os
import shutil
import pandas as pd

import sort_utils as util


def split_resized_images(textfile, image_src_parent, image_dest_parent):
    df = pd.read_csv(textfile, names=["image_file", "shape"], quotechar='"', dtype="string")
    total_image_count = 0
    shape_count = {}
    for row in df.itertuples():
        shape = row.shape
        img_src = os.path.join(image_src_parent, row.image_file)
        # Check if the image source file exists - there are more entries in the database than files in the folder
        # Only try to copy if the source file exists
        if os.path.isfile(img_src):
            # Add to the color count
            if shape in shape_count:
                shape_count[shape] = shape_count[shape] + 1
            else:
                shape_count[shape] = 1

            image_folder = image_dest_parent + f"\\{shape}"
            # Create the color directory if it doesn't already exist
            os.makedirs(image_folder, exist_ok=True)

            image_dest = os.path.join(image_folder, row.image_file)
            shutil.copyfile(img_src, image_dest)
            total_image_count = total_image_count + 1
        else:
            print(f"File {img_src} not found")

    print(f"Total images copied: {total_image_count}")
    print(shape_count)


def split_original_size_images_by_shape():
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"
    textfile = os.path.join(parent_folder, "shape", "jpg_c3pi_splimage_shape_sample.csv")
    image_dest_parent = os.path.join(parent_folder, "shape", "orig_size_splimage_sort")

    util.split_original_size_images(parent_folder, textfile, "shape", image_dest_parent,
                                    lambda row: row.shape)


if __name__ == "__main__":
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"
    # text_file = os.path.join(parent_folder, "shape", "jpg_no_splimage_good_shape.csv")
    # img_src_folder = parent_folder + r"\images\all_jpgs_640_by_640"
    # img_dest_folder = os.path.join(parent_folder, "shape", "640_by_640_no_splimage_sort")

    # text_file = os.path.join(parent_folder, "shape", "splimage_split_front_shape.csv")
    # img_src_folder = parent_folder + r"\images\split_spl_images\all_square"
    # img_dest_folder = os.path.join(parent_folder, "shape", "splimage_split_square_front_sort")

    # text_file = os.path.join(parent_folder, "shape", "all_jpg_shape.csv")
    # img_src_folder = parent_folder + r"\images\all_jpgs_640_by_640"
    # img_dest_folder = os.path.join(parent_folder, "shape", "640_by_640_all_sort")

    # text_file = os.path.join(parent_folder, "shape", "splimage_split_back_shape.csv")
    # img_src_folder = parent_folder + r"\images\split_spl_images\all_square"
    # img_dest_folder = os.path.join(parent_folder, "shape", "shape_splimage_split_square_back_sort")

    text_file = os.path.join(parent_folder, "shape", "splimage_split_all_shape.csv")
    img_src_folder = parent_folder + r"\images\split_spl_images\all_square"
    img_dest_folder = os.path.join(parent_folder, "shape", "shape_splimage_split_square_all_sort")

    split_resized_images(text_file, img_src_folder, img_dest_folder)
