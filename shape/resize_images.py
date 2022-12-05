import os
import shutil
from PIL import Image, ImageOps
import pandas as pd


def resize_image_crop(image_file: str, output_file: str, final_width: int, final_height: int) -> None:
    """
    Resizes the specified image to the specified size by shrinking/growing so that one edge matches the final size and
    the other edge is larger (maintaining the original aspect ratio), then cropping out the center of the image to the
    final size.  Saves the resized image to the specified output file.

    That is, if the initial size is 100 x 200 and the final size is 50 x 80, the image will first be shrunk to 50 x 100
    to preserve the original aspect ratio, then center cropped to 50 x 80.

    If the original image is already the final size, it's returned unmodified.

    :param image_file: image to be resized
    :param output_file: path to use when saving the resized image
    :param final_width: final width of the resized image
    :param final_height: final height of the resized image
    """
    img = Image.open(image_file)
    width = img.width
    height = img.height

    # Only convert and save the image if we had to resize or pad, since JPG is lossy
    if width != final_width or height != final_height:
        new_img = img
        # If both width and height are wrong, scale the image up or down so that the smaller side is the correct size
        if width != final_width and height != final_height:
            factor = max(float(final_width) / width, float(final_height) / height)
            new_width = round(factor * width)
            new_height = round(factor * height)
            new_img = img.resize((new_width, new_height), Image.LANCZOS)
        # Now crop
        new_width = new_img.width
        new_height = new_img.height
        delta_width = new_width - final_width
        delta_height = new_height - final_height
        crop = (delta_width // 2, delta_height // 2,
                final_width + (delta_width // 2), final_height + (delta_height // 2))
        new_img = new_img.crop(crop)
        new_img.save(output_file, "JPEG", quality=95)

    # If the size is already correct, just copy the original image to the output file
    else:
        shutil.copyfile(image_file, output_file)


def resize_image_with_padding(image_file: str, output_file: str, final_width: int, final_height: int) -> None:
    """
    Resizes the specified image to the specified final width and height using padding to preserve the original aspect
    ratio, saving it as the specified output file.  First resizes the image so that the larger side matches the final
    size, and then pads the edges of the shorter side using the color of the (almost) top left corner pixel.

    That is, if the initial size is 100 x 200 and the final size is 50 x 80, the image will first be shrunk to 40 x 80
    to preserve the original aspect ratio, then padded to 50 x 80.

    If the original image is already the final size, it's simply copied to the output file path with no modifications.

    :param image_file: image file containing the image to be resized
    :param output_file: path to use when saving the resized image
    :param final_width: final width of the resized image
    :param final_height: final height of the resized image
    """
    img = Image.open(image_file)
    width = img.width
    height = img.height
    # Get the color of the top left corner pixel to use for fill if needed, before resizing
    pixel_color = img.getpixel((1, 1))

    # Only convert and save the image if we had to resize or pad, since JPG is lossy
    if width != final_width or height != final_height:
        # If both width and height are wrong, scale the image up or down so that one of the sides is
        # the correct size
        if width != final_width and height != final_height:
            factor = min(float(final_width) / width, float(final_height) / height)
            new_width = round(factor * width)
            new_height = round(factor * height)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        # Now calculate any padding needed to extend either side to the final size
        # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        new_width = img.width
        new_height = img.height
        delta_width = final_width - new_width
        delta_height = final_height - new_height
        padding = (delta_width // 2,
                   delta_height // 2,
                   delta_width - (delta_width // 2),
                   delta_height - (delta_height // 2))
        new_img = ImageOps.expand(img, padding, pixel_color)
        new_img.save(output_file, "JPEG", quality=95)

    # Otherwise the image is already the right size, so just copy the image as is
    else:
        shutil.copyfile(image_file, output_file)


def read_c3pi_to_dataframe(labels_filepath: str, parent_image_dir: str, category: str) -> pd.DataFrame:
    """
    Read a labels file and return a pandas DataFrame with a new "file_path" column containing the full path to the
    image file in that entry.

    Labels file assumed to be in the format (C3PI directory), (Image file name), "(associated value)"

    :param labels_filepath: path to labels file
    :param parent_image_dir: path to parent folder containing the C3PI data in the original C3PI directory structure
    :param category: name of the column for the associated values
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


def resize_c3pi_images_to_square(text_file: str, image_src_parent: str, dest_dir: str, size: int = 640):
    df = read_c3pi_to_dataframe(text_file, image_src_parent, "image_class")

    count = 0
    for row in df.itertuples():
        src_file = row.file_path
        dest_file = os.path.join(dest_dir, row.image_file)
        print(dest_file)
        # For C3PI_TEST, we'll crop instead of pad since these images always have extra background that can be removed
        if row.image_class == "C3PI_Test":
            resize_image_crop(src_file, dest_file, size, size)
        else:
            # For all other (reference) image types, the pill may be close to the edge and the background is fairly
            # uniform, so padding is safer
            resize_image_with_padding(src_file, dest_file, size, size)
        count = count + 1

    print(f"Images resized: {count}")


def pad_image_to_square(image_file: str, output_file: str) -> None:
    """
    Pads the specified image to be square without changing the size of the larger side, using the color from the
    (almost) top left corner pixel.

    That is, if the original image is 100x200, it will be padded to 200x200.  If the original image is 50x100, it will
    be padded to 100x100.

    If the image is already square, it's just copied to the output file path without modifications.

    :param image_file: path to the file containing the image to pad to square
    :param output_file: path to save the padded square image
    """
    img = Image.open(image_file)
    width = img.width
    height = img.height
    # Get the color of the almost top left corner pixel to use for fill if needed, before resizing
    # Using the very top left sometimes gets a slightly odd color, works better to use the next pixel in
    pixel_color = img.getpixel((1, 1))

    # Only convert and save the image if we had to pad (image isn't already square), since JPG is lossy
    if width != height:
        size = max(width, height)
        # Calculate any padding needed to extend either side to the final size
        # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        delta_width = size - width
        delta_height = size - height
        padding = (delta_width // 2,
                   delta_height // 2,
                   delta_width - (delta_width // 2),
                   delta_height - (delta_height // 2))
        new_img = ImageOps.expand(img, padding, pixel_color)
        new_img.save(output_file, "JPEG", quality=95)

    # Otherwise the image is already the right size, so just copy the image as is
    else:
        shutil.copyfile(image_file, output_file)


if __name__ == "__main__":
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"
    text_file = os.path.join(parent_folder, "all_jpg_images_with_class.csv")
    image_src_parent = parent_folder + r"\images\C3PI full data"
    dest_folder = parent_folder + r"\images\all_jpgs_640_by_640_fixed"
    resize_c3pi_images_to_square(text_file, image_src_parent, dest_folder)

    # parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"
    # text_file = os.path.join(parent_folder, "all_jpg_good_images_with_class_sample.csv")
    # image_src_parent = parent_folder + r"\images\C3PI full data"
    # dest_folder = parent_folder + r"\images\640_by_640_sample"
    # resize_c3pi_images_to_square(text_file, image_src_parent, dest_folder)
