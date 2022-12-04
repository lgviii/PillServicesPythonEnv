import os
import shutil
from PIL import Image, ImageOps
import sort_utils as util


def resize_image_crop(image_file: str, final_width: int, final_height: int) -> Image:
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
                new_width - (delta_width // 2), new_height - (delta_height // 2))
        new_img = new_img.crop(crop)
        return new_img
    # If the size is already correct, just return the opened image
    else:
        return img


def resize_image_with_padding(image_file: str, output_file: str, final_width: int, final_height: int):
    img = Image.open(image_file)
    width = img.width
    height = img.height
    # Get the color of the top left corner pixel to use for fill if needed, before resizing
    pixel_color = img.getpixel((0, 0))

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


def resize_image_based_on_class(image_file: str, image_class: str, output_file: str, final_width: int, final_height):
    img = Image.open(image_file)
    width = img.width
    height = img.height
    # Get the color of the top left corner pixel to use for fill if needed, before resizing
    pixel_color = img.getpixel((0, 0))

    # Only convert and save the image if we had to resize or pad, since JPG is lossy
    if width != final_width or height != final_height:
        new_img = img
        # If both width and height are wrong, scale the image up or down so that one of the sides is
        # the correct size
        if width != final_width and height != final_height:
            factor = 1.0
            # For C3PI_TEST, we'll crop instead of pad since these images always have extra background that can be lost
            # So we want to shrink/grow so that the smaller side is the final size
            if image_class == "C3PI_Test":
                factor = max(float(final_width) / width, float(final_height) / height)
            else:
                factor = min(float(final_width) / width, float(final_height) / height)
            new_width = round(factor * width)
            new_height = round(factor * height)
            new_img = img.resize((new_width, new_height), Image.LANCZOS)
        # Now either crop or pad, depending on the image class
        new_width = new_img.width
        new_height = new_img.height
        if image_class == "C3PI_Test":
            delta_width = new_width - final_width
            delta_height = new_height - final_height
            crop = (delta_width // 2, delta_height // 2,
                    new_width - (delta_width // 2), new_height - (delta_height // 2))
            new_img = new_img.crop(crop)
        else:
            delta_width = final_width - new_width
            delta_height = final_height - new_height
            padding = (delta_width // 2,
                       delta_height // 2,
                       delta_width - (delta_width // 2),
                       delta_height - (delta_height // 2))
            new_img = ImageOps.expand(new_img, padding, pixel_color)

        # Finally, save the modified image
        new_img.save(output_file, "JPEG", quality=95)

    # Otherwise the image is already the right size, so just copy the image as is
    else:
        shutil.copyfile(image_file, output_file)


def resize_with_aspect_ratio():
    parent_folder = r"E:\NoBackup\DGMD_E-14_FinalProject"
    text_file = os.path.join(parent_folder, "all_jpg_bad_images_with_class.txt")
    image_src_parent = parent_folder + r"\images\C3PI full data"

    df = util.read_c3pi_to_dataframe(text_file, image_src_parent, "image_class")

    count = 0
    dest_folder = parent_folder + r"\images\all_jpgs_640_by_640_B"
    for row in df.itertuples():
        src_file = row.file_path
        dest_file = os.path.join(dest_folder, row.image_file)
        print(dest_file)
        resize_image_based_on_class(src_file, row.image_class, dest_file, 640, 640)
        count = count + 1

    print(f"Images resized: {count}")


if __name__ == "__main__":
    resize_with_aspect_ratio()
