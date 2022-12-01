#*ABRENIO: RESIZES ALL IMAGES AT ONE TIME, TO SAVE TIME BY OMMITTING PYTORCH RESIZE TRANSFORMATION

import os
import sys
import time
from PIL import Image

original_folder = "/Users/labuser/Desktop/train_images/all_downloaded_pill_images"
target_folder = "/Users/labuser/Desktop/train_images/all_pill_images_32_by_32_REAL"

im = Image.open('red.JPG')
width, height = im.size

def absoluteFilePaths(directory):
    all_filez = []
    for dirpath,_, filenames in os.walk(directory):
        for f in filenames:
            all_filez.append((os.path.join(dirpath, f)))

    return all_filez

all_original_filez = absoluteFilePaths(original_folder)

print("--ALL ORIGINAL FILES--")
time.sleep(2)

for idx in all_original_filez:
    print(idx)

print("NUMBER OF FILES: " + str(len(all_original_filez)))

print("--RESIZING IMAGES--")

time.sleep(2)

for idx in all_original_filez:
    print(idx)
    try:
        all_parts = idx.split("/")
        f_name = all_parts[len(all_parts) - 1]
        full_fname = target_folder + "/" + f_name
        print("TARGET OUTPUT FILE: " + full_fname)
        im = Image.open(idx)
        imResize = im.resize((450,600), Image.ANTIALIAS)
        imResize.save(full_fname , 'JPEG', quality=90)
        print("SUCESS: IMAGE RESIZED")
    except Exception as e:
        print("COULD NOT RESIZE IMAGE")

print("SUCCESS: ALL IMAGES RESIZED!")