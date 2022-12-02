#**ABRENIO: FOR CLASS , WEARABLE DEVICES
#**THIS SCRIPT WILL DOWNLOAD THE PILLS AND SORT THEM BY SHAPE
#**YOU CAN ALSO SPECIFY A MAX AMOUNT OF IMAGES TO DOWNLOAD IN CASE SPACE AND TIME IS LIMITED!

import pandas as pd
import xml.etree.ElementTree as et


import xml.etree.ElementTree as ET
import pandas as pd

# 1. Import the requests library
import requests
import os
import sys

import shutil
import time

#**ALL THE XMLS YOU WANT TO PARSE, YOU CAN PUT THEM IN THIS LIST
#**of course theres other creative ways to populate this list with the xml names just do what you want.

##ONLY NEED TO ADJUST THESE PARAMETERS (BEGIN)

xml_file_name_list = ["PillProjectDisc1.xml", "PillProjectDisc10.xml", "PillProjectDisc30.xml", "PillProjectDisc31.xml"]

xml_file_name_list = ["PillProjectDisc10.xml", "PillProjectDisc30.xml"]

xml_file_name_list = ["PillProjectDisc31.xml", "PillProjectDisc1.xml"]

xml_file_name_list = ["PillProjectDisc100.xml", "PillProjectDisc101.xml", "PillProjectDisc102.xml", "PillProjectDisc103.xml", "PillProjectDisc104.xml", "PillProjectDisc105.xml"]

xml_file_name_list = ["PillProjectDisc10.xml", "PillProjectDisc30.xml"]

xml_file_name_list = []

for idx in range(1, 111):
    fname = "PillProjectDisc" + str(idx) + ".xml"
    xml_file_name_list.append(fname)

for idx in xml_file_name_list:
    print(idx)

xml_file_name_list.append("testbad")

time.sleep(.5)

download_output_dir = "all_pill_images_32_by_32_copy" #**IMAGES DOWNLOADED HERE. NO SORTING JUST RAW IMAGE DOWNLOADS
color_training_directory = "color_sorted_non_challenge" #**IMAGES SPLIT INTO COLOR CATEGORIES IN THIS DIRECTORY
shape_training_directory = "shape_sorted_challenge" #***IMAGES SPLIT BY SHAPES INTO HERE
local_image_cache_directory = "all_pill_images_32_by_32_copy" #**IF IMAGE IS HERE, WELL JUST COPY IT INTO THE PROPER SHAPE OR COLOR AND NOW DOWNLOAD IT AGAIN

download_them = True

sort_by_color = True

sort_by_shape = True

only_sort_existing = True

##ONLY NEED TO ADJUST THESE PARAMETERS (END)

#**we can just download the xml files to make it even simpler to get the data, WHY NOT?

print("--ENUMERATING LOCAL CACHE DIRECTORY (Avoid re downloading files)")

images_already_downloaded = []  #**SEARCH THIS STRUCTURE FIRST FOR THE IMAGE , BEFORE DOWNLOADING IT. SAVES BANDWIDTH ETC

import os
for folder, subfolders, files in os.walk(local_image_cache_directory):
    for file in files:
        try:
            filePath = os.path.abspath(os.path.join(folder, file))
            print(filePath, os.stat(filePath).st_uid)
            if filePath not in images_already_downloaded:
                images_already_downloaded.append(filePath)
        except Exception as e:
            print(e)

for folder, subfolders, files in os.walk(download_output_dir):
    for file in files:
        filePath = os.path.abspath(os.path.join(folder, file))
        print(filePath, os.stat(filePath).st_uid)
        if filePath not in images_already_downloaded:
            images_already_downloaded.append(filePath)

print("---DOWNLOADING REQUIRED XML FILES---")

time.sleep(.5)

good_xml_check_and_download = []

for idx in xml_file_name_list:
    URL = "https://data.lhncbc.nlm.nih.gov/public/Pills/ALLXML/" + idx
    response = requests.get(URL)
    xml_string_check = "<ImageExport>"
    xml_string_check = xml_string_check.encode('ASCII')
    good_check = response.content.find(xml_string_check)
    if good_check > 0: #**is it a good xml OR NOT?
        output_file = idx
        open(output_file, "wb").write(response.content)
        print("Downloaded: " + URL)
        good_xml_check_and_download.append(idx)
        time.sleep(.5)

xml_file_name_list = good_xml_check_and_download #**GET RID OF THE BAD XMLS NO NEED TO ENUMERATE THOSE!

starting_url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" #**well form the proper url using the xml name

files_parse_success = []

shape_files_parsed_success = []

def verify_iamge(filename):
    from PIL import Image
    try:
        im = Image.open(filename)
        # do stuff
        print("THIS IS A GOOD IMAGE")
    except IOError:
       	print("error: THIS IS A BAD IMAGE. WE MIGHT BE GETTING BLOCKED. REMOVING FROM DATASET")
        #os.remove(filename)
        print("Something is wrong with image verification , terminate and reassess")
        sys.exit(1)

# filename not an image file

def break_into_shape_dir(da_xml, da_image_dir, da_training_dir):
    specific_xml_file = da_xml

    prstree = ET.parse(specific_xml_file)
    root = prstree.getroot()

    # print(root)
    store_items = []
    all_items = []

    x = root.iter('ImageExport')

    d = 0

    for idx in x:
        print(idx)
        d = idx.iter("Image")

    unique_shapes = []
    all_shapes = []

    for idx in d:
        print(idx)
        image_shape = idx.find("Shape").text
        print(image_shape)
        if image_shape not in unique_shapes:
            unique_shapes.append(image_shape)
        x = idx.find("File")
        image_name = x.find("Name").text
        temp_tuple = (image_name, image_shape)
        if image_name.endswith(".JPG"):
            all_shapes.append(temp_tuple)

    print(unique_shapes)

    print("Number of images: " + str(len(all_shapes)))
    print(all_shapes)

    #*get all the images

    training_image_names = []

    for f in os.listdir(da_image_dir):
        training_image_names.append(f)

    for idx in training_image_names:
        for x in all_shapes:
            iname = x[0]
            ishape = x[1]
            if iname == idx:
                shape_dir = da_training_dir + "//" + ishape
                try:
                    os.mkdir(shape_dir)
                    print("SUCCESS: CREATED SHAPE DIR COPYING FILE IN")
                except Exception as e:
                    print("DIR MIGHT ALREADY EXIST, TRYING TO COPY FILE IN!")

                try:
                    or_file = da_image_dir + "//" + idx
                    tar_file = shape_dir + "//" + idx
                    shutil.copyfile(or_file, tar_file)
                    temp_tuple = (da_xml, tar_file)
                    shape_files_parsed_success.append(temp_tuple)
                except Exception as e:
                    print("ERROR: COULD NOT COPY FILE IN FOR TRAINING")
                    #sys.exit(1)


def break_into_color_dir(da_xml, da_image_dir, da_training_dir):

    specific_xml_file = da_xml

    prstree = ET.parse(specific_xml_file)
    root = prstree.getroot()

    # print(root)
    store_items = []
    all_items = []

    x = root.iter('ImageExport')

    d = 0

    for idx in x:
        print(idx)
        d = idx.iter("Image")

    unique_colors = []
    all_colors = []

    for idx in d:
        print(idx)
        t = idx.iter()
        color_tag_count = 0
        for x in t:
            s = str(x)
            if 'Color' in s:
                color_tag_count += 1
        if color_tag_count == 1:
            image_shape = idx.find("Color").text
            print(image_shape)
            if image_shape not in unique_colors:
                unique_colors.append(image_shape)
            x = idx.find("File")
            image_name = x.find("Name").text
            temp_tuple = (image_name, image_shape)
            if image_name.endswith(".JPG"):
                all_colors.append(temp_tuple)
        else:
            print("IMAGE OMMITED TOO MANY COLOR TAGS: ")

    print(unique_colors)

    print("Number of images: " + str(len(all_colors)))
    print(all_colors)

    #*get all the images

    training_image_names = []

    for f in os.listdir(da_image_dir):
        training_image_names.append(f)

    for idx in training_image_names:
        for x in all_colors:
            iname = x[0]
            ishape = x[1]
            if iname == idx:
                shape_dir = da_training_dir + "//" + ishape
                try:
                    os.mkdir(shape_dir)
                    print("SUCCESS: CREATED COLOR DIR COPYING FILE IN")
                except Exception as e:
                    print("DIR MIGHT ALREADY EXIST, TRYING TO COPY FILE IN!")

                try:
                    or_file = da_image_dir + "//" + idx
                    tar_file = shape_dir + "//" + idx
                    shutil.copyfile(or_file, tar_file)
                    print("Copying File Target: " + or_file + " DESTINATION: " + tar_file)
                    print("GREAT SUCESS COPYING FILE IN FOR TRAINING :)")
                    temp_tuple = (da_xml, tar_file)
                    files_parse_success.append(temp_tuple)
                except Exception as e:
                    print("ERROR: COULD NOT COPY FILE IN FOR TRAINING")
                    #sys.exit(1)
    print("FILES CORRECTLY CATEGORIZED")
    #temp_tuple = (da_xml, tar_file)
    for idx in files_parse_success:
        print(idx)



#**THIS FUNCTION DOWNLOADS, AND PUTS THE IMAGES INTO A TARGET DIRECTORY
def write_image(directory, fname, url):
    print("CHECKING LOCAL CACHE DIRECTORY FIRST")
    download_it = True
    local_file = ""
    for idx in images_already_downloaded:
        if fname in idx:
            print("IMAGE ALREADY DOWNLOADED dont download")
            local_file = idx
            download_it = False

    if download_it:
        try:
            if only_sort_existing:
                print("WERE ONLY SORTING EXISTING. NOT DOWNLOADING FILE: " + fname)
                return
            time.sleep(.2)  # **i think if we download too fast it erros out
            URL = url + fname
            print(URL)
            # 2. download the data behind the URL
            response = requests.get(URL)
            # 3. Open the response into a new file called instagram.ico
            output_file = directory + "//" + fname
            open(output_file, "wb").write(response.content)
            print("SUCCESS: DOWNLOADED: " + fname)
            verify_iamge(output_file) #**verify we actually downloaded an image
        except Exception as e:
            print("Could not download: " + fname)
            print(e) #**something is wrong with our downloads? need to know what! 
    else:
        output_file = directory + "//" + fname
        shutil.copyfile(local_file, output_file) #*jusst copy it into the training directory from local cache


def DO_the_xml_parse_for_download(specific_xml_file, download_only_output_folder, download_url):
    da_url = download_url
    da_dir = download_only_output_folder
    prstree = ET.parse(specific_xml_file)
    root = prstree.getroot()

    # print(root)
    store_items = []
    all_items = []

    x = root.iter('ImageExport')

    d = 0

    for idx in x:
        print(idx)
        d = idx.iter("Image")

    unique_colors = []
    all_colors = []

    for idx in d:
        print(idx)
        image_shape = idx.find("Color").text
        print(image_shape)
        if image_shape not in unique_colors:
            unique_colors.append(image_shape)
        x = idx.find("File")
        image_name = x.find("Name").text
        temp_tuple = (image_name, image_shape)
        if image_name.endswith(".JPG"):
            all_colors.append(temp_tuple)

    print(unique_colors)

    print("Number of images: " + str(len(all_colors)))
    print(all_colors)

    #da_url = "https://data.lhncbc.nlm.nih.gov/public/Pills/PillProjectDisc1/images/"
    #da_dir = "disk_images_1_color"

    #print("clearing directory")

    #for f in os.listdir(da_dir):
    #    os.remove(os.path.join(da_dir, f))

    for idx in all_colors:
        file_name = idx[0]
        write_image(da_dir, file_name, da_url)

pull_url = ""


#**THIS IS JUST GOING TO PARSE THE TARGET XML, AND THEN DOWNLOAD THE IMAGES BASED ON COLORS THAT HAVE JPG IMAGES
#**THE JPG IMAGES SEEM LIKE THE BEST QUALITY DATA SO WELL USE THOSE


#**HERE WE SIMPLY CHECK TO SEE IF THE OUTPUT PATH EXISTS. IF NOT, WE CREATE IT.
#**THE IMAGES WONT BE SORTED HERE FOR COLOR. JUST DOWNLOADED
if not os.path.exists(download_output_dir):
    print("ERROR: Path did not already exist. Creating now. ")
    os.makedirs(download_output_dir)
    print("SUCCESS: Created image Output DIR: " + download_output_dir)
else:
    print("Download directory already exists")

downloaded_path_collection = []

#download_them = True

if download_them:
    for idx in xml_file_name_list:
        specific_xml_file = idx
        temp_name = idx.replace(".xml", "")
        starting_url = "https://data.lhncbc.nlm.nih.gov/public/Pills/" + temp_name + "/" + "images/"
        print("CURRENT IMAGE DOWNLOAD URL: " + starting_url)
        print("----------------------------")
        time.sleep(.5)
        try:
            DO_the_xml_parse_for_download(specific_xml_file, download_output_dir, starting_url)
            downloaded_path_collection.append(starting_url)
        except Exception as e:
            print(e)

print("Successfully Downloaded Images from Following DIRs:")
for idx in downloaded_path_collection:
    print(idx)
time.sleep(.5)

print("-----RAW IMAGE DOWNLOAD COMPLETE. STEP 2, SORT BY SHAPES FOR NUERAL NETWORK TRAINING LETS GO!-----")

#**MAKE SURE COLOR TRAINING DIRECTOY EXISTS

if not os.path.exists(color_training_directory):
    print("ERROR: Color Path did not already exist. Creating now. ")
    os.makedirs(color_training_directory)
    print("SUCCESS: Created image Output DIR: " + color_training_directory)
else:
    print("Download directory already exists")

#**MAKE SURE SHAPE TRAINING DIRECTOY EXISTS

if not os.path.exists(shape_training_directory):
    print("ERROR: Color Path did not already exist. Creating now. ")
    os.makedirs(shape_training_directory)
    print("SUCCESS: Created image Output DIR: " + shape_training_directory)
else:
    print("Download directory already exists")


if sort_by_color:
    for idx in xml_file_name_list:
        da_xml = idx
        da_image_dir = download_output_dir
        da_training_dir = color_training_directory
        break_into_color_dir(da_xml, da_image_dir, da_training_dir)

    print("FILES CORRECTLY CATEGORIZED BY COLOR")
    for idx in files_parse_success:
        print(idx)

if sort_by_shape:
    for idx in xml_file_name_list:
        da_xml = idx
        da_image_dir = download_output_dir
        da_training_dir = shape_training_directory
        break_into_shape_dir(da_xml, da_image_dir, da_training_dir)

    print("FILES CORRECTLY CATEGORIZED BY SHAPE")
    for idx in shape_files_parsed_success:
        print(idx)
