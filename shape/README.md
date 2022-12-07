# Shape Model Training and Testing
This subfolder contains scripts used to train and validate the shape feature model.  

Note that all the scripts are written to run on my home system, so they assume the following folder structure exists:
* (parent directory)\shape - contains all shape-related label files, sorted image directories, and output
* (parent directory)\images\C3PI full data - directory in which the C3PI images have been downloaded, maintaining the
same directory structure as on the [C3PI download website](https://data.lhncbc.nlm.nih.gov/public/Pills/index.html)
* (parent directory)\images\all_jpgs_640_by_640 - directory containing the resized 640x640 images used for shape
model training
* (parent directory)\images\split_spl_images\all_square - directory containing the "split" SPL images, padded to be 
square for easier resize transforms when validating

Also, all model-related scripts (DataLoaders, etc) assume that images are already sorted into directories labeled with 
the feature class, with one directory per class.  The `shape_sort.py` module contains the functionality used to create
these directories, based on CSV files linking image file with shape label.  (See the csv_files directory containing the
CSV files used for sorting.)

# Requirements
The shape feature model is built using PyTorch.  Utility functions also rely on pandas and the Pillow fork of PIL.  
See requirements.txt for the minimal list of required libraries. 

Scripts are written assuming use of GPU - model, labels, and data are all sent to the GPU.  If not run on a GPU, scripts
will need to be modified accordingly.

# Modules
## `resize_images.py`
This module contains functionality used to resize images so that they'd be smaller and more consistently-sized, 
simplifying training pipelines and reducing the time needed to train.

### `resize_c3pi_images_to_square()`
Function used to resize the original C3PI JPG images to 640x640 squares, either cropping or padding based on the image 
class.

### `pad_image_to_square()`
Function used to resize the reserved split SPL images to make them square, so that validation pipeline resizing to 
224x224 wouldn't crop out any portion of the pill. 

## `shape_sort.py`
This module contains the `split_resized_images()` function used to sort the images into subdirectories by shape feature
class, such that each image is in a directory labeled with its shape class.

## `cnn_train_utils.py`
This module contains all functions used to set up PyTorch models and DataLoaders and run validation and model 
fine-tuning.

Note that all model-related scripts (DataLoaders, etc) assume that images are already sorted into directories labeled
with the feature class, with one directory per class.  The `shape_sort.py` module contains the functionality used to
create these directories, based on CSV files linking image file with shape label.

Some functions of note:
### `generate_dataloaders_with_reserve()`
Function that builds both training and validation dataloaders, based on separate training and (reserved) validation 
folders.  Builds the training dataloader by splitting the images in the specified training folder into separate train
and validation sets, and then combining the split validation set with the reserved images in the specified validation 
folder such that the total ratio of training images to validation images is 80/20.

That is, if the training folder has 110 images and the validation folder has 15 images, the final training dataloader 
will have 100 images from the training folder and the final validation dataloader will have 10 randomly-selected 
images from the training folder plus the 15 reserved images from the validation folder.

This function was the primary one used to build the training and validation dataloaders used in fine-tuning the shape
model.

### `generate_train_dataloader()`
Function that builds just a training dataloader, using all the images in the specified training folder.

This function was used for a couple of tests checking model performance when fine-tuning on all C3PI JPGs, using only
the reserved split SPL images for validation.

### `generate_valid_dataloader()`
Function that builds just a validation dataloader, using all the images in the specified validation folder.

This function was used for a couple of tests checking model performance when fine-tuning on all C3PI JPGs, using only
the reserved split SPL images for validation.

### `initialize_model()`
Function that creates a PyTorch model specified by name, loading it with the default pre-trained weights, freezing 
all but a specified number of final layers, and replacing the output layer to generate the desired number of classes.

This function was used to create the models used for fine-tuning and validation.  It was largely copied from the 
[PyTorch tutorial page](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

### `validate_it()`
Function used to calculate prediction accuracy of a model against all images in a specified validation DataLoader.

This function was used both in fine-tuning, to calculate the prediction after each epoch, and in later model accuracy 
calculation against different sets of images.

### `pytorch_train_and_evaluate()`
Function used to fine-tune a model with specified train/test DataLoaders.

This function was used for all shape model fine-tuning.

## `cm_shape_train.py`
This module contains all functions used to fine-tune and test models for shape recognition.

Key functions:
### `run_shape_train_densenet_spl_front()`
Function used to generate the final shape model used for the pill-matcher application, with images resized to 224x224 
and no WeightedRandomSampler.

### `test_densenet()`
Function used to calculate prediction accuracy of trained models against the split SPL images.

# CSV_Files
This directory contains the CSV files used for sorting images into shape class.  Each one contains the image file name
and the shape class of the associated pill.  All images are assumed to be in one common directory.

### all_jpg_shape.csv
This CSV includes all C3PI JPG images.

### jpg_no_splimage_good_shape.csv
This CSV includes only C3PI_Test and MC_CHALLENGE_V1.0 class JPG images.

### splimage_split_all_shape.csv
This CSV includes both "front" and "back" split SPL images, used for final validation.

### splimage_split_back_shape.csv
This CSV includes only "back" split SPL images.

### splimage_split_front_shape.csv
This CSV includes only "front" split SPL images.
