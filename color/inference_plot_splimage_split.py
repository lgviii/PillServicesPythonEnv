# -*- coding: utf-8 -*-
"""INFERENCE_PLOT_SPLIMAGE_SPLIT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t3REavdvKL9cGDqNnxiUkrx-uQ577cym
"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

import torch

print("--PLOTTED INFERENCES USING splimage_split_square_front_sort.zip--")

print("--WE NEED TO CALCULATE ACCURACY AND ALSO PLOT DISTRIBUTION OF CORRECT PREDICTIONS--")

print("BUILD DATA LOADER")

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

import torch

#!unzip /content/splimage_split_square_front_sort.zip -d /content/splimage_split_square_front_sort2

#**JUST BUILD SIMPLE DATA LOADER FOR JUST THIS 

classes = []
classes.append("BLACK")  #**YOU NEED TO MAKE SURE ALL YOUR IMAGE CLASSES IN HERE AND THAT THEY ARE IN ALPHABETICAL ORDER
classes.append("BLUE")
classes.append("BROWN")
classes.append("GRAY")
classes.append("GREEN")
classes.append("ORANGE")
classes.append("PINK")
classes.append("PURPLE")
classes.append("RED")
classes.append("TURQUOISE")
classes.append("WHITE")
classes.append("YELLOW")

BATCH_SIZE = 32
custom_val_path = "/content/splimage_split_square_front_sort2/splimage_split_square_front_sort"
print("VALIDATION PATH: " + custom_val_path)

# the validation transforms
custom_valid_transform = transforms.Compose([
    transforms.Resize((450,600)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# validation dataset
valid_dataset = datasets.ImageFolder(
    root=custom_val_path,
    transform=custom_valid_transform  
)

print("VALIDATION DATASET LENGTH: " + str(len(valid_dataset)))

# training data loaders
valid_loader_plot = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True
)

tracking_dict = {}

for idx in classes: #*initialize the counting
    tracking_dict[idx] = 0

print("--WE NEED TO LOAD MODEL--")

# pytorch libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.optim as optim

def load_model(model, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #ptimizer.load_state_dict[checkpoint['optimizer_state_dict']]
    return model

def define_pytorch_model(): #**most of model frozen, up to 300 layers,might take long to train
    torch.cuda.empty_cache()
    use_pretrained = True
    from torchvision import models, transforms
    model = models.densenet121(pretrained=use_pretrained)
    total_layers = 0
    for child in model.children():  # **freeze all existing layers, well just train the 2 new linear layers
        for param in child.parameters():
            total_layers += 1
            if total_layers <= 300: #**well freeze most of the model due to training time too long
                param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 12)
    model.train()
    model.cuda()
    print("DENSET, FIRST 300 LAYERS FROZEN")
    return model

model_architecture = define_pytorch_model()

model_path = "/content/model.pth5745821318_0.9032529444756029_.all_files"

model_for_use = load_model(model_architecture, model_path)

print(model_for_use)

print("SUCCESS: MODEL LOADED FROM DISK AND PLACED ON GPU FOR INFERENCES")

print("--LETS DO VALIDATION--")

tracking_dict = {}

for idx in classes: #*initialize the counting
    tracking_dict[idx] = 0

def validate_it(model):
    total_images = 0
    total_images_correct_predict = 0
  ### TESTING PORTION ###
    from timeit import default_timer as timer
    start = timer()
    print("validating model")
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in valid_loader_plot:
            images, labels = data
            # calculate outputs by running images through the network
            images = images.cuda()
            labels = labels.cuda()
            #outputs = model(images)
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            throttle = 0
            for idx in labels:
                total_images += 1
                temp_prediction = predicted[throttle]
                pred = temp_prediction.item()
                real_label = idx.item()
                if real_label == pred:
                    chosen_class = classes[real_label]
                    tracking_dict[chosen_class] += 1
                    total_images_correct_predict += 1
                throttle += 1
            #del images
            #del labels
            #images.detach()
            #labels.detach()
    acc = correct / total
    print(f'Accuracy of the pytorch model on the validation images: {100*acc:.2f}%')
    #print('Training Complete')
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282
    print("TOTAL IMAGES: " + str(total_images))
    print("TOTAL IMAGES CORRECTLY PREDICTED: " + str(total_images_correct_predict))
    return acc

validate_it(model_for_use)

print(tracking_dict)

import matplotlib.pyplot as plt

D = tracking_dict

plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))
plt.xticks(rotation=40, ha='right')
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()

print("--SIMPLE COLOR DISTRIBUTION OF SPLIMAGE_SPLIT_SQUARE_FRONT_SORT.ZIP--")

tracking_dict = {}

for idx in classes: #*initialize the counting
    tracking_dict[idx] = 0

throttle = 0
for i, data in enumerate(valid_loader_plot, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #if throttle == 10:
            #  break
            throttle += 1
            for idx in labels:
              da_label = idx.item()
              chosen_class = classes[da_label]
              #print(chosen_class)
              tracking_dict[chosen_class] += 1

print(tracking_dict)

import matplotlib.pyplot as plt

D = tracking_dict

plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))
plt.xticks(rotation=40, ha='right')
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()

print("BUILD SOME HISTOGRAMS OF PILL IMAGES")

import cv2
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display

#imgpath = "/content/histogram_images/!UBAAO8J6NXAEMYMFE67VHPS-GB-QT.JPG"

def absoluteFilePaths(directory):
    import os
    all_filez = []
    for dirpath,_, filenames in os.walk(directory):
        for f in filenames:
            all_filez.append((os.path.join(dirpath, f)))

    return all_filez

def reg_green_blue_channel_histograp(imgpath):
  x = Image(imgpath, width=300, height = 300)
  display(x)

  img = cv2.imread(imgpath)

  #cv2.imshow('Image', img)
  #cv2.waitKey(0)

  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  red_hist = cv2.calcHist([img], [0], None, [256], [0, 255])
  green_hist = cv2.calcHist([img], [1], None, [256], [0, 255])
  blue_hist = cv2.calcHist([img], [2], None, [256], [0, 255])

  plt.subplot(4, 1, 2)
  plt.plot(red_hist, color='r')
  plt.xlim([0, 255])
  plt.title('red histogram')

  plt.subplot(4, 1, 3)
  plt.plot(green_hist, color='g')
  plt.xlim([0, 255])
  plt.title('green histogram')

  plt.subplot(4, 1, 4)
  plt.plot(blue_hist, color='b')
  plt.xlim([0, 255])
  plt.title('blue histogram')

  plt.tight_layout()
  plt.show()

#imgpath = "/content/cat.jpeg"
#reg_green_blue_channel_histograp(imgpath)

print("--TEST CAT PIC HISTOGRAMS. SHOULD LOOK LIKE MOST NORMAL PHOTOS--")
img_path = "/content/cat_pics"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)



print("--ORANGE SAMPLE PILLS FROM THE DATASET--")
img_path = "/content/histogram_images"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--RED SAMPLE PILLS FROM THE DATASET--")
img_path = "/content/red_pills"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--BLUE SAMPLE PILLS FROM THE DATASET--")
img_path = "/content/blue_pills"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--YELLOW SAMPLE PILLS FROM THE DATASET--")
img_path = "/content/yellow_pills"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--YELLOW PILLS FROM VIDEO FRAME--")
img_path = "/content/video_backgrounds"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--green PILLS FROM VIDEO FRAME wood background--")
img_path = "/content/green_wood_background"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--green PILLS FROM VIDEO FRAME concrete background--")
img_path = "/content/green_concrete_background"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--GREEN PILLS from dataset--")
img_path = "/content/green_pills"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--ORANGE SORTED CHALLENGE--")
img_path = "/content/orange_sorted_challenge"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--PINK SORTED CHALLENGE--")
img_path = "/content/pink_sorted_challenge"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--BLUE SORTED CHALLENGE--")
img_path = "/content/blue_sorted_challenge"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--YELLOW, NON GRAY BACKGROUND, NON CHALLENGE--")
img_path = "/content/yellow_non_gray_background"
filez = absoluteFilePaths(img_path)

for idx in filez:
  reg_green_blue_channel_histograp(idx)

print("--MORE GRANULAR HISTOGRAMS OF RED CHALLENGE IMAGES PIXELS--")

import numpy as np
import matplotlib.pyplot as plt
import cv2

import cv2
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display

def absoluteFilePaths(directory):
    import os
    all_filez = []
    for dirpath,_, filenames in os.walk(directory):
        for f in filenames:
            all_filez.append((os.path.join(dirpath, f)))

    return all_filez

def mask_background(img_location):
    print("-----")
    print(img_location)
    x = Image(img_location, width=300, height = 300)
    display(x)
    #return
    # read image
    im = cv2.imread(img_location)

    print(im.shape)

    # calculate mean value from RGB channels and flatten to 1D array
    vals = im.mean(axis=2).flatten()

    #for idx in vals:
    #  if idx != 118:
    #    idx = 1

    # calculate histogram
    counts, bins = np.histogram(vals, range(500))
    # plot histogram centered on values 0..255
    plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    plt.xlim([-0.5, 255.5])
    plt.show()

    im = cv2.imread(img_location)
    from PIL import Image as i
    #im = i.fromarray(im)

    #print(im)

    im[im<119]=0

    im = i.fromarray(im[:,:,::-1].astype('uint8'))

    #im.save(img_location)

    display(im)

    #import scipy.misc
    #scipy.misc.toimage('outfile.jpg', im)

#img_location = '/content/red/B14JB94-MBPUZ0DIDD7-S9G7TH-1NX5.JPG'

directory = "/content/red2"
all_filez = absoluteFilePaths(directory)
for idx in all_filez:
  mask_background(idx)

print("--MORE GRANULAR HISTOGRAMS OF BLUE CHALLENGE IMAGES PIXELS--")

directory = "/content/blue"
all_filez = absoluteFilePaths(directory)
for idx in all_filez:
  mask_background(idx)

print("--MORE GRANULAR HISTOGRAMS OF PINK CHALLENGE IMAGES PIXELS--")

directory = "/content/pink"
all_filez = absoluteFilePaths(directory)
for idx in all_filez:
  mask_background(idx)

print("--EXPERIMENT WITH GREEN PILL ON WOOD BACKGROUND--")

directory = "/content/green_on_wood"
all_filez = absoluteFilePaths(directory)
for idx in all_filez:
  mask_background(idx)

print("--MORE GRANULAR HISTOGRAMS OF green CHALLENGE IMAGES PIXELS--")

directory = "/content/green"
all_filez = absoluteFilePaths(directory)
for idx in all_filez:
  mask_background(idx)

print("--MORE GRANULAR HISTOGRAMS OF green non CHALLENGE IMAGES PIXELS--")

directory = "/content/green_non_challenge"
all_filez = absoluteFilePaths(directory)

first_file = all_filez[0]
im = cv2.imread(first_file)
print(im)


for idx in all_filez:
  mask_background(idx)

print("--MORE GRANULAR HISTOGRAMS OF BROWN CHALLENGE IMAGES PIXELS--")

directory = "/content/brown"
all_filez = absoluteFilePaths(directory)

first_file = all_filez[0]
im = cv2.imread(first_file)
print(im)

for idx in all_filez:
  mask_background(idx)