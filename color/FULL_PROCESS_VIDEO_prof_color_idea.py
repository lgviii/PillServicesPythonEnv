#*ABRENIO: THIS SCRIPT CAN BE USED TO PROCESS VIDEOS TO DO THE PILL COLOR AND SHAPE INFERENCES AND PRINT THEM TO EACH FRAME IN
#NEW TARGET OUTPUT VIDEO

import os

import cv2
import numpy as np

import torchvision.transforms as transforms
from PIL import Image
import torch

import time
import sys


TARGET_VIDEO_NAME = "greenframe.mp4"
TARGET_VIDEO_NAME = "yellowtest_marked_prof_trick.mp4"
TARGET_VIDEO_NAME = "pink_pill_test.mp4"
TARGET_VIDEO_NAME = "greentest.mp4"
marked_file_name = "allofthem_marked_full.mp4"
CLEAN_UP_TEMP_FRAMES = False
color_model_path = "/Users/labuser/Desktop/train_images/color_train_records/model.pth_0.4430416288047001_84.53232365727106_6410265990.all_files"
color_model_path = "model.pth5745821318_0.9032529444756029_.all_files"
shape_model_path = "/Users/labuser/Desktop/train_images/train_records/model.pth_0.783861234309567_73.66630520295125_541253714.all_images"

color_model_path = "model.pth3098659428_none_.all_files"

color_model_path = "model.pth3098659428_none_.all_files"

color_model_path = "model.pth4691792945_none_.all_files"

color_model_path = "model.pth3772376093_none_.all_files"

thres = 0.45 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)

pull_test_images = True
write_to_video_debug = True

##**BEGIN DETECT COLOR HEADERS

print("LOADING COLOR MODEL")
time.sleep(3)

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

classes = []
classes = []
classes.append("green")
classes.append("orange")
classes.append("pink")
classes.append("yellow")

print("color classes: ")
print(classes)

# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((450,600)),  #*resize for the color classes
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# the validation transforms
color_valid_transform = transforms.Compose([
    transforms.Resize((450,600)),  #*resize for the color classes
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def load_model(model, model_path):
   checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['model_state_dict'])
   # ptimizer.load_state_dict[checkpoint['optimizer_state_dict']]
   return model


def define_pytorch_model():
  import torch.nn as nn
  torch.cuda.empty_cache()
  use_pretrained = True
  from torchvision import models, transforms
  model = models.densenet121(pretrained=use_pretrained)
  for child in model.children():   #**freeze all existing layers, well just train the 2 new linear layers
    for param in child.parameters():
        param.requires_grad = False
  num_ftrs = model.classifier.in_features
  model.classifier = nn.Linear(num_ftrs, 4)
  return model

model = define_pytorch_model()

model = define_pytorch_model()
model = load_model(model, color_model_path) #**professors trick

print("---COLOR MODEL LOADED---")
model.eval()
print(model)

print(time.sleep(2))

#**WILL USE THE COLOR MODEL
def pre_image(image_path,model):
   img = Image.open(image_path)
   img_normalized = color_valid_transform(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to("cpu")
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)  #*were actually pushing the image through that model here, its a NUMPY ARRAY

      sm = torch.nn.Softmax()
      probabilities = sm(output)
      top_p, top_class = probabilities.topk(1, dim=1)
      probabilty = top_p.item()
      probabilty = probabilty * 100
      index = output.data.cpu().numpy().argmax()
      #classes = train_ds.classes
      class_name = classes[index]
      #if class_name == "BLUE":
      print("--probabilities--")
      return class_name + " Probability: " + str(probabilty)

#***BEGIN WILL DETECT SHAPES!!!!!!!!!!!!!!!!!!

print("START LOADING SHAPE MODEL")
time.sleep(3)

classes_shapes = []
classes_shapes.append("BULLET")
classes_shapes.append("CAPSULE")
classes_shapes.append("DIAMOND")
classes_shapes.append("DOUBLE_CIRCLE")
classes_shapes.append("FREEFORM")
classes_shapes.append("HEXAGON_6_SIDED")
classes_shapes.append("OVAL")
classes_shapes.append("PENTAGON_5_SIDED")
classes_shapes.append("RECTANGLE")
classes_shapes.append("ROUND")
classes_shapes.append("SEMI-SQUARED")
classes_shapes.append("SQUARE")
classes_shapes.append("TEAR")
classes_shapes.append("TRAPEZOID")
classes_shapes.append("TRIANGLE")


print("shape classes: ")
print(classes_shapes)

# the validation transforms
valid_transform_shapes = transforms.Compose([
    #transforms.Grayscale(), #*lets make it gray scale since we just lookin for shaped
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def load_model_shape(model, model_path):
   checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['model_state_dict'])
   # ptimizer.load_state_dict[checkpoint['optimizer_state_dict']]
   return model


import torch.nn as nn
import pretrainedmodels as pm


#Now using the AlexNet
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

new_input = torch.nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)) #*need to replace first layer for grayscale 1 demension

for child in AlexNet_model.children():   #**freeze all existing layers, well just train the 2 new linear layers
    for param in child.parameters():
        param.requires_grad = False


AlexNet_model.classifier[4] = nn.Linear(4096,1024)

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
AlexNet_model.classifier[6] = nn.Linear(1024,16)

model_alex = AlexNet_model

saved_shape_model_path = shape_model_path
shape_model = load_model_shape(model_alex, saved_shape_model_path)

print("---SHAPE MODEL LOADED---")
print(shape_model)
time.sleep(3)

def predict_image_shapes(image_path, model):  #**THIS IS FOR PREDICTING SHAPES. USES A DIFFERENT TRANSFORMATION
   img = Image.open(image_path)
   img_normalized = valid_transform_shapes(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   # input = Variable(image_tensor)
   img_normalized = img_normalized.to("cpu")
   # print(img_normalized.shape)
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)  #*were actually pushing the image through that model here, its a NUMPY ARRAY

      sm = torch.nn.Softmax()
      probabilities = sm(output)
      #print(probabilities)  # Converted to probabilities
      top_p, top_class = probabilities.topk(1, dim=1)
      probabilty = top_p.item()
      probabilty = probabilty * 100
      #output = torch.nn.functional.softmax(output, dim=1)
      #output = torch.nn.Softmax(output)
      index = output.data.cpu().numpy().argmax()
      #classes = train_ds.classes
      class_name = classes_shapes[index]
      return class_name + " Probability: " + str(probabilty)


#**AFTER WE LOAD THE MODELS ABOVE NOW WE NEED TO BREAK THE VIDEO UP INTO FRAMES

def write_to_video(image_folder, video_name):
    frame_count = 0
    keep_going = True
    images = []
    frame_cant_find_count = 0
    while keep_going:
        if frame_count == 3541:
            break
        file_name = image_folder +"//frame_test" + str(frame_count) + ".png"
        print(file_name)
        frame_count += 1
        images.append(file_name)

    while True:
        for file_name in images:
            try:
                frame = cv2.imread(os.path.join(file_name)) #**jUST READ ONE TO GET THE FRAME SIZE
                height, width, layers = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
                break
            except Exception as e:
                pass
        break

    for image in images:
        try:
            x = cv2.imread(os.path.join(image))
            if x.size:
                video.write(cv2.imread(os.path.join(image)))
                print("SUCCESS: FRAME WRITTEN TO VIDEO")
                frame_cant_find_count = 0
                #os.remove(os.path.join(image))
                print("DELETED MARKED FRAME AFTER WRITING TO VIDEO: " + os.path.join(image)) #*TRANSITION TO FULL AUTOMATION PIPELINE
        except Exception as e:
            print("ERROR: FRAME NOT AVAILABLE")
            frame_cant_find_count += 1
            if frame_cant_find_count == 12:
                print("PROB NO MORE FRAMES.")
                break

    cv2.destroyAllWindows()
    video.release()

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      file_name = pathOut + "//frame%d.jpg" % count
      print(file_name)
      try:
        cv2.imwrite(file_name, image)     # save frame as JPEG file
      except Exception as e:
          print(e)

      count += 1
    print("FRAMES WRITTEN COMPLETED")

import random
temp_extract_folder = random.randint(0, 9999999999999999999)
temp_extract_folder = str(temp_extract_folder)
os.mkdir(temp_extract_folder) #**well create a random folder name so that we can store our extracted frames in there

print("RAW FRAME EXTRACTION FOLDER: " + temp_extract_folder) #**raw video frames will be put HERE!

print("EXTRACTED FRAMES FROM VIDE0: " + TARGET_VIDEO_NAME)
print("TO FOLDER: " + temp_extract_folder)

if pull_test_images:
    extractImages(TARGET_VIDEO_NAME, temp_extract_folder)
    print("done")
    #sys.exit(1)

classNames= []
classFile = "coco.names"
#with open(classFile,"rt") as f:
#    classNames = f.read().rstrip("n").split("n")

fh = open('coco.names', 'r')
name_linez = fh.readlines()
fh.close()
for idx in name_linez:
    idx = idx.strip()
    classNames.append(idx)

#x = cv2.imread("for_download//frame0.jpg")

#print(classNames)
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

keep_going = True
process_image = True

frame_count = 0

width = 0
height = 0

output_folder = "output_folder_drone"

def pull_out_boxed_image(X,Y, W, H, img):
    #H = H + 12
    #W = W + 12
    #Y = Y + 8
    #X = X - 10

    cropped_image = img[Y:Y + H, X:X + W]
    print([X, Y, W, H])
    #plt.imshow(cropped_image)
    img_name = "contour1.png"
    img_name = "azz.jpg"
    cv2.imwrite(img_name, cropped_image)
    #cv2.imshow("Output", cropped_image)
    #cv2.waitKey(1)
    pred = pre_image(img_name, model) #**PREDICT COLOR!
    shape_pred = predict_image_shapes(img_name, shape_model) #**PREDICT IMAGE SHAPE!
    total_pred = pred + "-----" + shape_pred
    os.remove(img_name) #*just delete it after prediction
    #pred = "PREDICTION"
    return total_pred

def prediction_not_boxed(img):
    img_name = "contour1.png"
    cv2.imwrite(img_name, img)
    pred = pre_image(img_name, model)  # **PREDICT COLOR!
    shape_pred = predict_image_shapes(img_name, shape_model)  # **PREDICT IMAGE SHAPE!
    total_pred = pred + "-----" + shape_pred
    os.remove(img_name)  # *just delete it after prediction
    # pred = "PREDICTION"
    return total_pred

def just_predict_entire_image(img_name):
    pred = pre_image(img_name, model)
    return pred

def do_single_image_prediction():
    pass


marked_folder_name = str(random.randint(0,9999999999999))
os.mkdir(marked_folder_name)

print("MARKED IMAGE FRAMES FOLDER: " + marked_folder_name)

input_folder = temp_extract_folder  #**THESE ARE JUST YOUR RAW UNMARKED IMAGE FRAMES
output_folder = marked_folder_name  #***THE APP WILL TRY TO MARK FRAMES AND PUT THEM HERE

exception_count = 0

while keep_going:
    file_name = input_folder + "//frame" + str(frame_count) + ".jpg"
    try:
        print(file_name)
        img = cv2.imread(file_name)
        width = img.shape[0]
        height = img.shape[1]
        if img.all() == None:
            process_image = False
        else:
            process_image = True
    except Exception as e:
        #print(e)
        process_image = False
        frame_count += 1
        exception_count += 1
        if exception_count == 50:
            break
    if process_image:
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))

        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        print(indices)

        try:
            for i in indices:
                #i = i[0]
                box = bbox[i]
                x,y,w,h = box[0],box[1],box[2],box[3]

                da_prediction = pull_out_boxed_image(x, y, w, h, img) #*do prediction of color

                #**TRY ENTIRE IMAGE
                #da_prediction = just_predict_entire_image(file_name)
                #w = w + 60
                #h = h + 60
                #x = x - 10
                #y = y + 10
                w = w-5      #**ADJUST THE BOUNDING BOX HERE TO GET MORE OR LESS OF THE BACKGROUND IN THE CROPPED IMAGE
                h = h-20
                cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
                #cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                name = classIds[i] - 1
                #cv2.putText(img, classNames[classIds[i] - 1].upper(), (box[0] + 10, box[1] + 30),
                #cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                da_prediction = da_prediction.split("-----")

                cv2.putText(img, da_prediction[0], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(img, da_prediction[1], (box[0] + 10, box[1] + 70),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            if len(indices) == 0:
                da_prediction = prediction_not_boxed(img)
                da_prediction = da_prediction.split("-----")
                cv2.putText(img, da_prediction[0], (10 + 10, 10 + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(img, da_prediction[1], (10 + 10, 10 + 90),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)



        except Exception as e:
            print(e)
            exception_count += 1
            if exception_count == 50:
                break

        output_file_name = output_folder + "//frame_test" + str(frame_count) + ".png"
        cv2.imwrite(output_file_name, img)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
        frame_count += 1

print("ALL FRAMES SHOULD BE MARKED")
print("CHECK FOLLOWING VIDEO FILE FOR PREDICTIONS: " + marked_file_name)
if write_to_video_debug:
    write_to_video(marked_folder_name, marked_file_name)
    print("SUCCESS: DONE PROCESSING VIDEO")
print("CHECK FOLLOWING VIDEO FILE FOR PREDICTIONS: " + marked_file_name)

if CLEAN_UP_TEMP_FRAMES:
    print("CLEANING UP TEMPORARY FRAMES")
    time.sleep(2)
    import shutil
    try:
        shutil.rmtree(marked_folder_name)
        shutil.rmtree(temp_extract_folder)
        print("SUCCESS: CLEANED UP DIRECTORIES")
        print(marked_folder_name)
        print(temp_extract_folder)
        time.sleep(2)
    except Exception as e:
        print(e)
        print("ERROR: COULD NOT CLEANED UP DIRECTORIES")
        print(marked_folder_name)
        print(temp_extract_folder)
        time.sleep(2)
else:
    print("MARKED FRAMES FOLDER: " + temp_extract_folder)
    print("UNMARKED FRAMES FOLDER: " + marked_folder_name)