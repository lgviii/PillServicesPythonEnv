import os
import subprocess
from typing import List
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models


classes = [
        "BLACK",
        "BLUE",
        "BROWN",
        "GRAY",
        "GREEN",
        "ORANGE",
        "PINK",
        "PURPLE",
        "RED",
        "TURQUOISE",
        "WHITE",
        "YELLOW",
    ]

def initialize_model(num_classes: int):
    print("initialize_model")
    """
    Create the densenet model that will be used to make inferences, providing the
    expected number of outputs.
    """
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    n_layers = 0

    """ Densenet121 - 364 layers
    """
    model_ft = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Freeze all the layers since we aren't training, just using a pre-trained model
    for param in model_ft.parameters():
        param.requires_grad = False
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    print("Densenet 121, all layers frozen")
    return model_ft


def load_model_cpu(model, model_path):
    print("load_model_cpu")
    """
    Load pre-trained weights into the model, intended for use with the CPU.
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def load_model_gpu(model, model_path):

    print("load_model_gpu")
    """
    Load pre-trained weights into the model, intended for use with the GPU.
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def generate_model(num_classes: int, model_path: str):
    print("generate_model")

    model = initialize_model(num_classes)

    if torch.cuda.is_available():
        print("Using GPU")
        load_model_gpu(model, model_path)
    else:
        load_model_cpu(model, model_path)
    model.eval()
    return model

def generate_transform(size=224):
    print("generate_transform")
    # The new improved models are trained on 224x224 images, so default to that
    # This will scale the image so the short edge is the specified size, then crop the center
    # to generate a square result of that size
    img_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return img_transform


def generate_inferences_for_image(model, softmax, img_transform, classes: List[str], image_path: str):
    print("generate_inferences_for_image")
    """
    Uses the specified model, softmax layer, validation transform, and classes
    to generate the class probabilities for the specified image_path.

    Outputs result as a dict mapping class -> probability, ordered from highest
    probability to lowest probability.

    :param model: the initialized model with weights loaded
    :param softmax: torch.nn.Softmax() layer used to translate output to probabilities,
                    pulled out separately to make batch usage of this method easier
    :param img_transform: the preprocessing transformation that should be run on the image
                          to ensure it matches the expected model input
    :param classes: ordered List of the classes to which the image should be assigned,
                    must be in alphabetical order to translate model output correctly
                    (since that's the way the models were trained)
    :image_path: path to the image for which class probabilities should be generated
    :return: dict that maps class name -> probability, ordered from highest
             probability to lowest probability
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    print("width")
    print(width)

    pixels = img.load()
    print("pixels")
    print(pixels)


    img_normalized = img_transform(img).float()
    # Create a mini batch to provide the expected data structure to the model
    input_batch = img_normalized.unsqueeze_(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        model.eval()
        output = model(input_batch)
        # Convert output to probabilities using the softmax layer
        probabilities = softmax(output)
        print(probabilities)
        # Use topk to sort the probabilities in descending order
        top_p, top_class = probabilities.topk(len(classes), dim=1)
        result = {}
        for i in range(0, len(classes)):
            prob = top_p[0][i].item()
            class_index = top_class[0][i].item()
            class_name = classes[class_index]
            result[class_name] = prob

        print("result")
        print(result)
        return list(result.items())
        

def generate_color_inferences(model, image_path: str):
    print("generate_color_inferences")
    transform = generate_transform()
    softmax = torch.nn.Softmax(dim=1)
    # MAKE SURE ALL YOUR IMAGE CLASSES IN HERE AND THAT THEY ARE IN ALPHABETICAL ORDER
    

    result = generate_inferences_for_image(model, softmax=softmax, 
                                           img_transform=transform,
                                           classes=classes, 
                                           image_path=image_path)
    print(result)
    return result


def run_color_inferences(image_path):
    print("run_color_inferences")

    model = generate_model(len(classes), "C:/Users/lgvii/source/repos/PillServicesPythonEnv/PillServicesPythonEnv/pytorch-models/color_model_80_20_model.pth5745821318_0.9032529444756029_.all_files")

    return generate_color_inferences(model, image_path)


