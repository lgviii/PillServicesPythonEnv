from datetime import datetime
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from timeit import default_timer as timer

# batch sizes used for DataLoaders
BATCH_SIZE = 32
BATCH_SIZE_VALID = 32


def get_default_normalization():
    """
    Generates the default PyTorch normalization used - this is the normalization used by both Inception v3 and Densenet.

    :return: normalization transform with default mean/std values for PyTorch models.
    """
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )


def generate_valid_transform(normalize=get_default_normalization(), resize_transform=None):
    """
    Generates the transformation pipeline to use for validation images - should be used both for the validation set
    during training, and on any images for which inferences will be generated from the pre-trained model.

    :param normalize: normalization transform to use, defaults to the "default" PyTorch normalization if not specified
    :param resize_transform: resize transform to use, skips the resize if not provided
    :return: transformation pipeline which should be run on all validation/inference images
    """
    valid_transforms = [transforms.ToTensor(),
                        normalize]
    if resize_transform:
        valid_transforms.insert(0, resize_transform)
    return transforms.Compose(valid_transforms)


def generate_train_transform(normalize=get_default_normalization(), resize_transform=None):
    """
    Generates the transformation pipeline to be used with the training image dataset, including some random
    permutations on the images.

    :param normalize: normalization transform to use, defaults to the "default" PyTorch normalization if not specified
    :param resize_transform: resize transform to use, skips the resize if not provided
    :return: transformation pipeline which should be used for training datasets
    """
    train_transform_list = [transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomRotation(degrees=(30, 70)),
                            transforms.ToTensor(),
                            normalize]
    if resize_transform:
        train_transform_list.insert(0, resize_transform)
    train_transform = transforms.Compose(train_transform_list)
    return train_transform


def calc_split_factor(len_train: int, len_reserved_valid: int,
                      total_valid_percent: float = 0.2) -> float:
    """
    Calculates the percentage of training images to split out for the validation set during training to generate the
    specified final validation set percentage, assuming that there's a separate "reserved" set of images that are NOT
    included in the training set, but will additionally be used for validation.

    That, this assumes that the total images include both a "training" set and a "reserved" set, which will be used only
    for validation, and that we want the final validation set used during training to be the specified percentage of
    the total images.

    Calculates the split factor s such that
    s * (len_train) + (len_reserved_valid) = (total_valid_percent) * (len_train + len_reserved_valid)

    :param len_train: number of images in the training set
    :param len_reserved_valid: number of images in the reserved set used for validation only
    :param total_valid_percent: final percentage of the total images which should be in the validation set used during
           training
    :return: split factor to use when splitting the training set
    """
    len_total = len_train + len_reserved_valid
    total_valid = len_total * total_valid_percent
    remaining_valid = total_valid - len_reserved_valid
    return remaining_valid / len_train


def generate_valid_dataloader(val_path: str, normalize=get_default_normalization(),
                              resize_transform=None) -> DataLoader:
    """
    Generates a dataloader for validation data contained in the specified validation path, using the specified
    normalization transformation and resize transformation, if any.

    Assumes that the data is sorted into folders, each labeled with the name of the image class, suitable for use with
    the PyTorch ImageFolder dataset.

    :param val_path: directory containing the sorted validation images
    :param normalize: normalization transform to use, defaults to the "default" PyTorch normalization if not specified
    :param resize_transform: resize transform to use, skips the resize if not provided
    :return: validation dataloader based on sorted images in the specified val_path directory
    """
    print("VALIDATION PATH: " + val_path)

    # the validation transforms
    valid_transform = generate_valid_transform(normalize, resize_transform)

    valid_dataset = datasets.ImageFolder(
        root=val_path,
        transform=valid_transform
    )
    print(f"Validation dataset: {len(valid_dataset)}")
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE_VALID, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print("VALIDATION DATASET BATCHES: " + str(len(valid_data_loader)))
    return valid_data_loader


def generate_weight_sampler(len_train: int, train_dataset) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler inversely proportional to the prevalence of each class in the specified dataset.

    Note that this sampler DOES use replacement, so the same image may be selected multiple times.

    :param len_train: length of the training set, used as a check that the generation of weights is the correct size
    :param train_dataset: training dataset for which the WeightedRandomSampler should be generated
    :return: WeightedRandomSampler with replacement, with weights set to be inversely proportional to the prevalence
             of each class in the specified dataset
    """
    y_train_indices = train_dataset.indices
    # Since train_dataset is a subset, need to access inner dataset to get the folder names
    y_train = [train_dataset.dataset.targets[i] for i in y_train_indices]

    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1. / class_sample_count

    samples_weight = np.array([weight[t] for t in y_train])
    assert len(samples_weight) == len_train
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler


def generate_dataloaders_with_reserve(train_path: str, val_path: str, normalize=get_default_normalization(),
                                      resize_transform=None, weight_samples=True) -> Tuple[DataLoader, DataLoader]:
    """
    Builds training and validation dataloaders by splitting the images in the specified train_path into separate train
    and validation sets, and combining the split validation set with the reserved images in the specified val_path such
    that the total ratio of training images to validation images is 80/20.

    That is, if the train_path has 110 images and the val_path has 15 images, the final training dataloader will have
    100 images from the train_path and the final validation dataloader will have 10 images from the train_path plus the
    15 reserved images from the val_path.

    :param train_path: file path to a parent directory containing images sorted into subdirectories by shape, to be used
                       for training the model
    :param val_path: file path to a parent directory containing images sorted into subdirectories by shape, to be used
                     for validating the trained model
    :param normalize: normalization transform to use for both training and validation dataloaders, defaults to
                      get_default_normalization() if not specified
    :param resize_transform: resize transform to use for both training and validation dataloaders, defaults to None if
                             not specified
    :param weight_samples: True if the WeightedRandomSampler should be used for the training dataloader, with weights
                           inversely proportional to the class distribution, False if it shouldn't be, defaults to True
    :return: tuple containing the training and validation dataloaders built from the specified image directories and
             options
    """
    print("TRAINING PATH: " + train_path)
    print("VALIDATION PATH: " + val_path)

    # the training transforms
    train_transform = generate_train_transform(normalize, resize_transform)

    # the validation transforms
    valid_transform = generate_valid_transform(normalize, resize_transform)

    # training dataset
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )
    # partial validation dataset, containing images split from the full train to make an 80/20 split
    # set root as train_path - we'll split using indexes instead of the using the random_split method
    # to ensure that it has the right transform
    valid_dataset = datasets.ImageFolder(
        root=train_path,
        transform=valid_transform
    )

    # The rest of the validation dataset, containing the SPL images
    valid_dataset2 = datasets.ImageFolder(
        root=val_path,
        transform=valid_transform
    )

    # https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/7
    # Split by indexes instead of the using the random_split method to ensure that it has the right transform
    len_full_train = len(train_dataset)
    indices = torch.randperm(len_full_train)

    # Use factor so that when the split validation images from the training set are added
    # to the reserved SPL images the full validation set will be 20% of the total images
    factor = calc_split_factor(len_full_train, len(valid_dataset2))
    val_size = round(len_full_train * factor)

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-val_size:])

    len_train = len(train_dataset)
    len_valid = len(valid_dataset)
    print(f"Training dataset: {len_train}")
    print(f"Partial validation dataset: {len_valid}")

    sampler = None
    if weight_samples:
        sampler = generate_weight_sampler(len_train, train_dataset)

    print(f"Weighted sampler used: {weight_samples}")

    # training data loaders
    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True
    )
    # validation data loaders
    # Combine the valid images split from the training with the SPL images
    valid_dataset_combined = torch.utils.data.ConcatDataset([valid_dataset, valid_dataset2])
    len_full_valid = len(valid_dataset_combined)
    print(f"Full validation dataset: {len_full_valid}")

    valid_data_loader = DataLoader(
        valid_dataset_combined, batch_size=BATCH_SIZE_VALID, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print("TRAINING DATASET BATCHES: " + str(len(train_data_loader)))
    print("VALIDATION DATASET BATCHES: " + str(len(valid_data_loader)))

    return train_data_loader, valid_data_loader


def generate_dataloaders_no_reserve(train_path: str, normalize=get_default_normalization(),
                                    resize_transform=None, weight_samples=True) -> Tuple[DataLoader, DataLoader]:
    """
    Generates training and validation dataloaders by doing a straight 80/20 train/test split on the images in the
    specified train_path.

    :param train_path: file path to a parent directory containing images sorted into subdirectories by shape, to be used
                       for training and testing the model
    :param normalize: normalization transform to use for both training and validation dataloaders, defaults to
                      get_default_normalization() if not specified
    :param resize_transform: resize transform to use for both training and validation dataloaders, defaults to None if
                             not specified
    :param weight_samples: True if the WeightedRandomSampler should be used for the training dataloader, with weights
                           inversely proportional to the class distribution, False if it shouldn't be, defaults to True
    :return: tuple containing the training and validation dataloaders built from the specified image directory and
             options
    """
    print("TRAINING PATH: " + train_path)

    # the training transforms
    train_transform = generate_train_transform(normalize, resize_transform)

    # the validation transforms
    valid_transform = generate_valid_transform(normalize, resize_transform)

    # training dataset
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )
    # partial validation dataset, containing images split from the full train to make an 80/20 split
    # set root as train_path - we'll split using indexes instead of the using the random_split method
    # to ensure that it has the right transform
    valid_dataset = datasets.ImageFolder(
        root=train_path,
        transform=valid_transform
    )

    # https://discuss.pytorch.org/t/changing-transforms-after-creating-a-dataset/64929/7
    # Split by indexes instead of the using the random_split method to ensure that it has the right transform
    # By shuffling the indices this way, it also effectively shuffles the images
    len_full_train = len(train_dataset)
    indices = torch.randperm(len_full_train)

    # Use 20% factor to do 80/20 split of images
    factor = 0.2
    val_size = round(len_full_train * factor)

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-val_size:])

    len_train = len(train_dataset)
    len_valid = len(valid_dataset)
    print(f"Training dataset: {len_train}")
    print(f"Validation dataset: {len_valid}")

    sampler = None
    if weight_samples:
        sampler = generate_weight_sampler(len_train, train_dataset)

    print(f"Weighted sampler used: {weight_samples}")

    # training data loader
    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True
    )
    # validation data loader
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE_VALID, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print("TRAINING DATASET BATCHES: " + str(len(train_data_loader)))
    print("VALIDATION DATASET BATCHES: " + str(len(valid_data_loader)))

    return train_data_loader, valid_data_loader


def generate_train_dataloader(train_path: str, normalize=get_default_normalization(), resize_transform=None,
                              weight_samples: bool = True, shuffle: bool = False) -> DataLoader:
    """
    Generates the training dataloader using all the images in the specified train_path.

    :param train_path: file path to a parent directory containing images sorted into subdirectories by shape, to be used
                       for training the model
    :param normalize: normalization transform to use for the training dataloader, defaults to
                      get_default_normalization() if not specified
    :param resize_transform: resize transform to use for the training dataloader, defaults to None if not specified
    :param weight_samples: True if the WeightedRandomSampler should be used for the dataloader, with weights inversely
                           proportional to the class distribution, False if it shouldn't be, defaults to True
    :return: training dataloader built from the specified image directory and options
    """
    print("TRAINING PATH: " + train_path)

    # the training transforms
    train_transform = generate_train_transform(normalize, resize_transform)

    # training dataset
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=train_transform
    )

    # Shuffle the indices in the training dataset so that the output drawn sequentially will still be random
    len_train = len(train_dataset)
    indices = torch.randperm(len_train)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)

    print(f"Training dataset: {len_train}")

    # **BEGIN TESTING RANDOM WEIGHTED SAMPLER
    sampler = None
    if weight_samples:
        sampler = generate_weight_sampler(len_train, train_dataset)
        # If a WeightedRandomSampler is used, do NOT shuffle, as it will cause an error in the DataLoader
        # (not allowed to use sampler and shuffle = True together)
        shuffle = False

    # #**END TESTING, RANDOM WEIGHTED SAMPLER

    print(f"Weighted sampler used: {weight_samples}, Shuffle used: {shuffle}")

    # training data loader
    train_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=shuffle,
        num_workers=2, pin_memory=True
    )

    print("TRAINING DATASET BATCHES: " + str(len(train_data_loader)))

    return train_data_loader


def use_last_n_layers(model, n_unfrozen_layers: int) -> int:
    """
    Freezes all but the last n_unfrozen_layers on the specified model.

    If the model has fewer layers than n_unfrozen_layers, leaves all layers unfrozen.

    Based on method in https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    :param model: PyTorch CNN model for which the last n_unfrozen_layers should be left active for finetuning.
    :param n_unfrozen_layers: number of layers to leave unfrozen
    :return: total number of layers in the model
    """
    # First count the layers (model.parameters() doesn't have a len implementation, so do this manually)
    n_layers = 0
    for _ in model.parameters():
        n_layers += 1

    layers_to_freeze = 0 if n_layers < n_unfrozen_layers else n_layers - n_unfrozen_layers
    layer_count = 0
    for param in model.parameters():
        layer_count += 1
        if layer_count <= layers_to_freeze:
            param.requires_grad = False

    return n_layers


def freeze_layers(model, n_unfrozen_layers: int) -> int:
    """
    Freezes all but the specified number of layers, or leaves all layers unfrozen if n_unfrozen_layers is None.

    :param model: model whose layers should be frozen
    :param n_unfrozen_layers: number of layers to leave unfrozen, set to 0 to freeze all layers (as for feature
                              detection), or to None to leave all layers unfrozen (as for full fine-tuning)
    :return: total number of layers in the model
    """
    n_layers = 0
    if n_unfrozen_layers is not None:
        n_layers = use_last_n_layers(model, n_unfrozen_layers)
    else:
        for _ in model.parameters():
            n_layers += 1

    return n_layers


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def initialize_model(model_name: str, num_classes: int, n_unfrozen: int = None):
    """
    Load the model with the specified name and initializes it using default pre-trained weights, modifying it to have
    the final layer output the specified num_classes, optionally specifying the number of layers to leave unfrozen.

    Copied from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html with a few changes.

    :param model_name: name of the model to load and initialize with PyTorch default pre-trained weights for that model
    :param num_classes: number of classes that the final model layer should output
    :param n_unfrozen: number of layers to leave unfrozen, specify 0 to freeze all layers or None to leave all layers
                       unfrozen, defaults to None
    :return: tuple containing the initialized model and the default image size to use for model input
    """
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    n_layers = 0

    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            """ Resnet18 - 62 layers
            """
            model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_name == "resnet50" or model_name == "resnet":
            """ Resnet50 - 161 layers
            """
            model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        n_layers = freeze_layers(model_ft, n_unfrozen)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet - 16 layers
        """
        model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        n_layers = freeze_layers(model_ft, n_unfrozen)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg11" or model_name == "vgg":
        """ VGG11_bn - 38 layers
        """
        model_ft = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        n_layers = freeze_layers(model_ft, n_unfrozen)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet - 52 layers
        """
        model_ft = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        n_layers = freeze_layers(model_ft, n_unfrozen)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name.startswith("densenet"):
        if model_name == "densenet" or model_name == "densenet121":
            """ Densenet121 - 364 layers
            """
            model_ft = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        elif model_name == "densenet161":
            """ Densenet161 - 484 layers
            """
            model_ft = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        n_layers = freeze_layers(model_ft, n_unfrozen)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 - 292 layers
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        n_layers = freeze_layers(model_ft, n_unfrozen)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print(f"Invalid model name {model_name}, exiting...")
        exit()

    layer_message = ""
    if n_unfrozen is None:
        layer_message = f"0 of {n_layers} layers frozen, all layers active for optimization"
    elif n_layers == n_unfrozen:
        layer_message = f"ALL {n_layers} layers frozen"
    else:
        layer_message = f"{n_layers - n_unfrozen} layers of {n_layers} frozen, " \
                        f"last {n_unfrozen} layers active for optimization"

    print(f"{model_name}, {layer_message}")
    return model_ft, input_size


def generate_timestamp() -> str:
    """
    Generates a timestamp string of the current date and time, for use in labeling saved model files.

    :return: timestamp string of the current date and time, for use in labeling saved model files
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def save_model(epochs, model, optimizer, criterion, model_path: str, val_acc: float = None) -> None:
    """
    Saves the trained model to disk, including epochs, model state_dict, optimizer state_dict, and criterion.

    Saved object structure:
    {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }

    Builds the name of the model file from the specified model_path as a prefix, adding a timestamp and the validation
    accuracy to the model file name.

    For later use in validation or generating inferences, the model_state_dict should be loaded into an initialized
    model of the same type.
    """
    extension = generate_timestamp()
    extension = extension + "_" + str(val_acc) + "_"
    extension = extension + ".all_files"
    model_path = model_path + extension  # *add a file extension so we can easily keep savin new models why not?
    print("SAVING MODEL: " + model_path)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, model_path)


def load_model(model, model_path: str):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def validate_it(model, test_loader: DataLoader, get_outputs=lambda outputs: outputs) -> float:
    """
    Runs all images in the specified test_loader DataLoader against the specified model, tabulating the overall
    prediction accuracy of the model predictions against the actual labels, in range [0, 1].

    :param model: PyTorch model against which the images in the specified test_loader should be run
    :param test_loader: PyTorch dataloader containing images to test for accuracy against the specified model
    :param get_outputs: function that extracts the actual model outputs from the object generated by calling the
                        model on a set of images from the data loader, present in case the model output structure
                        contains multiple parts (as for the Inception v3 CNN), defaults to simply returning the
                        outputs object
    :return: accuracy for predictions by the model across all images in the specified test_loader, in range [0, 1]
    """
    ### TESTING PORTION ###
    start = timer()
    print("validating model")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(get_outputs(outputs), 1)
            # print(f"Predicted: {predicted}")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'Accuracy of the pytorch model on the validation images: {100 * acc:.2f}%')
    end = timer()
    print(end - start)  # Time in seconds, e.g. 5.38091952400282
    return acc


def pytorch_train_and_evaluate(model, train_loader: DataLoader, test_loader: DataLoader,
                               model_path_prefix: str, epoch_num=10,
                               get_outputs=lambda outputs: outputs,
                               calc_loss=lambda criterion, outputs, labels: criterion(outputs, labels)) -> None:
    """
    Fine-tunes the specified PyTorch model using the images in the specified train_loader, validating after each epoch
    using the images in the specified test_loader.  Runs for the specified number of epochs.  After each epoch, saves
    the current model state using the specified model path prefix.

    For Densenet, the get_outputs and calc_loss lambda arguments can be left as default - these were added to allow use
    with Inception v3, which has outputs in a different format, and has two separate optimizers whose loss should be
    combined.

    :param model: PyTorch model to fine tune and validate using the specified images
    :param train_loader: PyTorch DataLoader containing images to be used in training the model
    :param test_loader: PyTorch DataLoader containing images to be using in validating the model
    :param model_path_prefix: prefix for the saved model file
    :param epoch_num: number of epochs to fine tune, defaults to 10
    :param get_outputs: lambda used to extract the actual predictions from the model outputs, defaults to just returning
                        the outputs object
    :param calc_loss: lambda used to calculate the loss for the data run through the training, defaults to the value
                      returned by running the criterion with the outputs and labels
    """
    ### TRAINING PORTION ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("perform initial validation check: ")
    val_acc = validate_it(model, test_loader, get_outputs)
    print("perform model save check: ")
    save_model(0, model, optimizer, criterion, model_path_prefix, val_acc=val_acc)
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        print("TRAINING EPOCH: " + str(epoch))
        running_loss = 0.0
        start = timer()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = calc_loss(criterion, outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        val_acc = validate_it(model, test_loader, get_outputs)  # *validate after each epoch
        save_model(epoch, model, optimizer, criterion, model_path_prefix, val_acc=val_acc)
        end = timer()
        print("EPOCH TIME")
        print("Seconds: " + str(end - start))  # Time in seconds, e.g. 5.38091952400282
    print("ending runtime")
