import gc
import os
import torch
from torchvision import transforms

import cnn_train_utils as utils


def run_shape_train_densenet_splimage_reserve(epoch_num: int = 10):
    """
    Function used to train the Densenet121 CNN to predict shape classes.

    Freezes all but the last 100 layers of the Densenet model.

    Uses the 640x640 C3PI_Test and MC_CHALLENGE_V1.0 images for the training set, with the MC_SPL_SPLIMAGE_V3.0 images
    as the reserved validation set, with a random split of the training set added to make the final ratio 80/20.

    Resizes all images to 480x480 (since 640x640 was too large and caused it to crash).

    Uses the WeightedRandomSampler for the training DataLoader.

    :param epoch_num: number of epochs to train, defaults to 10
    """
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_no_splimage_sort"
    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_splimage_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models\shape_"

    # The 640x640 is apparently too big, causes it to crash with a weird storage resize error
    resize_transform = transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.LANCZOS)
    train_loader, valid_loader = utils.generate_dataloaders_with_reserve(train_path, val_path,
                                                                         resize_transform=resize_transform)

    assert abs(len(train_loader.dataset) - 39754) < 5
    assert abs(len(valid_loader.dataset) - 9938) < 5

    model, img_size = utils.initialize_model("densenet", 13, n_unfrozen=100)
    model.train()
    model.cuda()

    print(f"Using GPU: {torch.cuda.is_available()}")
    assert torch.cuda.is_available()

    # print(model)

    # utils.validate_it(model, valid_loader, get_outputs=lambda outputs: outputs.data)
    utils.pytorch_train_and_evaluate(model, train_loader, valid_loader, model_path, epoch_num=epoch_num,
                                     get_outputs=lambda outputs: outputs.data)


def run_shape_train_densenet_spl_front(size: int, weight_samples: bool, epoch_num: int = 10,
                                       train_unsplit: bool = False, n_unfrozen: int = 150):
    """
    Function used to train the Densenet121 CNN to predict shape classes.

    Uses the 640x640 C3PI_Test and MC_CHALLENGE_V1.0 images for the training set, with the "front" split SPL images as
    the reserved validation set.  If train_unsplit is True, only the split SPL images are used for validation.  If
    train_unsplit is False, a random split of the training set is added to make the final ratio 80/20.

    Resizes all images to the square size specified.
    Optionally uses the WeightedRandomSampler in training if specified.  Note that image order will be shuffled even if
    the sampler isn't used.

    :param size: square size to which the DataLoaders should resize images
    :param weight_samples: True if the RandomWeightedSampler should be used, False if not
    :param epoch_num: number of epochs to train, defaults to 10
    :param train_unsplit: True if all the images in the training set should be used for fine-tuning, False if some of
                          them should be split to form a combined validation set with the reserved split SPL images,
                          defaults to False
    :param n_unfrozen: number of layers to leave unfrozen for fine-tuning, defaults to 150
    """
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_no_splimage_sort"
    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\shape_splimage_split_square_front_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models" \
                 + f"\\shape_{size}x{size}{'_unsplit' if train_unsplit else ''}_{'w' if weight_samples else 'nw'}_"
    print(f"Model path: {model_path}")

    # The 640x640 is apparently too big, causes it to crash with a weird storage resize error
    resize_transform = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS)

    if train_unsplit:
        train_loader = utils.generate_train_dataloader(train_path, resize_transform=resize_transform,
                                                       weight_samples=weight_samples)
        valid_loader = utils.generate_valid_dataloader(val_path, resize_transform=resize_transform)

        assert abs(len(train_loader.dataset) - 44995) < 5
        assert abs(len(valid_loader.dataset) - 4106) < 5
    else:
        train_loader, valid_loader = utils.generate_dataloaders_with_reserve(train_path, val_path,
                                                                             resize_transform=resize_transform,
                                                                             weight_samples=weight_samples)
        assert abs(len(train_loader.dataset) - 39281) < 5
        assert abs(len(valid_loader.dataset) - 9820) < 5

    model, img_size = utils.initialize_model("densenet", 13, n_unfrozen=n_unfrozen)
    model.train()
    model.cuda()

    print(f"Using GPU: {torch.cuda.is_available()}")
    assert torch.cuda.is_available()

    # utils.validate_it(model, valid_loader, get_outputs=lambda outputs: outputs.data)
    utils.pytorch_train_and_evaluate(model, train_loader, valid_loader, model_path, epoch_num=epoch_num,
                                     get_outputs=lambda outputs: outputs.data)


def run_shape_train_densenet_all(weight_samples: bool, epoch_num: int = 10, train_unsplit: bool = False):
    """
    Function used to train the Densenet121 CNN to predict shape classes.

    Freezes all but the last 150 layers of the Densenet model.

    Uses all the resized 640x640 C3PI JPG images, with a random 80/20 split of the images for training/validation

    Resizes all images to the standard 224x224 size for the Densenet CNN.

    :param weight_samples: True if the RandomWeightedSampler should be used, False if not
    :param epoch_num: number of epochs to train, defaults to 10
    :param train_unsplit: True if all the C3PI JPG images should be used for training, with the split SPL images used
                          for validation, False if the C3PI JPG images should be split 80/20 for training/validation
    """
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_all_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models" \
                 + f"\\shape_224x224_all{'_unsplit' if train_unsplit else ''}_{'w' if weight_samples else 'nw'}_"

    # Resize down to 224x224, min size for densenet
    resize_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS)
    if train_unsplit:
        val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\shape_splimage_split_square_all_sort"
        train_loader = utils.generate_train_dataloader(train_path, resize_transform=resize_transform,
                                                       weight_samples=weight_samples)
        valid_loader = utils.generate_valid_dataloader(val_path, resize_transform=resize_transform)

        assert abs(len(train_loader.dataset) - 59086) < 5
        assert abs(len(valid_loader.dataset) - 9036) < 5
    else:
        train_loader, valid_loader = utils.generate_dataloaders_no_reserve(train_path, resize_transform=resize_transform,
                                                                           weight_samples=weight_samples)

        assert abs(len(train_loader.dataset) - 47269) < 5
        assert abs(len(valid_loader.dataset) - 11817) < 5

    model, img_size = utils.initialize_model("densenet", 13, n_unfrozen=150)
    model.train()
    model.cuda()

    print(f"Using GPU: {torch.cuda.is_available()}")
    assert torch.cuda.is_available()

    utils.pytorch_train_and_evaluate(model, train_loader, valid_loader, model_path, epoch_num=epoch_num,
                                     get_outputs=lambda outputs: outputs.data)


# Inception v3 has two separate outputs when training, need to include both in the loss
def calc_inception_loss(criterion, outputs, labels):
    """
    Function used to calculate loss when fine-tuning the Inception v3 CNN.  Since this model has two separate outputs
    used when training, the loss calculation must be modified to include both for proper fine-tuning.

    :param criterion: object used to calculate loss for a given set of outputs and labels
    :param outputs: outputs generated by the model for a set of images
    :param labels: labels associated with the images
    :return: total loss to be fed back into the model for fine-tuning
    """
    loss1 = criterion(outputs[0], labels)
    loss2 = criterion(outputs[1], labels)
    return loss1 + loss2


def run_shape_train_inception(epoch_num: int = 10) -> None:
    """
    Function used to train the Inception v3 CNN to predict shape classes.

    Leaves all layers unfrozen for fine-tuning.

    Uses the 640x640 C3PI_Test and MC_CHALLENGE_V1.0 images for the training set, with the MC_SPL_SPLIMAGE_V3.0 as the
    reserved validation set to which a random split of the training set is added to make the final ratio 80/20.

    Resizes all images to the 299x299 size required by the Inception v3 CNN.

    Uses the WeightedRandomSampler in training.

    :param epoch_num: number of epochs to train, defaults to 10
    """
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_no_splimage_sort"
    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_splimage_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models\shape_inception_"

    model, img_size = utils.initialize_model("inception", 13)
    model.train()
    model.cuda()

    # Inception v3 requires images to be resized to a specific size
    resize_transform = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS)
    train_loader, valid_loader = utils.generate_dataloaders_with_reserve(train_path, val_path,
                                                                         resize_transform=resize_transform)

    assert abs(len(train_loader.dataset) - 39754) < 5
    assert abs(len(valid_loader.dataset) - 9938) < 5

    print(f"Using GPU: {torch.cuda.is_available()}")
    assert torch.cuda.is_available()

    # When training, the Inception model outputs InceptionOutput, with the predictions in the initial (logits) tensor
    # Also, the loss must be calculated using both outputs
    utils.pytorch_train_and_evaluate(model, train_loader, valid_loader, model_path, epoch_num=epoch_num,
                                     get_outputs=lambda outputs: outputs[0], calc_loss=calc_inception_loss)


def test_densenet(val_path: str, model_dir: str, model_prefix: str, size: int):
    """
    Function used to calculate the prediction accuracy for the images in the specified val_path using all saved models
    in the specified model_dir where the files start with the specified model_prefix.

    Images will be resized to size x size by the PyTorch DataLoader.

    :param val_path: directory containing validation images sorted into subdirectories by class
    :param model_dir: directory containing saved model files to check
    :param model_prefix: file name prefix used to select the model files to load and check
    :param size: size of the image that should be run through the model (images will be resized to size x size by the
                 DataLoader)
    """
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    print(val_path)
    model, _ = utils.initialize_model("densenet", 13, n_unfrozen=0)
    model.eval()
    model.cuda()

    models = os.listdir(model_dir)
    model_paths = [f for f in models if f.startswith(model_prefix)]
    for model_path in model_paths:
        print(f"model: {model_path}")
        utils.load_model(model, os.path.join(model_dir, model_path))

        # The 640x640 is apparently too big, causes it to crash with a weird storage resize error
        resize_transform = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS)
        valid_loader = utils.generate_valid_dataloader(val_path, resize_transform=resize_transform)
        utils.validate_it(model, valid_loader)
        print()


if __name__ == "__main__":
    # run_shape_train_inception()
    # run_shape_train_densenet_splimage_reserve()
    # run_shape_train_densenet_all(weight_samples=False, train_unsplit=True)
    # run_shape_train_densenet_spl_front(size=224, weight_samples=False)

    # val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_splimage_sort"

    parent_dir = r"E:\NoBackup\DGMD_E-14_FinalProject\shape"
    val_path = os.path.join(parent_dir, "shape_splimage_split_square_back_sort")
    model_dir = os.path.join(parent_dir, "models")
    test_densenet(val_path, model_dir, "shape_224x224_nw_20221123_123637_0.6524439918533604_", 224)
