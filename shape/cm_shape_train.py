import gc
import os
import torch
from torchvision import transforms

import cnn_train_utils as utils


def run_shape_train_densenet_2(epoch_num: int = 10):
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


def run_shape_train_densenet_all(epoch_num: int = 10):
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_all_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models\shape_224x224_all_nw_"

    # The 640x640 is apparently too big, causes it to crash with a weird storage resize error
    # resize_transform = transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.LANCZOS)
    # Resize down to 224x224, min size for densenet
    resize_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS)
    train_loader, valid_loader = utils.generate_dataloaders_no_reserve(train_path, resize_transform=resize_transform)

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
    loss1 = criterion(outputs[0], labels)
    loss2 = criterion(outputs[1], labels)
    return loss1 + loss2


def run_shape_train_inception(epoch_num=10):
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    train_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_no_splimage_sort"
    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_splimage_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models\shape_"

    model, img_size = utils.initialize_model("inception", 13)
    model.train()
    model.cuda()

    # Inception v3 requires images to be resized
    resize_transform = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS)
    train_loader, valid_loader = utils.generate_dataloaders_with_reserve(train_path, val_path,
                                                                         resize_transform=resize_transform)

    assert abs(len(train_loader.dataset) - 39754) < 5
    assert abs(len(valid_loader.dataset) - 9938) < 5

    print(f"Using GPU: {torch.cuda.is_available()}")
    assert torch.cuda.is_available()

    # utils.validate_it(model, valid_loader, lambda outputs: outputs[0])

    # When training, the Inception model outputs InceptionOutput, with the predictions in the initial (logits) tensor
    # Also, the loss must be calculated using both outputs
    utils.pytorch_train_and_evaluate(model, train_loader, valid_loader, model_path, epoch_num=epoch_num,
                                     get_outputs=lambda outputs: outputs[0], calc_loss=calc_inception_loss)


# def canny_edge_preprocess(image)
def check_splimage_densenet_valid():
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\640_by_640_splimage_sort"
    model_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models"

    model, img_size = utils.initialize_model("densenet", 13, n_unfrozen=0)
    model.eval()
    model.cuda()

    utils.load_model(model, os.path.join(model_path, "shape_20221122_202125_0.34041054538136445_.all_files"))

    # The 640x640 is apparently too big, causes it to crash with a weird storage resize error
    resize_transform = transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.LANCZOS)
    valid_loader = utils.generate_valid_dataloader(val_path, resize_transform=resize_transform)
    utils.validate_it(model, valid_loader)


def check_splimage_densenet_valid_square(model_prefix: str, size: int):
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\shape_splimage_split_square_front_sort"
    model_dir = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models"

    print(val_path)
    model, img_size = utils.initialize_model("densenet", 13, n_unfrozen=0)
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


def check_splimage_densenet_valid_square_all(model_prefix: str, size: int):
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()

    val_path = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\shape_splimage_split_square_all_sort"
    model_dir = r"E:\NoBackup\DGMD_E-14_FinalProject\shape\models"

    print(val_path)
    model, img_size = utils.initialize_model("densenet", 13, n_unfrozen=0)
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
    # run_shape_train_densenet_2()
    # check_splimage_densenet_valid()
    # run_shape_train_densenet_spl_front()
    # check_splimage_densenet_valid_square("shape_224x224_all_nw_2022", 224)
    # run_shape_train_densenet_spl_front_224()
    # run_shape_train_densenet_all()
    # run_shape_train_densenet_spl_front(size=480, weight_samples=False)
    # run_shape_train_densenet_spl_front(size=224, weight_samples=True)
    # run_shape_train_densenet_spl_front(size=224, weight_samples=False, train_unsplit=True)
    # run_shape_train_densenet_spl_front(size=224, weight_samples=True, n_unfrozen=0)
    check_splimage_densenet_valid_square_all("shape_224x224_nw_2022", 224)
