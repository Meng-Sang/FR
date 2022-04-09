import os

import torch.nn

from model_util import get_model_recognition
from conf.config import get_config
from torchvision.utils import save_image
from torchvision import transforms

from util.utils import cv_imread

INSTANCE_FOLDER = None
INSTANCE_FOLDER_SUM = None
IMAGE_FOLDER = "./save_image"
modules_for_plot = (
    # torch.nn.ReLU6,
    torch.nn.Conv2d,
    # torch.nn.MaxPool2d,
    # torch.nn.AdaptiveMaxPool2d
)
BASE_TRANS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112))
])


def hook_func(module, input, output):
    image_name, sum_image_name = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    sum_data = torch.sum(data, dim=0, keepdim=True)
    save_image(data, image_name, pad_value=0.5)
    save_image(sum_data, sum_image_name, pad_value=0.5)


def get_image_name_for_hook(module):
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    os.makedirs(INSTANCE_FOLDER_SUM, exist_ok=True)
    base_name = str(module).split("(")[0]
    index = 0
    image_name = "."
    sum_image_name = "."
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(INSTANCE_FOLDER, f"{base_name}_{index}.png")
        sum_image_name = os.path.join(INSTANCE_FOLDER_SUM, f"{base_name}_{index}.png")
    return image_name, sum_image_name


def read_image(paths, device="cpu", trans=BASE_TRANS):
    imgs = []
    if isinstance(paths, str):
        img = cv_imread(paths)
        img.div_(255).sub_(0.5).div_(0.5)
        return [torch.stack((trans(img),), dim=0).to(device)]
    elif isinstance(paths, list):
        for path in paths:
            img = cv_imread(path)
            img.div_(255).sub_(0.5).div_(0.5)
            imgs.append(torch.stack(trans(img), dim=0).to(device))
        return imgs


def register_hook(model):
    for name, module in model.named_modules():
        if isinstance(module, modules_for_plot):
            module.register_forward_hook(hook_func)


if __name__ == "__main__":
    cfg = get_config()
    model = get_model_recognition(cfg.recognition)
    register_hook(model)
    images = read_image(r"F:\Pycharm\experience\FR\DB\img_embeddings\余凯龙_0.jpg", device=cfg.recognition.device)
    for index, image in enumerate(images):
        INSTANCE_FOLDER = os.path.join("visual_feature", f"{index}", "along")
        INSTANCE_FOLDER_SUM = os.path.join("visual_feature", f"{index}", "sum")
        model(image.half())
