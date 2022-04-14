import os

import cv2
import numpy as np
import torch.nn

from model_util import get_model_recognition
from conf.config import get_config
from torchvision.utils import save_image
from torchvision import transforms

from util.utils import cv_imread

INSTANCE_FOLDER = None
INSTANCE_FOLDER_HEAT_MAP = None
INSTANCE_FOLDER_SUM = None
IMAGE_FOLDER = "./save_image"
IMAGE = None
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

FC = None


def filter_condition(name):
    if name == "layer4.2.conv2":
        return True
    return False


def hook_func(module, input, output):
    image_name, sum_image_name, head_map = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    sum_data = torch.sum(data, dim=0, keepdim=True)
    save_image(data, image_name, pad_value=0.5)
    save_image(sum_data, sum_image_name, pad_value=0.5)
    save_heatmap(output, head_map, IMAGE)


def get_a_heatmap(heatmap):
    global FC
    a = np.sum(FC, axis=0)
    heatmap = a.reshape((512, 7, 7)) * heatmap
    return heatmap


def save_heatmap(output, filename, image):
    heatmap = output[0].cpu().detach().numpy()
    if MODE_INDEX == 1:
        heatmap = get_a_heatmap(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap = np.mean(heatmap, axis=0)
    max_value = np.max(heatmap)
    if max_value != 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap -= heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heat_image = cv2.addWeighted(image, 1, heatmap, 0.5, 0)
    cv2.imwrite(filename, heat_image)


def get_image_name_for_hook(module):
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    os.makedirs(INSTANCE_FOLDER_SUM, exist_ok=True)
    os.makedirs(INSTANCE_FOLDER_HEAT_MAP, exist_ok=True)
    base_name = str(module).split("(")[0]
    index = 0
    image_name = "."
    sum_image_name = "."
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(INSTANCE_FOLDER, f"{base_name}_{index}.png")
        sum_image_name = os.path.join(INSTANCE_FOLDER_SUM, f"{base_name}_{index}.png")
        heat_map = os.path.join(INSTANCE_FOLDER_HEAT_MAP, f"{base_name}_{index}.png")
    return image_name, sum_image_name, heat_map


def read_image(paths, device="cpu", trans=BASE_TRANS):
    imgs = []
    if isinstance(paths, str):
        img = cv_imread(paths)
        if trans is not None:
            img = (img / 255 - 0.5) / 0.5
            return [torch.stack((trans(img),), dim=0).to(device)]
        else:
            return [img]
    elif isinstance(paths, list):
        for path in paths:
            img = cv_imread(path)
            if trans is not None:
                img = (img / 255 - 0.5) / 0.5
                imgs.append(torch.stack((trans(img),), dim=0).to(device))
            else:
                imgs.append(img)
        return imgs


def register_hook(model):
    for name, module in model.named_modules():
        if isinstance(module, modules_for_plot):
            if MODE_INDEX ==1:
                if filter_condition(name):
                    module.register_forward_hook(hook_func)
            else:
                module.register_forward_hook(hook_func)


sign = None


def get_FC(model):
    global FC
    try:
        FC = model.fc.g_lines.weight.clone().cpu().detach().numpy()
    except:
        FC = model.fc.weight.clone().cpu().detach().numpy()


MODE = ["ordinary", "specific"]
MODE_INDEX = 1

if __name__ == "__main__":
    cfg = get_config()
    sign = cfg.encodings.sign
    model = get_model_recognition(cfg.recognition)
    if MODE_INDEX == 1:
        get_FC(model)
    register_hook(model)
    files = [r"C:\Users\DELL\Desktop\rotate.jpg", r"C:\Users\DELL\Desktop\positive.jpg"]
    images = read_image(files, device=cfg.recognition.device)
    old_images = read_image(files, trans=None)
    for index, image in enumerate(images):
        IMAGE = old_images[index]
        INSTANCE_FOLDER = os.path.join("visual_feature", sign, f"{index}", "along")
        INSTANCE_FOLDER_SUM = os.path.join("visual_feature", sign, f"{index}", "sum")
        INSTANCE_FOLDER_HEAT_MAP = os.path.join("visual_feature", sign, f"{index}", "heat_map")
        model(image.half())
