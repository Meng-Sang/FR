import os

import torch

from recognition_net.arcface.backbones import get_model
from conf.config import get_config
from nets_retinaface.retinaface import RetinaFace

def get_model_detect(retina):
    retina_net = RetinaFace(retina.net_cfg, pre_train=False, phase="eval")
    retina_net.eval()
    retina_net.to(retina.device)
    if os.path.exists(retina.model_path):
        retina_net.load_state_dict(torch.load(retina.model_path, retina.device))
    else:
        raise Exception("model file is not exist")
    return retina_net


def get_model_recognition(recognition):
    if recognition.model_type == "arcface":
        resnet = get_model(recognition.network, dropout=recognition.dropout, fp16=recognition.fp16,
                           split=recognition.split).to(recognition.device)
    elif recognition.model_type == "insightface_pytorch":
        from recognition_net.insightface_pytorch.model import Backbone
        resnet = Backbone(recognition, recognition.net_depth, recognition.drop_ratio, recognition.net_mode).to(recognition.device)
    else:
        raise Exception("not exist this type")
    resnet.eval()
    if os.path.exists(os.path.join(recognition.model_dir, recognition.model_type, recognition.model_name)):
        resnet.load_state_dict(
            torch.load(os.path.join(recognition.model_dir, recognition.model_type, recognition.model_name),
                       recognition.device))
    else:
        raise Exception("model file is not exist")
    return resnet
    # pass


if __name__ == "__main__":
    conf = get_config()
    get_model_detect(conf.retina)
    get_model_recognition(conf.recognition)
