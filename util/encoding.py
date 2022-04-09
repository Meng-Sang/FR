import os

import numpy as np
import torch
from cv2 import cv2

from conf.config import get_config
from util import Embeddings


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


class Encodings(object):
    def __init__(self, cfg):
        super(Encodings, self).__init__()
        self.source_dir = cfg.source_dir
        self.sign = cfg.sign
        if not os.path.exists(self.source_dir):
            raise Exception("encoding dir is not exist")
        self.target_dir = os.path.join(cfg.target_dir, self.sign)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
        else:
            raise Exception("sign is exist")
        self.embeddings = []
        self.names = []

    def encodings(self, net):
        files = os.listdir(self.source_dir)
        for file in files:
            file_path = os.path.join(self.source_dir, file)
            img = cv_imread(file_path)
            name = str(file).split("_")[0]
            embeddings = net.get_embeddings(img)[0]
            self.embeddings.append(embeddings)
            self.names.extend([name] * len(embeddings))
        self.embeddings = torch.cat(self.embeddings,0).numpy()
        self.names = np.array(self.names)
        np.save(os.path.join(self.target_dir, "embeddings.npy"), self.embeddings)
        np.save(os.path.join(self.target_dir, "ids.npy"), self.names)


if __name__ == "__main__":
    cfg = get_config()
    net = Embeddings(cfg)
    encodings = Encodings(cfg.encodings)
    encodings.encodings(net)
