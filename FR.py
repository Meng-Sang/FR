import os
import _thread

import numpy as np
import torch
from cv2 import cv2

from conf.config import get_config
from util import get_stream, Embeddings
from util.utils import compare_dist, plot

cfg = get_config()
cap_name, cap = get_stream(cfg)
net = Embeddings(cfg)
threshold = cfg.recognition.threshold
db_embeddings = np.load(os.path.join(cfg.encodings.target_dir, cfg.encodings.sign, "embeddings.npy"), allow_pickle=True)
ids = np.load(os.path.join(cfg.encodings.target_dir, cfg.encodings.sign, "ids.npy"), allow_pickle=True)


def deal_result(frame):
    img = np.array(frame)
    old_image = img.copy()
    embeddings, boxes_conf_landmarks = net.get_embeddings(img)
    if embeddings is None:
        cv2.waitKey(old_image)
        cv2.waitKey(0)
        return
    names = compare_dist(embeddings.numpy(), threshold, db_embeddings, ids)
    img = plot(old_image, names, boxes_conf_landmarks)
    cv2.imshow(str(cfg.source), img)
    cv2.waitKey(1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    # _thread.start_new_thread(deal_result, (frame,))
    img = np.array(frame)
    old_image = img.copy()
    embeddings, boxes_conf_landmarks = net.get_embeddings(img)
    if embeddings is None:
        cv2.imshow(str(cfg.source),old_image)
        cv2.waitKey(0)
        continue
    names = compare_dist(embeddings.numpy(), threshold, db_embeddings, ids)
    img = plot(old_image, names, boxes_conf_landmarks)
    cv2.imshow(str(cfg.source), img)
    cv2.waitKey(1)
    del ret
    del frame

