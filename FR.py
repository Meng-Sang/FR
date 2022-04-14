import os
import _thread

import numpy as np
import torch
from cv2 import cv2

from conf.config import get_config
from util import get_stream, Embeddings
from util.utils import compare_dist, plot

cfg = get_config()
net = Embeddings(cfg)
threshold = cfg.recognition.threshold
db_embeddings = np.load(os.path.join(cfg.encodings.target_dir, cfg.encodings.sign, "embeddings.npy"), allow_pickle=True)
ids = np.load(os.path.join(cfg.encodings.target_dir, cfg.encodings.sign, "ids.npy"), allow_pickle=True)
cap_name, cap = get_stream(cfg)
if cfg.save:
    dir_name,ext = os.path.splitext(cfg.source)
    file_name = dir_name+"_"+cfg.encodings.sign+".avi"
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if cfg.save:
            writer.release()
        cap.release()
        break
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
    if cfg.save:
        writer.write(img)
    cv2.imshow(str(cfg.source), img)
    cv2.waitKey(1)
    del ret
    del frame

