import numpy as np
import torch

from util import get_model_detect, get_model_recognition
from util.utils import image_process_for_detect, get_scale_output, get_anchors, decode, decode_landmarks, \
    non_max_suppression, image_process_for_recognition, l2_norm, retinaface_correct_boxes, alignment


class Embeddings(object):
    def __init__(self, cfg):
        super(Embeddings, self).__init__()
        self.cfg = cfg
        self.retina = get_model_detect(cfg.retina)
        self.recognition = get_model_recognition(cfg.recognition)

    def get_embeddings(self, img):
        old_image = img.copy()
        scale, scale_for_landmarks = get_scale_output(img)
        anchors = get_anchors(self.cfg.retina)
        img = image_process_for_detect(img, self.cfg)
        with torch.no_grad():
            # img = torch.from_numpy(img).type(torch.FloatTensor)
            img = img.to(self.cfg.retina.device)
            loc, conf, landmarks = self.retina(img)
            loc, conf, landmarks = loc.cpu(), conf.cpu(), landmarks.cpu()
            boxes = decode(loc.data.squeeze(0), anchors, self.cfg.retina.net_cfg['variance'], scale)
            conf = conf.data.squeeze(0)[:, 1:2].numpy()
            landmarks = decode_landmarks(landmarks.data.squeeze(0), anchors, self.cfg.retina.net_cfg['variance'],
                                         scale_for_landmarks)
            boxes_conf_landmarks = np.concatenate([boxes, conf, landmarks], -1)
            boxes_conf_landmarks = non_max_suppression(boxes_conf_landmarks, self.cfg.retina.confidence)
        if len(boxes_conf_landmarks) < 1:
            return None, None
        boxes_conf_landmarks = retinaface_correct_boxes(boxes_conf_landmarks, np.array(self.cfg.retina.input_shape[:2]), np.array(old_image.shape[:2]))
        face_embeddings = []
        for boxes_conf_landmark in boxes_conf_landmarks:
            boxes_conf_landmark = np.maximum(boxes_conf_landmark, 0)
            recognition_image = image_process_for_recognition(old_image, boxes_conf_landmark, self.cfg)
            with torch.no_grad():
                recognition_image = recognition_image.type(torch.FloatTensor)
                recognition_image = recognition_image.to(self.cfg.recognition.device)
                if self.cfg.recognition.split:
                    embeddings = self.recognition(recognition_image)[1]
                else:
                    embeddings = self.recognition(recognition_image)
                face_embeddings.append(embeddings.cpu())
        face_embeddings = torch.cat(face_embeddings, 0)
        face_embeddings = l2_norm(face_embeddings, -1)
        return face_embeddings, boxes_conf_landmarks
