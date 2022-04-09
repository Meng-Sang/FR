# -*- coding: utf-8 -*-
from glob import glob
import os
import cv2
import numpy as np
from tqdm import tqdm


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def defect_face(img, w, h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('util/files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray)
    if len(faces) == 0 or faces[0][2] < w or faces[0][3] < h:
        return False
    return True


class BatchCut(object):
    def __init__(self, impress):
        self.ratio = impress.ratio
        self.type = impress.type
        if impress.type == "count":
            self.cnt = impress.cnt
        elif impress.type == "resolution":
            self.resolution = impress.resolution
        else:
            raise Exception("this type is not exist")
        self.source_dir = impress.source_dir
        if not os.path.exists(self.source_dir):
            raise Exception("file is not exits")
        self.target_dir = impress.target_dir
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
        else:
            for root, dirs, files in tqdm(os.walk(self.target_dir)):
                for name in files:
                    if name.endswith(".jpg"):
                        os.remove(os.path.join(root, name))

    def cutImage(self):
        img_paths = os.path.join(self.source_dir, '*.jpg')
        img_paths = glob(img_paths)
        dict_name = {

        }
        for img_path in img_paths:
            image = cv_imread(img_path)
            file_name = os.path.basename(img_path)
            name = file_name.split('_', 1)[0]
            if name not in dict_name:
                dict_name[name] = 0
            if self.type == "count":
                self.impress_by_cnt(image, dict_name, name)
            elif self.type == "resolution":
                self.impress_by_resolution(image, dict_name, name)

    def impress_by_cnt(self, img, dict_name, name):
        for i in tqdm(range(self.cnt)):
            h, w, _ = img.shape
            size = (int(w * (self.ratio ** i)), int(h * (self.ratio ** i)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            dict_name[name] += 1
            s = str(dict_name[name])
            target_img = os.path.join(os.path.abspath(self.target_dir), name + '_' + s + '.jpg')
            cv2.imencode('.jpg', img)[1].tofile(target_img)

    def impress_by_resolution(self, img, dict_name, name):
        h, w, _ = img.shape
        i = 0
        while True:
            size = (int(w * (self.ratio ** i)), int(h * (self.ratio ** i)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if not defect_face(img, self.resolution[0], self.resolution[1]):
                break
            target_img = os.path.join(self.target_dir, name + '_' + str(dict_name[name]) + '.jpg')
            cv2.imencode('.jpg', img)[1].tofile(target_img)
            i += 1
            dict_name[name] += 1


if __name__ == '__main__':
    demo = BatchCut(None)
    demo.cutImage()
