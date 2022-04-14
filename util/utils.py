# image process
import math

import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw
from cv2 import cv2
from util.anchors import Anchors


def get_scale_output(img):
    scale = torch.Tensor([np.shape(img)[1], np.shape(img)[0], np.shape(img)[1], np.shape(img)[0]])
    scale_for_landmark = torch.Tensor(
        [np.shape(img)[1], np.shape(img)[0], np.shape(img)[1], np.shape(img)[0],
         np.shape(img)[1], np.shape(img)[0], np.shape(img)[1], np.shape(img)[0],
         np.shape(img)[1], np.shape(img)[0]])
    return scale, scale_for_landmark


def get_anchors(retina):
    anchors = Anchors(retina.net_cfg, image_size=[retina.input_shape[1], retina.input_shape[0]]).get_anchors()
    return anchors


def image_process_for_detect(img, cfg):
    img = letterbox_image(img, [cfg.retina.input_shape[1], cfg.retina.input_shape[0]])
    img = preprocess_input(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
    return img


def image_process_for_recognition(img, boxes_conf_landmark, cfg):
    img = np.array(img)[int(boxes_conf_landmark[1]):int(boxes_conf_landmark[3]),
          int(boxes_conf_landmark[0]):int(boxes_conf_landmark[2])]
    if cfg.recognition.alignment:
        img = alignment(img, np.reshape(boxes_conf_landmark[5:], (5, 2)))
    # img = alignment(img, landmark)
    # img = letterbox_image(img, [cfg.recognition.input_shape[0], cfg.recognition.input_shape[0]])
    img = cv2.resize(img, [cfg.recognition.input_shape[0], cfg.recognition.input_shape[0]])
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float()
    img.div_(255).sub_(0.5).div_(0.5)
    imgs = torch.stack((img,))
    return imgs


def normalize(embeddings):
    assert len(embeddings.shape) == 2
    norm_embeddings = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / norm_embeddings


# compare dist

def compare_dist(embeddings, threshold, db_embeddings, ids):
    assert type(embeddings) == np.ndarray
    assert type(db_embeddings) == np.ndarray
    assert type(ids) == np.ndarray
    dist_matrix = 2 - 2 * embeddings @ db_embeddings.T
    min_index = np.argmin(dist_matrix, axis=-1)
    row_index = np.arange(0, dist_matrix.shape[0])
    min_dist = dist_matrix[row_index, min_index].view()
    name = ids[min_index]
    name[min_dist > threshold] = "Unknown"
    # print(min_dist)
    return name

def compare_dist_two(embeddings,embeddings_arc,threshold, db_embeddings,db_embeddings_arc, ids,max_dist,max_dist_arc):
    assert type(embeddings) == np.ndarray
    assert type(db_embeddings) == np.ndarray
    assert type(ids) == np.ndarray
    assert type(embeddings_arc) == np.ndarray
    assert type(db_embeddings_arc) == np.ndarray
    dist_matrix = (2 - 2 * embeddings @ db_embeddings.T)/max_dist
    dist_matrix_arc = (2 - 2 * embeddings_arc @ db_embeddings_arc.T)/max_dist_arc
    dist_matrix += dist_matrix_arc
    min_index = np.argmin(dist_matrix, axis=-1)
    row_index = np.arange(0, dist_matrix.shape[0])
    min_dist = dist_matrix[row_index, min_index].view()
    name = ids[min_index]
    name[min_dist > threshold] = "Unknown"
    # print(min_dist)
    return name

def preprocess_input(image):
    image -= np.array((104, 117, 123), np.float32)
    return image


# deal image by letterbox
def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(image, (nw, nh), Image.BICUBIC)
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2, (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


# decode location information
def decode(loc, priors, variances, scale):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    boxes = boxes * scale
    return boxes.cpu().numpy()


# decode landmarks location information
def decode_landmarks(pre, priors, variances, scale_for_landmarks):
    # 关键点解码
    landmarks = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                           priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                           ), dim=1)
    landmarks = landmarks * scale_for_landmarks
    landmarks = landmarks.cpu().numpy()
    return landmarks


def non_max_suppression(boxes, conf_thres=0.5, nms_thres=0.3):
    detection = boxes
    # 1、找出该图片中得分大于门限函数的框。在进行重合框筛选前就进行得分的筛选可以大幅度减少框的数量。
    mask = detection[:, 4] >= conf_thres
    detection = detection[mask]
    if not np.shape(detection)[0]:
        return []

    best_box = []
    scores = detection[:, 4]
    # 2、根据得分对框进行从大到小排序。
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]

    while np.shape(detection)[0] > 0:
        # 3、每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = iou(best_box[-1], detection[1:])
        detection = detection[1:][ious < nms_thres]

    return np.array(best_box)


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou


def alignment(img, landmark):
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算它的弧度制
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0] * landmark[i, 0] + RotationMatrix[0, 1] * landmark[i, 1] + RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0] * landmark[i, 0] + RotationMatrix[1, 1] * landmark[i, 1] + RotationMatrix[1, 2])
        new_landmark.append(pts)

    return new_img


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def plot(old_image, names, boxes_conf_landms):
    for i, b in enumerate(boxes_conf_landms):
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(old_image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(old_image, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(old_image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(old_image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(old_image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(old_image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(old_image, (b[13], b[14]), 1, (255, 0, 0), 4)

        name = names[i]

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(old_image, name, (b[0] , b[3] - 15), font, 0.75, (255, 255, 255), 2)
        # --------------------------------------------------------------#
        #   cv2不能写中文，加上这段可以，但是检测速度会有一定的下降。
        #   如果不是必须，可以换成cv2只显示英文。
        # --------------------------------------------------------------#
        old_image = cv2ImgAddText(old_image, name, b[0] + 5, b[3] - 25)
    return old_image


def cv2ImgAddText(img, label, left, top, textColor=(255, 255, 255)):
    img = Image.fromarray(np.uint8(img))
    # 设置字体
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label, 'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)


def retinaface_correct_boxes(result, input_shape, image_shape):
    # 它的作用是将归一化后的框坐标转换成原图的大小
    scale_for_offset_for_boxs = np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
    scale_for_landmarks_offset_for_landmarks = np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0],
                                                         image_shape[1], image_shape[0], image_shape[1], image_shape[0],
                                                         image_shape[1], image_shape[0]])

    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = [scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1],
                           scale[0]]

    offset_for_boxs = np.array([offset[1], offset[0], offset[1], offset[0]]) * scale_for_offset_for_boxs
    offset_for_landmarks = np.array(
        [offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1],
         offset[0]]) * scale_for_landmarks_offset_for_landmarks

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
