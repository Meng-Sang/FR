from cv2 import cv2

from conf.config import get_config


def get_stream(cfg):
    return get_single_stream(cfg)


def get_single_stream(cfg):
    cap = cv2.VideoCapture(cfg.source)
    cap.set(cv2.CAP_PROP_FOCUS,cv2.VideoWriter_fourcc(*"MJPG"))
    if not cap.isOpened():
        raise Exception("stream dont opem normally")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cfg.source, cap


if __name__ == "__main__":
    conf = get_config()
    stream_name, stream_cap = get_stream(conf)
    while stream_cap.isOpened():
        ret, frame = stream_cap.read()
        cv2.imshow(str(stream_name), frame)
        cv2.waitKey(1)