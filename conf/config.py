import torch
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp
from conf.retina_config import cfg_re50, cfg_mnet


def get_config():
    config = edict()
    # 数据信息
    # config.source = "rtsp://admin:kmxt123456@192.168.5.100:554/h264/ch1/main/av_stream"
    config.source = r"C:\Users\DELL\Web\RecordFiles\2022-04-09\curtain.mp4"

    # 检测网络信息
    retina = edict()
    retina.name = "mnet"
    retina.confidence = 0.5
    retina.input_shape = [640, 640, 3]
    retina.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    retina.model_path = r"F:\Pycharm\experience\FR\models\retina\mobilenet0.25_Final.pth"
    if retina.name == "r50":
        retina.net_cfg = cfg_re50
    elif retina.name == "mnet":
        retina.net_cfg = cfg_mnet
    else:
        raise Exception("not exist type of net")
    config.retina = retina

    # 识别网络信息
    recognition = edict()
    recognition.model_type = "arcface"
    recognition.split = True

    recognition.model_dir = r"F:\Pycharm\experience\FR\models\recognition"
    recognition.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    recognition.alignment = True

    if recognition.model_type == "arcface":
        recognition.threshold = 1.5
        recognition.input_shape = [112, 112]
        recognition.network = "r100"
        recognition.dropout = 0.0
        recognition.fp16 = True
        # dir format:
        #       dataset[_way]_net
        recognition.model_name = "M2_SPLIT_R100/model.pt"
    elif recognition.model_type == "insightface_pytorch":
        recognition.net_depth = 50
        recognition.drop_ratio = 0
        recognition.net_mode = "ir_se"
        recognition.model_name = "model_2022-03-19-12-05_accuracy&0.924_step&153320_final.pth"
        if recognition.split:
            recognition.lc_embedding_size = 16
            recognition.lc_size = 49
    else:
        raise Exception("not exist this type")

    config.recognition = recognition

    # 压缩信息
    impress = edict()
    impress.ratio = 0.7
    impress.type = "resolution"  # resolution
    if impress.type == "count":
        impress.cnt = 1
    elif impress.type == "resolution":
        impress.resolution = [30, 30]
    impress.source_dir = r"F:\Pycharm\experience\FR\DB\img"
    impress.target_dir = r"F:\Pycharm\experience\FR\DB\img_embeddings"
    config.impress = impress

    # 编码信息
    encodings = edict()
    encodings.source_dir = impress.target_dir
    encodings.target_dir = r"F:\Pycharm\experience\FR\DB\encoding"
    # sign format : dataset[_way]_net_project
    encodings.sign = "M2_SPLIT_R100_Insightface" + "_ALIGN" if recognition.alignment else ""
    config.encodings = encodings
    return config
