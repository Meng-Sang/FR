import torch
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp
from conf.retina_config import cfg_re50, cfg_mnet

used_proper = {
    "model_dir": "M2_SPLIT_R100",
    "alignment": False,
    "split": True
}


def get_config(used_proper=used_proper):
    config = edict()
    # 数据信息
    # config.source = "rtsp://admin:kmxt123456@192.168.5.100:554/h264/ch1/main/av_stream"
    # 流地址信息
    config.source = r"C:\Users\DELL\Web\RecordFiles\2022-04-09\curtain.mp4"

    # 是否保存结果，需要自己实现
    config.save = True

    # retina网络信息
    retina = edict()
    # 使用backbone网络类别
    retina.name = "mnet"
    # 检测框置信度
    retina.confidence = 0.5
    # 输入网络的检测图像大小
    retina.input_shape = [640, 640, 3]
    # 网络所在设备
    retina.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # retina网络权重文件
    retina.model_path = r"F:\Pycharm\experience\FR\models\retina\mobilenet0.25_Final.pth"
    # 使用的具体网络模型
    if retina.name == "r50":
        retina.net_cfg = cfg_re50
    elif retina.name == "mnet":
        retina.net_cfg = cfg_mnet
    else:
        raise Exception("not exist type of net")
    config.retina = retina

    # 识别网络信息
    recognition = edict()
    # 人脸识别的优化方式
    recognition.model_type = "arcface"
    # 是否使用人脸分割模型
    recognition.split = used_proper["split"]

    # 模型权重所在文件夹
    recognition.model_dir = r"F:\Pycharm\experience\FR\models\recognition"
    # 模型需要加载的设备
    recognition.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 是否对人脸进行对齐操作
    recognition.alignment = used_proper["alignment"]
    # 人脸识别的最小阈值
    recognition.threshold = 1.0
    if recognition.model_type == "arcface":
        # arcface网络输入的图片大小
        recognition.input_shape = [112, 112]
        # arcface使用的具体网络
        recognition.network = "r100"
        # 网络最后一层的参数丢弃率
        recognition.dropout = 0.0
        # 是否使用amp加速运算
        recognition.fp16 = True
        # dir format:
        #       dataset[_way]_net
        # 模型权重的具体名称
        recognition.model_name = used_proper["model_dir"] + "/model.pt"
    elif recognition.model_type == "insightface_pytorch":
        # 网络深度
        recognition.net_depth = 50
        # 网络最后一层的参数丢弃率
        recognition.drop_ratio = 0
        # 使用魔性的种类
        recognition.net_mode = "ir_se"
        # 模型权重的具体名称
        recognition.model_name = "model_2022-03-19-12-05_accuracy&0.924_step&153320_final.pth"
    else:
        raise Exception("not exist this type")

    config.recognition = recognition

    # 压缩信息
    impress = edict()
    # 压缩率
    impress.ratio = 0.7
    # 压缩的类型、可以选择根据分辨率压缩或者根据次数进行压缩
    impress.type = "resolution"  # resolution
    if impress.type == "count":
        # 进行压缩的次数
        impress.cnt = 1
    elif impress.type == "resolution":
        # 人脸图片压缩的最小阈值、人脸的大小不是整张图片的大小
        impress.resolution = [30, 30]
    # 需要进行压缩文件的文件夹
    impress.source_dir = r"F:\Pycharm\experience\FR\DB\img"
    # 压缩后进行报存的文件夹
    impress.target_dir = r"F:\Pycharm\experience\FR\DB\img_embeddings"
    config.impress = impress

    # 编码信息
    encodings = edict()
    # 需要编码的图片使用的文件夹
    encodings.source_dir = impress.target_dir
    # 编码后数据进行保存的文件夹
    encodings.target_dir = r"F:\Pycharm\experience\FR\DB\encoding"
    # sign format : dataset[_way]_net_project
    # 编码的签名、用于表示使用的数据集、网络、优化方式等
    encodings.sign = used_proper["model_dir"] + "_" + recognition.model_type + (
        "_ALIGN" if recognition.alignment else "")
    config.encodings = encodings
    return config
