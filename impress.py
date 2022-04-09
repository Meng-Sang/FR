from conf.config import get_config
from util import BatchCut

cfg = get_config()
cut = BatchCut(cfg.impress)
cut.cutImage()
