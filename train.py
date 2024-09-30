import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import gc
import nibabel as nib
import tqdm as tqdm
from utils.meter import AverageMeter
from utils.general import save_checkpoint, load_pretrained_model, resume_training
from brats import get_datasets

from monai.data import  decollate_batch
import torch
import torch.nn as nn
from torch.backends import cudnn

from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from research.models.ResUNetpp.model import ResUnetPlusPlus
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet, UNETR
from research.models.ResUNetpp.model import ResUnetPlusPlus
from research.models.UNet.model import UNet3D
from research.models.UX_Net.network_backbone import UXNET
from research.models.nnformer.nnFormer_tumor import nnFormer

from functools import partial
from utils.augment import DataAugmenter
from utils.schedulers import SegResNetScheduler, PolyDecayScheduler

# Configure logger
import logging
import hydra
from omegaconf import DictConfig

# __name__은 현재 모듈의 이름을 나타내는 내장 변수입니다.
# train.py를 직접 실행하면 __main__이 되고, 다른 모듈에서 import해서 사용하면 모듈의 이름이 됩니다.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("logger", exist_ok=True)
file_handler = logging.FileHandler(filename="logger/train.log")                           #  로그 메시지를 파일(logger/train_logger.log)에 기록하는 로그 메시지를 기록 하는 Handler
stream_handler = logging.StreamHandler()                                                  # 콘솔에 로그 메시지를 출력하는 Handler
fomatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S") # 로그 메시지의 출력 형식을 지정하는 Formatter
file_handler.setFormatter(fomatter)                                                       # 로그 메시지를 파일에 기록하는 Handler
stream_handler.setFormatter(fomatter)                                                     # 로그 메시지를 콘솔에 출력하는 Handler

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class Solver:
    """list of optimizers for training Nerual Networks"""
    def __init__(self, )

