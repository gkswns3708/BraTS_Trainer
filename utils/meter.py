import torch
import numpy as np
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve() # 현재 파일의 절대 경로
# TODO : /mnt/hjchoi/Segmentation/BraTS/BraTS2023/Brain_Tumors_Segmentation_practice/utils/meter.py인데 여기서는 2번만 parents를 호출하는가?
# Answer: 프로젝트의 root directory를 추가하기 위해 해당 방법을 사용함
# 루트 디렉토를 추가한다면, 오류없이 모듈을 찾을 수 있음.
ROOT = FILE.parents[0].parents[0] # 현재 파일의 상위 폴더의 상위 폴더
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from metrics.metrics import dice_metric, jaccard_metric

class Meter:
    """
    Calculate IOU and Dice Scores

    Parameters
    ----------
    threshold: float = 0.5
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold: float = threshold
        self.dice_scores: list = []
        self.iou_scores: list = []

    def update(self, )