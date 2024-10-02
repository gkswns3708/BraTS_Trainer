import os
import random


import torch
import torch.nn as nn
from torch.backends import cudnn

import numpy as np


def save_best_model(dir_name, model, name="best_model"):
    """save best model weights"""
    save_path = os.path.join(dir_name, name)
    torch.save(model.state_dict(), f"{save_path}/{name}.pkl")
    
def save_checkpoint(dir_name, state, name="checkpoint"):
    """save checkpoint with each epoch to resume"""
    save_path = os.path.join(dir_name, name)
    torch.save(state, f"{save_path}/{name}.pth.tar")
 
def compute_loss(loss, preds, label):
        loss = loss(preds[0], label)
        for i, pred in enumerate(preds[1:]):
            downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
            loss += 0.5 ** (i + 1) * loss(pred, downsampled_label)
        c_norm = 1 / (2 - 2 ** (-len(preds)))
        return c_norm * loss

def create_dirs(dir_name):
    """create experiment directory storing
    checkpoint and best weights"""
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "best-model"), exist_ok=True)

def init_random(seed):
    """randomly initialize some options"""
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False         
    cudnn.deterministic = True