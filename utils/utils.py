import random
import os
import numpy as np
import torch
import os.path as osp
import yaml

def seed_setting(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA 10.2+
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)

def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)

def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)
        
def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path, weights_only=False)
    return None

