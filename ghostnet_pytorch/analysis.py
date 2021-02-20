import os
import torch
import torchvision
import torch.nn as nn
import argparse
from tqdm import tqdm
from datetime import datetime
from config import Config
from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
import numpy as np

from model import Net
from dataloader.LoadDataTrain import ReadData
from dataloader.LoadDataVal import ReadValData
from dataloader.LoadDataTest import ReadTestData
cfg = Config()

def get_mean_std(dataloader, ratio=0.5):
    """Get mean and std by sample ratio
    """
    # import pdb;pdb.set_trace()
    # print(iter(dataloader).next())
    train = iter(dataloader).next()['result']   # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    return mean, std

train_image_file = cfg.train_image_file
val_image_file = cfg.val_image_file
# training_dataset = ReadData(train_image_file, cfg.train_image_size)
# data_loader = DataLoader(training_dataset, batch_size=int(0.5*(len(training_dataset))),
#                                   shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
# val_dataset = ReadValData(val_image_file, cfg.train_image_size)
# data_loader = DataLoader(val_dataset, batch_size=int(0.5*(len(val_dataset))), shuffle=True,
#                              num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
# print("==> finish loading data")
# test_image_file = cfg.test_image_file
# test_image_file = cfg.test2_image_file
test_image_file = cfg.test3_image_file
test_dataset = ReadTestData(test_image_file, cfg.test_image_size)
data_loader = DataLoader(test_dataset, batch_size=int((len(test_dataset))), shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
# print("==> finish loading test data")

train_mean, train_std = get_mean_std(data_loader)
print(train_mean,train_std)