import os
import numpy as np
import glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def list_all_files(rootdir):
    # 返回某目录下的所以文件（包括子目录下的）
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

class ReadValData(Dataset):
    def __init__(self, image_root, image_size, data_augumentation=None):
        Class_Mesothelial_Dir = image_root + "/Mesothelial/"
        Class_Cancer_Dir = image_root + "/Cancer/"

        pic_paths = []
        labels    = []
        Class_Mesothelial_Files = list_all_files(Class_Mesothelial_Dir)
        pic_paths += Class_Mesothelial_Files
        labels += [0 for i in range(len(Class_Mesothelial_Files))]

        Class_Cancer_Files = list_all_files(Class_Cancer_Dir)
        pic_paths += Class_Cancer_Files
        labels += [1 for i in range(len(Class_Cancer_Files))]


        print("==> [in LoadDataVal] len(pic): {}, len(labels): {}".format(len(pic_paths), len(labels)))
        print("==> [in LoadDataVal] num Mesothelial: {}, num Cancer: {}".format(len(Class_Mesothelial_Files), len(Class_Cancer_Files)))

        self.data = [(pic_path, label) for pic_path, label in zip(pic_paths, labels)]
        self.data_augumentation = data_augumentation
        self.image_size = image_size
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        (path, label) = self.data[idx]
        temp_img = Image.open(path)
        picture_h_w = self.image_size
        if self.data_augumentation:
            result = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.CenterCrop((300, 300)),
                transforms.Resize((picture_h_w, picture_h_w)),
                # transforms.CenterCrop((picture_h_w, picture_h_w)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize([0.8025915], [0.16253494])
            ])(temp_img)
        else:
            result = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((picture_h_w, picture_h_w)),
                # transforms.CenterCrop((picture_h_w, picture_h_w)),
                transforms.ToTensor(),
                transforms.Normalize([0.8025915], [0.16253494])
            ])(temp_img)

        return {'result':result,'label':torch.LongTensor([label])}