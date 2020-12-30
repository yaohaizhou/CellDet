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

class ReadData(Dataset):
    def __init__(self, image_root, image_size, data_augumentation=None):
        Class_Zero_Dir = image_root + "/0/"
        Class_One_Dir = image_root + "/1/"
        Class_Two_Dir = image_root + "/2/"

        pic_paths = []
        labels    = []
        Class_Zero_Files = list_all_files(Class_Zero_Dir)
        pic_paths += Class_Zero_Files
        labels += [0 for i in range(len(Class_Zero_Files))]

        Class_One_Files = list_all_files(Class_One_Dir)
        pic_paths += Class_One_Files
        labels += [1 for i in range(len(Class_One_Files))]

        Class_Two_Files = list_all_files(Class_Two_Dir)
        pic_paths += Class_Two_Files
        labels += [2 for i in range(len(Class_Two_Files))]

        print("==> [in LoadDataTrain] len(pic): {}, len(labels): {}".format(len(pic_paths), len(labels)))
        print("==> [in LoadDataTrain] num Zero: {}, num One: {}, num Two: {}".format(len(Class_Zero_Files), len(Class_One_Files), len(Class_Two_Files)))

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
                transforms.CenterCrop((250, 250)),
                transforms.Resize((picture_h_w, picture_h_w)),
                # transforms.CenterCrop((picture_h_w, picture_h_w)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(20),
                transforms.ToTensor(),
                # transforms.Normalize([0, 0, 0], [1, 1, 1]) 
            ])(temp_img)
        else:
            result = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((picture_h_w, picture_h_w)),
                # transforms.CenterCrop((picture_h_w, picture_h_w)),
                transforms.ToTensor(),
                # transforms.Normalize([0, 0, 0], [1, 1, 1]) 
            ])(temp_img)

        return {'result':result,'label':torch.LongTensor([label])}