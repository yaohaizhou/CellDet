import os
import random
import shutil
from shutil import copy2

# /data01/zyh/CellDet/datasets/expr2/data
# Cancer
# Mesothelial

Cancer_DIR = "/data01/zyh/CellDet/datasets/北大胸腔第四次/578P/癌细胞/"

# for file in os.listdir(Cancer_DIR):
#     os.rename(Cancer_DIR+file,Cancer_DIR+Cancer_DIR.split("/")[-3]+"_"+file)

# for file in os.listdir(Cancer_DIR):
#     copy2(Cancer_DIR+file,"/data01/zyh/CellDet/datasets/expr2/data/Cancer")

Mesothelial_DIR = "/data01/zyh/CellDet/datasets/北大胸腔第四次/P209663N/间皮细胞/"

for file in os.listdir(Mesothelial_DIR):
    os.rename(Mesothelial_DIR+file,Mesothelial_DIR+Mesothelial_DIR.split("/")[-3]+"_"+file)

for file in os.listdir(Mesothelial_DIR):
    copy2(Mesothelial_DIR+file,"/data01/zyh/CellDet/datasets/expr2/data/Mesothelial")