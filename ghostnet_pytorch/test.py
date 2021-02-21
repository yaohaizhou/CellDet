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

from model import Net
from dataloader.LoadDataTrain import ReadData
from dataloader.LoadDataVal import ReadValData
from dataloader.LoadDataTest import ReadTestData
def load_model(model, optimizer, ckp_path):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--path', type=int)
    return parser.parse_args()

cfg = Config()
args = init_args()
model_save_path = args.model
test_path = args.path
# ===============================================
#            1. Load Data 
# ===============================================
print("=====================================")
print(cfg.model_type)
if test_path == 1:
    test_image_file = cfg.test_image_file
    print("Path 1")
elif test_path == 2:
    test_image_file = cfg.test2_image_file
    print("Path 2")
elif test_path == 3:
    test_image_file = cfg.train_image_file
    print("Path train")
elif test_path == 4:
    test_image_file = cfg.val_image_file
    print("Path val")
# elif test_path == 5:
#     test_image_file = cfg.test3_image_file
#     print("Path test new")

test_dataset = ReadTestData(test_image_file, cfg.test_image_size, cfg.crop_image_size)
test_data_loader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading test data")
# ===============================================
#            2. Load Model 
# ===============================================
model=Net(cfg)
loss_fc = model.loss().cuda()
validator_function = model.validator_function()
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
model = nn.DataParallel(model, cfg.device_ids)
current_epoch = 1
if model_save_path:
    if os.path.exists(model_save_path):
        model, optimizer, current_epoch, loss = load_model(model, optimizer, model_save_path)
model = model.cuda()
# ===============================================
#            3. Test model 
# ===============================================
model.eval()
correct = 0
Mesothelial_correct = 0
Cancer_correct = 0
Mesothelial_wrong = 0
Cancer_wrong = 0
total = 0
with torch.no_grad():
    col_key = []
    col_pre = []
    col_label = []

    for sample_batched in tqdm(test_data_loader):
        input_data = Variable(sample_batched['result']).cuda()
        labels = Variable(sample_batched['label']).cuda()
        paths = sample_batched['path']
        # print(sample_batched['path'])
        outputs = model(input_data)
        count_tmp, predict_index_list, Mesothelial_correct_tmp, Cancer_correct_tmp, Mesothelial_wrong_tmp, Cancer_wrong_tmp = validator_function(outputs, labels, paths)
        correct += count_tmp
        Mesothelial_correct += Mesothelial_correct_tmp
        Cancer_correct += Cancer_correct_tmp
        Mesothelial_wrong += Mesothelial_wrong_tmp
        Cancer_wrong += Cancer_wrong_tmp
        total += int(len(labels))

print("correct is ", correct)
max_test_acc = correct / total
message_test = "==> [TESTING] acc {} ".format(max_test_acc)
print(message_test)
message_res = ("Mesothelial_correct {}, Cancer_correct {}, Mesothelial_wrong {}, Cancer_wrong {}".format(Mesothelial_correct, Cancer_correct, Mesothelial_wrong, Cancer_wrong))
print(message_res)
