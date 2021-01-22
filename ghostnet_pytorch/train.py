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
    return parser.parse_args()

def evaluate_on_valset(epoch, model):
    model.eval()
    correct = 0
    total = 0
    global validator_function

    with torch.no_grad():
        col_key   = []
        col_pre   = []
        col_label = []
        
        for sample_batched in tqdm(test_data_loader):
            input_data = Variable(sample_batched['result']).cuda()
            labels = Variable(sample_batched['label']).cuda()
            # import pdb;pdb.set_trace()

            outputs = model(input_data)
            average_volumns = torch.sum(outputs.data, 1)
            count_tmp, predict_index_list = validator_function(outputs, labels)
            # print(outputs,labels)
            correct += count_tmp
            total += int(len(labels))
    
    print("correct is ", correct)
    acc = correct / total
    message = "==> [evaluate on val] epoch {}, acc {} ".format(epoch, acc)
    print(message)
    with open(cfg.log_txt_path, 'a') as f:
        f.write(message + "\n")
    return acc

cfg = Config()
with open(cfg.log_txt_path, 'a') as f:
    f.write(cfg.config_message + "\n")
    print(cfg.config_message)
args = init_args()
model_save_path = args.model
# ===============================================
#            1. Load Data 
# ===============================================
train_image_file = cfg.train_image_file
test_image_file   = cfg.test_image_file
training_dataset = ReadData(train_image_file, cfg.train_image_size)
training_data_loader = DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
test_dataset = ReadTestData(test_image_file, cfg.test_image_size)
test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading data")
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
#            3. Train model 
# ===============================================
temp_max_acc = 0
for epoch in range(current_epoch,cfg.NUM_EPOCHS+1):
    print("epoch ", epoch)
    model.train()
    i_batch = 0
    correct = 0
    total_loss = 0
    total = 0
    for sample_batched in tqdm(training_data_loader):
        i_batch+=1
        input_data = Variable(sample_batched['result']).cuda()
        labels = Variable(sample_batched['label']).cuda()
        # print(input_data.shape)
        optimizer.zero_grad()
        result = model(input_data)
        # import pdb;pdb.set_trace()
        loss = loss_fc(result,labels.squeeze(1))
        total_loss+=loss
        loss.backward()
        optimizer.step()

        count_tmp, predict_index_list = validator_function(result,labels)
        correct+=count_tmp
        total+=int(len(labels))
        
    acc = correct/total
    message = "==> [train] epoch {}, total_loss {}, train_acc {}".format(epoch, total_loss, acc)
    print(message)
    with open(cfg.log_txt_path, 'a') as f:
        f.write(message + "\n")

    if epoch % cfg.evaluate_epoch == 0:
        val_acc = evaluate_on_valset(epoch,model)
        if val_acc > temp_max_acc:
            temp_max_acc = val_acc
            message = "epoch {}, acc {}, saving model".format(epoch, val_acc)
            print(message)
            with open(cfg.log_txt_path, 'a') as f:
                f.write(message + "\n")
            # if temp_max_acc > 0.2:
            #     torch.save({
            #     "epoch": epoch,
            #     "model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     'loss': loss,
            #     }, "./weight/" + cfg.model_type + "_epoch_" + str(epoch) + "_acc_" + str(val_acc) + ".tar")