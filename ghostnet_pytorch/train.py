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

from model import Net
from dataloader.LoadDataTrain import ReadData
from dataloader.LoadDataVal import ReadValData
from dataloader.LoadDataTest import ReadTestData
from tensorboardX import SummaryWriter

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


def evaluate_on_valset(epoch, model, dataloader):
    model.eval()
    correct = 0
    total = 0
    Mesothelial_correct = 0
    Cancer_correct = 0
    Mesothelial_wrong = 0
    Cancer_wrong = 0
    global validator_function

    with torch.no_grad():
        col_key = []
        col_pre = []
        col_label = []

        for sample_batched in tqdm(dataloader):
            input_data = Variable(sample_batched['result']).cuda()
            labels = Variable(sample_batched['label']).cuda()
            outputs = model(input_data)
            # average_volumns = torch.sum(outputs.data, 1)
            count_tmp, predict_index_list, Mesothelial_correct_tmp, Cancer_correct_tmp, Mesothelial_wrong_tmp, Cancer_wrong_tmp = validator_function(outputs, labels)
            # print(outputs,labels)
            correct += count_tmp
            Mesothelial_correct += Mesothelial_correct_tmp
            Cancer_correct += Cancer_correct_tmp
            Mesothelial_wrong += Mesothelial_wrong_tmp
            Cancer_wrong += Cancer_wrong_tmp
            total += int(len(labels))

    print("correct is ", correct)
    acc = correct / total
    message = "==> [evaluate on val] epoch {}, acc {} ".format(epoch, acc)
    print(message)
    with open(cfg.log_txt_path, 'a') as f:
        f.write(message + "\n")
    return acc, Mesothelial_correct/(Mesothelial_correct+Mesothelial_wrong), Cancer_correct/(Cancer_correct+Cancer_wrong)

writer = SummaryWriter()
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
val_image_file = cfg.val_image_file
training_dataset = ReadData(train_image_file, cfg.train_image_size, cfg.crop_image_size)
training_data_loader = DataLoader(training_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
val_dataset = ReadValData(val_image_file, cfg.train_image_size, cfg.crop_image_size)
val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading data")
# test_image_file = cfg.test2_image_file
test_image_file = cfg.test_image_file
test_dataset = ReadTestData(test_image_file, cfg.test_image_size, cfg.crop_image_size)
test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
print("==> finish loading test data")
# ===============================================
#            2. Load Model
# ===============================================
model = Net(cfg)
loss_fc = model.loss().cuda()
validator_function = model.validator_function()
# optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
model = nn.DataParallel(model, cfg.device_ids)
current_epoch = 1
if model_save_path:
    if os.path.exists(model_save_path):
        model, optimizer, current_epoch, loss = load_model(model, optimizer, model_save_path)
model = model.cuda()
# ===============================================
#            3. Train model
# ===============================================
max_val_acc = 0
for epoch in range(current_epoch, cfg.NUM_EPOCHS+1):
    print("epoch ", epoch)
    model.train()
    i_batch = 0
    correct = 0
    total_loss = 0
    total = 0
    for sample_batched in tqdm(training_data_loader):
        i_batch += 1
        input_data = Variable(sample_batched['result']).cuda()
        labels = Variable(sample_batched['label']).cuda()
        # print(input_data.shape)
        optimizer.zero_grad()
        result = model(input_data)
        loss = loss_fc(result, labels.squeeze(1))
        # import pdb;pdb.set_trace()
        total_loss += loss
        loss.backward()
        optimizer.step()

        count_tmp, predict_index_list, _, _, _, _ = validator_function(result, labels)
        correct += count_tmp
        total += int(len(labels))
    acc = correct/total
    writer.add_scalar('data/loss', total_loss, epoch)
    writer.add_scalar('data/train_acc', acc, epoch)

    message = "==> [train] epoch {}, total_loss {}, train_acc {}".format(
        epoch, total_loss, acc)
    print(message)
    with open(cfg.log_txt_path, 'a') as f:
        f.write(message + "\n")

    if epoch % cfg.evaluate_epoch == 0:
        val_acc,Mesothelial_acc, Cancer_acc = evaluate_on_valset(epoch, model, val_data_loader)#test_data_loader val_data_loader
        writer.add_scalar('data/val_acc', val_acc, epoch)
        writer.add_scalar('data/Mesothelial_acc', Mesothelial_acc, epoch)
        writer.add_scalar('data/Cancer_acc', Cancer_acc, epoch)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_train_acc = acc
            best_model_epoch = epoch
            best_model = copy.deepcopy(model)
            message = "epoch {}, acc {}, saving model".format(epoch, val_acc)
            print(message)
            with open(cfg.log_txt_path, 'a') as f:
                f.write(message + "\n")
            if max_val_acc > 0.95:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    'loss': loss,
                }, cfg.save_path + cfg.model_type + "_epoch_" + str(epoch) + "_acc_" + str(val_acc) + ".tar")
writer.export_scalars_to_json("./boardlog/all_scalars.json")
writer.close()
# ===============================================
#            4. Test model
# ===============================================
# test_image_file = cfg.test_image_file
# # test_image_file = cfg.test2_image_file
# test_dataset = ReadTestData(test_image_file, cfg.test_image_size)
# test_data_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
#                               num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory)
# print("==> finish loading test data")

model = best_model
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
        outputs = model(input_data)
        # average_volumns = torch.sum(outputs.data, 1)
        count_tmp, predict_index_list, Mesothelial_correct_tmp, Cancer_correct_tmp, Mesothelial_wrong_tmp, Cancer_wrong_tmp = validator_function(outputs, labels)
        correct += count_tmp
        Mesothelial_correct += Mesothelial_correct_tmp
        Cancer_correct += Cancer_correct_tmp
        Mesothelial_wrong += Mesothelial_wrong_tmp
        Cancer_wrong += Cancer_wrong_tmp
        total += int(len(labels))

print("correct is ", correct)
max_test_acc = correct / total
message_test = "==> [TESTING] epoch {}, acc {} ".format(epoch, max_test_acc)
print(message_test)
message_best_model = ("==> [Best Model] epoch {}, train_acc {}, val_acc {}, test_acc {}".format(
    best_model_epoch, max_train_acc, max_val_acc, max_test_acc))
print(message_best_model)
message_res = ("Mesothelial_correct {}, Cancer_correct {}, Mesothelial_wrong {}, Cancer_wrong {}".format(Mesothelial_correct, Cancer_correct, Mesothelial_wrong, Cancer_wrong))
print(message_res)

with open(cfg.log_txt_path, 'a') as f:
    f.write(message_test + "\n")
    f.write(message_best_model + "\n")
    f.write(message_res + "\n")
