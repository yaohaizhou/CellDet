import torch                                                                                          
import time
import os
import shutil     
import re
from glob import glob                                                                                       
import torch
torch.multiprocessing.set_sharing_strategy('file_system')     

class Config(object):                                                                                
    def __init__(self):
        self.batch_size       =   1#64#
        self.eval_batch_size  =   1#64#
        self.num_workers      =   1#64#
        self.eval_num_workers =   1#64#
        self.USE_CUDA         =   torch.cuda.is_available()                                     
        self.NUM_EPOCHS       =   500
        self.evaluate_epoch   =   1 
        self.lr               =   5e-4
        self.save_path        =   "./weight/"
        self.log_path         =   "./log"
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        self.log_txt_path     =   self.log_path + "/log-"+timestr+".txt"
        self.ckp_path         =   self.save_path + "1"
        self.pin_memory       =   False
        self.load_ckp         =   None

        self.DEBUG            =   True
        self.model_type       =   "GhostNet"
        self.numclasses       =   3
        self.train_image_size =   368
        self.test_image_size  =   368        #original image size 368

        self.data_augumentation = False

        self.train_image_file  =  "/data01/zyh/CellDet/datasets/train_old"        
        self.test_image_file   =  "/data01/zyh/CellDet/datasets/test_old"

        self.device_ids       =   [0]
        self.main_gpu_id      =   0
        torch.cuda.set_device(self.main_gpu_id)

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.config_message = "Model {}, input_size {}, batch_size {}, evaluate_batch_size {}, NUM_EPOCHS {}, lr {}, device_ids {}, ckp_path {}, load_ckp {}".format(self.model_type, self.train_image_size, self.batch_size, self.eval_batch_size, self.NUM_EPOCHS, self.lr, self.device_ids, self.ckp_path, self.load_ckp)
