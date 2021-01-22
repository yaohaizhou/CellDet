import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from .ghostnet import ghostnet
from .MobileNetV3 import MobileNetV3_Small
from .ResNet import resnet18
# import sys
# sys.path.append("../")
# from config import Config

class Net(nn.Module):
    def __init__(self,Config):
        super(Net,self).__init__()
        if Config.model_type == "GhostNet":
            self.backbone = ghostnet()
        elif Config.model_type == "MobileNetV3":
            self.backbone = MobileNetV3_Small()
        elif Config.model_type == "Resnet18":
            self.backbone = resnet18()

        #function to initialize the weights and biases of each module. Matches the
        #classname with a regular expression to determine the type of the module, then
        #initializes the weights for it.
        def weights_init(m):
            classname = m.__class__.__name__
            if re.search("Conv[123]d", classname):
                m.weight.data.normal_(0.0, 0.02)
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                m.bias.data.fill_(0)
        #Apply weight initialization to every module in the model.
        self.apply(weights_init)

    def forward(self, input):
        output = self.backbone(input)
        return output
    
    def loss(self):
        return self.backbone.loss
    
    def validator_function(self):
        return self.backbone.validator

# model=Net()
# # print(model)
# input = torch.randn(2,3,224,224)
# output = model(input)
# print(output.size())#torch.Size([32, 3])

# loss_fc = model.loss()
# print(loss_fc(output.squeeze(1),output.squeeze(1)))