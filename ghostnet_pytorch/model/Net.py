import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from .ghostnet import ghostnet
from .MobileNetV3New import mobilenet
from .ResNet import resnet18

class Net(nn.Module):
    def __init__(self,Config):
        super(Net,self).__init__()
        if Config.model_type == "GhostNet":
            self.backbone = ghostnet()
        elif Config.model_type == "MobileNetV3_Small":
            self.backbone = mobilenet(mymode='small')
        elif Config.model_type == "MobileNetV3_Large":
            self.backbone = mobilenet(mymode='large')
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
        if Config.model_type is "MobileNetV3_Small" or "MobileNetV3_Large":
            pass
        else:
            self.apply(weights_init)

    def forward(self, input):
        output = self.backbone(input)
        return output
    
    def loss(self):
        return self.backbone.loss
    
    def validator_function(self):
        return self.backbone.validator


# import sys,os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
# from config import Config
# cfg = Config()
# model=Net(cfg)
# # from torchsummary import summary
# # summary(model, input_size=(1, 500, 500), device="cpu")
# input = torch.randn(2,1,500,500)
# output = model(input)
# print(output.size())#torch.Size([2, 2])

# loss_fc = model.loss()
# print(loss_fc(output.squeeze(1),output.squeeze(1)))