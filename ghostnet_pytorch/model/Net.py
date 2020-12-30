import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from .ghostnet import ghostnet

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.backbone = ghostnet()

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
# input = torch.randn(1,3,320,256)
# output = model(input)
# print(output.size())#torch.Size([32, 3])
# loss_fc = model.loss()
# print(loss_fc(output.squeeze(1),output.squeeze(1)))