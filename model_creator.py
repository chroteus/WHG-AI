import math
import torch
import torch.nn as nn
import globals

def size_after_conv(h,w, kernel_size, dilation=(1,1),stride=(1,1), padding=(0,0)):
    if type(kernel_size) == int:
        kernel_size = (kernel_size,kernel_size)
    if type(dilation) == int:
        dilation = (dilation,dilation)
    if type(stride) == int:
        stride = (stride,stride)
    if type(padding) == int:
        padding = (padding,padding)

    new_h = h + (2*padding[0]) - (dilation[0]*(kernel_size[0]-1)) - 1
    new_h /= stride[0]
    new_h = math.floor(new_h + 1)

    new_w = w + (2*padding[1]) - (dilation[1]*(kernel_size[1]-1)) - 1
    new_w /= stride[1]
    new_w = math.floor(new_w + 1)

    return (new_h,new_w)


def size_after_pool(h,w,  kernel_size, dilation=(1,1), stride=False, padding=(0,0)):
    if not stride: stride = kernel_size
    return size_after_conv(h,w, kernel_size, dilation,stride,padding)

def flat_size_after_conv(conv_module, h,w):
    last_outch = -1
    for m in conv_module:
        if m.__class__.__name__ == "Conv2d":
            h,w = size_after_conv(h,w, m.kernel_size, m.dilation, m.stride, m.padding)
            last_outch = m.out_channels
        elif m.__class__.__name__ == "MaxPool2d":
            h,w = size_after_pool(h,w, m.kernel_size, m.dilation, m.stride, m.padding)
    return h*w*last_outch


# WHG model will have two inputs: Game screen itself, and delta of of movement
class WHGModel(nn.Module):
    def __init__(self, output_num=globals.OUTPUT_NUM):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            # has extra "delta" channel
            nn.Conv2d(globals.IMAGE_CHANNELS+1, 20, kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),

            nn.Conv2d(20, 25, kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),

            nn.Conv2d(25, 30, kernel_size=(3,3)),
            nn.ReLU(),
        )

        self.fc_input_size = flat_size_after_conv(self.conv, globals.IMAGE_HEIGHT, globals.IMAGE_WIDTH)

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 200),
            nn.ReLU(),

            nn.Linear(200, 200),
            nn.ReLU(),

            nn.Linear(200, output_num),
            #nn.Softmax(), # CrossE loss is used which softmaxes output for us
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,self.fc_input_size)
        return self.fc(x)


def create_model():
    return WHGModel()



class ScoreModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.Conv2d(globals.IMAGE_CHANNELS, 20, kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),

            nn.Conv2d(20, 25, kernel_size=(3,3)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),

            nn.Conv2d(25, 30, kernel_size=(3,3)),
            #nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
        )

        self.fc_input_size = flat_size_after_conv(self.conv, globals.IMAGE_HEIGHT,globals.IMAGE_WIDTH)
        output_num = 1 if globals.VALUENET_REGRESSOR else 10

        self.fc = nn.Sequential(
            nn.Linear(self.size_for_fc, 500),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(500,200),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(200,output_num), # set output to 1 when using regressor
            nn.Sigmoid()
        )
        # if globals.VALUENET_REGRESSOR:
        #     self.fc.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_input_size)
        return self.fc(x)

def create_score_model():
    return ScoreModel()
