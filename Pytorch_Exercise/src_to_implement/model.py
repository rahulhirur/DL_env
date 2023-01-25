# Build a custom Resnet Class

# Import required libraries

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


# Define a sub Res block class which contains a convolution layer, batch normalization layer and a relu activation function

class SubResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.res_sub_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.res_sub_block(x)


# Define a ResBlock class

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fwd_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.fwd_batch_norm = nn.BatchNorm2d(out_channels)
        self.fwd_relu = nn.ReLU()

        self.res_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU()
                                       )

    # Define a ResBlock class forward function
    def forward(self, x):
        val1 = self.fwd_conv1(x)
        val1 = self.fwd_batch_norm(val1)

        val2 = self.res_block(x)
        return self.fwd_relu(val2 + val1)


# Define a ResBlock class with drop out layer in the first sub block

class ResBlock_d1(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, drop_out=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fwd_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.fwd_batch_norm = nn.BatchNorm2d(out_channels)
        self.fwd_relu = nn.ReLU()

        self.res_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_out),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU()
                                       )

    # Define a ResBlock class forward function
    def forward(self, x):
        val1 = self.fwd_conv1(x)
        val1 = self.fwd_batch_norm(val1)

        val2 = self.res_block(x)
        return self.fwd_relu(val2 + val1)

# Define a ResBlock class with drop out layer in the second sub block

class ResBlock_d2(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, drop_out=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fwd_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.fwd_batch_norm = nn.BatchNorm2d(out_channels)
        self.fwd_relu = nn.ReLU()

        self.res_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_out)
                                       )

    # Define a ResBlock class forward function
    def forward(self, x):
        val1 = self.fwd_conv1(x)
        val1 = self.fwd_batch_norm(val1)

        val2 = self.res_block(x)
        return self.fwd_relu(val2 + val1)

# Define a ResBlock class with drop out layer in both the sub blocks

class ResBlock_d3(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, drop_out=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.fwd_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.fwd_batch_norm = nn.BatchNorm2d(out_channels)
        self.fwd_relu = nn.ReLU()

        self.res_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_out),
                                       nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(),
                                       nn.Dropout(p=drop_out)
                                       )

    # Define a ResBlock class forward function
    def forward(self, x):
        val1 = self.fwd_conv1(x)
        val1 = self.fwd_batch_norm(val1)

        val2 = self.res_block(x)
        return self.fwd_relu(val2 + val1)


# Define a ResNet class with custom ResBlock

class ResNet(nn.Module):

    # Define a ResNet class constructor
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ResBlock(in_channels=64, out_channels=64, stride=1),
                                   ResBlock(in_channels=64, out_channels=128, stride=2),
                                   ResBlock(in_channels=128, out_channels=256, stride=2),
                                   ResBlock(in_channels=256, out_channels=512, stride=2),
                                   nn.AvgPool2d(kernel_size=10, stride=1),
                                   nn.Flatten(),
                                   nn.Linear(512, 2),
                                   nn.Sigmoid()
                                   )

    # Define a ResNet class forward function
    def forward(self, x):
        return self.model(x)
