
# Build a custom Resnet Class

# Import required libraries

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


# Define the ResNet class
class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return torch.argmax(x, dim=1)

    def predict_proba(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)

    def loss(self, x, y):
        x = self.forward(x)
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        pred = self.predict(x)
        return torch.mean((pred == y).float())

