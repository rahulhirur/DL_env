import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Hyper-parameters for ResNet
testSize = 0.3
batchSize = 2
learnRate = 0.001
wtDecay = 0.0001
Patience = 50
Epochs = 1000
val_threshold = 0.6

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

tab = pd.read_csv('data.csv', sep=';')
train_tab, val_tab = train_test_split(tab, test_size=testSize, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and
# ChallengeDataset objects

train_dl = t.utils.data.DataLoader(ChallengeDataset(train_tab, 'train'), batch_size=batchSize, shuffle=True)
val_dl = t.utils.data.DataLoader(ChallengeDataset(val_tab, 'val'), batch_size=batchSize, shuffle=True)

# create an instance of our ResNet model
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=learnRate, weight_decay=wtDecay)

# create an object of type Trainer and set its early stopping criterion
train_1 = Trainer(model, criterion, optimizer, train_dl=train_dl, val_test_dl=val_dl, cuda=True, early_stopping_patience=Patience, val_threshold=val_threshold))

# go, go, go... call fit on trainer
res = train_1.fit(epochs=Epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()

dtStamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# save the figure with name that includes date and time
fig_file = 'loss_' + dtStamp + '.png'
onnx_file = 'resnet_' + dtStamp + '.onnx'

plt.savefig(fig_file)

# save the model as onnx file
train_1.save_onnx(onnx_file)

# Model Text File
txt_file = dtStamp + '.txt'
lines = [testSize, batchSize, learnRate, wtDecay, Patience, Epochs]
with open(txt_file, 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
f.close
