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
import random

random_state = 42

t.manual_seed(random_state)

def worker_init_fn(worker_id):
    worker_seed = t.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

T_gen = t.Generator()
T_gen.manual_seed(random_state)

# Hyper-parameters for ResNet
testSize = 0.3
batchSize = 2
learnRate = 1e-5
wtDecay = 1e-5
Patience = 40
Epochs = 1000
val_threshold = 0.55

print(testSize, batchSize, learnRate, wtDecay, Patience, Epochs, val_threshold)

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

tab = pd.read_csv('data.csv', sep=';')
train_tab, val_tab = train_test_split(tab, test_size=testSize, random_state=random_state)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and
# ChallengeDataset objects

train_dl = t.utils.data.DataLoader(ChallengeDataset(train_tab, 'train'), batch_size=batchSize, shuffle=True, worker_init_fn=worker_init_fn, generator=T_gen)
val_dl = t.utils.data.DataLoader(ChallengeDataset(val_tab, 'val'), batch_size=batchSize, shuffle=True, worker_init_fn=worker_init_fn, generator=T_gen)

for i, (x, y) in (enumerate(train_dl)):
    if i < 5:
        print('train_dl batch: ', i, 'x: ', x[:,0,0,0])
    else:
        break
    
																																																																																

# create an instance of our ResNet model
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=learnRate, weight_decay=wtDecay)

# create an object of type Trainer and set its early stopping criterion
train_1 = Trainer(model, criterion, optimizer, train_dl=train_dl, val_test_dl=val_dl, cuda=True, early_stopping_patience=Patience, val_threshold=val_threshold)

chckpt_epoch = 0

if chckpt_epoch >0: 
    train_1.restore_checkpoint(chckpt_epoch)
    print('Restoring chkpts')
       
# go, go, go... call fit on trainer
res = train_1.fit(epochs=Epochs)

# save the model as onnx file
dtStamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
onnx_file = 'resnet_' + dtStamp + '.onnx'
train_1.save_onnx(onnx_file)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()



# save the figure with name that includes date and time
fig_file = 'loss_' + dtStamp + '.png'
onnx_file = 'resnet_' + dtStamp + '.onnx'

print('Saving Figure')
plt.savefig(fig_file)



# Model Text File
txt_file = 'Hyp_Params_' + dtStamp + '.txt'
f = open(txt_file, 'w')
f.write("Test Size = {}\n".format(testSize) +
        "Batch Size = {}\n".format(batchSize) +
        "Learning Rate = {}\n".format(learnRate) +
        "Weight Decay = {}\n".format(wtDecay) +
        "Patience = {}\n".format(Patience) +
        "Epochs = {}\n".format(Epochs) +
        "Validation Threshold = {}\n".format(val_threshold) +
        "No Dropout Layer")
f.close
