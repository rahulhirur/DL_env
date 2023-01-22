import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules

# TODO

tab = pd.read_csv('data.csv', sep=';')
train_tab, val_tab = train_test_split(tab, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO

train_dl = t.utils.data.DataLoader(ChallengeDataset(train_tab, 'train'), batch_size=10)
val_dl = t.utils.data.DataLoader(ChallengeDataset(val_tab, 'val'), batch_size=10)

# create an instance of our ResNet model
# TODO

model = model.ResNet()


# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)

# create an object of type Trainer and set its early stopping criterion
train_1 = Trainer(model, criterion, optimizer, train_dl=train_dl, val_test_dl=val_dl, cuda=True, early_stopping_patience=20)
# TODO

# go, go, go... call fit on trainer
res = train_1.fit(epochs=1000)

# #TODO


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

train_1.save_onnx('checkpoints/')
