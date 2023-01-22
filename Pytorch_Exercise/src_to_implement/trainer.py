from typing import List, Any

import torch as t
from sklearn.metrics import f1_score
from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping

        self.epoch = 0
        self._train_losses = []
        self._val_losses = []
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO

        self._optim.zero_grad()
        self._model.zero_grad()

        y_pred = self._model(x)

        loss = self._crit(y_pred, y.float())
        loss.backward()

        self._optim.step()

        return loss

    def val_test_step(self, x, y):

        # Predict the output for the given input
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO

        y_val_pred = self._model(x)
        loss_val = self._crit(y_val_pred, y)

        return loss_val, y_val_pred

    def train_epoch(self):

        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # TODO
        train_loss = []

        for i, (x, y) in tqdm(enumerate(self._train_dl)):
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)

            if i % 200 == 0:
                print('Batch: ', i, ' Loss: ', loss)

            train_loss.append(loss)

        return sum(train_loss) / len(train_loss)

    def val_test(self):

        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO
        val_loss = []
        f1_scores = []

        self._model.eval()
        with t.no_grad():
            for i, (x, y) in tqdm(enumerate(self._val_test_dl)):
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss_val, y_val_pred = self.val_test_step(x, y)
                val_loss.append(loss_val)
                f1_val = f1_score(y, y_val_pred, average='macro')
                f1_scores.append(f1_val)

                if i % 100 == 0:
                    print('Batch: ', i, ' Loss: ', loss_val, ' F1: ', f1_val)

        print('Validation Test Call complete')
        print('F1 score: ', sum(f1_scores) / len(f1_scores))
        return sum(val_loss) / len(val_loss)

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        # TODO

        self._train_losses = []
        self._val_losses = []

        self.epoch = 0

        while True:
            print('Fit Epoch : ', self.epoch)
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            # TODO
            if 0 < epochs < self.epoch:
                break

            train_loss = self.train_epoch()
            self.val_test()

            self._train_losses.append(train_loss)
            self._val_losses.append(sum(self._val_losses) / len(self._val_losses))

            if self.epoch % 100 == 0:
                self.save_checkpoint(self.epoch)

            if self.epoch > self._early_stopping_patience:
                if self._val_losses[-1] > self._val_losses[-self._early_stopping_patience]:
                    break

            self.epoch += 1

        return self._train_losses, self._val_losses
