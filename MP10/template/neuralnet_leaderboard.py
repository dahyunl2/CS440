# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        
        self.layer1 = nn.Linear(in_size, 64)
        self.layer2 = nn.Linear(64, out_size)
        

        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # self.layer1 = nn.Linear(64 * 5 * 5, 128)  # Adjust the input size based on your image dimensions
        # self.layer2 = nn.Linear(128, out_size)

        # # Use LeakyReLU for hidden layers
        # self.leaky_relu = nn.LeakyReLU()

        # # Use CrossEntropyLoss, which includes softmax
        # self.loss_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=lrate, weight_decay=0.00005)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)
        x = self.layer1(x)
        x = nn.LeakyReLU()(x)
        x = self.layer2(x)
        x = nn.Softmax()(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # raise NotImplementedError("You need to write this part!")
        # return 0.0
        x = (x-x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)
        self.opt.zero_grad()
        target = self.forward(x)
        loss = self.loss_fn(target, y)
        loss.backward()
        self.opt.step()
        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # raise NotImplementedError("You need to write this part!")
    # return [],[],None

    N, inSize = train_set.shape
    outSize = 4
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(0.001, loss_fn, inSize, outSize)
    losses = []
    estim = []

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 0}
    training = get_dataset_from_arrays(train_set, train_labels)

    for epoch in range(epochs):
        generate_training = torch.utils.data.DataLoader(training, **params)
        loss_ = []
        for i in generate_training:
            x = i['features']
            y = i['labels']
            loss_.append(net.step(x, y))
        loss_mean = np.mean(loss_)
        losses.append(loss_mean)

    estim = net(dev_set).argmax(dim=1).detach().numpy()

    return losses, estim, net