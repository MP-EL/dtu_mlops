import argparse
import sys

from matplotlib import pyplot as plt


from torch import nn, optim

import torch
from torch._C import Block

# from data import mnist
from models.model import MyAwesomeModel
# from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime
import os
import sys

from util import *

base_dir = os.getcwd()
print("base_dir: ", base_dir)
    
def train():
    
    model = MyAwesomeModel()
    # train_images, train_labels, _ = mnist()
    train_images, train_labels = unpack_npz("/data/processed/train.npz")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    epochs = 2
    train_losses = []

    for e in range(epochs):
        print(f'Training epoch {e}/{epochs}')
        tot_train_loss = 0
        for images, labels in zip(train_images, train_labels):
            optimizer.zero_grad()
            
            log_ps = model.forward(images.float().unsqueeze(dim=2).unsqueeze(dim=3))
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            tot_train_loss += loss.item()
        train_losses.append(tot_train_loss/len(train_images))
        print("loss: ", tot_train_loss / len(train_images))
    
    save_model(model)
    save_losses(train_losses)

if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
    
    