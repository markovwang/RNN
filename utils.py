# utils.py
# first we define some useful functions
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F


def load_training_data(path='training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


if __name__ == '__main__':
    x, y = load_training_data()
    print(x)
