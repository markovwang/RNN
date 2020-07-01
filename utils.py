# utils.py
# useful functions
# don't stop a dannchouuuuuuuuu
# ₘₙⁿ
# ▏n
# █▏　､⺍
# █▏ ⺰ʷʷｨ
# █◣▄██◣
# ◥██████▋
# 　◥████ █▎
# 　　███▉ █▎
# 　◢████◣⌠ₘ℩
# 　　██◥█◣\≫
# 　　██　◥█◣
# 　　█▉　　█▊
# 　　█▊　　█▊
# 　　█▊　　█▋
# 　　 █▏　　█▙
# 　　 █
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F


def load_training_data(path='training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


def load_testing_data(path='testing_data.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        # X = [sen.split(' ') for sen in X]
    return X


def evaluation(outputs, labels):
    # if out > 0.5 means negtive
    outputs[outputs > 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


if __name__ == '__main__':
    x = load_testing_data()
    print(x[0])
