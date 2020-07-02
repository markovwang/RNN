# test.py
# 這個block用來對testing_data.txt做預測
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大於等於0.5為負面
            outputs[outputs < 0.5] = 0  # 小於0.5為正面
            ret_output += outputs.int().tolist()

    return ret_output