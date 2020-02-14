
import json
import torch
import torch.nn as nn

import numpy as np
from model import VGG
from torchsummary import summary

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

net = VGG('VGG13')
summary(net, (3, 32, 32))
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH, map_location))

idx = 0
weights = {}
weights['weights'] = []

for module in net.modules():
    if type(module) != VGG and type(module) != nn.MaxPool2d and type(module) != nn.Sequential \
            and type(module) != nn.ReLU and type(module) != nn.BatchNorm2d:
        weight = module.weight
        if type(module) == nn.Linear:
            weight = weight.t()

        shape = list(weight.shape)
        print("shape ", shape)
        weight_object = {}
        weight_object['shape'] = shape
        weight_object['value'] = weight.reshape(np.prod(np.array(weight.shape)),).tolist()
        weights['weights'].append(weight_object)
        idx += 1

with open('weights.json', 'w') as outfile:
    json.dump(weights, outfile)
