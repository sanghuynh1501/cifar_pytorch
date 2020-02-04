import json
import base64
import os
import numpy as np
from PIL import Image, ImageFilter

import torch
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from optimizer import PlainRAdam
from load_data import load_cifar_10_data, generation_data

batch_size = 100

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512 * 2 * 2, 512, bias=False)
        self.classifier = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           # nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
        return nn.Sequential(*layers)


def imshow(img):
    img = img / 2+0.5     # unnormalize
    npimg = (img.numpy() * 255).astype("int")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def array2base64(data):
    dataStr = json.dumps(data).encode()
    base64EncodedStr = base64.b64encode(dataStr)
    return base64EncodedStr.decode()[:5]

net = VGG('VGG11')
summary(net, (3, 32, 32))
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))

cifar_10_dir = 'data/cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar_10_data()

print("Test data: ", train_data.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test labels: ", test_labels.shape)

isTrain = False

if isTrain:
    criterion = nn.CrossEntropyLoss()
    optimizer = PlainRAdam(net.parameters())

    for epoch in range(15):  # loop over the dataset multiple times

        running_loss =0.0
        total = 0
        with tqdm(total=50000) as pbar:
            for i, data in enumerate(generation_data(batch_size, train_data, train_labels), 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = torch.from_numpy(inputs).float()
                labels = torch.from_numpy(labels)
                # # zero the parameter gradients
                # # zero the parameter gradients
                optimizer.zero_grad()
                #
                # # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                total += 1
                # if i >= 500:
                #     break
                pbar.update(batch_size)

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / total))

        print('Finished Training')
        torch.save(net.state_dict(), PATH)

# correct = 0
# total = 0
# dem = 0

# net.eval()
# with torch.no_grad():
#     for data in generation_data(batch_size, test_data, test_labels):
#         images, labels = data
#         inputs = torch.from_numpy(images).float()
#         labels = torch.from_numpy(labels)
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         dem += batch_size
#         # if dem >= 10000:
#         #     break

# print('Accuracy of the network on the 10000 test images: %f %%' % (float(correct) / total))

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
        weight_object = {}
        weight_object['shape'] = shape
        weight_object['value'] = weight.reshape(np.prod(np.array(weight.shape)),).tolist()
        weights['weights'].append(weight_object)
        idx += 1

with open('weights.json', 'w') as outfile:
    json.dump(weights, outfile)