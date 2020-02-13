import json
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from model import VGG

from tqdm import tqdm

import numpy as np

from optimizer import PlainRAdam
from load_data import load_cifar_10_data, generation_data

batch_size = 32

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = VGG('VGG13')
summary(net, (3, 32, 32))
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))

cifar_10_dir = 'data/cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar_10_data()

print("Test data: ", train_data.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test labels: ", test_labels.shape)

criterion = nn.CrossEntropyLoss()
optimizer = PlainRAdam(net.parameters())

for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    total = 0
    with tqdm(total=50000) as pbar:
        for i, data in enumerate(generation_data(batch_size, train_data, train_labels), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels)

            # # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total += 1
            pbar.update(batch_size)

    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss / total))

    print('Finished Training')
    torch.save(net.state_dict(), PATH)

correct = 0
total = 0
dem = 0

net.eval()
with torch.no_grad():
    with tqdm(total=10000) as pbar:
      for data in generation_data(batch_size, test_data, test_labels):
          images, labels = data
          inputs = torch.from_numpy(images).float()
          labels = torch.from_numpy(labels)
          outputs = net(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          dem += batch_size
          pbar.update(1)

print('Accuracy of the network on the 10000 test images: %f %%' % (float(correct) / total))