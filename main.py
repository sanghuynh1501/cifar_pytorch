import json

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1600, 512)
        print(self.fc1.weight.shape)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Layer 0
        x = F.relu(self.conv1(x))

        # Layer 1
        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Dropout(0.25)(x)

        # Layer 2
        x = F.relu(self.conv3(x))

        # Layer 3
        x = self.pool(F.relu(self.conv4(x)))
        x = nn.Dropout(0.25)(x)

        x = x.view(-1, 1600)

        # Layer 4
        x = F.relu(self.fc1(x))
        x = nn.Dropout(0.5)(x)

        # Layer 5
        x = F.softmax(self.fc2(x), dim=1)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = (img.numpy() * 255).astype("int")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


net = Net()

PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
#
isTrain = False
if isTrain:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total += 1

        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss / total))

    print('Finished Training')
    torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (float(correct) / total))

idx = 0
weights = {}
weights['weights'] = {}
for module in net.modules():
    if type(module) != Net and type(module) != nn.MaxPool2d:
        weight = module.weight
        if type(module) == nn.Linear:
            weight = weight.t()
        weights['weights']['layer_' + str(idx)] = {}

        weights['weights']['layer_' + str(idx)]["weight"] = {}
        weights['weights']['layer_' + str(idx)]["weight"]['shape'] = weight.shape
        weights['weights']['layer_' + str(idx)]["weight"]['value'] = weight.reshape(np.prod(np.array(weight.shape)),).tolist()

        weights['weights']['layer_' + str(idx)]["bias"] = {}
        if module.bias is not None:
            bias = module.bias
            bias = bias.unsqueeze(0)
            if type(module) == nn.Conv2d:
                bias = bias.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
                bias = bias.repeat(1, 1, weight.shape[2], weight.shape[3])
            print('layer_' + str(idx), weight.shape, bias.shape)
            weights['weights']['layer_' + str(idx)]["bias"]['shape'] = bias.shape
            weights['weights']['layer_' + str(idx)]["bias"]['value'] = bias.reshape(np.prod(np.array(bias.shape)), ).tolist()
        else:
            print('layer_' + str(idx), weight.shape)
            weights['weights']['layer_' + str(idx)]["bias"]['shape'] = []
            weights['weights']['layer_' + str(idx)]["bias"]['value'] = []

        idx += 1

with open('weights.json', 'w') as outfile:
    json.dump(weights, outfile)


